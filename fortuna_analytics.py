#!/usr/bin/env python3
"""
fortuna_analytics.py
Race result harvesting and performance analysis engine for Fortuna.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Set, Tuple, Type
from zoneinfo import ZoneInfo

import structlog
from pydantic import Field, model_validator
from selectolax.parser import HTMLParser, Node

import fortuna

# -- CONSTANTS ----------------------------------------------------------------

EASTERN = ZoneInfo("America/New_York")
DEFAULT_DB_PATH: Final[str] = os.environ.get("FORTUNA_DB_PATH", "fortuna.db")
STANDARD_BET: Final[float] = 2.00

PLACE_POSITIONS_BY_FIELD_SIZE: Final[Dict[int, int]] = {
    4: 1,           # ≤4 runners: win only
    7: 2,           # 5-7 runners: top 2
    10000: 3,       # 8+: top 3
}

_currency_logger = structlog.get_logger("currency_parser")

_CASHED_VERDICTS: Final[frozenset] = frozenset({"CASHED", "CASHED_ESTIMATED"})
_LOSS_VERDICTS:   Final[frozenset] = frozenset({"BURNED"})

_VERDICT_DISPLAY: Final[Dict[str, str]] = {
    "CASHED":           "✅ WIN ",
    "CASHED_ESTIMATED": "✅ WIN~",
    "BURNED":           "❌ LOSS",
    "VOID":             "⚪ VOID",
}


# -- HELPER FUNCTIONS ---------------------------------------------------------

def now_eastern() -> datetime:
    return datetime.now(EASTERN)


def to_eastern(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=EASTERN)
    return dt.astimezone(EASTERN)


def get_optimal_region_at_time(dt: datetime) -> str:
    """US 9am–11pm ET, otherwise international."""
    et_hour = dt.astimezone(EASTERN).hour
    return "USA" if 9 <= et_hour < 23 else "INT"


def parse_position(pos_str: Optional[str]) -> Optional[int]:
    """``'1st'`` → 1, ``'2/12'`` → 2, ``'W'`` → 1, etc."""
    if not pos_str:
        return None
    s = str(pos_str).upper().strip()
    direct = {
        "W": 1, "1": 1, "1ST": 1,
        "P": 2, "2": 2, "2ND": 2,
        "S": 3, "3": 3, "3RD": 3,
        "4": 4, "4TH": 4,
        "5": 5, "5TH": 5,
    }
    if s in direct:
        return direct[s]
    m = re.search(r"^(\d+)", s)
    return int(m.group(1)) if m else None


def get_places_paid(field_size: int) -> int:
    for max_size, places in sorted(PLACE_POSITIONS_BY_FIELD_SIZE.items()):
        if field_size <= max_size:
            return places
    return 3


def parse_currency_value(value_str: str) -> float:
    """``'$1,234.56'`` → 1234.56"""
    if not value_str:
        return 0.0
    try:
        raw = str(value_str).strip()
        # Allow standard currency symbols and codes (GBP, EUR, USD, ZAR)
        if re.search(r"[^\d.,$£€\sA-Z]", raw):
            _currency_logger.debug("unexpected_currency_format", value=raw)
            # If it contains truly invalid characters for currency, return 0
            if re.search(r"[^\d.,$£€\sA-Z\-]", raw):
                 return 0.0

        if "," in raw and "." in raw:
            if raw.rfind(",") > raw.rfind("."):
                # European style: 1.234,56
                cleaned = raw.replace(".", "").replace(",", ".")
            else:
                # US style: 1,234.56
                cleaned = raw.replace(",", "")
        elif "," in raw and "." not in raw and re.search(r",\d{2}$", raw):
            # European format: 12,34
            cleaned = raw.replace(",", ".")
        else:
            cleaned = raw.replace(",", "")

        cleaned = re.sub(r"[^\d.]", "", cleaned)
        return float(cleaned) if cleaned else 0.0
    except (ValueError, TypeError):
        _currency_logger.warning("failed_parsing_currency", value=value_str)
        return 0.0


def validate_date_format(date_str: str) -> bool:
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


# -- MODELS -------------------------------------------------------------------

class ResultRunner(fortuna.Runner):
    """Runner extended with finishing position and payouts."""

    position: Optional[str] = None
    position_numeric: Optional[int] = None
    final_win_odds: Optional[float] = None
    win_payout: Optional[float] = None
    place_payout: Optional[float] = None
    show_payout: Optional[float] = None

    @model_validator(mode="after")
    def compute_position_numeric(self) -> ResultRunner:
        if self.position and self.position_numeric is None:
            self.position_numeric = parse_position(self.position)
        return self


class ResultRace(fortuna.Race):
    """Race with full result data."""

    runners: List[ResultRunner] = Field(default_factory=list)
    official_dividends: Dict[str, float] = Field(default_factory=dict)
    chart_url: Optional[str] = None
    is_fully_parsed: bool = False

    trifecta_payout: Optional[float] = None
    trifecta_cost: float = 1.00
    trifecta_combination: Optional[str] = None
    exacta_payout: Optional[float] = None
    exacta_combination: Optional[str] = None
    superfecta_payout: Optional[float] = None
    superfecta_combination: Optional[str] = None

    @property
    def canonical_key(self) -> str:
        d = self.start_time.strftime("%Y%m%d")
        t = self.start_time.strftime("%H%M")
        disc = (self.discipline or "T")[:1].upper()
        return f"{fortuna.get_canonical_venue(self.venue)}|{self.race_number}|{d}|{t}|{disc}"

    @property
    def relaxed_key(self) -> str:
        d = self.start_time.strftime("%Y%m%d")
        disc = (self.discipline or "T")[:1].upper()
        return f"{fortuna.get_canonical_venue(self.venue)}|{self.race_number}|{d}|{disc}"

    def get_top_finishers(self, n: int = 5) -> List[ResultRunner]:
        ranked = [r for r in self.runners if r.position_numeric is not None]
        ranked.sort(key=lambda r: r.position_numeric)
        return ranked[:n]


# -- AUDITOR ENGINE -----------------------------------------------------------

class AuditorEngine:
    """Matches predicted tips against actual race results via SQLite."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db = fortuna.FortunaDB(db_path or DEFAULT_DB_PATH)
        self.logger = structlog.get_logger(self.__class__.__name__)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    # -- data access -------------------------------------------------------

    async def get_unverified_tips(
        self, lookback_hours: int = 48,
    ) -> List[Dict[str, Any]]:
        return await self.db.get_unverified_tips(lookback_hours)

    async def get_all_audited_tips(self) -> List[Dict[str, Any]]:
        return await self.db.get_all_audited_tips()

    async def get_recent_tips(self, limit: int = 15) -> List[Dict[str, Any]]:
        return await self.db.get_recent_tips(limit)

    async def close(self) -> None:
        await self.db.close()

    # -- audit pipeline ----------------------------------------------------

    async def audit_races(
        self,
        results: List[ResultRace],
        unverified: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        results_map = self._build_results_map(results)
        self.logger.debug("Built results map", count=len(results_map))

        if unverified is None:
            unverified = await self.get_unverified_tips()

        audited: List[Dict[str, Any]] = []
        outcomes_to_batch: List[Tuple[str, Dict[str, Any]]] = []

        for tip in unverified:
            try:
                race_id = tip.get("race_id")
                if not race_id:
                    continue

                tip_key = self._tip_canonical_key(tip)
                if not tip_key:
                    continue

                result = self._match_tip_to_result(tip_key, results_map, race_id)
                if not result:
                    continue

                outcome = self._evaluate_tip(tip, result)
                outcomes_to_batch.append((race_id, outcome))
                audited.append({**tip, **outcome, "audit_completed": True})

            except Exception as exc:
                self.logger.error(
                    "Error during audit",
                    tip_id=tip.get("race_id"),
                    error=str(exc),
                    exc_info=True,
                )

        if outcomes_to_batch:
            self.logger.info("Batch updating audit results", count=len(outcomes_to_batch))
            await self.db.update_audit_results_batch(outcomes_to_batch)

        return audited

    @staticmethod
    def _build_results_map(
        results: List[ResultRace],
    ) -> Dict[str, ResultRace]:
        mapping: Dict[str, ResultRace] = {}
        log = structlog.get_logger("AuditorEngine")
        for r in results:
            # Canonical key: full precision
            mapping[r.canonical_key] = r

            if r.relaxed_key != r.canonical_key:
                # Relaxed key: Venue|Race|Date|Disc (no time)
                if r.relaxed_key in mapping:
                    existing = mapping[r.relaxed_key]
                    if existing.canonical_key != r.canonical_key:
                        log.debug(
                            "Relaxed key collision",
                            key=r.relaxed_key,
                            existing=existing.canonical_key,
                            new=r.canonical_key,
                        )
                        # Prefer existing canonical over new relaxed if collision
                        continue
                mapping[r.relaxed_key] = r
        return mapping


    def _match_tip_to_result(
        self,
        tip_key: str,
        results_map: Dict[str, ResultRace],
        race_id: str,
    ) -> Optional[ResultRace]:
        # Exact match
        result = results_map.get(tip_key)
        if result:
            return result

        parts = tip_key.split("|")

        # Fallback 1: drop time (keep discipline)
        if len(parts) >= 5:
            relaxed = f"{parts[0]}|{parts[1]}|{parts[2]}|{parts[4]}"
            result = results_map.get(relaxed)
            if result:
                self.logger.info(
                    "Time-relaxed fallback match",
                    race_id=race_id,
                    match_key=result.canonical_key,
                )
                return result

        # Fallback 2: drop discipline (keep time)
        if len(parts) >= 4:
            prefix = "|".join(parts[:4])
            matches = [obj for key, obj in results_map.items() if key.startswith(prefix)]
            if matches:
                if len(matches) > 1:
                    self.logger.warning(
                        "Multiple discipline fallback matches",
                        race_id=race_id,
                        count=len(matches),
                    )
                # Deterministic return: the one with exact time match if available
                return matches[0]

        return None

    # -- key generation ----------------------------------------------------

    @staticmethod
    def _tip_canonical_key(tip: Dict[str, Any]) -> Optional[str]:
        venue = tip.get("venue")
        race_number = tip.get("race_number")
        start_raw = tip.get("start_time")
        disc = (tip.get("discipline") or "T")[:1].upper()

        if not all([venue, race_number, start_raw]):
            return None
        try:
            st = datetime.fromisoformat(
                str(start_raw).replace("Z", "+00:00"),
            )
            return (
                f"{fortuna.get_canonical_venue(venue)}"
                f"|{race_number}"
                f"|{st.strftime('%Y%m%d')}"
                f"|{st.strftime('%H%M')}"
                f"|{disc}"
            )
        except (ValueError, TypeError):
            return None

    # -- evaluation --------------------------------------------------------

    def _evaluate_tip(
        self, tip: Dict[str, Any], result: ResultRace,
    ) -> Dict[str, Any]:
        selection_num = self._extract_selection_number(tip)
        selection_name = tip.get("selection_name")

        top_finishers = result.get_top_finishers(5)
        actual_top_5 = [str(r.number) for r in top_finishers]

        top1_place = (
            top_finishers[0].place_payout if len(top_finishers) >= 1 else None
        )
        top2_place = (
            top_finishers[1].place_payout if len(top_finishers) >= 2 else None
        )

        actual_2nd_fav_odds = self._find_actual_2nd_fav_odds(result)

        # Find our selection in result runners
        sel_result = self._find_selection_runner(
            result, selection_num, selection_name,
        )

        verdict, profit = self._compute_verdict(sel_result, result)

        return {
            "actual_top_5": ", ".join(actual_top_5),
            "actual_2nd_fav_odds": actual_2nd_fav_odds,
            "verdict": verdict,
            "net_profit": round(profit, 2),
            "selection_position": (
                sel_result.position_numeric if sel_result else None
            ),
            "audit_timestamp": datetime.now(EASTERN).isoformat(),
            "trifecta_payout": result.trifecta_payout,
            "trifecta_combination": result.trifecta_combination,
            "superfecta_payout": result.superfecta_payout,
            "superfecta_combination": result.superfecta_combination,
            "top1_place_payout": top1_place,
            "top2_place_payout": top2_place,
        }

    @staticmethod
    def _find_actual_2nd_fav_odds(result: ResultRace) -> Optional[float]:
        runners_list = sorted(
            (
                r for r in result.runners
                if r.final_win_odds and r.final_win_odds > 0 and not r.scratched
            ),
            key=lambda r: r.final_win_odds,
        )
        if len(runners_list) < 2:
            return None
        fav_odds = runners_list[0].final_win_odds
        higher = [r for r in runners_list if r.final_win_odds > fav_odds]
        return higher[0].final_win_odds if higher else None

    @staticmethod
    def _find_selection_runner(
        result: ResultRace,
        number: Optional[int],
        name: Optional[str],
    ) -> Optional[ResultRunner]:
        if number is not None:
            by_num = next(
                (r for r in result.runners if r.number == number), None,
            )
            if by_num:
                return by_num
        if name:
            return next(
                (
                    r for r in result.runners
                    if r.name.lower() == name.lower()
                ),
                None,
            )
        return None

    @staticmethod
    def _compute_verdict(
        sel: Optional[ResultRunner],
        result: ResultRace,
    ) -> Tuple[str, float]:
        if sel is None:
            return "VOID", 0.0
        if sel.position_numeric is None:
            return "BURNED", -STANDARD_BET

        active = [r for r in result.runners if not r.scratched]
        places_paid = get_places_paid(len(active))

        if sel.position_numeric > places_paid:
            return "BURNED", -STANDARD_BET

        # CASHED — calculate profit
        if sel.place_payout and sel.place_payout > 0:
            return "CASHED", sel.place_payout - STANDARD_BET

        # Heuristic: ~1/5 of win odds for place
        # Claude Fix: Mark as ESTIMATED to avoid corrupting real audit trail
        odds = sel.final_win_odds or 2.75
        place_roi = max(0.1, (odds - 1.0) / 5.0)
        return "CASHED_ESTIMATED", place_roi * STANDARD_BET

    @staticmethod
    def _extract_selection_number(tip: Dict[str, Any]) -> Optional[int]:
        sel = tip.get("selection_number")
        if sel is not None:
            try:
                return int(sel)
            except (ValueError, TypeError):
                pass
        top_five = tip.get("top_five", "")
        if top_five:
            first = str(top_five).split(",")[0].strip()
            try:
                return int(first)
            except (ValueError, TypeError):
                pass
        return None


# -- SHARED RESULT-PARSING UTILITIES ------------------------------------------

def parse_fractional_odds(text: str) -> float:
    """``'5/2'`` → 3.5, ``'2.5'`` → 2.5, anything else → 0.0."""
    val = fortuna.parse_odds_to_decimal(text)
    return float(val) if val is not None else 0.0


def build_start_time(
    date_str: str,
    time_str: Optional[str] = None,
    *,
    tz: ZoneInfo = EASTERN,
) -> datetime:
    """Build a tz-aware datetime from ``YYYY-MM-DD`` + optional ``HH:MM``."""
    try:
        base = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        structlog.get_logger("build_start_time").warning("unparseable_date", date=date_str)
        base = datetime.now(tz)
    hour, minute = 12, 0
    if time_str:
        try:
            parts = time_str.strip().split(":")
            hour, minute = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            structlog.get_logger("build_start_time").warning("malformed_time_string", time=time_str)
            pass
    return base.replace(hour=hour, minute=minute, tzinfo=tz)


def find_nested_value(
    obj: Any,
    key_fragment: str,
    *,
    _depth: int = 0,
    _max_depth: int = 20,
) -> Optional[float]:
    """Recursively search dicts/lists for a key containing *key_fragment*
    whose value is numeric.  Depth-guarded."""
    if _depth > _max_depth:
        return None
    frag = key_fragment.lower()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if frag in k.lower() and isinstance(v, (int, float, str)):
                parsed = (
                    parse_currency_value(str(v))
                    if isinstance(v, str)
                    else float(v)
                )
                if parsed:
                    return parsed
            found = find_nested_value(
                v, key_fragment, _depth=_depth + 1, _max_depth=_max_depth,
            )
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found = find_nested_value(
                item, key_fragment, _depth=_depth + 1, _max_depth=_max_depth,
            )
            if found is not None:
                return found
    return None


_BET_ALIASES: Final[Dict[str, List[str]]] = {
    "superfecta": ["superfecta", "first 4", "first four"],
    "trifecta":   ["trifecta", "tricast"],
    "exacta":     ["exacta", "forecast"],
}


def extract_exotic_payouts(
    tables: list[Node],
) -> Dict[str, Tuple[Optional[float], Optional[str]]]:
    """Scan *tables* for exotic-bet dividend rows.

    Returns ``{"trifecta": (payout, combination), ...}`` for each found type.
    Callers with an ``HTMLParser`` pass ``parser.css("table")``.
    Equibase passes a slice of tables between race boundaries.
    """
    results: Dict[str, Tuple[Optional[float], Optional[str]]] = {}
    for table in tables:
        text = table.text().lower()
        for bet_type, aliases in _BET_ALIASES.items():
            if bet_type in results:
                continue
            if not any(a in text for a in aliases):
                continue
            for row in table.css("tr"):
                row_text = row.text().lower()
                if not any(a in row_text for a in aliases):
                    continue
                cols = row.css("td")
                combo: Optional[str] = None
                payout = 0.0
                if len(cols) >= 3:
                    combo  = fortuna.clean_text(cols[1].text())
                    payout = parse_currency_value(cols[2].text())
                elif len(cols) >= 2:
                    combo  = fortuna.clean_text(cols[0].text())
                    payout = parse_currency_value(cols[1].text())
                if payout > 0:
                    results[bet_type] = (payout, combo)
                    break
    return results


def _extract_race_number_from_text(
    parser: HTMLParser,
    url: str = "",
) -> Optional[int]:
    """Best-effort race-number from page text or URL."""
    m = re.search(r"Race\s+(\d+)", parser.text(), re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"/R(\d+)(?:[/?#]|$)", url)
    if m:
        return int(m.group(1))
    return None


# -- PageFetchingResultsAdapter BASE ------------------------------------------

class PageFetchingResultsAdapter(
    fortuna.BrowserHeadersMixin,
    fortuna.DebugMixin,
    fortuna.RacePageFetcherMixin,
    fortuna.BaseAdapterV3,
):
    """Common base for results adapters that:

    1. Fetch an index / listing page for a date
    2. Extract links to individual result pages
    3. Fetch all pages concurrently
    4. Parse each page into one or more :class:`ResultRace`

    **Subclasses must set** ``SOURCE_NAME``, ``BASE_URL``, ``HOST``
    and implement :meth:`_discover_result_links`.

    For single-race pages override :meth:`_parse_race_page`.
    For multi-race pages (e.g. Equibase) override :meth:`_parse_page`.
    """

    ADAPTER_TYPE: Final[str] = "results"

    # -- subclass must set -------------------------------------------------
    SOURCE_NAME: str
    BASE_URL: str
    HOST: str

    # -- subclass may override ---------------------------------------------
    TIMEOUT: int = 60
    IMPERSONATE: Optional[str] = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            source_name=self.SOURCE_NAME,
            base_url=self.BASE_URL,
            **kwargs,
        )
        self._target_venues: Optional[Set[str]] = None

    # -- target venues property --------------------------------------------

    @property
    def target_venues(self) -> Optional[Set[str]]:
        return self._target_venues

    @target_venues.setter
    def target_venues(self, value: Optional[Set[str]]) -> None:
        self._target_venues = value

    # -- framework hooks (identical across every legacy adapter) -----------

    def _configure_fetch_strategy(self) -> fortuna.FetchStrategy:
        # Use CURL_CFFI as primary (faster) but keep PLAYWRIGHT as fallback via SmartFetcher (Project Hardening)
        return fortuna.FetchStrategy(
            primary_engine=fortuna.BrowserEngine.CURL_CFFI,
            enable_js=True,
            stealth_mode="camouflage",
            timeout=self.TIMEOUT,
            network_idle=True,
        )

    async def make_request(
        self, method: str, url: str, **kwargs: Any,
    ) -> Any:
        if self.IMPERSONATE:
            kwargs.setdefault("impersonate", self.IMPERSONATE)
        return await super().make_request(method, url, **kwargs)

    def _get_headers(self) -> dict:
        return self._get_browser_headers(host=self.HOST)

    def _validate_and_parse_races(self, raw_data: Any) -> List[ResultRace]:
        return self._parse_races(raw_data)

    # -- fetch pipeline ----------------------------------------------------

    async def _fetch_data(
        self, date_str: str,
    ) -> Optional[Dict[str, Any]]:
        links = await self._discover_result_links(date_str)
        if not links:
            self.logger.warning(
                "No result links found",
                source=self.SOURCE_NAME,
                date=date_str,
            )
            return None
        return await self._fetch_link_pages(links, date_str)

    async def _discover_result_links(self, date_str: str) -> Set[str]:
        """Return URLs for individual result pages.  **Must be overridden.**"""
        raise NotImplementedError

    async def _fetch_link_pages(
        self, links: Set[str], date_str: str,
    ) -> Optional[Dict[str, Any]]:
        absolute = list(dict.fromkeys(
            lnk if lnk.startswith("http") else f"{self.BASE_URL}{lnk}"
            for lnk in links
        ))
        self.logger.info(
            "Fetching result pages",
            source=self.SOURCE_NAME,
            count=len(absolute),
        )
        metadata = [{"url": u, "race_number": 0} for u in absolute]
        pages = await self._fetch_race_pages_concurrent(
            metadata, self._get_headers(),
        )
        return {"pages": pages, "date": date_str}

    # -- parse pipeline ----------------------------------------------------

    def _parse_races(self, raw_data: Any) -> List[ResultRace]:
        if not raw_data:
            return []
        date_str = raw_data.get(
            "date", datetime.now(EASTERN).strftime("%Y-%m-%d"),
        )
        races: List[ResultRace] = []
        for item in raw_data.get("pages", []):
            html = item.get("html") if isinstance(item, dict) else None
            url  = item.get("url", "") if isinstance(item, dict) else ""
            if not html:
                continue
            try:
                races.extend(self._parse_page(html, date_str, url))
            except Exception as exc:
                self.logger.warning(
                    "Failed to parse result page",
                    source=self.SOURCE_NAME,
                    url=url,
                    error=str(exc),
                )
        return races

    def _parse_page(
        self, html: str, date_str: str, url: str,
    ) -> List[ResultRace]:
        """Parse a page into one or more races.

        Default delegates to :meth:`_parse_race_page` (single race per page).
        Override for multi-race pages (e.g. Equibase track summaries).
        """
        race = self._parse_race_page(html, date_str, url)
        return [race] if race else []

    def _parse_race_page(
        self, html: str, date_str: str, _url: str,
    ) -> Optional[ResultRace]:
        """Parse a single-race page.  Override for most adapters."""
        raise NotImplementedError

    # -- shared helpers ----------------------------------------------------

    def _venue_matches(self, text: str, href: str = "") -> bool:
        """Check whether a link matches target venue filters."""
        if not self.target_venues:
            # If no targets specified, we accept everything (Project Directive: Keep fetchers fetching)
            return True

        # Try exact canonical match on text (e.g. track name in link)
        canon_text = fortuna.get_canonical_venue(text)
        if canon_text != "unknown" and canon_text in self.target_venues:
            return True

        # Check if any target venue slug is present in the href
        # This handles links that are just times (e.g. /results/track-slug/date/time)
        href_clean = href.lower().replace("-", "").replace("_", "")
        for v in self.target_venues:
            if v and v in href_clean:
                return True

        # Last resort: check alphanumeric chunks for track codes (e.g. 'GP')
        # We also check the text for short codes
        for source in [href, text]:
            for chunk in re.findall(r'[A-Za-z0-9]{2,}', source):
                canon_chunk = fortuna.get_canonical_venue(chunk)
                if canon_chunk != "unknown" and canon_chunk in self.target_venues:
                    return True

        return False

    def _make_race_id(
        self,
        prefix: str,
        venue: str,
        date_str: str,
        race_num: int,
    ) -> str:
        canon = fortuna.get_canonical_venue(venue)
        return f"{prefix}_{canon}_{date_str.replace('-', '')}_R{race_num}"


# -- EQUIBASE RESULTS ADAPTER ------------------------------------------------
#
# Special case: index → track summary pages → multiple race tables per page.
# Overrides _parse_page (multi-race) instead of _parse_race_page.
# -----------------------------------------------------------------------------

class EquibaseResultsAdapter(PageFetchingResultsAdapter):
    """Equibase summary charts — primary US thoroughbred results source."""

    SOURCE_NAME = "EquibaseResults"
    BASE_URL    = "https://www.equibase.com"
    HOST        = "www.equibase.com"
    IMPERSONATE = "chrome120"
    TIMEOUT     = 60

    def _configure_fetch_strategy(self) -> fortuna.FetchStrategy:
        # Equibase uses Instart Logic / Imperva; PLAYWRIGHT_LEGACY with network_idle is robust
        return fortuna.FetchStrategy(
            primary_engine=fortuna.BrowserEngine.PLAYWRIGHT_LEGACY,
            enable_js=True,
            stealth_mode="camouflage",
            timeout=self.TIMEOUT,
            network_idle=True,
        )

    def _get_headers(self) -> dict:
        # Equibase is sensitive to header order/content; let SmartFetcher handle it via browserforge
        return {"Referer": "https://www.equibase.com/"}

    # -- link discovery (complex: multiple index URL patterns) -------------

    async def _discover_result_links(self, date_str: str) -> set:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            self.logger.error("Invalid date format", date=date_str)
            return set()

        index_urls = [
            f"/static/chart/summary/index.html?SAP=TN",
            f"/static/chart/summary/index.html?date={dt.strftime('%m/%d/%Y')}",
            f"/static/chart/summary/{dt.strftime('%m%d%y')}sum.html",
            f"/static/chart/summary/{dt.strftime('%Y%m%d')}sum.html",
        ]

        resp = None
        for url in index_urls:
            # Try multiple impersonations to bypass Imperva/Cloudflare
            for imp in ["chrome120", "chrome110", "safari15_5"]:
                try:
                    resp = await self.make_request(
                        "GET", url, headers=self._get_headers(), impersonate=imp
                    )
                    if (
                        resp and resp.text
                        and len(resp.text) > 2000 # Increased threshold for real content
                        and "Pardon Our Interruption" not in resp.text
                        and "<table" in resp.text.lower() # Verify presence of data tables
                    ):
                        break
                    else:
                        resp = None
                except Exception:
                    continue
            if resp:
                break

        if not resp or not resp.text:
            self.logger.warning("No response from Equibase index", date=date_str)
            return set()

        self._save_debug_snapshot(resp.text, f"eqb_results_index_{date_str}")
        initial_links = self._extract_track_links(resp.text, dt)

        # Resolve any RaceCardIndex links to actual sum.html files
        resolved_links = set()
        index_links = [ln for ln in initial_links if "RaceCardIndex" in ln]
        sum_links = [ln for ln in initial_links if "RaceCardIndex" not in ln]

        resolved_links.update(sum_links)

        if index_links:
            self.logger.info("Resolving track indices", count=len(index_links))
            metadata = [{"url": ln, "race_number": 0} for ln in index_links]
            index_pages = await self._fetch_race_pages_concurrent(
                metadata, self._get_headers(),
            )
            for p in index_pages:
                html = p.get("html")
                if not html: continue
                # Extract all sum.html links from this track index
                date_short = dt.strftime("%m%d%y")
                for m in re.findall(r'href="([^"]+)"', html):
                    normalised = m.replace("\\/", "/").replace("\\", "/")
                    if date_short in normalised and "sum.html" in normalised:
                        resolved_links.add(self._normalise_eqb_link(normalised))
                for m in re.findall(r'"URL":"([^"]+)"', html):
                    normalised = m.replace("\\/", "/").replace("\\", "/")
                    if date_short in normalised and "sum.html" in normalised:
                        resolved_links.add(self._normalise_eqb_link(normalised))

        return resolved_links

    def _extract_track_links(self, html: str, dt: datetime) -> set:
        """Pull track-summary URLs from the index page."""
        parser = HTMLParser(html)
        raw_links: set = set()
        date_short = dt.strftime("%m%d%y")

        # Source 1 — inline JSON in <script> tags
        for url_match in re.findall(r'"URL":"([^"]+)"', html):
            normalised = url_match.replace("\\/", "/").replace("\\", "/")
            if date_short in normalised and (
                "sum.html" in normalised or "EQB.html" in normalised or "RaceCardIndex" in normalised
            ):
                raw_links.add(normalised)

        # Source 2 — <a> tags matching known patterns
        selectors_and_patterns = [
            ('table.display a[href*="sum.html"]', None),
            ('a[href*="/static/chart/summary/"]', lambda h: "index.html" not in h and "calendar.html" not in h),
            ("a", lambda h: (
                re.search(r"[A-Z]{3}\d{6}(?:sum|EQB)\.html", h)
                or (date_short in h and ("sum.html" in h.lower() or "eqb.html" in h.lower()))
            ) and "index.html" not in h and "calendar.html" not in h),
        ]
        for selector, extra_filter in selectors_and_patterns:
            for a in parser.css(selector):
                href = (a.attributes.get("href") or "").replace("\\", "/")
                if not href:
                    continue
                if extra_filter and not extra_filter(href):
                    continue
                if not self._venue_matches(a.text(), href):
                    continue
                raw_links.add(href)

        if not raw_links:
            self.logger.warning("No track links found in index", date=str(dt.date()))
            return set()

        self.logger.info("Track links extracted", count=len(raw_links))
        return {self._normalise_eqb_link(lnk) for lnk in raw_links}

    def _normalise_eqb_link(self, link: str) -> str:
        """Turn a relative Equibase link into an absolute URL."""
        if link.startswith("http"):
            return link
        path = link.lstrip("/")
        if "static/chart/summary/" not in path:
            if path.startswith("../"):
                path = "static/chart/" + path.replace("../", "")
            elif not path.startswith("static/"):
                path = f"static/chart/summary/{path}"
        path = re.sub(r"/+", "/", path)
        return f"{self.BASE_URL}/{path}"

    # -- multi-race page parsing -------------------------------------------

    def _parse_page(
        self, html: str, date_str: str, url: str,
    ) -> List[ResultRace]:
        """A track summary page contains multiple race tables."""
        parser = HTMLParser(html)

        # Venue from page header
        track_node = parser.css_first("h3") or parser.css_first("h2")
        if not track_node:
            self.logger.debug("No track header found", url=url)
            return []
        venue = fortuna.normalize_venue_name(track_node.text(strip=True))
        if not venue:
            return []

        # Identify race tables and their indices among ALL tables
        all_tables = parser.css("table")
        indexed_race_tables: List[Tuple[int, Node]] = []
        for i, table in enumerate(all_tables):
            header = table.css_first("thead tr th")
            if header and "Race" in header.text():
                indexed_race_tables.append((i, table))

        races: List[ResultRace] = []
        for j, (idx, race_table) in enumerate(indexed_race_tables):
            try:
                # Dividend tables sit between this race and the next
                next_idx = (
                    indexed_race_tables[j + 1][0]
                    if j + 1 < len(indexed_race_tables)
                    else len(all_tables)
                )
                dividend_tables = all_tables[idx + 1 : next_idx]
                exotics = extract_exotic_payouts(dividend_tables)

                race = self._parse_race_table(
                    race_table, venue, date_str, exotics,
                )
                if race:
                    races.append(race)
            except Exception as exc:
                self.logger.debug(
                    "Failed to parse race table", error=str(exc),
                )
        return races

    def _parse_race_table(
        self,
        table: Node,
        venue: str,
        date_str: str,
        exotics: Dict[str, Tuple[Optional[float], Optional[str]]],
    ) -> Optional[ResultRace]:
        header = table.css_first("thead tr th")
        if not header:
            return None
        header_text = header.text()

        race_match = re.search(r"Race\s+(\d+)", header_text)
        if not race_match:
            return None
        race_num = int(race_match.group(1))

        # Start time from header or fallback
        start_time = self._parse_header_time(header_text, date_str)

        runners = [
            r for row in table.css("tbody tr")
            if (r := self._parse_runner_row(row)) is not None
        ]
        if not runners:
            return None

        tri = exotics.get("trifecta", (None, None))
        exa = exotics.get("exacta", (None, None))
        sup = exotics.get("superfecta", (None, None))

        return ResultRace(
            id=self._make_race_id("eqb_res", venue, date_str, race_num),
            venue=venue,
            race_number=race_num,
            start_time=start_time,
            runners=runners,
            source=self.SOURCE_NAME,
            is_fully_parsed=True,
            trifecta_payout=tri[0],
            trifecta_combination=tri[1],
            exacta_payout=exa[0],
            exacta_combination=exa[1],
            superfecta_payout=sup[0],
            superfecta_combination=sup[1],
        )

    @staticmethod
    def _parse_header_time(header_text: str, date_str: str) -> datetime:
        m = re.search(r"(\d{1,2}:\d{2})\s*([APM]{2})", header_text, re.I)
        if m:
            try:
                t = datetime.strptime(
                    f"{m.group(1)} {m.group(2).upper()}", "%I:%M %p",
                ).time()
                d = datetime.strptime(date_str, "%Y-%m-%d")
                return datetime.combine(d, t).replace(tzinfo=EASTERN)
            except ValueError:
                pass
        return build_start_time(date_str)

    def _parse_runner_row(self, row: Node) -> Optional[ResultRunner]:
        try:
            cols = row.css("td")
            if len(cols) < 3:
                return None

            name = fortuna.clean_text(cols[2].text())
            if not name or name.upper() in ("HORSE", "NAME", "RUNNER"):
                return None

            pos_text = fortuna.clean_text(cols[0].text())
            num_text = fortuna.clean_text(cols[1].text())
            number = int(num_text) if num_text.isdigit() else 0

            odds_text = (
                fortuna.clean_text(cols[3].text()) if len(cols) > 3 else ""
            )
            final_odds = fortuna.parse_odds_to_decimal(odds_text)

            win_pay = place_pay = show_pay = 0.0
            if len(cols) >= 7:
                win_pay   = parse_currency_value(cols[4].text())
                place_pay = parse_currency_value(cols[5].text())
                show_pay  = parse_currency_value(cols[6].text())

            return ResultRunner(
                name=name,
                number=number,
                position=pos_text,
                final_win_odds=final_odds,
                win_payout=win_pay,
                place_payout=place_pay,
                show_payout=show_pay,
            )
        except Exception as exc:
            self.logger.warning(
                "Failed parsing runner row", error=str(exc),
            )
            return None


# -- RACING POST RESULTS ADAPTER ---------------------------------------------

class RacingPostResultsAdapter(PageFetchingResultsAdapter):
    """Racing Post results — UK / IRE thoroughbred and jumps."""

    SOURCE_NAME = "RacingPostResults"
    BASE_URL    = "https://www.racingpost.com"
    HOST        = "www.racingpost.com"
    IMPERSONATE = "chrome120"
    TIMEOUT     = 60

    def _configure_fetch_strategy(self) -> fortuna.FetchStrategy:
        strategy = super()._configure_fetch_strategy()
        # RacingPost is JS-heavy and has strong bot detection; keep CURL_CFFI primary but ensure fallback (Project Hardening)
        strategy.primary_engine = fortuna.BrowserEngine.CURL_CFFI
        return strategy

    # -- link discovery ----------------------------------------------------

    async def _discover_result_links(self, date_str: str) -> Set[str]:
        resp = await self.make_request(
            "GET", f"/results/{date_str}", headers=self._get_headers(),
        )
        if not resp or not resp.text:
            return set()
        self._save_debug_snapshot(resp.text, f"rp_results_index_{date_str}")
        parser = HTMLParser(resp.text)
        return self._extract_rp_links(parser)

    def _extract_rp_links(self, parser: HTMLParser) -> set:
        links: set = set()

        _SELECTORS = [
            'a[data-test-selector="RC-meetingItem__link_race"]',
            'a[href*="/results/"]',
            ".ui-link.rp-raceCourse__panel__race__time",
            "a.rp-raceCourse__panel__race__time",
            ".rp-raceCourse__panel__race__time a",
            ".RC-meetingItem__link",
        ]
        for selector in _SELECTORS:
            for a in parser.css(selector):
                href = a.attributes.get("href", "")
                if not href:
                    continue
                if not self._venue_matches(a.text(), href):
                    continue
                if self._is_rp_race_link(href):
                    links.add(href)

        # Last-resort fallback
        if not links:
            for a in parser.css('a[href*="/results/"]'):
                href = a.attributes.get("href", "")
                if len(href.split("/")) >= 3:
                    links.add(href)

        return links

    @staticmethod
    def _is_rp_race_link(href: str) -> bool:
        return bool(
            re.search(r"/results/.*?\d{5,}", href)
            or re.search(r"/results/\d+/", href)
            or re.search(r"/\d{4}-\d{2}-\d{2}/", href)
            or ("/results/" in href and len(href.split("/")) >= 4)
        )

    # -- single-race page parsing ------------------------------------------

    def _parse_race_page(
        self, html: str, date_str: str, _url: str,
    ) -> Optional[ResultRace]:
        parser = HTMLParser(html)

        venue_node = parser.css_first(".rp-raceTimeCourseName__course")
        if not venue_node:
            return None
        venue = fortuna.normalize_venue_name(venue_node.text(strip=True))

        dividends = self._parse_tote_dividends(parser)
        trifecta_pay, trifecta_combo = self._exotic_from_dividends(
            dividends, "trifecta",
        )
        superfecta_pay, superfecta_combo = self._exotic_from_dividends(
            dividends, "superfecta",
        )

        race_num = self._extract_rp_race_number(parser)

        runners = self._parse_rp_runners(parser, dividends)
        if not runners:
            return None

        return ResultRace(
            id=self._make_race_id("rp_res", venue, date_str, race_num),
            venue=venue,
            race_number=race_num,
            start_time=build_start_time(date_str),
            runners=runners,
            source=self.SOURCE_NAME,
            trifecta_payout=trifecta_pay,
            trifecta_combination=trifecta_combo,
            superfecta_payout=superfecta_pay,
            superfecta_combination=superfecta_combo,
            official_dividends={
                k: parse_currency_value(v) for k, v in dividends.items()
            },
        )

    # -- RP-specific helpers -----------------------------------------------

    @staticmethod
    def _parse_tote_dividends(parser: HTMLParser) -> Dict[str, str]:
        """Extract label→value pairs from the Tote Returns panel."""
        container = (
            parser.css_first('div[data-test-selector="RC-toteReturns"]')
            or parser.css_first(".rp-toteReturns")
        )
        if not container:
            return {}

        dividends: Dict[str, str] = {}
        for row in (
            container.css("div.rp-toteReturns__row")
            or container.css(".rp-toteReturns__row")
        ):
            label_node = (
                row.css_first("div.rp-toteReturns__label")
                or row.css_first(".rp-toteReturns__label")
            )
            val_node = (
                row.css_first("div.rp-toteReturns__value")
                or row.css_first(".rp-toteReturns__value")
            )
            if label_node and val_node:
                label = fortuna.clean_text(label_node.text())
                value = fortuna.clean_text(val_node.text())
                if label and value:
                    dividends[label] = value
        return dividends

    @staticmethod
    def _exotic_from_dividends(
        dividends: Dict[str, str],
        bet_type: str,
    ) -> Tuple[Optional[float], Optional[str]]:
        aliases = _BET_ALIASES.get(bet_type, [bet_type])
        for label, val in dividends.items():
            if any(a in label.lower() for a in aliases):
                payout = parse_currency_value(val)
                combo = (
                    val.split("£")[-1].strip() if "£" in val else None
                )
                return payout, combo
        return None, None

    @staticmethod
    def _extract_rp_race_number(parser: HTMLParser) -> int:
        # Priority 1 — navigation bar active item
        for i, link in enumerate(
            parser.css('a[data-test-selector="RC-raceTime"]'),
        ):
            cls = link.attributes.get("class", "")
            if "active" in cls or "rp-raceTimeCourseName__time" in cls:
                return i + 1
        # Priority 2 — text fallback
        return _extract_race_number_from_text(parser) or 1

    def _parse_rp_runners(
        self,
        parser: HTMLParser,
        dividends: Dict[str, str],
    ) -> List[ResultRunner]:
        runners: List[ResultRunner] = []
        for row in parser.css(".rp-horseTable__table__row"):
            try:
                name_node = row.css_first(".rp-horseTable__horse__name")
                if not name_node:
                    continue
                name = fortuna.clean_text(name_node.text())

                pos_node = row.css_first(".rp-horseTable__pos__number")
                pos = (
                    fortuna.clean_text(pos_node.text()) if pos_node else None
                )

                num_node = row.css_first(".rp-horseTable__saddleClothNo")
                number = 0
                if num_node:
                    try:
                        number = int(fortuna.clean_text(num_node.text()))
                    except ValueError:
                        pass

                # Place payout from dividends map
                place_payout: Optional[float] = None
                for lbl, val in dividends.items():
                    if (
                        "place" in lbl.lower()
                        and name.lower() in lbl.lower()
                    ):
                        place_payout = parse_currency_value(val)
                        break

                sp_node = row.css_first(".rp-horseTable__horse__sp")
                final_odds = 0.0
                if sp_node:
                    final_odds = parse_fractional_odds(
                        fortuna.clean_text(sp_node.text()),
                    )

                runners.append(ResultRunner(
                    name=name,
                    number=number,
                    position=pos,
                    place_payout=place_payout,
                    final_win_odds=final_odds,
                ))
            except Exception:
                continue
        return runners


# -- AT THE RACES RESULTS ADAPTER ---------------------------------------------

class AtTheRacesResultsAdapter(PageFetchingResultsAdapter):
    """At The Races results — UK / IRE."""

    SOURCE_NAME = "AtTheRacesResults"
    BASE_URL    = "https://www.attheraces.com"
    HOST        = "www.attheraces.com"
    IMPERSONATE = "chrome120"
    TIMEOUT     = 60

    def _configure_fetch_strategy(self) -> fortuna.FetchStrategy:
        strategy = super()._configure_fetch_strategy()
        # ATR uses Cloudflare; keep CURL_CFFI primary but ensure fallback (Project Hardening)
        strategy.primary_engine = fortuna.BrowserEngine.CURL_CFFI
        return strategy

    # -- link discovery (multi-URL index) ----------------------------------

    async def _discover_result_links(self, date_str: str) -> set:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        index_urls = [
            f"/results/{date_str}",
            f"/results/{dt.strftime('%d-%B-%Y')}",
            f"/results/international/{date_str}",
            f"/results/international/{dt.strftime('%d-%B-%Y')}",
            f"/results/yesterday",
            f"/results/international/yesterday",
        ]

        links: set = set()
        for url in index_urls:
            try:
                resp = await self.make_request(
                    "GET", url, headers=self._get_headers(),
                )
                if not resp or not resp.text:
                    continue
                self._save_debug_snapshot(
                    resp.text,
                    f"atr_index_{date_str}_{url.replace('/', '_')}",
                )
                links.update(self._extract_atr_links(resp.text))
            except Exception as exc:
                self.logger.debug(
                    "ATR index fetch failed", url=url, error=str(exc),
                )

        return links

    def _extract_atr_links(self, html: str) -> set:
        parser = HTMLParser(html)
        links: set = set()
        # Broad selectors for all possible result links (Council of Superbrains Directive)
        for selector in [
            "a[href*='/results/']",
            "a[data-test-selector*='result']",
            ".meeting-summary a",
            ".p-results__item a",
            ".p-meetings__item a",
            ".p-results-meeting a",
        ]:
            for a in parser.css(selector):
                href = a.attributes.get("href", "")
                if not href:
                    continue
                if not self._venue_matches(a.text(), href):
                    continue
                if self._is_atr_race_link(href):
                    links.add(
                        href if href.startswith("http")
                        else f"{self.BASE_URL}{href}"
                    )
        return links

    @staticmethod
    def _is_atr_race_link(href: str) -> bool:
        return bool(
            re.search(r"/results/.*?/\d{4}", href)
            or re.search(r"/results/\d{2}-.*?-\d{4}/", href)
            or re.search(r"/results/.*?/\d+$", href)
            or ("/results/" in href and len(href.split("/")) >= 4)
        )

    # -- single-race page parsing ------------------------------------------

    def _parse_race_page(
        self, html: str, date_str: str, url: str,
    ) -> Optional[ResultRace]:
        parser = HTMLParser(html)

        venue = self._extract_atr_venue(parser)
        if not venue:
            return None

        race_num = 1
        url_match = re.search(r"/R(\d+)$", url)
        if url_match:
            race_num = int(url_match.group(1))

        runners = self._parse_atr_runners(parser)

        # Dividends — use the shared exotic extractor on the
        # dedicated dividends table, then enrich with place payouts
        div_table = parser.css_first(".result-racecard__dividends-table")
        exotics: Dict[str, Tuple[Optional[float], Optional[str]]] = {}
        if div_table:
            exotics = extract_exotic_payouts([div_table])
            self._map_place_payouts(div_table, runners)

        if not runners:
            return None

        tri = exotics.get("trifecta", (None, None))
        sup = exotics.get("superfecta", (None, None))

        return ResultRace(
            id=self._make_race_id("atr_res", venue, date_str, race_num),
            venue=venue,
            race_number=race_num,
            start_time=build_start_time(date_str),
            runners=runners,
            trifecta_payout=tri[0],
            trifecta_combination=tri[1],
            superfecta_payout=sup[0],
            superfecta_combination=sup[1],
            source=self.SOURCE_NAME,
        )

    # -- ATR-specific helpers ----------------------------------------------

    @staticmethod
    def _extract_atr_venue(parser: HTMLParser) -> Optional[str]:
        header = (
            parser.css_first(".race-header__details--primary")
            or parser.css_first(".racecard-header")
            or parser.css_first(".race-header")
        )
        if not header:
            return None
        venue_node = (
            header.css_first("h2")
            or header.css_first("h1")
            or header.css_first(".track-name")
        )
        if not venue_node:
            return None
        return fortuna.normalize_venue_name(venue_node.text(strip=True))

    def _parse_atr_runners(
        self, parser: HTMLParser,
    ) -> List[ResultRunner]:
        rows = (
            parser.css(".result-racecard__row")
            or parser.css(".card-cell--horse")
            or parser.css("atr-result-horse")
            or parser.css("div[class*='RacecardResultItem']")
            or parser.css(".p-results__item")
        )

        runners: List[ResultRunner] = []
        for row in rows:
            try:
                name_node = (
                    row.css_first(".result-racecard__horse-name a")
                    or row.css_first(".horse-name a")
                    or row.css_first("a[href*='/horse/']")
                    or row.css_first("[class*='HorseName']")
                )
                if not name_node:
                    continue
                name = fortuna.clean_text(name_node.text())

                pos_node = (
                    row.css_first(".result-racecard__pos")
                    or row.css_first(".pos")
                    or row.css_first(".position")
                    or row.css_first("[class*='Position']")
                )
                pos = (
                    fortuna.clean_text(pos_node.text()) if pos_node else None
                )

                num_node = row.css_first(
                    ".result-racecard__saddle-cloth",
                )
                number = 0
                if num_node:
                    try:
                        number = int(fortuna.clean_text(num_node.text()))
                    except ValueError:
                        pass

                odds_node = row.css_first(".result-racecard__odds")
                final_odds = 0.0
                if odds_node:
                    final_odds = parse_fractional_odds(
                        fortuna.clean_text(odds_node.text()),
                    )

                runners.append(ResultRunner(
                    name=name,
                    number=number,
                    position=pos,
                    final_win_odds=final_odds,
                ))
            except Exception:
                continue
        return runners

    @staticmethod
    def _map_place_payouts(
        div_table: Node,
        runners: List[ResultRunner],
    ) -> None:
        """Enrich runners with place payouts from the dividends table."""
        for row in div_table.css("tr"):
            try:
                row_text = row.text().lower()
                if "place" not in row_text:
                    continue
                cols = row.css("td")
                if len(cols) < 2:
                    continue
                p_name = fortuna.clean_text(
                    cols[0].text().replace("Place", "").strip(),
                )
                p_val = parse_currency_value(cols[1].text())
                for runner in runners:
                    if (
                        runner.name.lower() in p_name.lower()
                        or p_name.lower() in runner.name.lower()
                    ):
                        runner.place_payout = p_val
            except Exception:
                continue


# -- Part 2: SportingLifeResultsAdapter → main()
# -- Continues from Part 1's final line
# -----------------------------------------------------------------------------


class SportingLifeResultsAdapter(PageFetchingResultsAdapter):
    """Sporting Life results (UK / IRE / International)."""

    SOURCE_NAME = "SportingLifeResults"
    BASE_URL    = "https://www.sportinglife.com"
    HOST        = "www.sportinglife.com"
    IMPERSONATE = "chrome120"
    TIMEOUT     = 45

    def _configure_fetch_strategy(self) -> fortuna.FetchStrategy:
        strategy = super()._configure_fetch_strategy()
        # SportingLife is JS-heavy; keep CURL_CFFI primary but ensure fallback (Project Hardening)
        strategy.primary_engine = fortuna.BrowserEngine.CURL_CFFI
        return strategy

    # -- link discovery ----------------------------------------------------

    async def _discover_result_links(self, date_str: str) -> Set[str]:
        resp = await self.make_request(
            "GET",
            f"/racing/results/{date_str}",
            headers=self._get_headers(),
        )
        if not resp or not resp.text:
            return set()
        self._save_debug_snapshot(resp.text, f"sl_results_index_{date_str}")
        return self._extract_sl_links(resp.text)

    def _extract_sl_links(self, html: str) -> set:
        links: set = set()
        for a in HTMLParser(html).css("a[href*='/racing/results/']"):
            href = a.attributes.get("href", "")
            if not href: continue
            if not self._venue_matches(a.text(), href):
                continue
            # /racing/results/2026-02-04/ludlow/901676/race-name
            if re.search(r"/results/\d{4}-\d{2}-\d{2}/.+/\d+/", href):
                links.add(href)
        return links

    # -- page parsing (two strategies) -------------------------------------

    def _parse_race_page(
        self, html: str, date_str: str, url: str,
    ) -> Optional[ResultRace]:
        parser = HTMLParser(html)

        # Strategy 1 — Next.js JSON payload (most reliable)
        script = parser.css_first("script#__NEXT_DATA__")
        if script:
            race = self._parse_from_next_data(script.text(), date_str)
            if race:
                return race

        # Strategy 2 — HTML scrape fallback
        return self._parse_from_html(parser, date_str)

    # -- Strategy 1: JSON -------------------------------------------------

    def _parse_from_next_data(
        self, script_text: str, date_str: str,
    ) -> Optional[ResultRace]:
        try:
            data = json.loads(script_text)
        except json.JSONDecodeError as exc:
            self.logger.debug("Invalid __NEXT_DATA__", error=str(exc))
            return None

        race_data = (
            data.get("props", {}).get("pageProps", {}).get("race", {})
        )
        if not race_data:
            return None

        summary    = race_data.get("race_summary", {})
        venue      = fortuna.normalize_venue_name(
            summary.get("course_name", "Unknown"),
        )
        race_num   = (
            race_data.get("race_number")
            or summary.get("race_number")
            or 1
        )
        date_val   = summary.get("date", date_str)
        start_time = build_start_time(date_val, summary.get("time"))

        runners = self._runners_from_json(race_data)
        if not runners:
            return None

        # Exotic payouts via recursive search
        trifecta_pay   = find_nested_value(race_data, "trifecta")
        superfecta_pay = find_nested_value(race_data, "superfecta")

        # Place payouts from CSV field
        self._apply_place_payouts_from_csv(
            race_data.get("place_win", ""), runners,
        )

        return ResultRace(
            id=self._make_race_id("sl_res", venue, date_val, race_num),
            venue=venue,
            race_number=race_num,
            start_time=start_time,
            runners=runners,
            trifecta_payout=trifecta_pay,
            superfecta_payout=superfecta_pay,
            source=self.SOURCE_NAME,
        )

    @staticmethod
    def _runners_from_json(race_data: dict) -> List[ResultRunner]:
        """Extract runners from ``rides`` (result pages) or ``runners``."""
        items = race_data.get("rides") or race_data.get("runners", [])
        runners: List[ResultRunner] = []
        for item in items:
            horse = item.get("horse", {})
            name  = horse.get("name") or item.get("name")
            if not name:
                continue
            sp_raw = (
                item.get("starting_price")
                or item.get("sp")
                or item.get("betting", {}).get("current_odds", "")
            )
            runners.append(ResultRunner(
                name=name,
                number=(
                    item.get("cloth_number")
                    or item.get("saddle_cloth_number", 0)
                ),
                position=str(item.get("finish_position", item.get("position", ""))),
                final_win_odds=parse_fractional_odds(str(sp_raw)),
            ))
        return runners

    @staticmethod
    def _apply_place_payouts_from_csv(
        place_csv: str,
        runners: List[ResultRunner],
    ) -> None:
        """Map comma-separated place payouts to runners by finishing position."""
        if not isinstance(place_csv, str) or not place_csv:
            return
        pays = [parse_currency_value(p) for p in place_csv.split(",")]
        for runner in runners:
            pos = runner.position_numeric
            if pos and 1 <= pos <= len(pays):
                runner.place_payout = pays[pos - 1]

    # -- Strategy 2: HTML fallback ----------------------------------------

    def _parse_from_html(
        self, parser: HTMLParser, date_str: str,
    ) -> Optional[ResultRace]:
        header = parser.css_first("h1")
        if not header:
            return None

        match = re.match(
            r"(\d{1,2}:\d{2})\s+(.+)\s+Result",
            fortuna.clean_text(header.text()),
        )
        if not match:
            return None

        time_str   = match.group(1)
        venue      = fortuna.normalize_venue_name(match.group(2))
        start_time = build_start_time(date_str, time_str)

        runners: List[ResultRunner] = []
        for row in parser.css(
            'div[class*="ResultRunner__StyledResultRunnerWrapper"]',
        ):
            name_node = row.css_first(
                'a[class*="ResultRunner__StyledHorseName"]',
            )
            if not name_node:
                continue
            pos_node = row.css_first(
                'div[class*="ResultRunner__StyledRunnerPositionContainer"]',
            )
            runners.append(ResultRunner(
                name=fortuna.clean_text(name_node.text()),
                number=0,
                position=(
                    fortuna.clean_text(pos_node.text()) if pos_node else None
                ),
            ))

        if not runners:
            return None

        return ResultRace(
            id=self._make_race_id("sl_res", venue, date_str, 1),
            venue=venue,
            race_number=1,
            start_time=start_time,
            runners=runners,
            source=self.SOURCE_NAME,
        )


class SkySportsResultsAdapter(PageFetchingResultsAdapter):
    """Sky Sports Racing results (UK / IRE)."""

    SOURCE_NAME = "SkySportsResults"
    BASE_URL    = "https://www.skysports.com"
    HOST        = "www.skysports.com"
    IMPERSONATE = "chrome120"
    TIMEOUT     = 45

    # -- link discovery ----------------------------------------------------

    async def _discover_result_links(self, date_str: str) -> Set[str]:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            url_dates = [dt.strftime("%d-%m-%Y"), (dt - timedelta(days=1)).strftime("%d-%m-%Y")]
        except ValueError:
            url_dates = [date_str]

        parser = HTMLParser(resp.text)
        self._save_debug_snapshot(resp.text, f"sky_results_index_{url_date}")
        return self._extract_sky_links(parser, date_str, url_date)

    def _extract_sky_links(self, parser: HTMLParser, date_str: str, url_date: str) -> set:
        links: set = set()
        # Broad selectors for SkySports results (Council of Superbrains Directive)
        for a in (parser.css("a[href*='/racing/results/']") + parser.css("a[href*='/full-result/']")):
            href = a.attributes.get("href", "")
            if not href: continue
            if not self._venue_matches(a.text(), href):
                continue

            # Match various result path patterns
            has_race_path = any(
                p in href for p in ("/full-result/", "/race-result/", "/results/full-result/")
            ) or re.search(r"/\d{6,}/", href)

            # Check if link belongs to requested date or is generally a result link
            has_date = date_str in href or url_date in href or re.search(r"/\d{6,}/", href)

            if has_race_path and has_date:
                links.add(href)
        return links

    # -- page parsing ------------------------------------------------------

    def _parse_race_page(
        self, html: str, date_str: str, url: str,
    ) -> Optional[ResultRace]:
        parser = HTMLParser(html)

        header = parser.css_first(".sdc-site-racing-header__name")
        if not header:
            return None

        match = re.match(
            r"(\d{1,2}:\d{2})\s+(.+)",
            fortuna.clean_text(header.text()),
        )
        if not match:
            return None

        time_str   = match.group(1)
        venue      = fortuna.normalize_venue_name(match.group(2))
        start_time = build_start_time(date_str, time_str)

        runners = self._parse_sky_runners(parser)
        if not runners:
            return None

        exotics  = extract_exotic_payouts(parser.css("table"))
        race_num = self._extract_sky_race_number(parser, url)

        tri = exotics.get("trifecta", (None, None))
        sup = exotics.get("superfecta", (None, None))

        return ResultRace(
            id=self._make_race_id("sky_res", venue, date_str, race_num),
            venue=venue,
            race_number=race_num,
            start_time=start_time,
            runners=runners,
            trifecta_payout=tri[0],
            superfecta_payout=sup[0],
            source=self.SOURCE_NAME,
        )

    @staticmethod
    def _parse_sky_runners(parser: HTMLParser) -> List[ResultRunner]:
        runners: List[ResultRunner] = []
        for row in parser.css(".sdc-site-racing-card__item"):
            name_node = row.css_first(".sdc-site-racing-card__name")
            if not name_node:
                continue

            pos_node    = row.css_first(".sdc-site-racing-card__position")
            number_node = row.css_first(".sdc-site-racing-card__number")
            odds_node   = row.css_first(".sdc-site-racing-card__odds")

            number = 0
            if number_node:
                try:
                    number = int(re.sub(r"\D", "", number_node.text()))
                except (ValueError, TypeError):
                    pass

            runners.append(ResultRunner(
                name=fortuna.clean_text(name_node.text()),
                number=number,
                position=(
                    fortuna.clean_text(pos_node.text()) if pos_node else None
                ),
                final_win_odds=parse_fractional_odds(
                    fortuna.clean_text(odds_node.text()) if odds_node else "",
                ),
            ))
        return runners

    @staticmethod
    def _extract_sky_race_number(parser: HTMLParser, url: str) -> int:
        """Try navigation index, then URL ID, then text fallback."""
        url_match = re.search(r"/(\d+)/", url)
        if url_match:
            nav_links = parser.css("a[href*='/racing/results/']")
            for i, link in enumerate(nav_links):
                if url_match.group(0) in (
                    link.attributes.get("href") or ""
                ):
                    return i + 1

        return _extract_race_number_from_text(parser, url) or 1


# -- REPORT GENERATION --------------------------------------------------------

_REPORT_WIDTH: Final[int] = 80
_REPORT_SEP:   Final[str] = "=" * _REPORT_WIDTH
_REPORT_DOT:   Final[str] = "." * _REPORT_WIDTH
_SECTION_SEP:  Final[str] = "-" * 40


def _format_tip_time(tip: Dict[str, Any]) -> str:
    """Best-effort ET time string from a tip's ``start_time``."""
    raw = tip.get("start_time", "")
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return to_eastern(dt).strftime("%Y-%m-%d %H:%M ET")
    except (ValueError, TypeError):
        return str(raw)[:16].replace("T", " ")


def generate_analytics_report(
    audited_tips: List[Dict[str, Any]],
    recent_tips: Optional[List[Dict[str, Any]]] = None,
    harvest_summary: Optional[Dict[str, Any]] = None,
    *,
    include_lifetime_stats: bool = False,
) -> str:
    """Build the human-readable performance audit report."""
    now_str = datetime.now(EASTERN).strftime("%Y-%m-%d %H:%M ET")
    lines: list[str] = [
        _REPORT_SEP,
        "🐎 FORTUNA INTELLIGENCE - PERFORMANCE AUDIT & VERIFICATION".center(_REPORT_WIDTH),
        f"Generated: {now_str}".center(_REPORT_WIDTH),
        _REPORT_SEP,
        "",
    ]

    if harvest_summary:
        _append_harvest_proof(lines, harvest_summary)

    if recent_tips:
        _append_pending_section(lines, recent_tips)

    audited_sorted = sorted(
        audited_tips,
        key=lambda t: t.get("start_time", ""),
        reverse=True,
    )
    if audited_sorted:
        _append_recent_performance(lines, audited_sorted[:15])

    _append_exotic_tracking(lines, audited_tips)

    if include_lifetime_stats and audited_tips:
        _append_lifetime_stats(lines, audited_tips)

    return "\n".join(lines)


def _append_harvest_proof(
    lines: list[str],
    harvest_summary: Dict[str, Any],
) -> None:
    lines.extend(["🔎 LIVE ADAPTER HARVEST PROOF", _SECTION_SEP])
    for adapter, data in harvest_summary.items():
        if isinstance(data, dict):
            count    = data.get("count", 0)
            max_odds = data.get("max_odds", 0.0)
        else:
            count, max_odds = data, 0.0

        status   = "✅ SUCCESS" if count > 0 else "⏳ PENDING/NO DATA"
        odds_str = (
            f"MaxOdds: {max_odds:>5.1f}" if max_odds > 0 else "Odds: N/A"
        )
        lines.append(
            f"{adapter:<25} | {status:<15} | Records: {count:<4} | {odds_str}",
        )
    lines.append("")


def _append_pending_section(
    lines: list[str],
    recent_tips: List[Dict[str, Any]],
) -> None:
    lines.extend([
        "⏳ PENDING VERIFICATION - RECENT DISCOVERIES",
        _SECTION_SEP,
        f"{'RACE TIME':<18} | {'VENUE':<20} | {'R#':<3} | {'GM?':<4} | STATUS",
        _REPORT_DOT,
    ])
    for tip in recent_tips[:25]:
        st_str = _format_tip_time(tip)
        venue  = str(tip.get("venue", "Unknown"))[:20]
        rnum   = tip.get("race_number", "?")
        gm     = "GOLD" if tip.get("is_goldmine") else "----"
        status = (
            tip.get("verdict") if tip.get("audit_completed") else "WATCHING"
        )
        lines.append(
            f"{st_str:<18} | {venue:<20} | {rnum:<3} | {gm:<4} | {status}",
        )
    lines.append("")


def _append_recent_performance(
    lines: list[str],
    tips: List[Dict[str, Any]],
) -> None:
    lines.extend([
        "💰 RECENT PERFORMANCE PROOF (MATCHED RESULTS)",
        _SECTION_SEP,
        f"{'RESULT':<6} | {'RACE':<25} | {'PROFIT':<8} | PAYOUT/DETAILS",
        _REPORT_DOT,
    ])
    for tip in tips:
        verdict = tip.get("verdict", "?")
        emoji = _VERDICT_DISPLAY.get(verdict, "⚪ VOID")
        venue  = (
            f"{tip.get('venue', 'Unknown')[:18]} "
            f"R{tip.get('race_number', '?')}"
        )
        profit = f"${tip.get('net_profit', 0.0):+.2f}"

        detail_parts: list[str] = []

        # Place payouts
        p1 = tip.get("top1_place_payout")
        p2 = tip.get("top2_place_payout")
        if p1 or p2:
            detail_parts.append(f"P: {p1 or 0:.2f}/{p2 or 0:.2f}")

        # Odds drift
        po = tip.get("predicted_2nd_fav_odds")
        ao = tip.get("actual_2nd_fav_odds")
        if po is not None or ao is not None:
            po_s = f"{po:.1f}" if po is not None else "?"
            ao_s = f"{ao:.1f}" if ao is not None else "?"
            detail_parts.append(f"Odds: {po_s}->{ao_s}")

        # Exotics
        if tip.get("superfecta_payout"):
            detail_parts.append(f"Super: ${tip['superfecta_payout']:.2f}")
        elif tip.get("trifecta_payout"):
            detail_parts.append(f"Tri: ${tip['trifecta_payout']:.2f}")
        elif tip.get("actual_top_5"):
            detail_parts.append(f"Top 5: [{tip['actual_top_5']}]")

        payout_info = " | ".join(detail_parts)
        lines.append(
            f"{emoji:<6} | {venue:<25} | {profit:>8} | {payout_info}",
        )
    lines.append("")


def _append_exotic_tracking(
    lines: list[str],
    audited_tips: List[Dict[str, Any]],
) -> None:
    super_races = [t for t in audited_tips if t.get("superfecta_payout")]
    tri_races   = [t for t in audited_tips if t.get("trifecta_payout")]

    if super_races:
        payouts = [t["superfecta_payout"] for t in super_races]
        lines.extend([
            "🎯 SUPERFECTA PERFORMANCE PROOF",
            _SECTION_SEP,
            f"Superfecta Matches: {len(super_races)}",
            f"  Average Payout:   ${sum(payouts) / len(payouts):.2f}",
            f"  Maximum Payout:   ${max(payouts):.2f}",
            "",
        ])
    elif tri_races:
        payouts = [t["trifecta_payout"] for t in tri_races]
        lines.extend([
            "🎯 SECONDARY EXOTIC TRACKING (TRIFECTA)",
            _SECTION_SEP,
            f"Trifecta Matches:   {len(tri_races)} "
            f"(Avg: ${sum(payouts) / len(payouts):.2f})",
            "",
        ])


def _append_lifetime_stats(
    lines: list[str],
    audited_tips: List[Dict[str, Any]],
) -> None:
    total  = len(audited_tips)
    cashed = sum(1 for t in audited_tips if t.get("verdict") in _CASHED_VERDICTS)
    profit = sum(t.get("net_profit", 0.0) for t in audited_tips)
    sr     = (cashed / total * 100) if total else 0.0
    roi    = (profit / (total * 2.0) * 100) if total else 0.0

    lines.extend([
        "📊 SUMMARY METRICS (LIFETIME)",
        _SECTION_SEP,
        f"Total Verified Races: {total}",
        f"Overall Strike Rate:   {sr:.1f}%",
        f"Total Net Profit:     ${profit:+.2f} (Using $2.00 Base Unit)",
        f"Return on Investment:  {roi:+.1f}%",
        "",
    ])


# -- ADAPTER REGISTRY & LIFECYCLE ---------------------------------------------

def get_results_adapter_classes() -> List[Type[fortuna.BaseAdapterV3]]:
    """All concrete adapter classes with ``ADAPTER_TYPE == 'results'``."""

    def _all_subclasses(cls: type) -> set:
        subs = set(cls.__subclasses__())
        return subs.union(s for c in subs for s in _all_subclasses(c))

    return [
        c
        for c in _all_subclasses(fortuna.BaseAdapterV3)
        if not getattr(c, "__abstractmethods__", None)
        and getattr(c, "ADAPTER_TYPE", "discovery") == "results"
    ]


@asynccontextmanager
async def managed_adapters(
    region: Optional[str] = None,
    target_venues: Optional[set] = None,
):
    """Instantiate, optionally filter, yield, then tear down all results
    adapters."""
    classes = get_results_adapter_classes()
    logger = structlog.get_logger("managed_adapters")

    if region:
        allowed = (
            fortuna.USA_RESULTS_ADAPTERS
            if region == "USA"
            else fortuna.INT_RESULTS_ADAPTERS
        )
        classes = [
            c for c in classes
            if getattr(c, "SOURCE_NAME", "") in allowed
        ]

    adapters: list[fortuna.BaseAdapterV3] = []
    for cls in classes:
        adapter = cls()
        if target_venues:
            adapter.target_venues = target_venues  # type: ignore[attr-defined]
        adapters.append(adapter)

    try:
        yield adapters
    finally:
        for adapter in adapters:
            try:
                await adapter.close()
            except Exception as exc:
                logger.warning("Adapter cleanup failed", adapter=adapter.source_name, error=str(exc))
        try:
            await fortuna.GlobalResourceManager.cleanup()
        except Exception as exc:
            logger.error("Global resource cleanup failed", error=str(exc))


# -- ORCHESTRATION HELPERS ----------------------------------------------------

_analytics_logger = structlog.get_logger("run_analytics")
_MAX_CONCURRENT_FETCHES: Final[int] = 10


async def _harvest_results(
    adapters: List[fortuna.BaseAdapterV3],
    valid_dates: List[str],
    harvest_summary: Dict[str, Dict[str, Any]],
) -> List[ResultRace]:
    """Fetch results from all adapters × dates; populate *harvest_summary*."""
    sem = asyncio.Semaphore(_MAX_CONCURRENT_FETCHES)

    async def _fetch_one(
        adapter: fortuna.BaseAdapterV3,
        date_str: str,
    ) -> Tuple[str, List[ResultRace]]:
        async with sem:
            try:
                races = await adapter.get_races(date_str)
                _analytics_logger.debug(
                    "Fetched results",
                    adapter=adapter.source_name,
                    date=date_str,
                    count=len(races),
                )
                return adapter.source_name, races
            except Exception as exc:
                _analytics_logger.warning(
                    "Adapter fetch failed",
                    adapter=adapter.source_name,
                    date=date_str,
                    error=str(exc),
                )
                return adapter.source_name, []

    tasks = [
        _fetch_one(adapter, d)
        for d in valid_dates
        for adapter in adapters
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    all_races: List[ResultRace] = []
    for res in raw_results:
        if isinstance(res, Exception):
            _analytics_logger.warning("Task raised exception", error=str(res))
            continue

        name, races = res
        all_races.extend(races)

        # Track harvest metrics
        max_odds = max(
            (
                float(r.final_win_odds)
                for race in races
                for r in race.runners
                if r.final_win_odds
            ),
            default=0.0,
        )
        entry = harvest_summary.setdefault(
            name, {"count": 0, "max_odds": 0.0},
        )
        entry["count"] += len(races)
        entry["max_odds"] = max(entry["max_odds"], max_odds)

    return all_races


async def _save_harvest_summary(
    harvest_summary: Dict[str, Dict[str, Any]],
    auditor: AuditorEngine,
    region: Optional[str],
) -> None:
    """Persist harvest summary to JSON file and database."""
    try:
        with open("results_harvest.json", "w", encoding="utf-8") as f:
            json.dump(harvest_summary, f)
    except OSError as exc:
        _analytics_logger.debug(
            "Failed to write results_harvest.json", error=str(exc),
        )

    if harvest_summary:
        try:
            await auditor.db.log_harvest(harvest_summary, region=region)
        except Exception as exc:
            _analytics_logger.debug(
                "Failed to log harvest to DB", error=str(exc),
            )


async def _write_gha_summary(
    auditor: AuditorEngine,
    harvest_summary: dict,
) -> None:
    """Build and write the GitHub Actions Job Summary."""
    if "GITHUB_STEP_SUMMARY" not in os.environ:
        return

    try:
        pending_tips = await auditor.db.get_unverified_tips(lookback_hours=48)
        qualified: list[fortuna.Race] = []

        for tip in pending_tips:
            try:
                race = fortuna.Race(
                    id=tip["race_id"],
                    venue=tip["venue"],
                    race_number=tip["race_number"],
                    start_time=datetime.fromisoformat(
                        tip["start_time"].replace("Z", "+00:00"),
                    ),
                    runners=[],
                    source="Database",
                )
                race.metadata = {
                    "is_goldmine": bool(tip.get("is_goldmine")),
                    "selection_number": tip.get("selection_number"),
                    "selection_name": tip.get("selection_name"),
                    "predicted_2nd_fav_odds": tip.get(
                        "predicted_2nd_fav_odds",
                    ),
                }
                race.top_five_numbers = tip.get("top_five")
                qualified.append(race)
            except (KeyError, ValueError):
                continue

        predictions_md = fortuna.format_predictions_section(qualified)
        proof_md       = await fortuna.format_proof_section(auditor.db)
        harvest_md     = fortuna.build_harvest_table(
            harvest_summary, "🛰️ Results Harvest Performance",
        )
        artifacts_md   = fortuna.format_artifact_links()
        fortuna.write_job_summary(
            predictions_md, harvest_md, proof_md, artifacts_md,
        )
        _analytics_logger.info("GHA Job Summary written")
    except Exception as exc:
        _analytics_logger.error(
            "Failed to write GHA summary", error=str(exc),
        )


async def _generate_and_save_report(
    auditor: AuditorEngine,
    harvest_summary: Dict[str, Any],
    *,
    include_lifetime_stats: bool = False,
) -> None:
    """Fetch all audited data, generate report, print and persist."""
    all_audited = await auditor.get_all_audited_tips()
    recent_tips = await auditor.get_recent_tips(limit=20)

    report = generate_analytics_report(
        audited_tips=all_audited,
        recent_tips=recent_tips,
        harvest_summary=harvest_summary,
        include_lifetime_stats=include_lifetime_stats,
    )
    print(report)

    try:
        Path("analytics_report.txt").write_text(report, encoding="utf-8")
        _analytics_logger.info("Report saved", path="analytics_report.txt")
    except OSError as exc:
        _analytics_logger.error("Failed to save report", error=str(exc))

    if all_audited:
        _analytics_logger.info(
            "Analytics complete", total_audited=len(all_audited),
        )
    else:
        _analytics_logger.info("No audited tips found in history")


# -- TOP-LEVEL ORCHESTRATION --------------------------------------------------

async def run_analytics(
    target_dates: List[str],
    region: Optional[str] = None,
    *,
    include_lifetime_stats: bool = False,
) -> None:
    """Main analytics entry: harvest → audit → report → GHA summary."""
    valid_dates = [d for d in target_dates if validate_date_format(d)]
    if not valid_dates:
        _analytics_logger.error("No valid dates", input_dates=target_dates)
        return

    target_region = region or get_optimal_region_at_time(now_eastern())
    _analytics_logger.info(
        "Starting analytics audit",
        dates=valid_dates,
        region=target_region,
    )

    # Pre-populate harvest summary for regional visibility
    expected = (
        fortuna.USA_RESULTS_ADAPTERS
        if target_region == "USA"
        else fortuna.INT_RESULTS_ADAPTERS
    )
    harvest_summary: Dict[str, Dict[str, Any]] = {
        name: {"count": 0, "max_odds": 0.0} for name in expected
    }

    async with AuditorEngine() as auditor:
        unverified = await auditor.get_unverified_tips()
        target_venues = None

        if not unverified:
            _analytics_logger.info(
                "No unverified tips found in database; fetching all results for visibility (Project Directive)",
            )
        else:
            _analytics_logger.info("Tips to audit", count=len(unverified))
            target_venues = {
                fortuna.get_canonical_venue(t.get("venue"))
                for t in unverified
            }
            _analytics_logger.info(
                "Targeting venues", venues=list(target_venues),
            )

        async with managed_adapters(
            region=region, target_venues=target_venues,
        ) as adapters:
            try:
                all_results = await _harvest_results(
                    adapters, valid_dates, harvest_summary,
                )
                _analytics_logger.info(
                    "Total results harvested",
                    count=len(all_results),
                )
                if all_results and unverified:
                    await auditor.audit_races(all_results, unverified=unverified)
                elif not all_results:
                    _analytics_logger.warning(
                        "No results harvested from any source",
                    )
            finally:
                await _save_harvest_summary(
                    harvest_summary, auditor, region,
                )

        await _generate_and_save_report(
            auditor,
            harvest_summary,
            include_lifetime_stats=include_lifetime_stats,
        )
        await _write_gha_summary(auditor, harvest_summary)



# -- CLI ENTRY POINT ----------------------------------------------------------

def _build_target_dates(
    explicit_date: Optional[str],
    lookback_days: int,
) -> List[str]:
    """Return a list of ``YYYY-MM-DD`` date strings."""
    if explicit_date:
        if not validate_date_format(explicit_date):
            raise ValueError(
                f"Invalid date format '{explicit_date}'.  Use YYYY-MM-DD.",
            )
        return [explicit_date]
    now = datetime.now(EASTERN)
    return [
        (now - timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(lookback_days)
    ]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Fortuna Analytics Engine — "
            "Race result auditing and performance analysis"
        ),
    )
    parser.add_argument(
        "--date", type=str, help="Target date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--region",
        type=str,
        choices=["USA", "INT"],
        help="Filter results by region",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=2,
        help="Number of days to look back (default: 2)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to tip database (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Migrate data from legacy JSON to SQLite",
    )
    parser.add_argument(
        "--include-lifetime-stats",
        action="store_true",
        help="Include lifetime summary statistics in report",
    )
    args = parser.parse_args()

    # Set DB path before any initialization
    if args.db_path != DEFAULT_DB_PATH:
        os.environ["FORTUNA_DB_PATH"] = args.db_path

    log_level = logging.DEBUG if args.verbose else logging.INFO
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
    )

    # Handle migration sub-command
    if args.migrate:
        async def _do_migrate() -> None:
            db = fortuna.FortunaDB(args.db_path)
            try:
                await db.migrate_from_json()
                print("Migration complete.")
            except Exception as exc:
                print(f"Migration failed: {exc}")
            finally:
                await db.close()

        asyncio.run(_do_migrate())
        return

    # Build target dates
    try:
        target_dates = _build_target_dates(args.date, args.days)
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    # Auto-select region if not specified
    if not args.region:
        args.region = get_optimal_region_at_time(datetime.now(EASTERN))
        structlog.get_logger().info(
            "Auto-selected region", region=args.region,
        )

    asyncio.run(
        run_analytics(
            target_dates,
            region=args.region,
            include_lifetime_stats=args.include_lifetime_stats,
        )
    )


if __name__ == "__main__":
    main()
