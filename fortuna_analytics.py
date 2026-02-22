#!/usr/bin/env python3
"""
fortuna_analytics.py
Race result harvesting and performance analysis engine for Fortuna.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Set, Tuple, Type
from zoneinfo import ZoneInfo

import structlog
from pydantic import Field, model_validator
from selectolax.parser import HTMLParser, Node

from fortuna_utils import (
    EASTERN, VENUE_MAP, BET_TYPE_KEYWORDS, DISCIPLINE_KEYWORDS,
    clean_text, node_text, get_canonical_venue, normalize_venue_name,
    parse_odds_to_decimal, SmartOddsExtractor, is_placeholder_odds,
    is_valid_odds, now_eastern, to_eastern, get_places_paid
)
import fortuna

# -- CONSTANTS ----------------------------------------------------------------

DEFAULT_DB_PATH: Final[str] = os.environ.get("FORTUNA_DB_PATH", "fortuna.db")
STANDARD_BET: Final[float] = 2.00
DEFAULT_REGION: Final[str] = "GLOBAL"
_MAX_CURRENCY_INPUT_LEN: Final[int] = 64
_MAX_CONCURRENT_FETCHES: Final[int] = 10
_REPORT_WIDTH: Final[int] = 80
_REPORT_SEP: Final[str] = "=" * _REPORT_WIDTH
_REPORT_DOT: Final[str] = "." * _REPORT_WIDTH
_SECTION_SEP: Final[str] = "-" * 40


_BET_ALIASES: Final[Dict[str, list[str]]] = {
    "superfecta": ["superfecta", "first 4", "first four"],
    "trifecta":   ["trifecta", "tricast"],
    "exacta":     ["exacta", "forecast"],
}

_currency_logger = structlog.get_logger("currency_parser")

_POSITION_LOOKUP: Final[Dict[str, int]] = {
    "W": 1, "1": 1, "1ST": 1,
    "P": 2, "2": 2, "2ND": 2,
    "S": 3, "3": 3, "3RD": 3,
    "4": 4, "4TH": 4,
    "5": 5, "5TH": 5,
}


# -- VERDICT ENUM -------------------------------------------------------------

class Verdict(str, Enum):
    """Auditable tip outcomes."""
    CASHED           = "CASHED"
    CASHED_ESTIMATED = "CASHED_ESTIMATED"
    BURNED           = "BURNED"
    VOID             = "VOID"

    @property
    def is_win(self) -> bool:
        return self in _CASHED_VERDICTS

    @property
    def is_loss(self) -> bool:
        return self is Verdict.BURNED

    @property
    def display(self) -> str:
        return _VERDICT_DISPLAY[self]


_CASHED_VERDICTS: Final[frozenset[Verdict]] = frozenset(
    {Verdict.CASHED, Verdict.CASHED_ESTIMATED}
)
_LOSS_VERDICTS: Final[frozenset[Verdict]] = frozenset({Verdict.BURNED})

_VERDICT_DISPLAY: Final[Dict[Verdict, str]] = {
    Verdict.CASHED:           "✅ WIN ",
    Verdict.CASHED_ESTIMATED: "✅ WIN~",
    Verdict.BURNED:           "❌ LOSS",
    Verdict.VOID:             "⚪ VOID",
}


# -- HELPER FUNCTIONS ---------------------------------------------------------





def parse_position(pos_str: Optional[str]) -> Optional[int]:
    """'1st' → 1, '2/12' → 2, 'W' → 1, etc."""
    if not pos_str:
        return None
    s = str(pos_str).upper().strip()
    if s in _POSITION_LOOKUP:
        return _POSITION_LOOKUP[s]
    m = re.search(r"^(\d+)", s)
    return int(m.group(1)) if m else None




def parse_currency_value(value_str: str) -> float:
    """'$1,234.56' → 1234.56.  Returns 0.0 on unparseable input."""
    if not value_str:
        return 0.0
    raw = str(value_str).strip()
    if len(raw) > _MAX_CURRENCY_INPUT_LEN:
        _currency_logger.debug("currency_input_too_long", length=len(raw))
        return 0.0
    try:
        negative = raw.startswith("-") or raw.startswith("(")
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
        if not cleaned:
            return 0.0
        result = float(cleaned)
        return -result if negative else result
    except (ValueError, TypeError):
        _currency_logger.warning("failed_parsing_currency", value=value_str)
        return 0.0


def validate_date_format(date_str: str) -> bool:
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _safe_int(text: str, default: int = 0) -> int:
    """Extract leading digits from text, return *default* on failure."""
    cleaned = re.sub(r"\D", "", text)
    try:
        return int(cleaned) if cleaned else default
    except ValueError:
        return default


def _apply_exotics_to_race(
    exotics: Dict[str, Tuple[Optional[float], Optional[str]]],
) -> Dict[str, Any]:
    """Unpack exotic payout dict into keyword args for ResultRace."""
    tri = exotics.get("trifecta", (None, None))
    exa = exotics.get("exacta", (None, None))
    sup = exotics.get("superfecta", (None, None))
    return {
        "trifecta_payout": tri[0] if (tri[0] and tri[0] > 0) else None,
        "trifecta_combination": tri[1],
        "exacta_payout": exa[0] if (exa[0] and exa[0] > 0) else None,
        "exacta_combination": exa[1],
        "superfecta_payout": sup[0] if (sup[0] and sup[0] > 0) else None,
        "superfecta_combination": sup[1],
    }


def _race_quality(race: "ResultRace") -> int:
    """Heuristic to score a result by completeness (higher is better)."""
    score = len(race.runners)
    # Give significant weight to having odds and numeric positions
    score += sum(10 for r in race.runners if r.final_win_odds and r.final_win_odds > 0)
    score += sum(5 for r in race.runners if r.place_payout and r.place_payout > 0)
    score += sum(3 for r in race.runners if r.position_numeric is not None)
    # Exotic payouts are a good indicator of high-quality data (e.g. Equibase)
    if race.trifecta_payout:
        score += 20
    if race.superfecta_payout:
        score += 20
    return score


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
    race_type: Optional[str] = None
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
        """Venue|Race|Date|Disc — no time component for fuzzy matching."""
        d = self.start_time.strftime("%Y%m%d")
        disc = (self.discipline or "T")[:1].upper()
        return f"{fortuna.get_canonical_venue(self.venue)}|{self.race_number}|{d}|{disc}"

    def get_top_finishers(self, n: int = 5) -> List[ResultRunner]:
        ranked = [r for r in self.runners if r.position_numeric is not None]
        ranked.sort(key=lambda r: r.position_numeric)  # type: ignore[arg-type]
        return ranked[:n]

    @property
    def active_runners(self) -> List[ResultRunner]:
        """Non-scratched runners."""
        return [r for r in self.runners if not r.scratched]


# -- AUDITOR ENGINE -----------------------------------------------------------

class AuditorEngine:
    """Matches predicted tips against actual race results via SQLite."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db = fortuna.FortunaDB(db_path or DEFAULT_DB_PATH)
        self.logger = structlog.get_logger(self.__class__.__name__)

    async def __aenter__(self) -> AuditorEngine:
        return self

    async def __aexit__(self, *exc: object) -> None:
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
        if hasattr(self, "db"):
            await self.db.close()

    # -- audit pipeline ----------------------------------------------------

    async def audit_races(
        self,
        results: List[ResultRace],
        unverified: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        results_map, secondary_index = self._build_results_map(results)
        self.logger.debug(
            "Results map built",
            canonical_keys=len(results_map),
            secondary_keys=len(secondary_index),
        )

        if unverified is None:
            unverified = await self.get_unverified_tips()

        for tip in unverified[:10]:
            tip_key = self._tip_canonical_key(tip)
            self.logger.debug(
                "Tip key vs results",
                tip_venue=tip.get("venue"),
                tip_key=tip_key,
                matched=tip_key in results_map if tip_key else False,
            )

        audited: List[Dict[str, Any]] = []
        outcomes_to_batch: List[Tuple[str, Dict[str, Any]]] = []

        for tip in unverified:
            race_id = tip.get("race_id")
            if not race_id:
                continue

            tip_key = self._tip_canonical_key(tip)
            if not tip_key:
                continue

            try:
                # Item 8: Algorithm optimization
                result, confidence = self._match_tip_to_result(
                    tip_key, results_map, race_id, secondary_index=secondary_index
                )
                if not result:
                    continue

                outcome = self._evaluate_tip(tip, result, confidence=confidence)
                outcomes_to_batch.append((race_id, outcome))
                audited.append({**tip, **outcome, "audit_completed": True})

            except Exception:
                self.logger.error(
                    "Error during audit",
                    tip_id=race_id,
                    exc_info=True,
                )

        if outcomes_to_batch:
            self.logger.info("Updating audit results", count=len(outcomes_to_batch))
            await self._persist_outcomes(outcomes_to_batch)

        return audited

    async def _persist_outcomes(
        self, outcomes: List[Tuple[str, Dict[str, Any]]],
    ) -> None:
        """Batch-persist or fall back to individual updates."""
        if hasattr(self.db, "update_audit_results_batch"):
            await self.db.update_audit_results_batch(outcomes)
        else:
            for race_id, outcome in outcomes:
                await self.db.update_audit_result(race_id, outcome)

    @staticmethod
    def _build_results_map(
        results: List[ResultRace],
    ) -> Tuple[Dict[str, ResultRace], Dict[str, List[ResultRace]]]:
        mapping: Dict[str, ResultRace] = {}
        secondary: Dict[str, List[ResultRace]] = defaultdict(list)
        log = structlog.get_logger("AuditorEngine")
        for r in results:
            mapping[r.canonical_key] = r

            # Item 8: Build secondary index by venue+date for faster fallback matching
            # relaxed_key is {venue}|{race}|{date}|{disc}
            parts = r.relaxed_key.split('|')
            if len(parts) >= 3:
                venue_date_key = f"{parts[0]}|{parts[2]}"
                secondary[venue_date_key].append(r)

            rk = r.relaxed_key
            if rk == r.canonical_key:
                continue
            if rk in mapping:
                existing = mapping[rk]
                if existing.canonical_key in mapping:
                    log.debug(
                        "Relaxed key collision — keeping canonical",
                        key=rk,
                        existing=existing.canonical_key,
                        new=r.canonical_key,
                    )
                    continue
            mapping[rk] = r
        return mapping, secondary

    def _match_tip_to_result(
        self,
        tip_key: str,
        results_map: Dict[str, ResultRace],
        race_id: str,
        secondary_index: Optional[Dict[str, List[ResultRace]]] = None,
    ) -> Tuple[Optional[ResultRace], str]:
        """Matches a tip to a result, returning the result and match confidence."""
        # Exact canonical match
        result = results_map.get(tip_key)
        if result:
            return result, "exact"

        parts = tip_key.split("|")
        if len(parts) < 5:
            return None, "none"

        venue, race_str, date_str, time_str, disc = parts

        # Fallback 1: closest time match within 90 minutes (same discipline)
        # Item 8: Use secondary index to avoid O(N) iteration
        if secondary_index:
            venue_date_key = f"{venue}|{date_str}"
            candidates = []
            try:
                tip_minutes = int(time_str[:2]) * 60 + int(time_str[2:])
                for obj in secondary_index.get(venue_date_key, []):
                    # Match race number and discipline
                    if str(obj.race_number) == race_str and (obj.discipline or "T")[:1].upper() == disc:
                        res_time = obj.start_time.strftime("%H%M")
                        res_minutes = int(res_time[:2]) * 60 + int(res_time[2:])
                        delta = abs(tip_minutes - res_minutes)
                        if delta <= 90:
                            candidates.append((delta, obj))

                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    res = candidates[0][1]
                    self.logger.info(
                        "Time-window fallback match (optimized)",
                        race_id=race_id,
                        match_key=res.canonical_key,
                    )
                    return res, "relaxed"
            except (ValueError, IndexError):
                pass

        # Fallback 1.1: drop time entirely, keep discipline (any time)
        relaxed = f"{venue}|{race_str}|{date_str}|{disc}"
        result = results_map.get(relaxed)
        if result:
            self.logger.info(
                "Time-relaxed fallback match",
                race_id=race_id,
                match_key=result.canonical_key,
            )
            return result, "relaxed"

        # Fallback 2: drop discipline, keep venue|race|date
        matches = []
        if secondary_index:
            venue_date_key = f"{venue}|{date_str}"
            for obj in secondary_index.get(venue_date_key, []):
                if str(obj.race_number) == race_str:
                    matches.append(obj)

        if len(matches) == 1:
            self.logger.info(
                "Discipline-relaxed fallback match",
                race_id=race_id,
                match_key=matches[0].canonical_key,
            )
            return matches[0], "relaxed"
        if len(matches) > 1:
            self.logger.warning(
                "Ambiguous discipline fallback — skipping",
                race_id=race_id,
                candidate_count=len(matches),
            )

        # Fallback 3: ±1 race number, same venue/date/discipline
        try:
            venue_part, race_str, date_str, _, disc = parts

            # Guard: only allow shift if the original race was NOT found anywhere
            # (checked via relaxed key)
            original_relaxed = f"{venue_part}|{race_str}|{date_str}|{disc}"
            if original_relaxed in results_map:
                # The result exists for the correct race number but time/discipline didn't match perfectly.
                # Don't shift to a different race.
                return None, "none"

            for delta in (-1, +1):
                shifted_num = int(race_str) + delta
                if shifted_num < 1:
                    continue
                shifted_key = f"{venue_part}|{shifted_num}|{date_str}|{disc}"
                result = results_map.get(shifted_key)
                if result:
                    self.logger.warning(
                        "Race-number-shifted fallback match",
                        race_id=race_id,
                        original_race_num=race_str,
                        matched_race_num=shifted_num,
                        match_key=result.canonical_key,
                    )
                    return result, "shifted"
        except (ValueError, IndexError):
            pass

        if results_map:
            tip_venue = parts[0] if parts else ""
            venue_candidates = [k for k in results_map if k.startswith(tip_venue)][:5]
            self.logger.warning(
                "No result match found",
                race_id=race_id,
                tip_key=tip_key,
                venue_candidates_in_results=venue_candidates,
            )

        return None, "none"

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
        self,
        tip: Dict[str, Any],
        result: ResultRace,
        confidence: str = "exact",
    ) -> Dict[str, Any]:
        selection_num = self._extract_selection_number(tip)
        selection_name = tip.get("selection_name")

        top_finishers = result.get_top_finishers(5)
        actual_top_5 = [str(r.number) for r in top_finishers]

        top1_place = top_finishers[0].place_payout if top_finishers else None
        top2_place = (
            top_finishers[1].place_payout if len(top_finishers) >= 2 else None
        )

        actual_2nd_fav_odds = self._find_actual_2nd_fav_odds(result)

        sel_result = self._find_selection_runner(
            result, selection_num, selection_name,
        )

        verdict, profit = self._compute_verdict(sel_result, result)

        return {
            "field_size": len(result.active_runners),
            "match_confidence": confidence,
            "actual_top_5": ", ".join(actual_top_5),
            "actual_2nd_fav_odds": actual_2nd_fav_odds,
            "verdict": verdict.value,
            "net_profit": round(profit, 2),
            "selection_position": (
                sel_result.position_numeric if sel_result else None
            ),
            "audit_timestamp": now_eastern().isoformat(),
            "trifecta_payout": result.trifecta_payout,
            "trifecta_combination": result.trifecta_combination,
            "superfecta_payout": result.superfecta_payout,
            "superfecta_combination": result.superfecta_combination,
            "top1_place_payout": top1_place,
            "top2_place_payout": top2_place,
        }

    @staticmethod
    def _find_actual_2nd_fav_odds(result: ResultRace) -> Optional[float]:
        """
        Derives the actual second-favorite's odds from the result data.
        Handles cases where odds are identical (co-favorites) and fallbacks to payouts.
        """
        runners_list = []
        for r in result.runners:
            if r.scratched:
                continue

            odds = r.final_win_odds
            # Fallback to win_payout if odds are missing (common for Harness - GPT5 Fix)
            if (not odds or odds <= 0) and r.win_payout and r.win_payout > 0:
                odds = round(r.win_payout / 2.0, 2)

            if odds and odds > 0:
                runners_list.append((r, odds))

        if not runners_list:
            return None

        # Sort by odds ascending
        runners_list.sort(key=lambda x: x[1])

        if len(runners_list) < 2:
            # If we only have odds for one runner (e.g. the winner), we can't find 2nd fav
            return None

        # Return the odds of the actual second favorite (Claude Fix)
        return float(runners_list[1][1])

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
            name_lower = name.lower()
            return next(
                (r for r in result.runners if r.name.lower() == name_lower),
                None,
            )
        return None

    @staticmethod
    def _compute_verdict(
        sel: Optional[ResultRunner],
        result: ResultRace,
    ) -> Tuple[Verdict, float]:
        if sel is None:
            return Verdict.VOID, 0.0
        if sel.position_numeric is None:
            # Fallback (Item 4): If position is missing but we have payouts, they placed!
            if (sel.place_payout and sel.place_payout > 0.01) or (sel.win_payout and sel.win_payout > 0.01):
                if sel.place_payout and sel.place_payout > 0.01:
                    return Verdict.CASHED, sel.place_payout - STANDARD_BET
                # We know they placed because they won or have a win payout
                return Verdict.CASHED_ESTIMATED, round((STANDARD_BET * max(1.1, 1.0 + ((sel.final_win_odds or fortuna.DEFAULT_ODDS_FALLBACK) - 1.0) / 5.0)) - STANDARD_BET, 2)

            # Missing position often means parsing gap, not a loss (Claude Fix)
            return Verdict.VOID, 0.0

        places_paid = get_places_paid(len(result.active_runners), is_handicap=result.is_handicap)

        if sel.position_numeric > places_paid:
            return Verdict.BURNED, -STANDARD_BET

        # Actual place payout available (GPT5 Fix: Handle zero payouts)
        if sel.place_payout and sel.place_payout > 0.01:
            return Verdict.CASHED, sel.place_payout - STANDARD_BET

        # Heuristic estimate: ~1/5 of (win odds - 1) for place return
        odds = sel.final_win_odds or DEFAULT_ODDS_FALLBACK
        if not sel.final_win_odds:
            structlog.get_logger("AuditorEngine").debug(
                "odds_defaulted", runner=sel.name, fallback=DEFAULT_ODDS_FALLBACK
            )
        estimated_place_return = STANDARD_BET * max(1.1, 1.0 + (odds - 1.0) / 5.0)
        return Verdict.CASHED_ESTIMATED, round(estimated_place_return - STANDARD_BET, 2)

    @staticmethod
    def _extract_selection_number(tip: Dict[str, Any]) -> Optional[int]:
        sel = tip.get("selection_number")
        if sel is not None:
            try:
                return int(sel)
            except (ValueError, TypeError):
                pass
        # Fallback: first entry in top_five prediction
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
    """'5/2' → 3.5, '2.5' → 2.5, anything else → 0.0."""
    val = fortuna.parse_odds_to_decimal(text)
    return float(val) if val is not None else 0.0


def build_start_time(
    date_str: str,
    time_str: Optional[str] = None,
    *,
    tz: ZoneInfo = EASTERN,
) -> datetime:
    """Build a tz-aware datetime from YYYY-MM-DD + optional HH:MM."""
    _log = structlog.get_logger("build_start_time")
    try:
        base = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        _log.warning("unparseable_date", date=date_str)
        base = datetime.now(tz)

    hour, minute = 12, 0
    if time_str:
        parts = time_str.strip().split(":")
        try:
            hour, minute = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            _log.warning("malformed_time_string", time=time_str)
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


def extract_exotic_payouts(
    tables: list[Node],
) -> Dict[str, Tuple[Optional[float], Optional[str]]]:
    """Scan *tables* for exotic-bet dividend rows.

    Returns {"trifecta": (payout, combination), ...} for each found type.
    """
    results: Dict[str, Tuple[Optional[float], Optional[str]]] = {}
    for table in tables:
        text = fortuna.node_text(table).lower()
        for bet_type, aliases in _BET_ALIASES.items():
            if bet_type in results:
                continue
            if not any(a in text for a in aliases):
                continue
            for row in table.css("tr"):
                row_text = fortuna.node_text(row).lower()
                if not any(a in row_text for a in aliases):
                    continue
                cols = row.css("td")
                combo: Optional[str] = None
                payout = 0.0
                if len(cols) >= 3:
                    combo = fortuna.clean_text(fortuna.node_text(cols[1]))
                    payout = parse_currency_value(fortuna.node_text(cols[2]))
                elif len(cols) >= 2:
                    combo = fortuna.clean_text(fortuna.node_text(cols[0]))
                    payout = parse_currency_value(fortuna.node_text(cols[1]))
                if payout > 0:
                    results[bet_type] = (payout, combo)
                    break
    return results


def _extract_race_number_from_text(
    parser: HTMLParser,
    url: str = "",
) -> Optional[int]:
    """Best-effort race-number from page text or URL."""
    m = re.search(r"Race\s+(\d+)", fortuna.node_text(parser), re.I)
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
    """Common base for results adapters that fetch, discover, and parse.

    Subclasses **must** set SOURCE_NAME, BASE_URL, HOST
    and implement :meth:`_discover_result_links`.

    Override :meth:`_parse_race_page` for single-race pages or
    :meth:`_parse_page` for multi-race pages.
    """

    ADAPTER_TYPE: Final[str] = "results"

    _BLOCK_SIGNATURES: Final[tuple[str, ...]] = (
        "pardon our interruption",
        "checking your browser",
        "just a moment",
        "cloudflare",
        "access denied",
        "captcha",
        "please verify",
        "attention required",
        "one more step",
        "incapsula",
        "incident id",
    )
    _BLOCK_MAX_LENGTH: Final[int] = 15_000

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

    # -- target venues -----------------------------------------------------

    @property
    def target_venues(self) -> Optional[Set[str]]:
        return self._target_venues

    @target_venues.setter
    def target_venues(self, value: Optional[Set[str]]) -> None:
        self._target_venues = value

    # -- anti-bot detection ------------------------------------------------

    def _check_for_block(self, html: str, url: str) -> bool:
        """Detect anti-bot block pages."""
        if len(html) > self._BLOCK_MAX_LENGTH:
            return False

        html_lower = html.lower()
        # Check full text for very short pages (likely block pages)
        if len(html) < 5000:
            for sig in self._BLOCK_SIGNATURES:
                if sig in html_lower:
                    self.logger.error(
                        "Bot blocked (body text match)",
                        source=self.SOURCE_NAME,
                        url=url,
                        signature=sig,
                    )
                    return True

        # For medium pages, check only title/h1
        parser = HTMLParser(html)
        for selector in ("title", "h1"):
            node = parser.css_first(selector)
            if node:
                text = fortuna.node_text(node).lower()
                for sig in self._BLOCK_SIGNATURES:
                    if sig in text:
                        self.logger.error(
                            "Bot blocked (title/h1 match)",
                            source=self.SOURCE_NAME,
                            url=url,
                            signature=sig,
                        )
                        return True
        return False

    # -- date guard --------------------------------------------------------

    def _link_matches_date(self, href: str, date_str: str) -> bool:
        """Verify a discovered link belongs to the target date.

        Checks for ISO date and common compact variants in the URL.
        Subclasses with non-standard URL date formats should override.
        """
        if date_str in href:
            return True

        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return False

        compact_formats = (
            dt.strftime("%Y%m%d"),       # 20260209
            dt.strftime("%m%d%y"),       # 020926
            dt.strftime("%d%m%Y"),       # 09022026
            dt.strftime("%d-%m-%Y"),     # 09-02-2026
            dt.strftime("%d-%B-%Y"),     # 09-February-2026
            dt.strftime("%m/%d/%Y"),     # 02/09/2026
        )
        href_lower = href.lower()
        return any(fmt.lower() in href_lower for fmt in compact_formats)

    # -- framework hooks ---------------------------------------------------

    def _configure_fetch_strategy(self) -> fortuna.FetchStrategy:
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

    async def _fetch_data(self, date_str: str) -> Optional[Dict[str, Any]]:
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
        # GPT5 Fix: Clean and deduplicate links to avoid net::ERR_INVALID_ARGUMENT
        clean_absolute = [u.strip() for u in absolute if u and u.strip()]
        metadata = [{"url": u, "race_number": 0} for u in clean_absolute]
        pages = await self._fetch_race_pages_concurrent(
            metadata, self._get_headers(),
        )
        return {"pages": pages, "date": date_str}

    # -- parse pipeline ----------------------------------------------------

    def _parse_races(self, raw_data: Any) -> List[ResultRace]:
        if not raw_data:
            return []
        date_str = raw_data.get(
            "date", now_eastern().strftime("%Y-%m-%d"),
        )
        races: List[ResultRace] = []
        for item in raw_data.get("pages", []):
            if not isinstance(item, dict):
                continue
            html = item.get("html")
            url = item.get("url", "")
            if not html:
                continue
            if self._check_for_block(html, url):
                continue
            try:
                races.extend(self._parse_page(html, date_str, url))
            except Exception:
                self.logger.warning(
                    "Failed to parse result page",
                    source=self.SOURCE_NAME,
                    url=url,
                    exc_info=True,
                )
        return races

    def _parse_page(
        self, html: str, date_str: str, url: str,
    ) -> List[ResultRace]:
        """Default: single race per page.  Override for multi-race."""
        race = self._parse_race_page(html, date_str, url)
        return [race] if race else []

    def _parse_race_page(
        self, html: str, date_str: str, _url: str,
    ) -> Optional[ResultRace]:
        """Parse a single-race page.  Override in subclasses."""
        raise NotImplementedError

    # -- shared helpers ----------------------------------------------------

    def _venue_matches(self, text: str, href: str = "") -> bool:
        """Check whether a link matches target venue filters."""
        if not self._target_venues:
            return True

        canon_text = fortuna.get_canonical_venue(text)
        if canon_text != "unknown" and canon_text in self._target_venues:
            return True

        href_clean = href.lower().replace("-", "").replace("_", "")
        return any(v and v in href_clean for v in self._target_venues)

    def _make_race_id(
        self,
        prefix: str,
        venue: str,
        date_str: str,
        race_num: int,
    ) -> str:
        canon = fortuna.get_canonical_venue(venue)
        return f"{prefix}_{canon}_{date_str.replace('-', '')}_R{race_num}"


# -- DRF RESULTS ADAPTER -----------------------------------------------------

class DRFResultsAdapter(PageFetchingResultsAdapter):
    """
    Adapter for DRF.com text charts (structured pipe-delimited data).
    Covers all USA Thoroughbred tracks with official pari-mutuel results.
    """
    SOURCE_NAME = "DRFResults"
    BASE_URL = "https://www1.drf.com"
    HOST = "www1.drf.com"

    def _configure_fetch_strategy(self) -> fortuna.FetchStrategy:
        return fortuna.FetchStrategy(
            primary_engine=fortuna.BrowserEngine.HTTPX,
            enable_js=False,
            timeout=45
        )

    def _get_headers(self) -> Dict[str, str]:
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/plain, application/octet-stream, */*",
            "Referer": "https://www1.drf.com/drfTextChart.do"
        }

    async def _fetch_data(self, date_str: str) -> Optional[Any]:
        # YYYY-MM-DD -> YYYYMMDD
        drf_date = date_str.replace("-", "")
        url = f"/drfTextChartDownload.do?TRK=ALL&CY=USA&DATE={drf_date}"
        try:
            resp = await self.smart_fetcher.fetch(
                url,
                method="GET",
                headers=self._get_headers()
            )
            if not resp or not resp.text: return None
            # Check for redirect to login or auth page
            if "Please login" in resp.text or "DrfAuthService" in resp.text:
                self.logger.warning("DRF fetch redirected to login page", url=url)
                return None
            return {"content": resp.text, "date": date_str}
        except Exception as e:
            self.logger.error("DRFResults fetch failed", error=str(e))
            return None

    def _parse_races(self, raw_data: Any) -> List[ResultRace]:
        if not raw_data: return []
        content = raw_data.get("content", "")
        date_str = raw_data.get("date")

        races: List[ResultRace] = []
        current_venue = "Unknown"
        # Use a dict to group runners by race_num for the current venue
        venue_races = defaultdict(lambda: {"runners": [], "info": {}})

        import csv
        import io

        # Detect delimiter
        delim = "|" if "|" in content[:2000] else ","

        reader = csv.reader(io.StringIO(content), delimiter=delim, quotechar='"')

        for fields in reader:
            if not fields: continue
            rtype = fields[0]

            if rtype == "H":
                # New track card starts. Emit previous track races if any.
                if venue_races:
                    races.extend(self._finalize_venue_races(venue_races, current_venue, date_str))
                    venue_races = defaultdict(lambda: {"runners": [], "info": {}})

                track_code = fields[2] if len(fields) > 2 else ""
                raw_venue_name = fields[6] if len(fields) > 6 else track_code

                # Use DRF_VENUE_MAP for high-confidence mapping
                from fortuna_utils import DRF_VENUE_MAP
                if track_code in DRF_VENUE_MAP:
                    current_venue = DRF_VENUE_MAP[track_code]
                else:
                    current_venue = fortuna.normalize_venue_name(raw_venue_name)

            elif rtype == "R":
                if len(fields) < 3: continue
                # Breed filter: TB only
                if fields[2] != "TB": continue

                try:
                    race_num = int(fields[1])
                    venue_races[race_num]["info"] = {
                        "distance": fields[29] if len(fields) > 29 else "",
                        "unit": fields[30] if len(fields) > 30 else "",
                        "surface": fields[31] if len(fields) > 31 else "",
                        "post_time": fields[37] if len(fields) > 37 else "",
                        "is_handicap": "HANDICAP" in " ".join(fields).upper()
                    }
                except (ValueError, IndexError):
                    continue

            elif rtype == "S":
                if len(fields) < 4: continue
                try:
                    race_num = int(fields[1])
                    if race_num not in venue_races: continue

                    # Scratch check (field 82 / index 81)
                    scratch_reason = fields[81].strip() if len(fields) > 81 else ""
                    if scratch_reason: continue

                    name = fields[3]
                    # Odds to $1 are in field 37 (index 36)
                    raw_odds = fields[36] if len(fields) > 36 else "0"
                    try:
                        # DRF odds are usually multiplied by 100 in text charts
                        decimal_odds = (float(raw_odds) / 100.0) + 1.0
                    except ValueError:
                        decimal_odds = None

                    # Payouts (fields 68, 69, 70 -> indices 67, 68, 69)
                    win_pay = parse_currency_value(fields[67]) if len(fields) > 67 else 0.0
                    place_pay = parse_currency_value(fields[68]) if len(fields) > 68 else 0.0
                    show_pay = parse_currency_value(fields[69]) if len(fields) > 69 else 0.0

                    finish_pos = fields[50] if len(fields) > 50 else ""
                    # Program number (field 81 / index 80)
                    prog_num_str = fields[80] if len(fields) > 80 else "0"
                    try:
                        prog_num = int("".join(filter(str.isdigit, prog_num_str)))
                    except ValueError:
                        prog_num = 0

                    venue_races[race_num]["runners"].append(ResultRunner(
                        name=name,
                        number=prog_num,
                        position=finish_pos,
                        final_win_odds=decimal_odds,
                        win_payout=win_pay,
                        place_payout=place_pay,
                        show_payout=show_pay,
                        odds_source="extracted"
                    ))
                except (ValueError, IndexError):
                    continue

        # Finalize last venue
        if venue_races:
            races.extend(self._finalize_venue_races(venue_races, current_venue, date_str))

        return races

    def _finalize_venue_races(self, venue_races: Dict[int, Any], venue: str, date_str: str) -> List[ResultRace]:
        finalized = []
        for race_num, data in venue_races.items():
            if not data["runners"]: continue

            info = data["info"]
            post_time = info.get("post_time", "12:00")
            # DRF times like "0100" (1:00 PM).
            if len(post_time) == 4 and post_time.isdigit():
                hr = int(post_time[:2])
                if hr < 11: hr += 12
                formatted_time = f"{hr}:{post_time[2:]}"
            else:
                formatted_time = "12:00"

            start_time = build_start_time(date_str, formatted_time)

            finalized.append(ResultRace(
                id=fortuna.generate_race_id("drf", venue, start_time, race_num),
                venue=venue,
                race_number=race_num,
                start_time=start_time,
                runners=data["runners"],
                discipline="Thoroughbred",
                is_handicap=info.get("is_handicap"),
                source=self.SOURCE_NAME,
                is_fully_parsed=True
            ))
        return finalized


# -- NYRABETS RESULTS ADAPTER ------------------------------------------------

class NYRABetsResultsAdapter(PageFetchingResultsAdapter):
    """
    DEPRECATED: Internal JSON API (brk0201) returns 403.
    Scheduled for replacement with Next.js __NEXT_DATA__ parser.
    """
    SOURCE_NAME = "NYRABetsResults"
    BASE_URL = "https://www.nyrabets.com"
    API_URL = "https://brk0201-iapi-webservice.nyrabets.com"

    def _configure_fetch_strategy(self) -> fortuna.FetchStrategy:
        return fortuna.FetchStrategy(
            primary_engine=fortuna.BrowserEngine.CURL_CFFI,
            timeout=45
        )

    def _get_headers(self) -> Dict[str, str]:
        # Minimal headers to satisfy the internal API (Fix 3)
        return {
            "Origin": "https://www.nyrabets.com",
            "Referer": "https://www.nyrabets.com/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Requested-With": "XMLHttpRequest"
        }

    async def _fetch_data(self, date_str: str) -> Optional[Any]:
        # 1. Get Cards for the date
        nyra_date = f"{date_str}T00:00:00.000"
        header = {
            "version": 2, "fragmentLanguage": "Javascript", "fragmentVersion": "", "clientIdentifier": "nyra.1b"
        }
        cards_payload = {
            "header": header, "cohort": "A--", "wageringCohort": "NBI",
            "cardDate": nyra_date, "wantFeaturedContent": True
        }
        try:
            resp = await self.smart_fetcher.fetch(
                f"{self.API_URL}/ListCards.ashx",
                method="POST",
                data={"request": json.dumps(cards_payload)},
                headers=self._get_headers()
            )
            if not resp or not resp.text: return None
            cards_data = json.loads(resp.text)
            card_ids = [c["cardId"] for c in cards_data.get("cards", [])]
            if not card_ids: return None

            # 2. List Races
            races_payload = {
                "header": header, "cohort": "A--", "wageringCohort": "NBI", "cardIds": card_ids
            }
            resp = await self.smart_fetcher.fetch(
                f"{self.API_URL}/ListRaces.ashx",
                method="POST",
                data={"request": json.dumps(races_payload)},
                headers=self._get_headers()
            )
            if not resp or not resp.text: return None
            list_races_data = json.loads(resp.text)
            race_ids = [r["raceId"] for r in list_races_data.get("races", [])]
            if not race_ids: return None

            # 3. GetResults (chunked)
            all_results = []
            for i in range(0, len(race_ids), 50):
                chunk = race_ids[i:i+50]
                get_results_payload = {
                    "header": header, "cohort": "A--", "wageringCohort": "NBI", "raceIds": chunk
                }
                resp = await self.smart_fetcher.fetch(
                    f"{self.API_URL}/GetResults.ashx",
                    method="POST",
                    data={"request": json.dumps(get_results_payload)},
                    headers=self._get_headers()
                )
                if resp and resp.text:
                    chunk_data = json.loads(resp.text)
                    all_results.extend(chunk_data.get("races", []))
            return all_results
        except Exception as e:
            self.logger.error("NYRABetsResults fetch failed", error=str(e))
            return None

    def _parse_races(self, raw_data: Any) -> List[ResultRace]:
        if not raw_data: return []
        results = []
        for r in raw_data:
            venue = fortuna.normalize_venue_name(r["raceMeetingName"])
            race_num = r["raceNumber"]
            try:
                # 2026-02-19T10:48:37Z
                start_time = datetime.strptime(r["postTime"], "%Y-%m-%dT%H:%M:%SZ")
            except Exception: continue
            runners = []
            for runner in r.get("runners", []):
                name = runner.get("runnerName")
                num_str = "".join(filter(str.isdigit, str(runner.get("programNumber", "0"))))
                number = int(num_str) if num_str else 0
                fpos = runner.get("finishPosition")
                pos = int(fpos) if fpos and fpos.isdigit() else None
                final_odds = runner.get("currentWinPrice")
                win_payout = (float(final_odds) * 2.0) if final_odds else None
                runners.append(ResultRunner(
                    name=name, number=number, position=str(pos) if pos else None,
                    position_numeric=pos, win_payout=win_payout, final_win_odds=final_odds,
                    odds_source="extracted" if final_odds else None
                ))
            if not runners: continue
            is_handicap = None
            if r.get("raceMeetingName") and "HANDICAP" in r["raceMeetingName"].upper():
                is_handicap = True

            results.append(ResultRace(
                id=fortuna.generate_race_id("nyrab", venue, start_time, race_num),
                venue=venue, race_number=race_num, start_time=start_time,
                runners=runners, discipline="Thoroughbred", is_handicap=is_handicap,
                source=self.SOURCE_NAME
            ))
        return results

class EquibaseResultsAdapter(PageFetchingResultsAdapter):
    """Equibase summary charts — primary US thoroughbred results source."""

    SOURCE_NAME = "EquibaseResults"
    BASE_URL = "https://www.equibase.com"
    HOST = "www.equibase.com"
    IMPERSONATE = "chrome120"
    TIMEOUT = 60

    _IMPERSONATION_FALLBACKS: Final[tuple[str, ...]] = (
        "chrome120", "chrome110", "safari15_5",
    )
    _MIN_CONTENT_LENGTH: Final[int] = 2000

    def _configure_fetch_strategy(self) -> fortuna.FetchStrategy:
        return fortuna.FetchStrategy(
            primary_engine=fortuna.BrowserEngine.CURL_CFFI,
            timeout=45
        )

    def _get_headers(self) -> dict:
        return {"Referer": "https://www.equibase.com/"}

    # -- link discovery ----------------------------------------------------

    async def _discover_result_links(self, date_str: str) -> Set[str]:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            self.logger.error("Invalid date format", date=date_str)
            return set()

        index_urls = [
            "/static/chart/summary/index.html?SAP=TN",
            f"/static/chart/summary/index.html?date={dt.strftime('%m/%d/%Y')}",
            f"/static/chart/summary/{dt.strftime('%m%d%y')}sum.html",
            f"/static/chart/summary/{dt.strftime('%Y%m%d')}sum.html",
        ]

        resp = await self._fetch_first_valid_index(index_urls)
        if not resp or not resp.text:
            self.logger.warning("No response from Equibase index", date=date_str)
            return set()

        self._save_debug_snapshot(resp.text, f"eqb_results_index_{date_str}")
        initial_links = self._extract_track_links(resp.text, dt)

        return await self._resolve_index_links(initial_links, dt)

    async def _fetch_first_valid_index(self, urls: list[str]) -> Any:
        """Try each URL with multiple impersonations until valid content."""
        for url in urls:
            # Try only primary impersonation first to save time
            for imp in self._IMPERSONATION_FALLBACKS[:1]:
                try:
                    resp = await self.make_request(
                        "GET", url, headers=self._get_headers(), impersonate=imp,
                    )
                    if (
                        resp and resp.text
                        and len(resp.text) > self._MIN_CONTENT_LENGTH
                        and "Pardon Our Interruption" not in resp.text
                        and "<table" in resp.text.lower()
                    ):
                        return resp

                    if resp and resp.text and "Pardon Our Interruption" in resp.text:
                         self.logger.warning("Equibase definitive block", url=url)
                         break # Try next URL instead of more impersonations
                except Exception:
                    continue
        return None

    async def _resolve_index_links(
        self, initial_links: Set[str], dt: datetime,
    ) -> Set[str]:
        """Resolve RaceCardIndex links to actual summary pages."""
        index_links = [ln for ln in initial_links if "RaceCardIndex" in ln]
        resolved = {ln for ln in initial_links if "RaceCardIndex" not in ln}

        if not index_links:
            return resolved

        self.logger.info("Resolving track indices", count=len(index_links))
        # GPT5 Fix: Clean and deduplicate links to avoid net::ERR_INVALID_ARGUMENT
        clean_index_links = [ln.strip() for ln in index_links if ln and ln.strip()]
        metadata = [{"url": ln, "race_number": 0} for ln in clean_index_links]
        index_pages = await self._fetch_race_pages_concurrent(
            metadata, self._get_headers(),
        )
        date_short = dt.strftime("%m%d%y")
        for p in index_pages:
            html = p.get("html")
            if not html:
                continue
            for pattern in (r'href="([^"]+)"', r'"URL":"([^"]+)"'):
                for m in re.findall(pattern, html):
                    normalised = m.replace("\\/", "/").replace("\\", "/")
                    if date_short in normalised and "sum.html" in normalised:
                        resolved.add(self._normalise_eqb_link(normalised))

        return resolved

    def _extract_track_links(self, html: str, dt: datetime) -> Set[str]:
        """Pull track-summary URLs from the index page."""
        parser = HTMLParser(html)
        raw_links: Set[str] = set()
        date_short = dt.strftime("%m%d%y")

        # Source 1: inline JSON
        for url_match in re.findall(r'"URL":"([^"]+)"', html):
            normalised = url_match.replace("\\/", "/").replace("\\", "/")
            if date_short in normalised and any(
                s in normalised for s in ("sum.html", "EQB.html", "RaceCardIndex")
            ):
                raw_links.add(normalised)

        # Source 2: <a> tags
        selectors_and_filters: list[tuple[str, Any]] = [
            ('table.display a[href*="sum.html"]', None),
            (
                'a[href*="/static/chart/summary/"]',
                lambda h: "index.html" not in h and "calendar.html" not in h,
            ),
            (
                "a",
                lambda h: (
                    re.search(r"[A-Z]{3}\d{6}(?:sum|EQB)\.html", h)
                    or (date_short in h and ("sum.html" in h.lower() or "eqb.html" in h.lower()))
                ) and "index.html" not in h and "calendar.html" not in h,
            ),
        ]
        for selector, extra_filter in selectors_and_filters:
            for a in parser.css(selector):
                href = (a.attributes.get("href") or "").replace("\\", "/")
                if not href:
                    continue
                if extra_filter and not extra_filter(href):
                    continue
                if not self._venue_matches(fortuna.node_text(a), href):
                    continue
                raw_links.add(href)

        if not raw_links:
            self.logger.warning("No track links in index", date=str(dt.date()))
            return set()

        self.logger.info("Track links extracted", count=len(raw_links))
        return {self._normalise_eqb_link(lnk) for lnk in raw_links}

    def _normalise_eqb_link(self, link: str) -> str:
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
        parser = HTMLParser(html)

        track_node = parser.css_first("h3") or parser.css_first("h2")
        if not track_node:
            self.logger.debug("No track header found", url=url)
            return []
        venue = fortuna.normalize_venue_name(fortuna.node_text(track_node))
        if not venue:
            return []

        all_tables = parser.css("table")
        indexed_race_tables: list[tuple[int, Node]] = []
        for i, table in enumerate(all_tables):
            header = table.css_first("thead tr th")
            if header and "Race" in fortuna.node_text(header):
                indexed_race_tables.append((i, table))

        races: List[ResultRace] = []
        for j, (idx, race_table) in enumerate(indexed_race_tables):
            try:
                next_idx = (
                    indexed_race_tables[j + 1][0]
                    if j + 1 < len(indexed_race_tables)
                    else len(all_tables)
                )
                dividend_tables = all_tables[idx + 1 : next_idx]
                exotics = extract_exotic_payouts(dividend_tables)

                race = self._parse_race_table(race_table, venue, date_str, exotics)
                if race:
                    races.append(race)
            except Exception:
                self.logger.debug("Failed to parse race table", exc_info=True)
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
        header_text = fortuna.node_text(header)

        race_match = re.search(r"Race\s+(\d+)", header_text)
        if not race_match:
            return None
        race_num = int(race_match.group(1))

        start_time = self._parse_header_time(header_text, date_str)

        runners = [
            r for row in table.css("tbody tr")
            if (r := self._parse_runner_row(row)) is not None
        ]
        if not runners:
            return None

        # S5 — extract race type (independent review item)
        race_type = None
        rt_match = re.search(r'(Maiden\s+\w+|Claiming|Allowance|Graded\s+Stakes|Stakes)', header_text, re.I)
        if rt_match: race_type = rt_match.group(1)

        is_handicap = None
        if "HANDICAP" in header_text.upper():
            is_handicap = True

        return ResultRace(
            id=self._make_race_id("eqb_res", venue, date_str, race_num),
            venue=venue,
            race_number=race_num,
            start_time=start_time,
            runners=runners,
            discipline="Thoroughbred",
            race_type=race_type,
            is_handicap=is_handicap,
            source=self.SOURCE_NAME,
            is_fully_parsed=True,
            **_apply_exotics_to_race(exotics),
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
        cols = row.css("td")
        if len(cols) < 3:
            return None

        name = fortuna.clean_text(fortuna.node_text(cols[2]))
        if not name or name.upper() in ("HORSE", "NAME", "RUNNER"):
            return None

        pos_text = fortuna.clean_text(fortuna.node_text(cols[0]))
        num_text = fortuna.clean_text(fortuna.node_text(cols[1]))

        odds_text = fortuna.clean_text(fortuna.node_text(cols[3])) if len(cols) > 3 else ""
        final_odds = fortuna.parse_odds_to_decimal(odds_text)

        odds_source = "extracted" if final_odds is not None else None

        # Advanced heuristic fallback (Jules Fix)
        if final_odds is None:
            final_odds = fortuna.SmartOddsExtractor.extract_from_node(row)
            if final_odds is not None:
                odds_source = "smart_extractor"

        win_pay = place_pay = show_pay = 0.0
        if len(cols) >= 7:
            win_pay = parse_currency_value(fortuna.node_text(cols[4]))
            place_pay = parse_currency_value(fortuna.node_text(cols[5]))
            show_pay = parse_currency_value(fortuna.node_text(cols[6]))

        return ResultRunner(
            name=name,
            number=_safe_int(num_text),
            position=pos_text,
            final_win_odds=final_odds,
            odds_source=odds_source,
            win_payout=win_pay,
            place_payout=place_pay,
            show_payout=show_pay,
        )


# -- RACING POST RESULTS ADAPTER ---------------------------------------------

class RacingPostResultsAdapter(PageFetchingResultsAdapter):
    """Racing Post results — UK / IRE thoroughbred and jumps."""

    SOURCE_NAME = "RacingPostResults"
    BASE_URL = "https://www.racingpost.com"
    HOST = "www.racingpost.com"
    IMPERSONATE = "chrome120"
    TIMEOUT = 60

    _RP_LINK_SELECTORS: Final[tuple[str, ...]] = (
        'a[data-test-selector="RC-meetingItem__link_race"]',
        'a[href*="/results/"]',
        ".ui-link.rp-raceCourse__panel__race__time",
        "a.rp-raceCourse__panel__race__time",
        ".rp-raceCourse__panel__race__time a",
        ".RC-meetingItem__link",
    )

    # -- link discovery ----------------------------------------------------

    async def _discover_result_links(self, date_str: str) -> Set[str]:
        url = f"/results/{date_str}"
        resp = await self.make_request("GET", url, headers=self._get_headers())
        if not resp or not resp.text:
            return set()
        if self._check_for_block(resp.text, url):
            return set()

        self._save_debug_snapshot(resp.text, f"rp_results_index_{date_str}")
        return self._extract_rp_links(HTMLParser(resp.text), date_str)

    def _extract_rp_links(self, parser: HTMLParser, date_str: str) -> Set[str]:
        links: Set[str] = set()
        for selector in self._RP_LINK_SELECTORS:
            for a in parser.css(selector):
                href = a.attributes.get("href", "")
                if not href:
                    continue
                if not self._venue_matches(fortuna.node_text(a), href):
                    continue
                if not self._link_matches_date(href, date_str):
                    continue
                if self._is_rp_race_link(href):
                    links.add(href)

        # Broad fallback — still date-gated
        if not links:
            for a in parser.css('a[href*="/results/"]'):
                href = a.attributes.get("href", "")
                if (
                    len(href.split("/")) >= 3
                    and self._link_matches_date(href, date_str)
                ):
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

        venue_node = (
            parser.css_first('*[data-test-selector="RC-courseHeader__name"]')
            or parser.css_first(".rp-raceTimeCourseName__course")
            or parser.css_first(".rp-course__name")
            or parser.css_first("h1")
        )
        if not venue_node:
            return None
        raw_venue = fortuna.node_text(venue_node)
        # RP often includes time and date in the venue node: "1:30 Market Rasen 17 Feb"
        # We capture the time (HH:MM) and then strip it and the date (DD MMM) patterns
        time_match = re.match(r"^(\d{1,2}:\d{2})\s*", raw_venue)
        race_time_str = time_match.group(1) if time_match else None

        raw_venue = re.sub(r"^\d{1,2}:\d{2}\s*", "", raw_venue)
        raw_venue = re.sub(r"[,\s]+\d{1,2}\s+[A-Z][a-z]{2}.*$", "", raw_venue)
        venue = fortuna.normalize_venue_name(raw_venue.strip())

        dividends = self._parse_tote_dividends(parser)
        trifecta_pay, trifecta_combo = self._exotic_from_dividends(dividends, "trifecta")
        superfecta_pay, superfecta_combo = self._exotic_from_dividends(dividends, "superfecta")

        race_num = self._extract_rp_race_number(parser)
        runners = self._parse_rp_runners(parser, dividends)
        if not runners:
            return None

        # S5 — extract race type (independent review item)
        race_type = None
        is_handicap = None
        header_node = parser.css_first(".rp-raceCourse__panel__race__info") or parser.css_first(".RC-course__info")
        if header_node:
            header_text = fortuna.node_text(header_node)
            rt_match = re.search(r'(Maiden\s+\w+|Claiming|Allowance|Graded\s+Stakes|Stakes)', header_text, re.I)
            if rt_match: race_type = rt_match.group(1)
            if "HANDICAP" in header_text.upper():
                is_handicap = True

        return ResultRace(
            id=self._make_race_id("rp_res", venue, date_str, race_num),
            venue=venue,
            race_number=race_num,
            start_time=build_start_time(date_str, race_time_str),
            runners=runners,
            discipline="Thoroughbred",
            race_type=race_type,
            is_handicap=is_handicap,
            source=self.SOURCE_NAME,
            trifecta_payout=trifecta_pay,
            trifecta_combination=trifecta_combo,
            superfecta_payout=superfecta_pay,
            superfecta_combination=superfecta_combo,
            official_dividends={
                k: parse_currency_value(v) for k, v in dividends.items()
            },
        )

    @staticmethod
    def _parse_tote_dividends(parser: HTMLParser) -> Dict[str, str]:
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
                label = fortuna.clean_text(fortuna.node_text(label_node))
                value = fortuna.clean_text(fortuna.node_text(val_node))
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
                combo = val.split("£")[-1].strip() if "£" in val else None
                return payout, combo
        return None, None

    @staticmethod
    def _extract_rp_race_number(parser: HTMLParser) -> int:
        for i, link in enumerate(
            parser.css('a[data-test-selector="RC-raceTime"]'),
        ):
            cls = link.attributes.get("class", "")
            if "active" in cls or "rp-raceTimeCourseName__time" in cls:
                return i + 1
        return _extract_race_number_from_text(parser) or 1

    def _parse_rp_runners(
        self,
        parser: HTMLParser,
        dividends: Dict[str, str],
    ) -> List[ResultRunner]:
        runners: List[ResultRunner] = []
        # Try both old and new selectors for runner rows
        rows = (
            parser.css(".rp-horseTable__table__row")
            or parser.css(".rp-horseTable__mainRow")
            or parser.css("tr.rp-horseTable__mainRow")
            or parser.css('.RC-runnerRow')
        )

        for row in rows:
            name_node = (
                row.css_first('*[data-test-selector="RC-cardPage-runnerName"]')
                or row.css_first(".rp-horseTable__horse__name")
                or row.css_first("a.rp-horseTable__horse__name")
                or row.css_first(".RC-runnerName")
            )
            if not name_node:
                continue
            name = fortuna.clean_text(fortuna.node_text(name_node))

            pos_node = (
                row.css_first('*[data-test-selector="RC-cardPage-runnerPosition"]')
                or row.css_first(".rp-horseTable__pos__number")
                or row.css_first(".rp-horseTable__pos")
            )
            pos = fortuna.clean_text(fortuna.node_text(pos_node)) if pos_node else None

            num_node = (
                row.css_first('*[data-test-selector="RC-cardPage-runnerNumber-no"]')
                or row.css_first(".rp-horseTable__saddleClothNo")
                or row.css_first(".RC-runnerNumber__no")
            )
            number = _safe_int(fortuna.node_text(num_node)) if num_node else 0

            place_payout = self._find_place_payout(name, dividends)

            sp_node = (
                row.css_first('*[data-test-selector="RC-cardPage-runnerPrice"]')
                or row.css_first(".rp-horseTable__horse__sp")
                or row.css_first(".RC-runnerPrice")
            )
            final_odds = (
                parse_fractional_odds(fortuna.clean_text(fortuna.node_text(sp_node)))
                if sp_node else 0.0
            )
            odds_source = "starting_price" if final_odds > 0.01 else None

            # Advanced heuristic fallback (Jules Fix)
            if final_odds <= 0.01:
                final_odds = fortuna.SmartOddsExtractor.extract_from_node(row) or 0.0
                if final_odds > 0.01:
                    odds_source = "smart_extractor"

            runners.append(ResultRunner(
                name=name,
                number=number,
                position=pos,
                place_payout=place_payout,
                final_win_odds=final_odds,
                odds_source=odds_source
            ))
        return runners

    @staticmethod
    def _find_place_payout(
        name: str, dividends: Dict[str, str],
    ) -> Optional[float]:
        name_lower = name.lower()
        for lbl, val in dividends.items():
            if "place" in lbl.lower() and name_lower in lbl.lower():
                return parse_currency_value(val)
        return None


class RacingPostUSAResultsAdapter(RacingPostResultsAdapter):
    """Racing Post results restricted to North American thoroughbred tracks.

    Uses Racing Post's battle-tested data (same format as UK results) as a
    more reliable replacement for Equibase, which aggressively blocks
    headless browsers.  Covers every major US/Canadian dirt and turf track
    including Turfway, Turf Paradise, Tampa Bay Downs, Gulfstream, Oaklawn,
    Aqueduct, and Woodbine.
    """

    SOURCE_NAME = "RacingPostUSAResults"
    # BASE_URL, HOST, IMPERSONATE, TIMEOUT, and all parsing methods are
    # inherited from RacingPostResultsAdapter — nothing else to change.

    # Racing Post uses hyphenated lowercase venue slugs in their result URLs.
    # e.g. https://www.racingpost.com/results/turfway-park/2026-02-18/834891/4
    _USA_TRACK_SLUGS: Final[frozenset[str]] = frozenset({
        # ── US thoroughbred ───────────────────────────────────────────────
        "aqueduct",
        "belmont-park",
        "belmont",
        "saratoga",
        "sar",
        "gulfstream-park",
        "gulfstream",
        "gulfstream-park-west",
        "santa-anita-park",
        "santa-anita",
        "del-mar",
        "dmr",
        "churchill-downs",
        "keeneland",
        "oaklawn-park",
        "oaklawn",
        "laurel-park",
        "laurel",
        "pimlico",
        "tampa-bay-downs",
        "tampa-bay",
        "tampa",
        "turfway-park",
        "turfway",
        "turf-paradise",
        "fair-grounds",
        "fair-grounds-race-course",
        "monmouth-park",
        "monmouth",
        "remington-park",
        "remington",
        "sam-houston-race-park",
        "sam-houston",
        "los-alamitos",
        "golden-gate-fields",
        "golden-gate",
        "penn-national",
        "charles-town",
        "presque-isle-downs",
        "presque-isle",
        "mahoning-valley",
        "finger-lakes",
        "hawthorne",
        "indiana-grand",
        "parx-racing",
        "parx",
        "suffolk-downs",
        "will-rogers-downs",
        "will-rogers",
        "emerald-downs",
        "sunland-park",
        "sunland",
        "lone-star-park",
        "lone-star",
        "delta-downs",
        "evangeline-downs",
        "evangeline",
        "ellis-park",
        "kentucky-downs",
        "thistledown",
        "belterra-park",
        "belterra",
        # ── US harness (in case RP covers them) ───────────────────────────
        "meadowlands",
        "yonkers-raceway",
        "pocono-downs",
        "dover-downs",
        "northfield-park",
        "scioto-downs",
        "miami-valley-raceway",
        "hoosier-park",
        # ── Canadian thoroughbred + harness ───────────────────────────────
        "woodbine",
        "fort-erie",
        "hastings-park",
        "woodbine-mohawk-park",
        "flamboro-downs",
        "western-fair",
        "rideau-carleton",
        "colonial-downs",
        "churchill-downs-synthetic",
        "del-mar-thoroughbred-club",
        "harrahs-louisiana-downs",
        "harrahs-philadelphia",
        "mountaineer-casino",
        "plainridge-park",
        "running-aces",
        "zia-park",
    })

    # Racing Post numeric track IDs corresponding to North American venues.
    # Added to handle new URL format /results/{id}/{slug}/{date}.
    _USA_TRACK_IDS: Final[frozenset[int]] = frozenset({
        393, # Gulfstream Park
        315, # Santa Anita
        37,  # Churchill Downs
        462, # Tampa Bay Downs
        1083,# Turfway Park
        28,  # Belmont Park
        12,  # Aqueduct
        311, # Saratoga
        380, # Oaklawn Park
        469, # Del Mar
        491, # Keeneland
        490, # Laurel Park
        182, # Pimlico
    })

    async def _discover_result_links(self, date_str: str) -> Set[str]:
        """Fetch RP's full results index and keep only North American links."""
        all_links = await super()._discover_result_links(date_str)
        usa_links = {
            link for link in all_links
            if self._is_usa_link(link)
        }
        self.logger.info(
            "RacingPostUSA: filtered links",
            total=len(all_links),
            usa=len(usa_links),
            date=date_str,
        )
        return usa_links

    @classmethod
    def _is_usa_link(cls, href: str) -> bool:
        """Return True if the RP result URL contains a known NA track slug or numeric ID."""
        href_lower = href.lower()

        # Extract the slug segment(s) from the URL path after /results/
        # Format can be /results/slug/date/... OR /results/id/slug/date/...
        parts = href_lower.split('/results/')
        if len(parts) < 2:
            return False

        segments = parts[1].split('/')
        if not segments:
            return False

        slug = segments[0]

        # Case 1: Numeric ID segment
        if slug.isdigit():
            track_id = int(slug)
            # If it's a known US ID, we're done
            if track_id in cls._USA_TRACK_IDS:
                return True

            # If not a known ID, check the next segment for a named slug
            if len(segments) > 1:
                slug = segments[1]
                # Fall through to named slug check
            else:
                return False

        # Case 2: Named slug match (exact segment match to prevent false positives)
        return slug in cls._USA_TRACK_SLUGS


# -- AT THE RACES RESULTS ADAPTER ---------------------------------------------

class AtTheRacesResultsAdapter(PageFetchingResultsAdapter):
    """At The Races results — UK / IRE."""

    SOURCE_NAME = "AtTheRacesResults"
    BASE_URL = "https://www.attheraces.com"
    HOST = "www.attheraces.com"
    IMPERSONATE = "chrome120"
    TIMEOUT = 60

    def _configure_fetch_strategy(self) -> fortuna.FetchStrategy:
        return fortuna.FetchStrategy(
            primary_engine=fortuna.BrowserEngine.CURL_CFFI,
            enable_js=False,
            stealth_mode="camouflage",
            timeout=self.TIMEOUT,
        )

    _ATR_LINK_SELECTORS: Final[tuple[str, ...]] = (
        "a[href*='/results/']",
        "a[href*='/racecard/']",
        "a[data-test-selector*='result']",
        ".meeting-summary a",
        ".p-results__item a",
        ".p-meetings__item a",
        ".p-results-meeting a",
    )

    _ATR_RUNNER_SELECTORS: Final[tuple[str, ...]] = (
        ".result-racecard__row",
        ".card-cell--horse",
        ".card-entry",
        "atr-result-horse",
        "div[class*='RacecardResultItem']",
        ".p-results__item",
    )

    # -- link discovery ----------------------------------------------------

    async def _discover_result_links(self, date_str: str) -> Set[str]:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        index_urls = [
            f"/results/{date_str}",
            f"/results/{dt.strftime('%d-%B-%Y')}",
            f"/results/international/{date_str}",
            f"/results/international/{dt.strftime('%d-%B-%Y')}",
        ]

        links: Set[str] = set()
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
                links.update(self._extract_atr_links(resp.text, date_str))
            except Exception:
                self.logger.debug("ATR index fetch failed", url=url, exc_info=True)
        return links

    def _extract_atr_links(self, html: str, date_str: str) -> Set[str]:
        parser = HTMLParser(html)
        links: Set[str] = set()
        for selector in self._ATR_LINK_SELECTORS:
            for a in parser.css(selector):
                href = a.attributes.get("href", "")
                if not href:
                    continue
                if not self._venue_matches(fortuna.node_text(a), href):
                    continue
                if not self._link_matches_date(href, date_str):
                    continue
                if self._is_atr_race_link(href):
                    full = href if href.startswith("http") else f"{self.BASE_URL}{href}"
                    links.add(full)
        return links

    @staticmethod
    def _is_atr_race_link(href: str) -> bool:
        if "userprofile" in href:
            return False
        return bool(
            re.search(r"/(?:results|racecard)/.*?/\d{4}", href)
            or re.search(r"/(?:results|racecard)/\d{2}-.*?-\d{4}/", href)
            or re.search(r"/(?:results|racecard)/.*?/\d+$", href)
            or (("/results/" in href or "/racecard/" in href) and len(href.split("/")) >= 4)
        )

    # -- single-race page parsing ------------------------------------------

    def _parse_race_page(
        self, html: str, date_str: str, url: str,
    ) -> Optional[ResultRace]:
        parser = HTMLParser(html)

        venue, race_time_str = self._extract_atr_venue(parser)
        if not venue:
            return None

        # Robust race number extraction from URL (last numeric part)
        url_match = re.search(r"/(\d+)$", url.rstrip("/"))
        race_num = int(url_match.group(1)) if url_match else 1

        runners = self._parse_atr_runners(parser)

        div_table = (
            parser.css_first(".result-racecard__dividends-table")
            or next((t for t in parser.css("table") if "Trifecta" in t.text()), None)
        )
        exotics: Dict[str, Tuple[Optional[float], Optional[str]]] = {}
        if div_table:
            exotics = extract_exotic_payouts([div_table])
            self._map_place_payouts(div_table, runners)

        if not runners:
            return None

        # S5 — extract race type (independent review item)
        race_type = None
        header_node = parser.css_first(".race-header__details--secondary") or parser.css_first(".race-header")
        if header_node:
            rt_match = re.search(r'(Maiden\s+\w+|Claiming|Allowance|Graded\s+Stakes|Stakes)', fortuna.node_text(header_node), re.I)
            if rt_match: race_type = rt_match.group(1)

        return ResultRace(
            id=self._make_race_id("atr_res", venue, date_str, race_num),
            venue=venue,
            race_number=race_num,
            start_time=build_start_time(date_str, race_time_str),
            runners=runners,
            discipline="Thoroughbred",
            race_type=race_type,
            source=self.SOURCE_NAME,
            **_apply_exotics_to_race(exotics),
        )

    @staticmethod
    def _extract_atr_venue(parser: HTMLParser) -> Tuple[Optional[str], Optional[str]]:
        header = (
            parser.css_first(".race-header__details--primary")
            or parser.css_first(".racecard-header")
            or parser.css_first(".race-header")
        )
        if not header:
            return None, None
        venue_node = (
            header.css_first("h2")
            or header.css_first("h1")
            or header.css_first(".track-name")
        )
        if not venue_node:
            return None, None
        venue_text = fortuna.node_text(venue_node)
        # Strip leading time if present (e.g. "14:20 Southwell")
        time_match = re.match(r"^(\d{1,2}:\d{2})\s*", venue_text)
        race_time = time_match.group(1) if time_match else None
        venue_text = re.sub(r"^\d{1,2}:\d{2}\s*", "", venue_text)
        return fortuna.normalize_venue_name(venue_text), race_time

    def _parse_atr_runners(self, parser: HTMLParser) -> List[ResultRunner]:
        # Favor the full-result tab if present (new layout)
        tab = parser.css_first("#tab-full-result")
        if tab:
            rows = tab.css(".card-entry")
        else:
            rows = []
            for selector in self._ATR_RUNNER_SELECTORS:
                rows = parser.css(selector)
                if rows:
                    break

        runners: List[ResultRunner] = []
        for row in rows:
            name_node = (
                row.css_first(".horse__link")
                or row.css_first(".result-racecard__horse-name a")
                or row.css_first(".horse-name a")
                or row.css_first("a[href*='/form/horse/']")
                or row.css_first("[class*='HorseName']")
            )
            if not name_node:
                continue

            pos_node = (
                row.css_first(".card-no-draw .p--large")
                or row.css_first(".result-racecard__pos")
                or row.css_first(".pos")
                or row.css_first(".position")
                or row.css_first("[class*='Position']")
            )

            num_node = (
                row.css_first(".card-cell--horse h2 span")
                or row.css_first(".result-racecard__saddle-cloth")
            )

            # Saddle cloth might have a dot, e.g. "1."
            num_text = fortuna.node_text(num_node)
            if num_text and "." in num_text:
                num_text = num_text.split(".")[0]

            odds_node = (
                row.css_first(".card-cell--odds")
                or row.css_first(".result-racecard__odds")
            )

            final_odds = (
                parse_fractional_odds(fortuna.clean_text(fortuna.node_text(odds_node)))
                if odds_node else 0.0
            )
            odds_source = "starting_price" if final_odds > 0.01 else None

            # Advanced heuristic fallback (Jules Fix)
            if final_odds <= 0.01:
                final_odds = fortuna.SmartOddsExtractor.extract_from_node(row) or 0.0
                if final_odds > 0.01:
                    odds_source = "smart_extractor"

            runners.append(ResultRunner(
                name=fortuna.clean_text(fortuna.node_text(name_node)),
                number=_safe_int(num_text) if num_text else 0,
                position=fortuna.clean_text(fortuna.node_text(pos_node)) if pos_node else None,
                final_win_odds=final_odds,
                odds_source=odds_source
            ))
        return runners

    @staticmethod
    def _map_place_payouts(div_table: Node, runners: List[ResultRunner]) -> None:
        for row in div_table.css("tr"):
            row_text = fortuna.node_text(row).lower()
            if "place" not in row_text:
                continue
            cols = row.css("td")
            if len(cols) < 2:
                continue
            p_name = fortuna.clean_text(fortuna.node_text(cols[0]).replace("Place", "").strip())
            p_val = parse_currency_value(fortuna.node_text(cols[1]))
            p_name_lower = p_name.lower()
            for runner in runners:
                if (
                    runner.name.lower() in p_name_lower
                    or p_name_lower in runner.name.lower()
                ):
                    runner.place_payout = p_val


# -- SPORTING LIFE RESULTS ADAPTER -------------------------------------------

class AtTheRacesGreyhoundResultsAdapter(PageFetchingResultsAdapter):
    """At The Races Greyhound results."""

    SOURCE_NAME = "AtTheRacesGreyhoundResults"
    BASE_URL = "https://greyhounds.attheraces.com"
    HOST = "greyhounds.attheraces.com"
    IMPERSONATE = "chrome120"
    TIMEOUT = 45

    async def _discover_result_links(self, date_str: str) -> Set[str]:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            url_date = dt.strftime("%d-%B-%Y")
        except ValueError:
            url_date = date_str

        index_url = f"/results/{url_date}"
        try:
            resp = await self.make_request(
                "GET", index_url, headers=self._get_headers(),
            )
            if not resp or not resp.text:
                return set()
            self._save_debug_snapshot(resp.text, f"atr_grey_results_index_{url_date}")
            parser = HTMLParser(resp.text)

            # Strategy 1: page-content JSON payload (most reliable)
            links = self._extract_grey_links_from_payload(parser, date_str)
            if links:
                return links

            # Strategy 2: standard <a> tag scraping
            return self._extract_grey_links_from_html(parser, date_str)
        except Exception:
            self.logger.debug("ATR Greyhound index fetch failed", exc_info=True)
            return set()

    def _extract_grey_links_from_payload(self, parser: HTMLParser, date_str: str) -> Set[str]:
        links: Set[str] = set()
        pc = parser.css_first("page-content")
        if not pc:
            return links

        items_raw = pc.attributes.get(":items") or pc.attributes.get(":modules")
        if not items_raw:
            return links

        try:
            # Unescape and parse JSON
            import html as py_html
            modules = json.loads(py_html.unescape(items_raw))
            for module in modules:
                # Look for result items
                for item in module.get("data", {}).get("items", []):
                    # Check both 'items' list and direct 'cta'
                    sub_items = item.get("items", [item])
                    for sub in sub_items:
                        href = sub.get("cta", {}).get("href")
                        if href and "/result/" in href:
                            if self._link_matches_date(href, date_str):
                                full = href if href.startswith("http") else f"{self.BASE_URL}{href}"
                                links.add(full)

            if not links:
                module_types = [m.get('type') for m in modules[:10]]
                self.logger.warning('ATR Grey: no links found in payload', module_types=module_types, date=date_str)

        except Exception:
            self.logger.debug("Failed to parse ATR Grey JSON payload", exc_info=True)

        return links

    def _extract_grey_links_from_html(self, parser: HTMLParser, date_str: str) -> Set[str]:
        links: Set[str] = set()
        # ATR Greyhound uses /result/ for individual pages and /results/ for index
        for a in parser.css("a[href*='/result/']"):
            href = a.attributes.get("href", "")
            if not href:
                continue
            if not self._link_matches_date(href, date_str):
                continue
            if re.search(r"/result/.*?/.*?/\d+", href):
                full = (
                    href if href.startswith("http") else f"{self.BASE_URL}{href}"
                )
                links.add(full)
        return links

    def _parse_page(
        self, html: str, date_str: str, url: str,
    ) -> List[ResultRace]:
        parser = HTMLParser(html)
        pc = parser.css_first("page-content")
        if pc:
            items_raw = pc.attributes.get(":items") or pc.attributes.get(
                ":modules",
            )
            if items_raw:
                try:
                    import html as py_html
                    modules = json.loads(py_html.unescape(items_raw))
                    return self._parse_from_modules(modules, date_str, url, html=html)
                except Exception:
                    self.logger.debug(
                        "Failed to parse ATR Grey JSON", exc_info=True,
                    )
        return []

    def _parse_races(self, raw_data: Any) -> List["ResultRace"]:
        """Override to assign sequential race numbers per venue.

        The parent implementation calls _parse_page → _parse_from_modules for
        each fetched URL.  Because ATR greyhound URLs contain no race number,
        every race comes back with race_number = 1.  Here we fix that by
        sorting each venue's races by start_time and numbering them 1, 2, 3…
        """
        races: List[ResultRace] = super()._parse_races(raw_data)
        if not races:
            return races

        # Group by canonical venue + date so multi-venue days don't bleed
        groups: Dict[tuple, List[ResultRace]] = defaultdict(list)
        for r in races:
            key = (
                fortuna.get_canonical_venue(r.venue),
                r.start_time.strftime("%Y%m%d"),
            )
            groups[key].append(r)

        fixed: List[ResultRace] = []
        for group in groups.values():
            group.sort(key=lambda r: r.start_time)
            for i, race in enumerate(group, start=1):
                date_str = race.start_time.strftime("%Y-%m-%d")
                fixed.append(
                    race.model_copy(update={
                        "race_number": i,
                        "id": self._make_race_id(
                            "atrg_res", race.venue, date_str, i,
                        ),
                    })
                )

        return fixed

    def _parse_from_modules(
        self, modules: List[Any], date_str: str, url: str, html: str = "",
    ) -> List[ResultRace]:
        races: List[ResultRace] = []
        venue, race_time_str, race_num = "", "", 1

        # Initial Heuristic: Extract venue and time from HTML title if modules are sparse
        if html:
            title_match = re.search(
                r"<title>\s*(\d{1,2}:\d{2})\s+([^|]+?)\s+Greyhound", html,
            )
            if title_match:
                race_time_str = title_match.group(1)
                venue = fortuna.normalize_venue_name(title_match.group(2))

        # URL fallback if race_num wasn't found in title (only check last segment)
        if race_num == 1:
            last_segment = url.rstrip("/").split("/")[-1]
            if last_segment.isdigit():
                val = int(last_segment)
                if 1 <= val <= 15:
                    race_num = val

        for module in modules:
            m_type, m_data = module.get("type"), module.get("data", {})
            if m_type == "RacecardHero":
                venue = fortuna.normalize_venue_name(m_data.get("track", ""))
                race_time_str = m_data.get("time", "")
                # Prefer race number from module data
                rn = m_data.get("raceNumber")
                if rn and isinstance(rn, (int, str)) and str(rn).isdigit():
                    race_num = int(rn)
            elif m_type in ("RaceResult", "RacecardResultForm"):
                runners: List[ResultRunner] = []
                # Handle both module structures
                items = m_data.get("items")
                if items is None and "form" in m_data:
                    items = m_data["form"].get("data", [])

                for item in items or []:
                    name = item.get("name", "")
                    num = item.get("trap") or item.get("number") or 0
                    pos = item.get("position")

                    # startingPrice might be "11/4" or numeric
                    sp_raw = str(item.get("sp") or item.get("startingPrice") or "0")
                    final_odds = parse_fractional_odds(sp_raw)

                    runners.append(ResultRunner(
                        name=name,
                        number=num,
                        position=str(pos) if pos else None,
                        final_win_odds=final_odds if final_odds > 0 else None,
                        odds_source="starting_price" if final_odds > 0 else None
                    ))

                if runners and venue:
                    races.append(ResultRace(
                        id=self._make_race_id(
                            "atr_grey_res", venue, date_str, race_num,
                        ),
                        venue=venue,
                        race_number=race_num,
                        start_time=build_start_time(date_str, race_time_str),
                        runners=runners,
                        discipline="Greyhound",
                        source=self.SOURCE_NAME,
                    ))
        return races


class SportingLifeResultsAdapter(PageFetchingResultsAdapter):
    """Sporting Life results (UK / IRE / International)."""

    SOURCE_NAME = "SportingLifeResults"
    BASE_URL = "https://www.sportinglife.com"
    HOST = "www.sportinglife.com"
    IMPERSONATE = "chrome120"
    TIMEOUT = 45

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
        return self._extract_sl_links(resp.text, date_str)

    def _extract_sl_links(self, html: str, date_str: str) -> Set[str]:
        links: Set[str] = set()
        for a in HTMLParser(html).css("a[href*='/racing/results/']"):
            href = a.attributes.get("href", "")
            if not href:
                continue
            if not self._venue_matches(fortuna.node_text(a), href):
                continue
            if not self._link_matches_date(href, date_str):
                continue
            if re.search(r"/results/\d{4}-\d{2}-\d{2}/.+/\d+/", href):
                links.add(href)
        return links

    # -- page parsing (two strategies) -------------------------------------

    def _parse_race_page(
        self, html: str, date_str: str, url: str,
    ) -> Optional[ResultRace]:
        parser = HTMLParser(html)

        # Strategy 1: Next.js JSON payload (most reliable)
        script = parser.css_first("script#__NEXT_DATA__")
        if script:
            race = self._parse_from_next_data(fortuna.node_text(script), date_str)
            if race:
                return race

        # Strategy 2: HTML scrape fallback
        return self._parse_from_html(parser, date_str)

    def _parse_from_next_data(
        self, script_text: str, date_str: str,
    ) -> Optional[ResultRace]:
        try:
            data = json.loads(script_text)
        except json.JSONDecodeError:
            self.logger.debug("Invalid __NEXT_DATA__", exc_info=True)
            return None

        race_data = data.get("props", {}).get("pageProps", {}).get("race", {})
        if not race_data:
            return None

        summary = race_data.get("race_summary", {})
        venue = fortuna.normalize_venue_name(
            summary.get("course_name", "Unknown"),
        )
        race_num = (
            race_data.get("race_number")
            or summary.get("race_number")
            or 1
        )
        date_val = summary.get("date", date_str)
        start_time = build_start_time(date_val, summary.get("time"))

        runners = self._runners_from_json(race_data)
        if not runners:
            return None

        trifecta_pay = find_nested_value(race_data, "trifecta")
        superfecta_pay = find_nested_value(race_data, "superfecta")

        self._apply_place_payouts_from_csv(
            race_data.get("place_win", ""), runners,
        )

        # S5 — extract race type (independent review item)
        race_type = None
        # Try summary header or card info
        header_text = summary.get("race_title") or summary.get("race_name") or ""
        rt_match = re.search(r'(Maiden\s+\w+|Claiming|Allowance|Graded\s+Stakes|Stakes)', header_text, re.I)
        if rt_match: race_type = rt_match.group(1)

        is_handicap = summary.get("has_handicap")
        return ResultRace(
            id=self._make_race_id("sl_res", venue, date_val, race_num),
            venue=venue,
            race_number=race_num,
            start_time=start_time,
            runners=runners,
            discipline="Thoroughbred",
            race_type=race_type,
            is_handicap=is_handicap,
            trifecta_payout=trifecta_pay,
            superfecta_payout=superfecta_pay,
            source=self.SOURCE_NAME,
        )

    @staticmethod
    def _runners_from_json(race_data: dict) -> List[ResultRunner]:
        items = race_data.get("rides") or race_data.get("runners", [])
        runners: List[ResultRunner] = []
        for item in items:
            horse = item.get("horse", {})
            name = horse.get("name") or item.get("name")
            if not name:
                continue
            sp_raw = (
                item.get("starting_price")
                or item.get("sp")
                or item.get("betting", {}).get("current_odds", "")
            )
            final_odds = parse_fractional_odds(str(sp_raw))
            runners.append(ResultRunner(
                name=name,
                number=item.get("cloth_number") or item.get("saddle_cloth_number", 0),
                position=str(item.get("finish_position", item.get("position", ""))),
                final_win_odds=final_odds,
                odds_source="starting_price" if final_odds else None
            ))
        return runners

    @staticmethod
    def _apply_place_payouts_from_csv(
        place_csv: str,
        runners: List[ResultRunner],
    ) -> None:
        if not isinstance(place_csv, str) or not place_csv:
            return
        pays = [parse_currency_value(p) for p in place_csv.split(",")]
        for runner in runners:
            pos = runner.position_numeric
            if pos and 1 <= pos <= len(pays):
                runner.place_payout = pays[pos - 1]

    def _parse_from_html(
        self, parser: HTMLParser, date_str: str,
    ) -> Optional[ResultRace]:
        header = parser.css_first("h1")
        if not header:
            return None

        match = re.match(
            r"(\d{1,2}:\d{2})\s+(.+)\s+Result",
            fortuna.clean_text(fortuna.node_text(header)),
        )
        if not match:
            return None

        time_str = match.group(1)
        venue = fortuna.normalize_venue_name(match.group(2))
        start_time = build_start_time(date_str, time_str)

        runners: List[ResultRunner] = []
        for row in parser.css(
            'div[class*="ResultRunner__StyledResultRunnerWrapper"]',
        ):
            name_node = row.css_first('a[class*="ResultRunner__StyledHorseName"]')
            if not name_node:
                continue
            pos_node = row.css_first(
                'div[class*="ResultRunner__StyledRunnerPositionContainer"]',
            )

            # Extract odds from HTML fallback (Jules Fix)
            final_odds = fortuna.SmartOddsExtractor.extract_from_node(row)

            runners.append(ResultRunner(
                name=fortuna.clean_text(fortuna.node_text(name_node)),
                number=0,
                position=fortuna.clean_text(fortuna.node_text(pos_node)) if pos_node else None,
                final_win_odds=final_odds,
                odds_source="smart_extractor" if final_odds else None
            ))

        if not runners:
            return None

        return ResultRace(
            id=self._make_race_id("sl_res", venue, date_str, 1),
            venue=venue,
            race_number=1,
            start_time=start_time,
            runners=runners,
            discipline="Thoroughbred",
            source=self.SOURCE_NAME,
        )


# -- STANDARDBRED CANADA RESULTS ADAPTER -------------------------------------

class StandardbredCanadaResultsAdapter(PageFetchingResultsAdapter):
    """Standardbred Canada harness results."""

    SOURCE_NAME = "StandardbredCanadaResults"
    BASE_URL = "https://standardbredcanada.ca"
    HOST = "standardbredcanada.ca"
    IMPERSONATE = "chrome120"
    TIMEOUT = 45

    _TRACK_CODES: Final[tuple[str, ...]] = (
        "lonn", "lon", "wbsbsn", "wbsb", "flmn", "flm", "flmdn", "ridcn", "rid",
        "trrn", "kdun", "geodn", "clntn", "hanon", "dresn", "grvr",
        "leam", "kaww", "wood",
    )

    _PAYOUT_RE = re.compile(
        r"\$?\d+\s+([\w\s-]+)\(([^)]+)\)\s+paid\s+([\d,.]+)",
        re.IGNORECASE,
    )

    _RUNNER_RE = re.compile(
        r"(\d+)-(.+?)\s{2,}([\d.]+)?\s*([\d.]+)?\s*([\d.]+)?\s*([\d.]+)?",
    )

    _BET_NAME_MAP: Final[Dict[str, str]] = {
        "EXACTOR": "exacta",
        "TRIACTOR": "trifecta",
        "SUPERFECTA": "superfecta",
    }

    def _configure_fetch_strategy(self) -> fortuna.FetchStrategy:
        return fortuna.FetchStrategy(
            primary_engine=fortuna.BrowserEngine.CURL_CFFI,
            enable_js=False,
            stealth_mode="fast",
            timeout=self.TIMEOUT,
        )

    async def _discover_result_links(self, date_str: str) -> Set[str]:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        date_short = dt.strftime("%m%d")
        return {
            f"/racing/results/data/r{date_short}{track}.dat"
            for track in self._TRACK_CODES
        }

    def _parse_page(
        self, html: str, date_str: str, url: str,
    ) -> List[ResultRace]:
        if "ALL RACES CANCELLED" in html.upper():
            self.logger.warning("Races cancelled for venue", url=url)
            return []

        parser = HTMLParser(html)
        # Relaxed venue detection: try headers first, then scan text if needed (Jules Fix)
        venue_node = parser.css_first("h1#condition-name") or parser.css_first("h1") or parser.css_first("h2") or parser.css_first("strong")

        venue_text = ""
        if venue_node:
            venue_text = fortuna.node_text(venue_node).split("-")[0].strip()

        if not venue_text:
            # Fallback: search for track name in first 1000 chars
            for track_match in ["Western Fair", "London", "Mohawk", "Flamboro", "Rideau", "Woodbine"]:
                if track_match.lower() in html[:1000].lower():
                    venue_text = track_match
                    break

        if not venue_text:
            # Last fallback: extract from URL if possible
            url_lower = url.lower()
            for track_code, track_name in [
                ("lon", "Western Fair"), ("wbsb", "Mohawk"), ("flm", "Flamboro"),
                ("rid", "Rideau"), ("wood", "Woodbine")
            ]:
                if track_code in url_lower:
                    venue_text = track_name
                    break

        if not venue_text:
            return []

        venue = fortuna.normalize_venue_name(venue_text)

        # Split by race anchors (Legacy format)
        blocks = re.split(r"<a name='N?(\d+)'></a>", html)

        races: List[ResultRace] = []
        if len(blocks) > 1:
            for i in range(1, len(blocks), 2):
                try:
                    race_num = int(blocks[i])
                    race_html = blocks[i + 1].split("<hr/>")[0]
                    race = self._parse_race_block(race_html, venue, date_str, race_num)
                    if race:
                        races.append(race)
                except (ValueError, IndexError):
                    self.logger.debug("Failed to parse SC race block", exc_info=True)

        # If no races found, try more aggressive splitting (Newer format)
        if not races:
            # Try splitting by "RACE #" or similar headers in the page text
            # We look for "RACE" followed by a number, often in bold or as a text header
            blocks = re.split(r"(?:RACE\s*#?\s*|Jump to race:.*?\s+)(\d+)\s*", html, flags=re.IGNORECASE)

            # If still nothing, try looking for the Horse table headers which usually mark a race
            if len(blocks) <= 1:
                blocks = re.split(r"(?:^|\n)\s*(\d+)\s+(?:[A-Z][a-z]+\s+){1,3}(?:\(L\)|\(ES\))?\s+\d+\s+[\d/]+[A-Z]*", html)

            if len(blocks) > 1:
                for i in range(1, len(blocks), 2):
                    try:
                        race_num = int(blocks[i])
                        if race_num > 20: continue # Sanity check

                        race_html = blocks[i + 1]
                        # Trim until the next "RACE" or end of page
                        next_race_match = re.search(r"RACE\s*#?\s*\d+", race_html, re.I)
                        if next_race_match:
                            race_html = race_html[:next_race_match.start()]

                        race = self._parse_race_block(race_html, venue, date_str, race_num)
                        if race:
                            races.append(race)
                    except (ValueError, IndexError):
                        pass

        return races

    def _parse_race_block(
        self, html: str, venue: str, date_str: str, race_num: int,
    ) -> Optional[ResultRace]:
        parser = HTMLParser(html)

        exotics: Dict[str, Tuple[Optional[float], Optional[str]]] = {}
        runners_map: Dict[int, ResultRunner] = {}

        # 1. Parse payouts and finishers from <strong> tags
        for s in parser.css("strong"):
            txt = fortuna.node_text(s).strip()

            # Check for payout line
            p_match = self._PAYOUT_RE.search(txt)
            if p_match:
                bet_name = p_match.group(1).strip().upper()
                combo = p_match.group(2).strip()
                payout = parse_currency_value(p_match.group(3))
                for key_name, exotic_key in self._BET_NAME_MAP.items():
                    if key_name in bet_name:
                        exotics[exotic_key] = (payout, combo)
                        break
                continue

            # Check for finisher line (Top 3)
            # Format: NUM-NAME   WIN_PAY  PLACE_PAY  SHOW_PAY  POOL
            m = self._RUNNER_RE.match(txt)
            if m:
                num = int(m.group(1))
                name = m.group(2).strip()

                # Handling column-aware payouts based on available numbers
                nums = [p for p in m.groups()[2:] if p is not None]
                win_pay = place_pay = show_pay = 0.0

                # Standardbred Canada format:
                # Winner: NUM-NAME WIN PLACE SHOW POOL
                # 2nd:    NUM-NAME PLACE SHOW POOL
                # 3rd:    NUM-NAME SHOW POOL
                if len(nums) == 4: # Winner
                    win_pay = parse_currency_value(nums[0])
                    place_pay = parse_currency_value(nums[1])
                    show_pay = parse_currency_value(nums[2])
                elif len(nums) == 3: # 2nd place
                    place_pay = parse_currency_value(nums[0])
                    show_pay = parse_currency_value(nums[1])
                elif len(nums) == 2: # 3rd place
                    show_pay = parse_currency_value(nums[0])

                runners_map[num] = ResultRunner(
                    name=name,
                    number=num,
                    win_payout=win_pay,
                    place_payout=place_pay,
                    show_payout=show_pay,
                )

        # 2. Parse Horse table for odds and positions (Source of Truth for ALL runners)
        in_horse_table = False
        for line in html.splitlines():
            clean_line = line.strip()
            if clean_line.startswith("Horse ") and "Odds" in clean_line:
                in_horse_table = True
                continue
            if in_horse_table:
                if not clean_line or "Time:" in clean_line or "Temp:" in clean_line or "---" in clean_line:
                    if clean_line and re.match(r"^\d+", clean_line):
                        pass # Continue if it looks like a runner
                    else:
                        in_horse_table = False
                        continue

                # Table line: 2   Forefather(L)               2    9/14T ... 8.60   T Schlatman
                # Relaxed regex to handle tight spacing in harness result tables (Jules Fix)
                rm = re.match(r"^(\d+)\s+(.+?)(?:\s{2,}|$)", clean_line)
                if rm:
                    num = int(rm.group(1))
                    name = rm.group(2).split("(")[0].strip()

                    # Extract odds (numeric value or fraction, possibly with *)
                    parts = clean_line.split()
                    final_odds = None
                    for p in reversed(parts[1:-1]): # Skip trainer at end
                        os = p.replace("*", "").strip()
                        if not os:
                            continue
                        try:
                            val = float(os)
                            if 0.0 <= val < 1000.0:
                                final_odds = val
                                break
                        except ValueError:
                            # Try parsing as fractional odds
                            f_val = fortuna.parse_odds_to_decimal(os)
                            if f_val is not None:
                                final_odds = float(f_val)
                                break

                    # Extract position from "Finish" column (usually 7th or 8th part)
                    # We pick the LAST one matching the pattern to avoid 1/4, 1/2, 3/4, Stretch calls
                    pos = None
                    for p in parts[3:]:
                        if "/" in p and re.match(r"^\d+[A-Z]*/", p):
                            pos = p.split("/")[0]

                    if num in runners_map:
                        runners_map[num].final_win_odds = final_odds
                        runners_map[num].odds_source = "extracted" if final_odds is not None else None
                        if pos: runners_map[num].position = pos
                    else:
                        runners_map[num] = ResultRunner(
                            name=name,
                            number=num,
                            position=pos,
                            final_win_odds=final_odds,
                            odds_source="extracted" if final_odds is not None else None
                        )

        runners = list(runners_map.values())
        if not runners:
            return None

        is_handicap = None
        if "HANDICAP" in html.upper():
            is_handicap = True

        return ResultRace(
            id=self._make_race_id("sc_res", venue, date_str, race_num),
            venue=venue,
            race_number=race_num,
            start_time=build_start_time(date_str),
            runners=runners,
            discipline="Harness",
            is_handicap=is_handicap,
            source=self.SOURCE_NAME,
            **_apply_exotics_to_race(exotics),
        )


# -- RACING AND SPORTS RESULTS ADAPTER ---------------------------------------

class RacingAndSportsResultsAdapter(PageFetchingResultsAdapter):
    """Racing & Sports results (AUS / NZ / International)."""

    SOURCE_NAME = "RacingAndSportsResults"
    BASE_URL = "https://www.racingandsports.com.au"
    HOST = "www.racingandsports.com.au"
    IMPERSONATE = "chrome120"
    TIMEOUT = 60

    async def _discover_result_links(self, date_str: str) -> Set[str]:
        url = f"/racing-results/{date_str}"
        resp = await self.make_request("GET", url, headers=self._get_headers())
        if not resp or not resp.text:
            return set()

        self._save_debug_snapshot(resp.text, f"ras_results_index_{date_str}")
        parser = HTMLParser(resp.text)
        links: Set[str] = set()

        for table in parser.css("table.table-index"):
            for row in table.css("tbody tr"):
                venue_cell = row.css_first("td.venue-name")
                if not venue_cell:
                    continue
                if not self._venue_matches(fortuna.node_text(venue_cell)):
                    continue
                for link in row.css("td a.race-link"):
                    race_url = link.attributes.get("href", "")
                    if race_url and self._link_matches_date(race_url, date_str):
                        links.add(race_url)
        return links

    def _parse_race_page(
        self, html: str, date_str: str, url: str,
    ) -> Optional[ResultRace]:
        parser = HTMLParser(html)

        header = parser.css_first("h1") or parser.css_first(".race-title")
        if not header:
            return None

        header_text = fortuna.node_text(header)
        parts = header_text.split("-")
        venue = fortuna.normalize_venue_name(parts[0].strip())

        race_num = 1
        if len(parts) > 1:
            num_match = re.search(r"Race\s+(\d+)", parts[1])
            if num_match:
                race_num = int(num_match.group(1))

        runners = self._parse_ras_runners_primary(parser)
        if not runners:
            runners = self._parse_ras_runners_fallback(parser)
        if not runners:
            return None

        exotics = extract_exotic_payouts(parser.css("table"))

        return ResultRace(
            id=self._make_race_id("ras_res", venue, date_str, race_num),
            venue=venue,
            race_number=race_num,
            start_time=build_start_time(date_str),
            runners=runners,
            discipline="Thoroughbred",
            source=self.SOURCE_NAME,
            **_apply_exotics_to_race(exotics),
        )

    @staticmethod
    def _parse_ras_runners_primary(parser: HTMLParser) -> List[ResultRunner]:
        runners: List[ResultRunner] = []
        for row in parser.css("tr.runner-row"):
            name_node = row.css_first(".runner-name")
            if not name_node:
                continue
            num_node = row.css_first(".runner-number")
            pos_node = row.css_first(".position")
            odds_node = row.css_first(".odds-win")
            win_node = row.css_first(".payout-win")
            place_node = row.css_first(".payout-place")

            final_odds = parse_fractional_odds(fortuna.node_text(odds_node)) if odds_node else 0.0
            odds_source = "starting_price" if final_odds > 0.01 else None

            # Advanced heuristic fallback (Jules Fix)
            if final_odds <= 0.01:
                final_odds = fortuna.SmartOddsExtractor.extract_from_node(row) or 0.0
                if final_odds > 0.01:
                    odds_source = "smart_extractor"

            runners.append(ResultRunner(
                name=fortuna.clean_text(fortuna.node_text(name_node)),
                number=_safe_int(fortuna.node_text(num_node)) if num_node else 0,
                position=fortuna.clean_text(fortuna.node_text(pos_node)) if pos_node else None,
                final_win_odds=final_odds,
                odds_source=odds_source,
                win_payout=parse_currency_value(fortuna.node_text(win_node)) if win_node else 0.0,
                place_payout=parse_currency_value(fortuna.node_text(place_node)) if place_node else 0.0,
            ))
        return runners

    @staticmethod
    def _parse_ras_runners_fallback(parser: HTMLParser) -> List[ResultRunner]:
        runners: List[ResultRunner] = []
        for row in parser.css("table.results-table tbody tr"):
            cols = row.css("td")
            if len(cols) < 5:
                continue
            runners.append(ResultRunner(
                name=fortuna.clean_text(fortuna.node_text(cols[2])),
                number=_safe_int(fortuna.node_text(cols[1])),
                position=fortuna.clean_text(fortuna.node_text(cols[0])),
                win_payout=parse_currency_value(fortuna.node_text(cols[-2])),
                place_payout=parse_currency_value(fortuna.node_text(cols[-1])),
            ))
        return runners


# -- TIMEFORM RESULTS ADAPTER ------------------------------------------------

class TimeformResultsAdapter(PageFetchingResultsAdapter):
    """Timeform results (UK / IRE / International)."""

    SOURCE_NAME = "TimeformResults"
    BASE_URL = "https://www.timeform.com"
    HOST = "www.timeform.com"
    IMPERSONATE = "chrome120"
    TIMEOUT = 60

    def _configure_fetch_strategy(self) -> fortuna.FetchStrategy:
        return fortuna.FetchStrategy(
            primary_engine=fortuna.BrowserEngine.PLAYWRIGHT,
            enable_js=True,
            stealth_mode="camouflage",
            timeout=self.TIMEOUT,
            network_idle=True,
        )

    async def _discover_result_links(self, date_str: str) -> Set[str]:
        url = f"/horse-racing/results/{date_str}"
        resp = await self.make_request("GET", url, headers=self._get_headers())
        if not resp or not resp.text:
            return set()

        self._save_debug_snapshot(resp.text, f"timeform_results_index_{date_str}")
        parser = HTMLParser(resp.text)
        links: Set[str] = set()

        for a in parser.css("a[href*='/horse-racing/result/']"):
            href = a.attributes.get("href", "")
            if not href:
                continue
            if "/horse-racing/result/" in href and len(href.split("/")) >= 6:
                if not self._link_matches_date(href, date_str):
                    continue
                if self._venue_matches(fortuna.node_text(a), href):
                    links.add(href)
        return links

    def _parse_race_page(
        self, html: str, date_str: str, url: str,
    ) -> Optional[ResultRace]:
        parser = HTMLParser(html)

        venue = ""
        title = parser.css_first("title")
        if title:
            match = re.search(r"\d{1,2}:\d{2}\s+([^|]+)", fortuna.node_text(title))
            if match:
                venue = fortuna.normalize_venue_name(match.group(1).strip())
        if not venue:
            return None

        race_num = 1
        # Extract race number from the last numeric part of the URL (e.g., .../33/3)
        # We assume race numbers are <= 100
        num_matches = re.findall(r"/(\d+)", url)
        if num_matches:
            for m in reversed(num_matches):
                val = int(m)
                if 1 <= val <= 100:
                    race_num = val
                    break

        runners: List[ResultRunner] = []
        for row in parser.css("tbody.rp-table-row"):
            name_node = row.css_first("a.rp-horse")
            if not name_node:
                continue
            pos_node = row.css_first(".rp-entry-number")
            num_node = row.css_first(".rp-saddle-cloth") or row.css_first(".rp-cloth")
            odds_node = row.css_first(".rp-odds") or row.css_first(".rp-sp")

            final_odds = parse_fractional_odds(fortuna.node_text(odds_node)) if odds_node else 0.0
            odds_source = "starting_price" if final_odds > 0.01 else None

            if final_odds <= 0.01:
                final_odds = fortuna.SmartOddsExtractor.extract_from_node(row) or 0.0
                if final_odds > 0.01:
                    odds_source = "smart_extractor"

            runners.append(ResultRunner(
                name=fortuna.clean_text(fortuna.node_text(name_node)),
                number=_safe_int(fortuna.node_text(num_node)) if num_node else 0,
                position=fortuna.clean_text(fortuna.node_text(pos_node)) if pos_node else None,
                final_win_odds=final_odds,
                odds_source=odds_source
            ))

        if not runners:
            return None

        return ResultRace(
            id=self._make_race_id("tf_res", venue, date_str, race_num),
            venue=venue,
            race_number=race_num,
            start_time=build_start_time(date_str),
            runners=runners,
            discipline="Thoroughbred",
            source=self.SOURCE_NAME,
        )


# -- SKY SPORTS RESULTS ADAPTER ----------------------------------------------

class SkySportsResultsAdapter(PageFetchingResultsAdapter):
    """Sky Sports Racing results (UK / IRE)."""

    SOURCE_NAME = "SkySportsResults"

    def _parse_races(self, raw_data: Any) -> List["ResultRace"]:
        """Override to assign sequential race numbers per venue (Jules Fix)."""
        races: List[ResultRace] = super()._parse_races(raw_data)
        if not races:
            return races

        groups: Dict[tuple, List[ResultRace]] = defaultdict(list)
        for r in races:
            key = (
                fortuna.get_canonical_venue(r.venue),
                r.start_time.strftime("%Y%m%d"),
            )
            groups[key].append(r)

        fixed: List[ResultRace] = []
        for group in groups.values():
            group.sort(key=lambda r: r.start_time)
            for i, race in enumerate(group, start=1):
                date_str = race.start_time.strftime("%Y-%m-%d")
                fixed.append(
                    race.model_copy(update={
                        "race_number": i,
                        "id": self._make_race_id(
                            "sky_res", race.venue, date_str, i,
                        ),
                    })
                )
        return fixed
    BASE_URL = "https://www.skysports.com"
    HOST = "www.skysports.com"
    IMPERSONATE = "chrome120"
    TIMEOUT = 45

    def _parse_races(self, raw_data: Any) -> List["ResultRace"]:
        """Override to assign sequential race numbers per venue (Jules Fix)."""
        races: List[ResultRace] = super()._parse_races(raw_data)
        if not races:
            return races

        groups: Dict[tuple, List[ResultRace]] = defaultdict(list)
        for r in races:
            key = (
                fortuna.get_canonical_venue(r.venue),
                r.start_time.strftime("%Y%m%d"),
            )
            groups[key].append(r)

        fixed: List[ResultRace] = []
        for group in groups.values():
            group.sort(key=lambda r: r.start_time)
            for i, race in enumerate(group, start=1):
                date_str = race.start_time.strftime("%Y-%m-%d")
                fixed.append(
                    race.model_copy(update={
                        "race_number": i,
                        "id": self._make_race_id(
                            "sky_res", race.venue, date_str, i,
                        ),
                    })
                )
        return fixed

    async def _discover_result_links(self, date_str: str) -> Set[str]:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            url_date = dt.strftime("%d-%m-%Y")
        except ValueError:
            url_date = date_str

        try:
            resp = await self.make_request(
                "GET", f"/racing/results/{url_date}", headers=self._get_headers(),
            )
            if not resp or not resp.text:
                return set()
            self._save_debug_snapshot(resp.text, f"sky_results_index_{url_date}")
            parser = HTMLParser(resp.text)
            return self._extract_sky_links(parser, date_str)
        except Exception:
            self.logger.debug("Sky index fetch failed", exc_info=True)
            return set()

    def _extract_sky_links(
        self, parser: HTMLParser, date_str: str,
    ) -> Set[str]:
        links: Set[str] = set()
        candidates = (
            parser.css("a[href*='/racing/results/']")
            + parser.css("a[href*='/full-result/']")
        )
        for a in candidates:
            href = a.attributes.get("href", "")
            if not href:
                continue
            if not self._venue_matches(fortuna.node_text(a), href):
                continue

            has_race_path = any(
                p in href
                for p in ("/full-result/", "/race-result/", "/results/full-result/")
            )

            if has_race_path and self._link_matches_date(href, date_str):
                links.add(href)
        return links

    def _parse_race_page(
        self, html: str, date_str: str, url: str,
    ) -> Optional[ResultRace]:
        parser = HTMLParser(html)

        header = parser.css_first(".sdc-site-racing-header__name")
        if not header:
            return None

        match = re.match(
            r"(\d{1,2}:\d{2})\s+(.+)",
            fortuna.clean_text(fortuna.node_text(header)),
        )
        if not match:
            return None

        time_str = match.group(1)
        venue = fortuna.normalize_venue_name(match.group(2))
        start_time = build_start_time(date_str, time_str)

        runners = self._parse_sky_runners(parser)
        if not runners:
            return None

        exotics = extract_exotic_payouts(parser.css("table"))
        race_num = self._extract_sky_race_number(parser, url)

        return ResultRace(
            id=self._make_race_id("sky_res", venue, date_str, race_num),
            venue=venue,
            race_number=race_num,
            start_time=start_time,
            runners=runners,
            discipline="Thoroughbred",
            source=self.SOURCE_NAME,
            **_apply_exotics_to_race(exotics),
        )

    @staticmethod
    def _parse_sky_runners(parser: HTMLParser) -> List[ResultRunner]:
        runners: List[ResultRunner] = []
        for row in parser.css(".sdc-site-racing-card__item"):
            name_node = row.css_first(".sdc-site-racing-card__name")
            if not name_node:
                continue

            pos_node = row.css_first(".sdc-site-racing-card__position")
            number_node = row.css_first(".sdc-site-racing-card__number")
            odds_node = row.css_first(".sdc-site-racing-card__odds")

            final_odds = parse_fractional_odds(
                fortuna.clean_text(fortuna.node_text(odds_node)) if odds_node else "",
            )
            odds_source = "starting_price" if final_odds > 0.01 else None

            # Advanced heuristic fallback (Jules Fix)
            if final_odds <= 0.01:
                final_odds = fortuna.SmartOddsExtractor.extract_from_node(row) or 0.0
                if final_odds > 0.01:
                    odds_source = "smart_extractor"

            runners.append(ResultRunner(
                name=fortuna.clean_text(fortuna.node_text(name_node)),
                number=_safe_int(fortuna.node_text(number_node)) if number_node else 0,
                position=fortuna.clean_text(fortuna.node_text(pos_node)) if pos_node else None,
                final_win_odds=final_odds,
                odds_source=odds_source
            ))
        return runners

    @staticmethod
    def _extract_sky_race_number(parser: HTMLParser, url: str) -> int:
        url_match = re.search(r"/(\d+)/", url)
        if url_match:
            nav_links = parser.css("a[href*='/racing/results/']")
            for i, link in enumerate(nav_links):
                if url_match.group(0) in (link.attributes.get("href") or ""):
                    return i + 1
        return _extract_race_number_from_text(parser, url) or 1


# -- REPORT GENERATION --------------------------------------------------------

def _format_tip_time(tip: Dict[str, Any]) -> str:
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
    now_str = now_eastern().strftime("%Y-%m-%d %H:%M ET")
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
            count = data.get("count", 0)
            max_odds = data.get("max_odds", 0.0)
        else:
            count, max_odds = data, 0.0

        status = "✅ SUCCESS" if count > 0 else "⏳ PENDING/NO DATA"
        odds_str = f"MaxOdds: {max_odds:>5.1f}" if max_odds > 0 else "Odds: N/A"
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
        venue = str(tip.get("venue", "Unknown"))[:20]
        rnum = tip.get("race_number", "?")
        gm = "GOLD" if tip.get("is_goldmine") else "----"
        status = tip.get("verdict") if tip.get("audit_completed") else "WATCHING"
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
        verdict_str = tip.get("verdict", "VOID")
        try:
            verdict = Verdict(verdict_str)
            emoji = verdict.display
        except ValueError:
            emoji = "⚪ VOID"

        venue = f"{tip.get('venue', 'Unknown')[:18]} R{tip.get('race_number', '?')}"
        profit = f"${tip.get('net_profit', 0.0):+.2f}"

        detail_parts: list[str] = []

        p1 = tip.get("top1_place_payout")
        p2 = tip.get("top2_place_payout")
        if p1 or p2:
            detail_parts.append(f"P: {p1 or 0:.2f}/{p2 or 0:.2f}")

        po = tip.get("predicted_2nd_fav_odds")
        ao = tip.get("actual_2nd_fav_odds")
        if po is not None or ao is not None:
            po_s = f"{po:.1f}" if po is not None else "?"
            ao_s = f"{ao:.1f}" if ao is not None else "?"
            detail_parts.append(f"Odds: {po_s}->{ao_s}")

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
    tri_races = [t for t in audited_tips if t.get("trifecta_payout")]

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
    total = len(audited_tips)
    cashed = sum(
        1 for t in audited_tips
        if t.get("verdict") in {v.value for v in _CASHED_VERDICTS}
    )
    profit = sum(t.get("net_profit", 0.0) for t in audited_tips)
    sr = (cashed / total * 100) if total else 0.0
    roi = (profit / (total * STANDARD_BET) * 100) if total else 0.0

    lines.extend([
        "📊 SUMMARY METRICS (LIFETIME)",
        _SECTION_SEP,
        f"Total Verified Races: {total}",
        f"Overall Strike Rate:   {sr:.1f}%",
        f"Total Net Profit:     ${profit:+.2f} (Using ${STANDARD_BET:.2f} Base Unit)",
        f"Return on Investment:  {roi:+.1f}%",
        "",
    ])


# -- ADAPTER REGISTRY & LIFECYCLE ---------------------------------------------

def get_results_adapter_classes() -> List[Type[fortuna.BaseAdapterV3]]:
    """All concrete adapter classes with ``ADAPTER_TYPE == 'results'``."""

    def _all_subclasses(cls: type) -> set[type]:
        subs = set(cls.__subclasses__())
        return subs.union(s for c in subs for s in _all_subclasses(c))

    return [
        c
        for c in _all_subclasses(fortuna.BaseAdapterV3)
        if not getattr(c, "__abstractmethods__", None)
        and getattr(c, "ADAPTER_TYPE", "discovery") == "results"
        and hasattr(c, "SOURCE_NAME") # Filter out base classes without SOURCE_NAME (GPT5 Fix)
    ]


@asynccontextmanager
async def managed_adapters(
    region: Optional[str] = None,
    target_venues: Optional[Set[str]] = None,
):
    """Instantiate, optionally filter, yield, then tear down all results adapters."""
    classes = get_results_adapter_classes()
    logger = structlog.get_logger("managed_adapters")

    if region and region != "GLOBAL":
        allowed = (
            set(fortuna.USA_RESULTS_ADAPTERS)
            if region == "USA"
            else set(fortuna.INT_RESULTS_ADAPTERS)
        )
        classes = [
            c for c in classes
            if getattr(c, "SOURCE_NAME", "") in allowed
        ]

    adapters: list[fortuna.BaseAdapterV3] = []
    for cls in classes:
        name = getattr(cls, "SOURCE_NAME", cls.__name__)
        adapter = cls()
        if target_venues:
            adapter.target_venues = target_venues  # type: ignore[attr-defined]
        adapters.append(adapter)

        # Optimization: Do NOT double-up mobile versions for results auditing
        # This prevents resource exhaustion and timeouts during heavy audits (Jules Fix)

    try:
        yield adapters
    finally:
        for adapter in adapters:
            try:
                await adapter.close()
            except Exception:
                logger.warning(
                    "Adapter cleanup failed",
                    adapter=adapter.source_name,
                    exc_info=True,
                )
        try:
            await fortuna.GlobalResourceManager.cleanup()
        except Exception:
            logger.error("Global resource cleanup failed", exc_info=True)


# -- ORCHESTRATION ------------------------------------------------------------

_analytics_logger = structlog.get_logger("run_analytics")


async def _harvest_results(
    adapters: List[fortuna.BaseAdapterV3],
    valid_dates: List[str],
    harvest_summary: Dict[str, Dict[str, Any]],
) -> List[ResultRace]:
    """Fetch results from all adapters × dates; populate *harvest_summary*."""
    sem = asyncio.Semaphore(_MAX_CONCURRENT_FETCHES)

    async def _fetch_one(
        adapter: fortuna.BaseAdapterV3, date_str: str,
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
            except Exception:
                _analytics_logger.warning(
                    "Adapter fetch failed",
                    adapter=adapter.source_name,
                    date=date_str,
                    exc_info=True,
                )
                return adapter.source_name, []

    tasks = [
        _fetch_one(adapter, d)
        for d in valid_dates
        for adapter in adapters
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    # canonical_key -> race
    seen_races: Dict[str, ResultRace] = {}

    for res in raw_results:
        if isinstance(res, Exception):
            _analytics_logger.warning("Task raised exception", error=str(res))
            continue

        name, races = res

        # Deduplicate across adapters with quality scoring
        deduped_for_this_adapter: List[ResultRace] = []
        for race in races:
            key = race.canonical_key
            if key in seen_races:
                # Keep the higher quality version
                if _race_quality(race) > _race_quality(seen_races[key]):
                    seen_races[key] = race
                    # Note: this might slightly under-count for the previous adapter
                    # and over-count for this one in harvest_summary, but the
                    # final all_races list will be optimal.
                    deduped_for_this_adapter.append(race)
            else:
                seen_races[key] = race
                deduped_for_this_adapter.append(race)

        max_odds = max(
            (
                float(r.final_win_odds)
                for race in deduped_for_this_adapter
                for r in race.runners
                if r.final_win_odds
            ),
            default=0.0,
        )
        entry = harvest_summary.setdefault(name, {"count": 0, "max_odds": 0.0})
        entry["count"] += len(deduped_for_this_adapter)
        entry["max_odds"] = max(entry["max_odds"], max_odds)

    return list(seen_races.values())


async def _save_harvest_summary(
    harvest_summary: Dict[str, Dict[str, Any]],
    auditor: AuditorEngine,
    region: Optional[str],
) -> None:
    try:
        Path(fortuna.get_writable_path("results_harvest.json")).write_text(
            json.dumps(harvest_summary, indent=2), encoding="utf-8",
        )
    except OSError:
        _analytics_logger.debug("Failed to write results_harvest.json", exc_info=True)

    if harvest_summary:
        try:
            await auditor.db.log_harvest(harvest_summary, region=region)
        except Exception:
            _analytics_logger.debug("Failed to log harvest to DB", exc_info=True)


async def _write_gha_summary(
    auditor: AuditorEngine,
    harvest_summary: dict,
) -> None:
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
                    "predicted_2nd_fav_odds": tip.get("predicted_2nd_fav_odds"),
                }
                race.top_five_numbers = tip.get("top_five")
                qualified.append(race)
            except (KeyError, ValueError):
                continue

        predictions_md = fortuna.format_predictions_section(qualified)
        proof_md = await fortuna.format_proof_section(auditor.db)
        harvest_md = fortuna.build_harvest_table(
            harvest_summary, "🛰️ Results Harvest Performance",
        )
        artifacts_md = fortuna.format_artifact_links()
        fortuna.write_job_summary(predictions_md, harvest_md, proof_md, artifacts_md)
        _analytics_logger.info("GHA Job Summary written")
    except Exception:
        _analytics_logger.error("Failed to write GHA summary", exc_info=True)


async def _generate_and_save_report(
    auditor: AuditorEngine,
    harvest_summary: Dict[str, Any],
    *,
    include_lifetime_stats: bool = False,
) -> None:
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
        Path(fortuna.get_writable_path("analytics_report.txt")).write_text(report, encoding="utf-8")
        _analytics_logger.info("Report saved", path="analytics_report.txt")
    except OSError:
        _analytics_logger.error("Failed to save report", exc_info=True)

    if all_audited:
        _analytics_logger.info("Analytics complete", total_audited=len(all_audited))
    else:
        _analytics_logger.info("No audited tips found in history")


async def run_analytics(
    target_dates: List[str],
    region: Optional[str] = None,
    *,
    include_lifetime_stats: bool = False,
    lookback_hours: Optional[int] = None,
    include_adapters: Optional[List[str]] = None,
    quality: Optional[str] = None,
) -> None:
    """Main analytics entry: harvest → audit → report → GHA summary."""
    valid_dates = [d for d in target_dates if validate_date_format(d)]
    if not valid_dates:
        _analytics_logger.error("No valid dates", input_dates=target_dates)
        return

    target_region = region or DEFAULT_REGION
    _analytics_logger.info(
        "Starting analytics audit",
        dates=valid_dates,
        region=target_region,
    )

    async with AuditorEngine() as auditor:
        if lookback_hours is None:
            lookback_hours = 72 if target_region == "GLOBAL" else 48
        unverified = await auditor.get_unverified_tips(lookback_hours=lookback_hours)
        target_venues: Optional[Set[str]] = None

        if not unverified:
            _analytics_logger.info("No unverified tips — fetching all results for visibility")
        else:
            _analytics_logger.info("Tips to audit", count=len(unverified))
            target_venues = {
                fortuna.get_canonical_venue(t.get("venue"))
                for t in unverified
            }
            target_venues.discard("unknown")

            if not target_venues:
                _analytics_logger.warning("All tip venues resolved to 'unknown' — fetching everything")
                target_venues = None
            else:
                _analytics_logger.info("Targeting venues", venues=sorted(target_venues))

        async with managed_adapters(region=region, target_venues=target_venues) as adapters:
            # Filter adapters by quality or include list (Council of Superbrains strategy)
            if include_adapters:
                adapters = [a for a in adapters if a.source_name in include_adapters]

            if quality:
                if quality == "solid":
                    adapters = [a for a in adapters if a.source_name in fortuna.SOLID_RESULTS_ADAPTERS]
                else:
                    adapters = [a for a in adapters if a.source_name not in fortuna.SOLID_RESULTS_ADAPTERS]
            harvest_summary: Dict[str, Dict[str, Any]] = {
                a.source_name: {"count": 0, "max_odds": 0.0}
                for a in adapters
            }
            try:
                all_results = await _harvest_results(adapters, valid_dates, harvest_summary)
                _analytics_logger.info("Total results harvested", count=len(all_results))

                if not all_results:
                    _analytics_logger.error(
                        "ZERO results harvested — audit impossible",
                        adapters_tried=[a.source_name for a in adapters],
                        dates=valid_dates,
                        region=target_region,
                    )
                elif not unverified:
                    _analytics_logger.warning("No unverified tips to audit against results")
                else:
                    matched = await auditor.audit_races(all_results, unverified=unverified)
                    _analytics_logger.info(
                        "Audit complete",
                        results_available=len(all_results),
                        tips_checked=len(unverified),
                        tips_matched=len(matched),
                        tips_still_unmatched=len(unverified) - len(matched),
                    )
            finally:
                await _save_harvest_summary(harvest_summary, auditor, region)

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
    if explicit_date:
        if not validate_date_format(explicit_date):
            raise ValueError(
                f"Invalid date format '{explicit_date}'.  Use YYYY-MM-DD.",
            )
        return [explicit_date]
    now = now_eastern()
    return [
        (now - timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(lookback_days)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fortuna Analytics Engine — Race result auditing and performance analysis",
    )
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD)")
    parser.add_argument(
        "--region",
        type=str,
        choices=["USA", "INT", "GLOBAL"],
        help="Filter results by region",
    )
    parser.add_argument(
        "--days", type=int, default=2,
        help="Number of days to look back (default: 2)",
    )
    parser.add_argument(
        "--db-path", type=str, default=DEFAULT_DB_PATH,
        help=f"Path to tip database (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--lookback-hours", type=int,
        help="Custom hours to look back for unverified tips",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--migrate", action="store_true", help="Migrate data from legacy JSON to SQLite")
    parser.add_argument(
        "--include-lifetime-stats", action="store_true",
        help="Include lifetime summary statistics in report",
    )
    parser.add_argument("--include", help="Comma-separated adapter names to include")
    parser.add_argument("--quality", choices=["solid", "lousy"], help="Filter by quality")
    args = parser.parse_args()

    if args.db_path != DEFAULT_DB_PATH:
        os.environ["FORTUNA_DB_PATH"] = args.db_path

    log_level = logging.DEBUG if args.verbose else logging.INFO
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
    )

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

    try:
        target_dates = _build_target_dates(args.date, args.days)
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    if not args.region:
        args.region = DEFAULT_REGION
        structlog.get_logger().info("Using default region", region=args.region)

    asyncio.run(
        run_analytics(
            target_dates,
            region=args.region,
            include_lifetime_stats=args.include_lifetime_stats,
            lookback_hours=args.lookback_hours,
            include_adapters=args.include.split(",") if args.include else None,
            quality=args.quality,
        ),
    )


if __name__ == "__main__":
    main()
