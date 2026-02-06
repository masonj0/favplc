from __future__ import annotations
# fortuna_analytics.py
# Race result harvesting and performance analysis engine for Fortuna

import argparse
import asyncio
import hashlib
import html
import json
import logging
import os
import random
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from io import StringIO
from pathlib import Path
from typing import (
    Any,
    Annotated,
    Callable,
    ClassVar,
    Dict,
    Final,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    Tuple,
)

import httpx
import pandas as pd
import structlog
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    WrapSerializer,
    field_validator,
)
from selectolax.parser import HTMLParser, Node
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# --- OPTIONAL IMPORTS ---
try:
    from curl_cffi import requests as curl_requests
except ImportError:
    curl_requests = None

try:
    from scrapling import AsyncFetcher, Fetcher
    from scrapling.fetchers import AsyncDynamicSession, AsyncStealthySession
    from scrapling.parser import Selector
    ASYNC_SESSIONS_AVAILABLE = True
except ImportError:
    ASYNC_SESSIONS_AVAILABLE = False
    Selector = None

try:
    from browserforge.headers import HeaderGenerator
    from browserforge.fingerprints import FingerprintGenerator
    BROWSERFORGE_AVAILABLE = True
except ImportError:
    BROWSERFORGE_AVAILABLE = False

# Reuse constants and utilities from fortuna.py
from fortuna import (
    DEFAULT_BROWSER_HEADERS,
    CHROME_USER_AGENT,
    CHROME_SEC_CH_UA,
    MIN_VALID_ODDS,
    MAX_VALID_ODDS,
    DEFAULT_ODDS_FALLBACK,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_CONCURRENT_REQUESTS,
    BET_TYPE_KEYWORDS,
    DISCIPLINE_KEYWORDS,
    FortunaBaseModel,
    OddsData,
    Runner,
    Race,
    BrowserEngine,
    FetchStrategy,
    SmartFetcher,
    GlobalResourceManager,
    BaseAdapterV3,
    BrowserHeadersMixin,
    DebugMixin,
    JSONParsingMixin,
    RacePageFetcherMixin,
    DataValidationPipeline,
    RaceValidator,
    normalize_venue_name,
    get_canonical_venue,
    clean_text,
    parse_odds_to_decimal,
    is_valid_odds,
    create_odds_data,
    generate_race_id,
    _get_best_win_odds
)

# --- MODELS FOR ANALYTICS ---

class ResultRunner(Runner):
    position: Optional[str] = None # e.g. "1", "2", "3", "4", "5", "W", "P", "S"
    final_win_odds: Optional[float] = None
    win_payout: Optional[float] = None
    place_payout: Optional[float] = None
    show_payout: Optional[float] = None

class ResultRace(Race):
    runners: List[ResultRunner] = Field(default_factory=list)
    official_dividends: Dict[str, str] = Field(default_factory=dict)
    chart_url: Optional[str] = None
    is_fully_parsed: bool = False

    # Trifecta data
    trifecta_payout: Optional[float] = None
    trifecta_cost: float = 1.00 # Standard cost for the reported payout
    trifecta_combination: Optional[str] = None # e.g. "4-2-9"

# --- AUDITOR LOGIC ---

def parse_position(pos_str: str) -> Optional[int]:
    """Extracts numeric position from strings like '1st', '2/12', 'W', etc."""
    if not pos_str: return None
    pos_str = str(pos_str).upper().strip()
    if pos_str in ("W", "1", "1ST"): return 1
    if pos_str in ("P", "2", "2ND"): return 2
    if pos_str in ("S", "3", "3RD"): return 3

    match = re.search(r"(\d+)", pos_str)
    if match:
        return int(match.group(1))
    return None

class AuditorEngine:
    def __init__(self, db_path: str = "hot_tips_db.json"):
        self.db_path = Path(db_path)
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.history: List[Dict[str, Any]] = []
        self._load_history()

    def _load_history(self):
        if self.db_path.exists():
            try:
                with open(self.db_path, "r") as f:
                    self.history = json.load(f)
                self.logger.info("Loaded tip history", count=len(self.history))
            except Exception as e:
                self.logger.error("Failed to load tip history", error=str(e))

    def get_unverified_tips(self) -> List[Dict[str, Any]]:
        """Returns tips that haven't been successfully audited yet."""
        # For now, just return everything from the last 48 hours
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=48)

        unverified = []
        for tip in self.history:
            try:
                start_time = datetime.fromisoformat(tip["start_time"])
                if start_time < now and start_time > cutoff:
                    if not tip.get("audit_completed"):
                        unverified.append(tip)
            except: continue
        return unverified

    async def audit_races(self, results: List[ResultRace]):
        """Match results to history and update audit status."""
        results_map = {}
        for r in results:
            # key: canonical_venue|date|race_num
            st = r.start_time
            date_str = st.strftime('%Y%m%d')
            key = f"{get_canonical_venue(r.venue)}|{date_str}|{r.race_number}"
            results_map[key] = r

        audit_results = []
        updated = False

        for tip in self.history:
            if tip.get("audit_completed"):
                continue

            try:
                st = datetime.fromisoformat(tip["start_time"])
                date_str = st.strftime('%Y%m%d')
                key = f"{get_canonical_venue(tip['venue'])}|{date_str}|{tip['race_number']}"

                if key in results_map:
                    result = results_map[key]
                    self.logger.info("Auditing tip", venue=tip['venue'], race=tip['race_number'])
                    outcome = self._evaluate_tip(tip, result)
                    tip.update(outcome)
                    tip["audit_completed"] = True
                    audit_results.append(tip)
                    updated = True
            except Exception as e:
                self.logger.error("Error during audit", tip=tip.get("race_id"), error=str(e))

        if updated:
            self._save_history()

        return audit_results

    def _evaluate_tip(self, tip: Dict[str, Any], result: ResultRace) -> Dict[str, Any]:
        """Compares predicted selection with actual result."""
        # Fallback to first runner in top_five if selection_number is missing
        selection_num = str(tip.get("selection_number", ""))
        if not selection_num and tip.get("top_five"):
            selection_num = tip["top_five"].split(",")[0].strip()

        # Sort result runners by position
        runners_with_pos = []
        for r in result.runners:
            p = parse_position(r.position)
            if p is not None:
                runners_with_pos.append((r, p))

        runners_with_pos.sort(key=lambda x: x[1])
        actual_top_5 = [str(r.number) for r, p in runners_with_pos[:5]]

        # Identify 2nd favorite based on final odds
        runners_with_odds = sorted([r for r in result.runners if r.final_win_odds], key=lambda x: x.final_win_odds)
        actual_2nd_fav_odds = runners_with_odds[1].final_win_odds if len(runners_with_odds) >= 2 else None

        verdict = "BURNED"
        profit = -2.00 # Standard 1 unit loss ($2.00)

        # Find our selection in the result
        selection_result = next((r for r in result.runners if str(r.number) == selection_num), None)

        if selection_result:
            pos = parse_position(selection_result.position)
            # Favorite To Place logic: usually top 3
            # If field size <= 4, only 1st counts
            # If field size 5-7, top 2 count
            # If field size 8+, top 3 count
            field_size = len(result.runners)
            places_paid = 3
            if field_size <= 4: places_paid = 1
            elif field_size <= 7: places_paid = 2

            if pos is not None and pos <= places_paid:
                verdict = "CASHED"
                # If we have official place payout, use it.
                # US charts often give $ payout for a $2 bet.
                payout = selection_result.place_payout or 0.0
                if payout > 0:
                    profit = payout - 2.00
                else:
                    # Heuristic: 1/5th of win odds + stake if no payout data
                    odds = selection_result.final_win_odds or 2.0
                    profit = (odds - 1.0) / 5.0 * 2.0 # simplified place profit
            else:
                verdict = "BURNED"
        else:
            # Selection not found in results (could be a scratch)
            verdict = "VOID"
            profit = 0.0

        return {
            "actual_top_5": ", ".join(actual_top_5),
            "actual_2nd_fav_odds": actual_2nd_fav_odds,
            "verdict": verdict,
            "net_profit": round(profit, 2),
            "audit_timestamp": datetime.now(timezone.utc).isoformat(),
            "trifecta_payout": result.trifecta_payout,
            "trifecta_combination": result.trifecta_combination
        }

    def _save_history(self):
        try:
            with open(self.db_path, "w") as f:
                json.dump(self.history, f, indent=4)
        except Exception as e:
            self.logger.error("Failed to save tip history", error=str(e))


# --- RESULTS ADAPTERS ---

class EquibaseResultsAdapter(BrowserHeadersMixin, DebugMixin, BaseAdapterV3):
    """
    Adapter for Equibase Results / Summary Charts.
    """
    def __init__(self, **kwargs):
        super().__init__(source_name="EquibaseResults", base_url="https://www.equibase.com", **kwargs)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return FetchStrategy(primary_engine=BrowserEngine.HTTPX, enable_js=False)

    def _get_headers(self) -> dict:
        return self._get_browser_headers(host="www.equibase.com")

    async def _fetch_data(self, date_str: str) -> Optional[Dict[str, Any]]:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        url = f"/static/chart/summary/index.html?date={dt.strftime('%m/%d/%Y')}"

        resp = await self.make_request("GET", url, headers=self._get_headers())
        if not resp or not resp.text:
            return None

        self._save_debug_snapshot(resp.text, f"eqb_results_index_{date_str}")
        parser = HTMLParser(resp.text)

        links = []
        for a in parser.css("a"):
            href = a.attributes.get("href", "")
            if "/static/chart/summary/" in href and href.endswith(".html") and "index.html" not in href:
                links.append(href)

        async def fetch_track_results(link):
            r = await self.make_request("GET", link, headers=self._get_headers())
            return (link, r.text if r else "")

        tasks = [fetch_track_results(link) for link in set(links)]
        pages = await asyncio.gather(*tasks)
        return {"pages": pages, "date": date_str}

    def _parse_races(self, raw_data: Any) -> List[ResultRace]:
        if not raw_data or not raw_data.get("pages"):
            return []

        races = []
        for link, html_content in raw_data["pages"]:
            if not html_content: continue
            try:
                parsed = self._parse_track_page(html_content, raw_data["date"])
                races.extend(parsed)
            except: continue
        return races

    def _parse_track_page(self, html_content: str, date_str: str) -> List[ResultRace]:
        parser = HTMLParser(html_content)
        races = []

        track_node = parser.css_first("h3") or parser.css_first("h2")
        if not track_node: return []
        venue = normalize_venue_name(track_node.text(strip=True))

        for race_table in parser.css("table"):
            header = race_table.css_first("thead tr th")
            if header and "Race" in header.text():
                race_num_match = re.search(r"Race\s+(\d+)", header.text())
                if not race_num_match: continue
                race_num = int(race_num_match.group(1))

                runners = []
                for row in race_table.css("tbody tr"):
                    cols = row.css("td")
                    if len(cols) >= 3:
                        pos_text = clean_text(cols[0].text())
                        num_text = clean_text(cols[1].text())
                        name = clean_text(cols[2].text())
                        odds_text = clean_text(cols[3].text()) if len(cols) > 3 else ""
                        if not name or name.upper() == "HORSE": continue
                        num = 0
                        if num_text.isdigit(): num = int(num_text)
                        final_odds = parse_odds_to_decimal(odds_text)
                        win_pay = 0.0
                        place_pay = 0.0
                        show_pay = 0.0
                        if len(cols) >= 7:
                            try:
                                win_pay = float(cols[4].text().replace("$", "").replace(",", ""))
                                place_pay = float(cols[5].text().replace("$", "").replace(",", ""))
                                if len(cols) > 6:
                                    show_pay = float(cols[6].text().replace("$", "").replace(",", ""))
                            except: pass

                        runners.append(ResultRunner(
                            name=name, number=num, position=pos_text,
                            final_win_odds=final_odds, win_payout=win_pay,
                            place_payout=place_pay, show_payout=show_pay
                        ))

                if runners:
                    trifecta_val = None
                    trifecta_comb = None
                    for next_table in parser.css("table"):
                        if "Trifecta" in next_table.text():
                            for r_row in next_table.css("tr"):
                                if "Trifecta" in r_row.text():
                                    r_cols = r_row.css("td")
                                    if len(r_cols) >= 3:
                                        trifecta_comb = clean_text(r_cols[1].text())
                                        try:
                                            trifecta_val = float(r_cols[2].text().replace("$", "").replace(",", ""))
                                        except: pass
                                    break

                    st = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=12, minute=0, tzinfo=timezone.utc)
                    races.append(ResultRace(
                        id=f"eqb_res_{get_canonical_venue(venue)}_{date_str.replace('-', '')}_R{race_num}",
                        venue=venue, race_number=race_num, start_time=st,
                        runners=runners, source=self.source_name, is_fully_parsed=True,
                        trifecta_payout=trifecta_val, trifecta_combination=trifecta_comb
                    ))
        return races

class RacingPostResultsAdapter(BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
    def __init__(self, **kwargs):
        super().__init__(source_name="RacingPostResults", base_url="https://www.racingpost.com", **kwargs)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return FetchStrategy(primary_engine=BrowserEngine.HTTPX, enable_js=False)
    def _get_headers(self) -> dict:
        return self._get_browser_headers(host="www.racingpost.com")
    async def _fetch_data(self, date_str: str) -> Optional[Dict[str, Any]]:
        url = f"/results/{date_str}"
        resp = await self.make_request("GET", url, headers=self._get_headers())
        if not resp: return None
        parser = HTMLParser(resp.text)
        links = [a.attributes.get("href") for a in parser.css("a[href*='/results/']") if "/results/" in a.attributes.get("href", "")]
        pages = await self._fetch_race_pages_concurrent([{"url": l} for l in set(links) if re.search(r"/\d+/", l)], self._get_headers())
        return {"pages": pages, "date": date_str}
    def _parse_races(self, raw_data: Any) -> List[ResultRace]:
        races = []
        for item in raw_data.get("pages", []):
            html_content = item.get("html")
            if not html_content: continue
            parser = HTMLParser(html_content)
            # Simplified RP parsing
            try:
                venue_node = parser.css_first(".rp-raceTimeCourseName__course")
                if not venue_node: continue
                venue = normalize_venue_name(venue_node.text(strip=True))
                runners = []
                for row in parser.css(".rp-horseTable__table__row"):
                    name = clean_text(row.css_first(".rp-horseTable__horse__name").text())
                    pos = clean_text(row.css_first(".rp-horseTable__pos__number").text())
                    runners.append(ResultRunner(name=name, position=pos))
                races.append(ResultRace(venue=venue, runners=runners, source=self.source_name, start_time=datetime.now(timezone.utc)))
            except: continue
        return races

class SportingLifeResultsAdapter(BaseAdapterV3):
    def __init__(self, **kwargs):
        super().__init__(source_name="SportingLifeResults", base_url="https://www.sportinglife.com", **kwargs)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return FetchStrategy(primary_engine=BrowserEngine.HTTPX, enable_js=False)

    async def _fetch_data(self, date_str: str): return {}
    def _parse_races(self, raw_data: Any): return []

class SkySportsResultsAdapter(BaseAdapterV3):
    def __init__(self, **kwargs):
        super().__init__(source_name="SkySportsResults", base_url="https://www.skysports.com", **kwargs)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return FetchStrategy(primary_engine=BrowserEngine.HTTPX, enable_js=False)

    async def _fetch_data(self, date_str: str): return {}
    def _parse_races(self, raw_data: Any): return []

class AtTheRacesResultsAdapter(BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
    def __init__(self, **kwargs):
        super().__init__(source_name="AtTheRacesResults", base_url="https://www.attheraces.com", **kwargs)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return FetchStrategy(primary_engine=BrowserEngine.HTTPX, enable_js=False)
    def _get_headers(self) -> dict:
        return self._get_browser_headers(host="www.attheraces.com")
    async def _fetch_data(self, date_str: str) -> Optional[Dict[str, Any]]:
        url = f"/results/{date_str}"
        resp = await self.make_request("GET", url, headers=self._get_headers())
        if not resp: return None
        parser = HTMLParser(resp.text)
        links = [a.attributes.get("href") for a in parser.css("a[href*='/results/']")]
        pages = await self._fetch_race_pages_concurrent([{"url": l} for l in set(links) if re.search(r"/\d{2}-[A-Za-z]{3}-\d{4}/", l)], self._get_headers())
        return {"pages": pages, "date": date_str}
    def _parse_races(self, raw_data: Any) -> List[ResultRace]:
        races = []
        for item in raw_data.get("pages", []):
            html_content = item.get("html")
            if not html_content: continue
            parser = HTMLParser(html_content)
            header = parser.css_first(".race-header__details--primary")
            if not header: continue
            venue = normalize_venue_name(header.css_first("h2").text(strip=True))
            runners = []
            for row in parser.css(".result-racecard__row"):
                name = clean_text(row.css_first(".result-racecard__horse-name a").text())
                pos = clean_text(row.css_first(".result-racecard__pos").text())
                runners.append(ResultRunner(name=name, position=pos))
            # Trifecta
            tri_pay = None
            div_table = parser.css_first(".result-racecard__dividends-table")
            if div_table:
                for r in div_table.css("tr"):
                    if "trifecta" in r.text().lower():
                        try: tri_pay = float(r.css("td")[1].text().replace("£", "").strip())
                        except: pass
            races.append(ResultRace(venue=venue, runners=runners, trifecta_payout=tri_pay, source=self.source_name, start_time=datetime.now(timezone.utc)))
        return races

# --- MAIN ORCHESTRATOR FOR ANALYTICS ---

async def run_analytics(target_dates: List[str]):
    logger = structlog.get_logger("run_analytics")
    logger.info("Starting Analytics Audit", dates=target_dates)

    auditor = AuditorEngine()
    unverified = auditor.get_unverified_tips()

    if not unverified:
        logger.info("No unverified tips found in history.")
        return

    logger.info("Tips to audit", count=len(unverified))

    adapters = [
        EquibaseResultsAdapter(),
        RacingPostResultsAdapter(),
        SportingLifeResultsAdapter(),
        SkySportsResultsAdapter(),
        AtTheRacesResultsAdapter(),
    ]

    all_results = []

    async def fetch_results(adapter, date_str):
        try:
            races = await adapter.get_races(date_str)
            return races
        except Exception as e:
            logger.error("Failed to fetch results", adapter=adapter.source_name, date=date_str, error=str(e))
            return []

    fetch_tasks = []
    for d in target_dates:
        for a in adapters:
            fetch_tasks.append(fetch_results(a, d))

    results_lists = await asyncio.gather(*fetch_tasks)
    for r_list in results_lists:
        all_results.extend(r_list)

    logger.info("Total results harvested", count=len(all_results))

    audit_matches = await auditor.audit_races(all_results)

    if audit_matches:
        logger.info("Audit completed", matches=len(audit_matches))
        report = generate_analytics_report(audit_matches)
        print(report)
        with open("analytics_report.txt", "w") as f:
            f.write(report)
    else:
        logger.info("No matches found during audit.")

    for a in adapters: await a.close()
    await GlobalResourceManager.cleanup()

def generate_analytics_report(audited_tips: List[Dict[str, Any]]) -> str:
    lines = ["FORTUNA PERFORMANCE ANALYTICS REPORT", "=====================================", ""]

    total = len(audited_tips)
    cashed = sum(1 for t in audited_tips if t.get("verdict") == "CASHED")
    burned = sum(1 for t in audited_tips if t.get("verdict") == "BURNED")
    total_profit = sum(t.get("net_profit", 0.0) for t in audited_tips)

    strike_rate = (cashed / total * 100) if total > 0 else 0.0

    lines.append(f"Total Audited: {total}")
    lines.append(f"Cashed: {cashed}")
    lines.append(f"Burned: {burned}")
    lines.append(f"Strike Rate: {strike_rate:.1f}%")
    lines.append(f"Net Profit (unit $2.00): ${total_profit:.2f}")
    lines.append("")
    lines.append("TRIFECTA PERFORMANCE:")
    tri_races = [t for t in audited_tips if t.get("trifecta_payout")]
    lines.append(f"Trifectas tracked: {len(tri_races)}")
    if tri_races:
        avg_tri = sum(t["trifecta_payout"] for t in tri_races) / len(tri_races)
        lines.append(f"Average Trifecta Payout: ${avg_tri:.2f}")
    lines.append("")
    lines.append("DETAIL LOG:")
    lines.append("-" * 40)

    for t in audited_tips:
        report_date = t.get("report_date", "N/A")[:10]
        lines.append(f"{report_date} | {t['venue']} R{t['race_number']}")
        lines.append(f"  Verdict: {t['verdict']} | Profit: ${t.get('net_profit', 0.0):.2f}")
        lines.append(f"  Actual Top 5: [{t.get('actual_top_5', 'N/A')}]")
        if t.get("trifecta_payout"):
            lines.append(f"  Trifecta: {t.get('trifecta_combination')} paid ${t['trifecta_payout']}")
        lines.append("")

    return "\n".join(lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fortuna Analytics Engine")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=2, help="Number of days to look back")
    args = parser.parse_args()

    target_dates = []
    if args.date:
        target_dates = [args.date]
    else:
        now = datetime.now(timezone.utc)
        for i in range(args.days):
            d = now - timedelta(days=i)
            target_dates.append(d.strftime("%Y-%m-%d"))

    asyncio.run(run_analytics(target_dates))
