from __future__ import annotations
# fortuna_analytics.py
# Race result harvesting and performance analysis engine for Fortuna

import argparse
import asyncio
import json
import logging
import os
import re
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import (
    Any,
    Dict,
    Final,
    List,
    Optional,
    Tuple,
)

import structlog
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from selectolax.parser import HTMLParser, Node

# --- OPTIONAL IMPORTS ---
try:
    from scrapling import AsyncFetcher
    ASYNC_SESSIONS_AVAILABLE = True
except ImportError:
    ASYNC_SESSIONS_AVAILABLE = False

# Import monolithic fortuna
import fortuna

# --- CONSTANTS ---
DEFAULT_DB_PATH: Final[str] = os.environ.get("FORTUNA_DB_PATH", "fortuna.db")
PLACE_POSITIONS_BY_FIELD_SIZE: Final[Dict[int, int]] = {
    4: 1,   # 4 or fewer runners: only win counts
    7: 2,   # 5-7 runners: top 2
    999: 3, # 8+ runners: top 3
}


# --- HELPER FUNCTIONS ---

def parse_position(pos_str: Optional[str]) -> Optional[int]:
    """
    Extract numeric position from strings like '1st', '2/12', 'W', 'P', 'S', etc.

    Returns:
        Integer position (1-based) or None if unparseable.
    """
    if not pos_str:
        return None

    pos_str = str(pos_str).upper().strip()

    # Direct mappings for common abbreviations
    position_map = {
        "W": 1, "1": 1, "1ST": 1,
        "P": 2, "2": 2, "2ND": 2,
        "S": 3, "3": 3, "3RD": 3,
        "4": 4, "4TH": 4,
        "5": 5, "5TH": 5,
    }

    if pos_str in position_map:
        return position_map[pos_str]

    # Extract first number from strings like "2/12" or "3rd"
    match = re.search(r"^(\d+)", pos_str)
    if match:
        return int(match.group(1))

    return None


def get_places_paid(field_size: int) -> int:
    """Determine how many places are paid based on field size."""
    for max_size, places in sorted(PLACE_POSITIONS_BY_FIELD_SIZE.items()):
        if field_size <= max_size:
            return places
    return 3  # Default


def parse_currency_value(value_str: str) -> float:
    """Parse currency strings like '$123.45', '£50.00', '1,234.56'."""
    if not value_str:
        return 0.0
    try:
        cleaned = re.sub(r'[^\d.]', '', value_str)
        return float(cleaned) if cleaned else 0.0
    except (ValueError, TypeError):
        return 0.0


def validate_date_format(date_str: str) -> bool:
    """Validate YYYY-MM-DD format."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


# --- MODELS FOR ANALYTICS ---

class ResultRunner(fortuna.FortunaBaseModel):
    """
    Extended runner with result information.

    Note: We don't inherit from Runner to avoid Pydantic type conflicts.
    Instead, we include all relevant fields directly.
    """
    name: str
    number: int = 0
    position: Optional[str] = None
    position_numeric: Optional[int] = None
    scratched: bool = False

    # Result-specific fields
    final_win_odds: Optional[float] = None
    win_payout: Optional[float] = None
    place_payout: Optional[float] = None
    show_payout: Optional[float] = None

    @model_validator(mode='after')
    def compute_position_numeric(self) -> 'ResultRunner':
        """Auto-compute numeric position from string."""
        if self.position and self.position_numeric is None:
            self.position_numeric = parse_position(self.position)
        return self


class ResultRace(fortuna.FortunaBaseModel):
    """
    Race with full result data.
    """
    id: str
    venue: str
    race_number: int
    start_time: datetime
    source: str
    discipline: Optional[str] = None

    runners: List[ResultRunner] = Field(default_factory=list)
    official_dividends: Dict[str, float] = Field(default_factory=dict)
    chart_url: Optional[str] = None
    is_fully_parsed: bool = False

    # Exotic bet payouts
    trifecta_payout: Optional[float] = None
    trifecta_cost: float = 1.00
    trifecta_combination: Optional[str] = None

    exacta_payout: Optional[float] = None
    exacta_combination: Optional[str] = None

    superfecta_payout: Optional[float] = None
    superfecta_combination: Optional[str] = None

    @property
    def canonical_key(self) -> str:
        """Generate a canonical key for matching."""
        date_str = self.start_time.strftime('%Y%m%d')
        return f"{fortuna.get_canonical_venue(self.venue)}|{date_str}|{self.race_number}"

    def get_top_finishers(self, n: int = 5) -> List[ResultRunner]:
        """Get top N finishers sorted by position."""
        with_position = [
            r for r in self.runners
            if r.position_numeric is not None
        ]
        sorted_runners = sorted(with_position, key=lambda x: x.position_numeric)
        return sorted_runners[:n]


# --- AUDITOR ENGINE ---

class AuditorEngine:
    """
    Matches tips from history against actual race results using SQLite.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db = fortuna.FortunaDB(db_path or DEFAULT_DB_PATH)
        self.logger = structlog.get_logger(self.__class__.__name__)

    async def get_unverified_tips(self, lookback_hours: int = 48) -> List[Dict[str, Any]]:
        """
        Returns tips that haven't been successfully audited yet.
        """
        return await self.db.get_unverified_tips(lookback_hours)

    async def get_all_audited_tips(self) -> List[Dict[str, Any]]:
        """
        Returns all audited tips from the database.
        """
        return await self.db.get_all_audited_tips()

    async def get_recent_tips(self, limit: int = 15) -> List[Dict[str, Any]]:
        """
        Returns the absolute latest tips recorded.
        """
        return await self.db.get_recent_tips(limit)

    async def close(self):
        """Cleanup resources."""
        await self.db.close()

    async def audit_races(self, results: List[ResultRace]) -> List[Dict[str, Any]]:
        """
        Match results to history and update audit status.
        """
        # Build lookup map: canonical_key -> ResultRace
        results_map: Dict[str, ResultRace] = {}
        for r in results:
            results_map[r.canonical_key] = r

        self.logger.debug("Built results map", count=len(results_map))

        unverified = await self.get_unverified_tips()
        audited_tips = []

        for tip in unverified:
            try:
                race_id = tip.get("race_id")
                if not race_id:
                    self.logger.warning("Tip missing race_id", tip=tip)
                    continue

                tip_key = self._get_tip_canonical_key(tip)
                if not tip_key or tip_key not in results_map:
                    continue

                result = results_map[tip_key]
                self.logger.info(
                    "Auditing tip",
                    venue=tip.get('venue'),
                    race=tip.get('race_number')
                )

                outcome = self._evaluate_tip(tip, result)
                await self.db.update_audit_result(race_id, outcome)

                # Enrich tip with outcome for the return list
                tip.update(outcome)
                tip["audit_completed"] = True
                audited_tips.append(tip)

            except Exception as e:
                self.logger.error(
                    "Error during audit",
                    tip_id=tip.get("race_id"),
                    error=str(e),
                    exc_info=True
                )

        return audited_tips

    def _get_tip_canonical_key(self, tip: Dict[str, Any]) -> Optional[str]:
        """Generate canonical key for a tip."""
        venue = tip.get("venue")
        race_number = tip.get("race_number")
        start_time_raw = tip.get("start_time")

        if not all([venue, race_number, start_time_raw]):
            return None

        try:
            st = datetime.fromisoformat(str(start_time_raw))
            date_str = st.strftime('%Y%m%d')
            return f"{fortuna.get_canonical_venue(venue)}|{date_str}|{race_number}"
        except (ValueError, TypeError):
            return None

    def _evaluate_tip(self, tip: Dict[str, Any], result: ResultRace) -> Dict[str, Any]:
        """
        Compare predicted selection with actual result.

        Returns dict with audit outcome fields.
        """
        # Determine selection number
        selection_num = self._extract_selection_number(tip)

        # Get sorted finishers
        top_finishers = result.get_top_finishers(5)
        actual_top_5 = [str(r.number) for r in top_finishers]

        # Capture place payouts for top 2 finishers
        top1_place_payout = top_finishers[0].place_payout if len(top_finishers) >= 1 else None
        top2_place_payout = top_finishers[1].place_payout if len(top_finishers) >= 2 else None

        # Find 2nd favorite by final odds
        runners_with_odds = [
            r for r in result.runners
            if r.final_win_odds is not None and r.final_win_odds > 0
        ]
        runners_with_odds.sort(key=lambda x: x.final_win_odds)
        actual_2nd_fav_odds = (
            runners_with_odds[1].final_win_odds
            if len(runners_with_odds) >= 2
            else None
        )

        # Default outcome
        verdict = "BURNED"
        profit = -2.00  # Standard 1 unit loss ($2.00)

        # Find our selection in results
        selection_result = next(
            (r for r in result.runners if r.number == selection_num),
            None
        )

        if selection_result is None:
            # Selection not found (likely scratched)
            verdict = "VOID"
            profit = 0.0
        elif selection_result.position_numeric is not None:
            # Calculate places paid based on field size
            active_runners = [r for r in result.runners if not r.scratched]
            places_paid = get_places_paid(len(active_runners))

            if selection_result.position_numeric <= places_paid:
                verdict = "CASHED"

                # Calculate profit
                if selection_result.place_payout and selection_result.place_payout > 0:
                    # Use official payout (usually for $2 bet)
                    profit = selection_result.place_payout - 2.00
                else:
                    # Heuristic: ~1/5 of win odds for place
                    odds = selection_result.final_win_odds or 2.0
                    profit = ((odds - 1.0) / 5.0) * 2.0

        return {
            "actual_top_5": ", ".join(actual_top_5),
            "actual_2nd_fav_odds": actual_2nd_fav_odds,
            "verdict": verdict,
            "net_profit": round(profit, 2),
            "selection_position": (
                selection_result.position_numeric
                if selection_result else None
            ),
            "audit_timestamp": datetime.now(timezone.utc).isoformat(),
            "trifecta_payout": result.trifecta_payout,
            "trifecta_combination": result.trifecta_combination,
            "superfecta_payout": result.superfecta_payout,
            "superfecta_combination": result.superfecta_combination,
            "top1_place_payout": top1_place_payout,
            "top2_place_payout": top2_place_payout,
        }

    def _extract_selection_number(self, tip: Dict[str, Any]) -> Optional[int]:
        """Extract selection number from tip data."""
        # Direct selection_number field
        selection = tip.get("selection_number")
        if selection is not None:
            try:
                return int(selection)
            except (ValueError, TypeError):
                pass

        # Fallback to first in top_five
        top_five = tip.get("top_five", "")
        if top_five:
            first = str(top_five).split(",")[0].strip()
            try:
                return int(first)
            except (ValueError, TypeError):
                pass

        return None


# --- RESULTS ADAPTERS ---

class EquibaseResultsAdapter(fortuna.BrowserHeadersMixin, fortuna.DebugMixin, fortuna.BaseAdapterV3):
    """
    Adapter for Equibase Results / Summary Charts.
    Primary source for US thoroughbred race results.
    """

    SOURCE_NAME = "EquibaseResults"
    BASE_URL = "https://www.equibase.com"

    def __init__(self, **kwargs):
        super().__init__(
            source_name=self.SOURCE_NAME,
            base_url=self.BASE_URL,
            **kwargs
        )
        self._semaphore = asyncio.Semaphore(5)

    def _configure_fetch_strategy(self) -> fortuna.FetchStrategy:
        return fortuna.FetchStrategy(
            primary_engine=fortuna.BrowserEngine.HTTPX,
            enable_js=False,
            timeout=30,
        )

    def _get_headers(self) -> dict:
        return self._get_browser_headers(host="www.equibase.com")

    async def _fetch_data(self, date_str: str) -> Optional[Dict[str, Any]]:
        """Fetch results index and all track pages for a date."""
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            self.logger.error("Invalid date format", date=date_str)
            return None

        url = f"/static/chart/summary/index.html?date={dt.strftime('%m/%d/%Y')}"

        resp = await self.make_request("GET", url, headers=self._get_headers())
        if not resp or not resp.text:
            self.logger.warning("No response from Equibase index", url=url)
            return None

        self._save_debug_snapshot(resp.text, f"eqb_results_index_{date_str}")
        parser = HTMLParser(resp.text)

        # Extract track-specific result page links
        links = set()
        for a in parser.css("a"):
            href = a.attributes.get("href", "")
            if (
                "/static/chart/summary/" in href
                and href.endswith(".html")
                and "index.html" not in href
            ):
                links.add(href)

        if not links:
            self.logger.warning("No track result links found", date=date_str)
            return None

        self.logger.info("Found track result pages", count=len(links))

        # Fetch all track pages concurrently
        async def fetch_track_page(link: str) -> Tuple[str, str]:
            async with self._semaphore:
                try:
                    r = await self.make_request("GET", link, headers=self._get_headers())
                    return (link, r.text if r else "")
                except Exception as e:
                    self.logger.warning("Failed to fetch track page", link=link, error=str(e))
                    return (link, "")

        tasks = [fetch_track_page(link) for link in links]
        pages = await asyncio.gather(*tasks)

        valid_pages = [(link, html) for link, html in pages if html]
        return {"pages": valid_pages, "date": date_str}

    def _parse_races(self, raw_data: Any) -> List[ResultRace]:
        """Parse all track pages into ResultRace objects."""
        if not raw_data or not raw_data.get("pages"):
            return []

        races = []
        for link, html_content in raw_data["pages"]:
            if not html_content:
                continue
            try:
                parsed = self._parse_track_page(html_content, raw_data["date"], link)
                races.extend(parsed)
            except Exception as e:
                self.logger.warning(
                    "Failed to parse track page",
                    link=link,
                    error=str(e),
                    exc_info=True
                )
        return races

    def _parse_track_page(
        self,
        html_content: str,
        date_str: str,
        source_url: str
    ) -> List[ResultRace]:
        """Parse a single track's results page."""
        parser = HTMLParser(html_content)
        races = []

        # Get venue from header
        track_node = parser.css_first("h3") or parser.css_first("h2")
        if not track_node:
            self.logger.debug("No track header found", url=source_url)
            return []

        venue = fortuna.normalize_venue_name(track_node.text(strip=True))
        if not venue:
            return []

        # Find all race tables - they typically have "Race X" in the header
        all_tables = parser.css("table")
        race_tables = []

        for table in all_tables:
            header = table.css_first("thead tr th")
            if header and "Race" in header.text():
                race_tables.append(table)

        for race_table in race_tables:
            try:
                race = self._parse_race_table(race_table, venue, date_str, parser)
                if race:
                    races.append(race)
            except Exception as e:
                self.logger.debug("Failed to parse race table", error=str(e))

        return races

    def _parse_race_table(
        self,
        race_table: Node,
        venue: str,
        date_str: str,
        page_parser: HTMLParser
    ) -> Optional[ResultRace]:
        """Parse a single race table into a ResultRace."""
        header = race_table.css_first("thead tr th")
        if not header:
            return None

        race_num_match = re.search(r"Race\s+(\d+)", header.text())
        if not race_num_match:
            return None

        race_num = int(race_num_match.group(1))

        # Parse runners
        runners = []
        for row in race_table.css("tbody tr"):
            runner = self._parse_runner_row(row)
            if runner:
                runners.append(runner)

        if not runners:
            return None

        # Parse exotic payouts from the dividends section
        # The dividends are typically in a separate table following the race results
        trifecta_payout, trifecta_combo = self._find_exotic_payout(
            race_table, page_parser, "trifecta"
        )
        exacta_payout, exacta_combo = self._find_exotic_payout(
            race_table, page_parser, "exacta"
        )
        superfecta_payout, superfecta_combo = self._find_exotic_payout(
            race_table, page_parser, "superfecta"
        )

        # Build start time
        try:
            race_date = datetime.strptime(date_str, "%Y-%m-%d")
            start_time = race_date.replace(
                hour=12, minute=0,
                tzinfo=timezone.utc
            )
        except ValueError:
            start_time = datetime.now(timezone.utc)

        return ResultRace(
            id=f"eqb_res_{fortuna.get_canonical_venue(venue)}_{date_str.replace('-', '')}_R{race_num}",
            venue=venue,
            race_number=race_num,
            start_time=start_time,
            runners=runners,
            source=self.SOURCE_NAME,
            is_fully_parsed=True,
            trifecta_payout=trifecta_payout,
            trifecta_combination=trifecta_combo,
            exacta_payout=exacta_payout,
            exacta_combination=exacta_combo,
            superfecta_payout=superfecta_payout,
            superfecta_combination=superfecta_combo,
        )

    def _parse_runner_row(self, row: Node) -> Optional[ResultRunner]:
        """Parse a single runner row from results table."""
        cols = row.css("td")
        if len(cols) < 3:
            return None

        pos_text = fortuna.clean_text(cols[0].text())
        num_text = fortuna.clean_text(cols[1].text())
        name = fortuna.clean_text(cols[2].text())

        # Skip header-like rows
        if not name or name.upper() in ("HORSE", "NAME", "RUNNER"):
            return None

        # Parse number
        try:
            number = int(num_text) if num_text.isdigit() else 0
        except ValueError:
            number = 0

        # Parse odds
        odds_text = fortuna.clean_text(cols[3].text()) if len(cols) > 3 else ""
        final_odds = fortuna.parse_odds_to_decimal(odds_text)

        # Parse payouts (columns 4, 5, 6 typically)
        win_pay = place_pay = show_pay = 0.0
        if len(cols) >= 7:
            win_pay = parse_currency_value(cols[4].text())
            place_pay = parse_currency_value(cols[5].text())
            show_pay = parse_currency_value(cols[6].text())

        return ResultRunner(
            name=name,
            number=number,
            position=pos_text,
            final_win_odds=final_odds,
            win_payout=win_pay,
            place_payout=place_pay,
            show_payout=show_pay,
        )

    def _find_exotic_payout(
        self,
        race_table: Node,
        page_parser: HTMLParser,
        bet_type: str
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Find exotic bet payout from dividend tables.

        This looks for a table following the race table that contains
        the specified bet type (e.g., "trifecta", "exacta").
        """
        # Get all tables and find the race table's position
        all_tables = page_parser.css("table")
        race_table_html = race_table.html if hasattr(race_table, 'html') else str(race_table)

        found_race_table = False
        for table in all_tables:
            table_html = table.html if hasattr(table, 'html') else str(table)

            if not found_race_table:
                # Check if this is our race table
                if table_html == race_table_html:
                    found_race_table = True
                continue

            # We're now past the race table - look for dividends
            table_text = table.text().lower()
            if bet_type.lower() not in table_text:
                continue

            # Found a table with our bet type - extract payout
            for row in table.css("tr"):
                row_text = row.text().lower()
                if bet_type.lower() in row_text:
                    cols = row.css("td")
                    if len(cols) >= 2:
                        combination = fortuna.clean_text(cols[0].text())
                        payout = parse_currency_value(cols[1].text())
                        if payout > 0:
                            return payout, combination

            # Only check the next table after race results
            break

        return None, None


class RacingPostResultsAdapter(fortuna.BrowserHeadersMixin, fortuna.DebugMixin, fortuna.RacePageFetcherMixin, fortuna.BaseAdapterV3):
    """Adapter for Racing Post UK/IRE results."""

    SOURCE_NAME = "RacingPostResults"
    BASE_URL = "https://www.racingpost.com"

    def __init__(self, **kwargs):
        super().__init__(
            source_name=self.SOURCE_NAME,
            base_url=self.BASE_URL,
            **kwargs
        )

    def _configure_fetch_strategy(self) -> fortuna.FetchStrategy:
        return fortuna.FetchStrategy(primary_engine=fortuna.BrowserEngine.HTTPX, enable_js=False)

    def _get_headers(self) -> dict:
        return self._get_browser_headers(host="www.racingpost.com")

    async def _fetch_data(self, date_str: str) -> Optional[Dict[str, Any]]:
        url = f"/results/{date_str}"
        resp = await self.make_request("GET", url, headers=self._get_headers())
        if not resp:
            return None

        parser = HTMLParser(resp.text)

        # Find individual race result links
        links = []
        for a in parser.css("a[href*='/results/']"):
            href = a.attributes.get("href", "")
            # Match links with numeric race ID
            if re.search(r"/results/\d+/", href):
                links.append(href)

        if not links:
            self.logger.warning("No result links found", date=date_str)
            return None

        # Deduplicate
        unique_links = list(set(links))
        self.logger.info("Found result links", count=len(unique_links))

        # Fetch race pages
        metadata = [{"url": link, "race_number": 0} for link in unique_links]
        pages = await self._fetch_race_pages_concurrent(metadata, self._get_headers())

        return {"pages": pages, "date": date_str}

    def _parse_races(self, raw_data: Any) -> List[ResultRace]:
        if not raw_data:
            return []

        races = []
        date_str = raw_data.get("date", datetime.now().strftime("%Y-%m-%d"))

        for item in raw_data.get("pages", []):
            html_content = item.get("html")
            if not html_content:
                continue

            try:
                race = self._parse_race_page(html_content, date_str, item.get("url", ""))
                if race:
                    races.append(race)
            except Exception as e:
                self.logger.warning("Failed to parse RP result", error=str(e))

        return races

    def _parse_race_page(
        self,
        html_content: str,
        date_str: str,
        url: str
    ) -> Optional[ResultRace]:
        """Parse a Racing Post result page."""
        parser = HTMLParser(html_content)

        # Get venue
        venue_node = parser.css_first(".rp-raceTimeCourseName__course")
        if not venue_node:
            return None
        venue = fortuna.normalize_venue_name(venue_node.text(strip=True))

        # Extract dividends
        dividends = {}
        tote_container = parser.css_first('div[data-test-selector="RC-toteReturns"]')
        if not tote_container:
             tote_container = parser.css_first('.rp-toteReturns')

        if tote_container:
            for row in (tote_container.css('div.rp-toteReturns__row') or tote_container.css('.rp-toteReturns__row')):
                label_node = row.css_first('div.rp-toteReturns__label') or row.css_first('.rp-toteReturns__label')
                val_node = row.css_first('div.rp-toteReturns__value') or row.css_first('.rp-toteReturns__value')
                if label_node and val_node:
                    label = fortuna.clean_text(label_node.text())
                    value = fortuna.clean_text(val_node.text())
                    if label and value:
                        dividends[label] = value

        # Extract exotic payouts
        trifecta_pay = trifecta_combo = None
        superfecta_pay = superfecta_combo = None

        for label, val in dividends.items():
            l_lower = label.lower()
            if "trifecta" in l_lower or "tricast" in l_lower:
                trifecta_pay = parse_currency_value(val)
                trifecta_combo = val.split("£")[-1].strip() if "£" in val else None
            elif "superfecta" in l_lower or "first 4" in l_lower:
                superfecta_pay = parse_currency_value(val)
                superfecta_combo = val.split("£")[-1].strip() if "£" in val else None

        # Extract race number from header or navigation
        race_num = 1
        # Priority 1: Navigation bar active time (most reliable on RP)
        time_links = parser.css('a[data-test-selector="RC-raceTime"]')
        found_in_nav = False
        for i, link in enumerate(time_links):
            cls = link.attributes.get("class", "")
            if "active" in cls or "rp-raceTimeCourseName__time" in cls:
                race_num = i + 1
                found_in_nav = True
                break

        if not found_in_nav:
            # Priority 2: Text search for "Race X"
            race_num_match = re.search(r'Race\s+(\d+)', parser.text())
            if race_num_match:
                race_num = int(race_num_match.group(1))

        # Parse runners
        runners = []
        for row in parser.css(".rp-horseTable__table__row"):
            try:
                name_node = row.css_first(".rp-horseTable__horse__name")
                pos_node = row.css_first(".rp-horseTable__pos__number")

                if not name_node:
                    continue

                name = fortuna.clean_text(name_node.text())
                pos = fortuna.clean_text(pos_node.text()) if pos_node else None

                # Try to get saddle number
                number_node = row.css_first(".rp-horseTable__saddleClothNo")
                number = 0
                if number_node:
                    num_text = fortuna.clean_text(number_node.text())
                    try:
                        number = int(num_text)
                    except ValueError:
                        pass

                # Check for place dividend in dividends map
                place_payout = None
                for label, val in dividends.items():
                    if "place" in label.lower() and name.lower() in label.lower():
                        place_payout = parse_currency_value(val)
                        break

                runners.append(ResultRunner(
                    name=name,
                    number=number,
                    position=pos,
                    place_payout=place_payout
                ))
            except Exception:
                continue

        if not runners:
            return None

        try:
            race_date = datetime.strptime(date_str, "%Y-%m-%d")
            start_time = race_date.replace(hour=12, minute=0, tzinfo=timezone.utc)
        except ValueError:
            start_time = datetime.now(timezone.utc)

        return ResultRace(
            id=f"rp_res_{fortuna.get_canonical_venue(venue)}_{date_str.replace('-', '')}_R{race_num}",
            venue=venue,
            race_number=race_num,
            start_time=start_time,
            runners=runners,
            source=self.SOURCE_NAME,
            trifecta_payout=trifecta_pay,
            trifecta_combination=trifecta_combo,
            superfecta_payout=superfecta_pay,
            superfecta_combination=superfecta_combo,
            official_dividends={k: parse_currency_value(v) for k, v in dividends.items()}
        )


class AtTheRacesResultsAdapter(fortuna.BrowserHeadersMixin, fortuna.DebugMixin, fortuna.RacePageFetcherMixin, fortuna.BaseAdapterV3):
    """Adapter for At The Races results (UK/IRE)."""

    SOURCE_NAME = "AtTheRacesResults"
    BASE_URL = "https://www.attheraces.com"

    def __init__(self, **kwargs):
        super().__init__(
            source_name=self.SOURCE_NAME,
            base_url=self.BASE_URL,
            **kwargs
        )

    def _configure_fetch_strategy(self) -> fortuna.FetchStrategy:
        return fortuna.FetchStrategy(primary_engine=fortuna.BrowserEngine.HTTPX, enable_js=False)

    def _get_headers(self) -> dict:
        return self._get_browser_headers(host="www.attheraces.com")

    async def _fetch_data(self, date_str: str) -> Optional[Dict[str, Any]]:
        url = f"/results/{date_str}"
        resp = await self.make_request("GET", url, headers=self._get_headers())
        if not resp:
            return None

        parser = HTMLParser(resp.text)

        # Find result page links
        links = []
        for a in parser.css("a[href*='/results/']"):
            href = a.attributes.get("href", "")
            # ATR format: /results/Venue/DD-Mon-YYYY/HHMM
            if re.search(r"/results/[^/]+/\d{2}-[A-Za-z]{3}-\d{4}/", href):
                links.append(href)

        unique_links = list(set(links))
        if not unique_links:
            return None

        metadata = [{"url": link, "race_number": 0} for link in unique_links]
        pages = await self._fetch_race_pages_concurrent(metadata, self._get_headers())

        return {"pages": pages, "date": date_str}

    def _parse_races(self, raw_data: Any) -> List[ResultRace]:
        if not raw_data:
            return []

        races = []
        date_str = raw_data.get("date", datetime.now().strftime("%Y-%m-%d"))

        for item in raw_data.get("pages", []):
            html_content = item.get("html")
            if not html_content:
                continue

            try:
                race = self._parse_race_page(html_content, date_str, item.get("url", ""))
                if race:
                    races.append(race)
            except Exception as e:
                self.logger.warning("Failed to parse ATR result", error=str(e))

        return races

    def _parse_race_page(
        self,
        html_content: str,
        date_str: str,
        url: str
    ) -> Optional[ResultRace]:
        """Parse an ATR result page."""
        parser = HTMLParser(html_content)

        header = parser.css_first(".race-header__details--primary")
        if not header:
            return None

        venue_node = header.css_first("h2")
        if not venue_node:
            return None
        venue = fortuna.normalize_venue_name(venue_node.text(strip=True))

        # Extract race number from URL: /results/Venue/Date/R1 or just time
        race_num = 1
        url_match = re.search(r'/R(\d+)$', url)
        if url_match:
            race_num = int(url_match.group(1))

        # Parse runners
        runners = []
        for row in parser.css(".result-racecard__row"):
            try:
                name_node = row.css_first(".result-racecard__horse-name a")
                pos_node = row.css_first(".result-racecard__pos")

                if not name_node:
                    continue

                name = fortuna.clean_text(name_node.text())
                pos = fortuna.clean_text(pos_node.text()) if pos_node else None

                # Saddle number
                num_node = row.css_first(".result-racecard__saddle-cloth")
                number = 0
                if num_node:
                    try:
                        number = int(fortuna.clean_text(num_node.text()))
                    except ValueError:
                        pass

                runners.append(ResultRunner(
                    name=name,
                    number=number,
                    position=pos,
                ))
            except Exception:
                continue

        # Parse exotic payouts from dividends table
        trifecta_pay = None
        trifecta_combo = None
        superfecta_pay = None
        superfecta_combo = None

        div_table = parser.css_first(".result-racecard__dividends-table")
        if div_table:
            for row in div_table.css("tr"):
                row_text = row.text().lower()
                cols = row.css("td")
                if len(cols) < 2: continue

                if "trifecta" in row_text:
                    trifecta_combo = fortuna.clean_text(cols[0].text())
                    trifecta_pay = parse_currency_value(cols[1].text())
                elif "superfecta" in row_text or "first 4" in row_text:
                    superfecta_combo = fortuna.clean_text(cols[0].text())
                    superfecta_pay = parse_currency_value(cols[1].text())
                elif "place" in row_text:
                    # Map place dividends to runners if possible
                    p_name = fortuna.clean_text(cols[0].text().replace("Place", "").strip())
                    p_val = parse_currency_value(cols[1].text())
                    for r in runners:
                        if r.name.lower() in p_name.lower() or p_name.lower() in r.name.lower():
                            r.place_payout = p_val

        if not runners:
            return None

        try:
            race_date = datetime.strptime(date_str, "%Y-%m-%d")
            start_time = race_date.replace(hour=12, minute=0, tzinfo=timezone.utc)
        except ValueError:
            start_time = datetime.now(timezone.utc)

        return ResultRace(
            id=f"atr_res_{fortuna.get_canonical_venue(venue)}_{date_str.replace('-', '')}_R{race_num}",
            venue=venue,
            race_number=race_num,
            start_time=start_time,
            runners=runners,
            trifecta_payout=trifecta_pay,
            trifecta_combination=trifecta_combo,
            superfecta_payout=superfecta_pay,
            superfecta_combination=superfecta_combo,
            source=self.SOURCE_NAME,
        )


# Placeholder adapters - implement as needed
class SportingLifeResultsAdapter(fortuna.BaseAdapterV3):
    """Placeholder for Sporting Life results adapter."""

    def __init__(self, **kwargs):
        super().__init__(
            source_name="SportingLifeResults",
            base_url="https://www.sportinglife.com",
            **kwargs
        )

    def _configure_fetch_strategy(self) -> fortuna.FetchStrategy:
        return fortuna.FetchStrategy(primary_engine=fortuna.BrowserEngine.HTTPX, enable_js=False)

    async def _fetch_data(self, date_str: str) -> Optional[Dict[str, Any]]:
        # TODO: Implement
        return {"pages": [], "date": date_str}

    def _parse_races(self, raw_data: Any) -> List[ResultRace]:
        return []


class SkySportsResultsAdapter(fortuna.BaseAdapterV3):
    """Placeholder for Sky Sports results adapter."""

    def __init__(self, **kwargs):
        super().__init__(
            source_name="SkySportsResults",
            base_url="https://www.skysports.com",
            **kwargs
        )

    def _configure_fetch_strategy(self) -> fortuna.FetchStrategy:
        return fortuna.FetchStrategy(primary_engine=fortuna.BrowserEngine.HTTPX, enable_js=False)

    async def _fetch_data(self, date_str: str) -> Optional[Dict[str, Any]]:
        # TODO: Implement
        return {"pages": [], "date": date_str}

    def _parse_races(self, raw_data: Any) -> List[ResultRace]:
        return []


# --- REPORT GENERATION ---

def generate_analytics_report(
    audited_tips: List[Dict[str, Any]],
    recent_tips: List[Dict[str, Any]] = None,
    harvest_summary: Dict[str, int] = None
) -> str:
    """Generate a high-impact human-readable performance audit report."""
    now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    lines = [
        "=" * 80,
        "🐎 FORTUNA INTELLIGENCE - PERFORMANCE AUDIT & VERIFICATION".center(80),
        f"Generated: {now_str}".center(80),
        "=" * 80,
        "",
    ]

    # --- 1. PROOF OF HARVESTING ---
    if harvest_summary:
        lines.extend([
            "🔎 LIVE ADAPTER HARVEST PROOF",
            "-" * 40,
        ])
        for adapter, count in harvest_summary.items():
            status = "✅ SUCCESS" if count > 0 else "⏳ PENDING/NO DATA"
            lines.append(f"{adapter:<25} | {status:<15} | Results Found: {count}")
        lines.append("")

    # --- 2. PENDING VERIFICATION (THE "WATCH" LIST) ---
    if recent_tips:
        lines.extend([
            "⏳ PENDING VERIFICATION - RECENT DISCOVERIES",
            "-" * 40,
            f"{'RACE TIME':<18} | {'VENUE':<20} | {'R#':<3} | {'GM?':<4} | {'STATUS'}",
            "." * 80
        ])
        for tip in recent_tips[:15]:
            st_str = str(tip.get("start_time", "N/A"))[:16].replace("T", " ")
            venue = str(tip.get("venue", "Unknown"))[:20]
            rnum = tip.get("race_number", "?")
            gm = "GOLD" if tip.get("is_goldmine") else "----"
            status = tip.get("verdict") if tip.get("audit_completed") else "WATCHING"
            lines.append(f"{st_str:<18} | {venue:<20} | {rnum:<3} | {gm:<4} | {status}")
        lines.append("")

    # --- 3. RECENT PERFORMANCE PROOF ---
    audited_recent = sorted(audited_tips, key=lambda x: x.get("start_time", ""), reverse=True)
    if audited_recent:
        lines.extend([
            "💰 RECENT PERFORMANCE PROOF (MATCHED RESULTS)",
            "-" * 40,
            f"{'RESULT':<6} | {'RACE':<25} | {'PROFIT':<8} | {'PAYOUT/DETAILS'}",
            "." * 80
        ])
        for tip in audited_recent[:15]:
            verdict = tip.get("verdict", "?")
            emoji = "✅ WIN " if verdict == "CASHED" else "❌ LOSS" if verdict == "BURNED" else "⚪ VOID"
            venue = f"{tip.get('venue', 'Unknown')[:18]} R{tip.get('race_number', '?')}"
            profit = f"${tip.get('net_profit', 0.0):+.2f}"

            payout_info = ""
            p1 = tip.get("top1_place_payout")
            p2 = tip.get("top2_place_payout")
            if p1 or p2:
                payout_info = f"P: {p1 or 0:.2f}/{p2 or 0:.2f} | "

            if tip.get("superfecta_payout"):
                payout_info += f"Super: ${tip['superfecta_payout']:.2f}"
            elif tip.get("trifecta_payout"):
                payout_info += f"Tri: ${tip['trifecta_payout']:.2f}"
            elif tip.get("actual_top_5"):
                payout_info += f"Top 5: [{tip['actual_top_5']}]"

            lines.append(f"{emoji:<6} | {venue:<25} | {profit:<8} | {payout_info}")
        lines.append("")

    # --- 4. SUPERFECTA & TRIFECTA TRACKING ---
    super_races = [t for t in audited_tips if t.get("superfecta_payout")]
    tri_races = [t for t in audited_tips if t.get("trifecta_payout")]

    if super_races or tri_races:
        lines.extend([
            "🎯 EXOTIC PAYOUT TRACKING",
            "-" * 40,
        ])
        if super_races:
            avg_super = sum(t["superfecta_payout"] for t in super_races) / len(super_races)
            max_super = max(t["superfecta_payout"] for t in super_races)
            lines.extend([
                f"Superfecta Matches: {len(super_races)}",
                f"  Average Payout:   ${avg_super:.2f}",
                f"  Maximum Payout:   ${max_super:.2f}",
            ])
        if tri_races:
            avg_tri = sum(t["trifecta_payout"] for t in tri_races) / len(tri_races)
            lines.append(f"Trifecta Matches:   {len(tri_races)} (Avg: ${avg_tri:.2f})")
        lines.append("")

    # --- 5. SUMMARY STATISTICS ---
    if audited_tips:
        total = len(audited_tips)
        cashed = sum(1 for t in audited_tips if t.get("verdict") == "CASHED")
        total_profit = sum(t.get("net_profit", 0.0) for t in audited_tips)
        strike_rate = (cashed / total * 100) if total > 0 else 0.0
        roi = (total_profit / (total * 2.0) * 100) if total > 0 else 0.0

        lines.extend([
            "📊 SUMMARY METRICS (LIFETIME)",
            "-" * 40,
            f"Total Verified Races: {total}",
            f"Overall Strike Rate:   {strike_rate:.1f}%",
            f"Total Net Profit:     ${total_profit:+.2f} (Using $2.00 Base Unit)",
            f"Return on Investment:  {roi:+.1f}%",
            ""
        ])

    return "\n".join(lines)


# --- MAIN ORCHESTRATOR ---

@asynccontextmanager
async def managed_adapters():
    """Context manager for adapter lifecycle using auto-discovery."""
    def get_all_adapter_classes(cls):
        return set(cls.__subclasses__()).union(
            [s for c in cls.__subclasses__() for s in get_all_adapter_classes(c)]
        )

    adapter_classes = [
        c for c in get_all_adapter_classes(fortuna.BaseAdapterV3)
        if not getattr(c, "__abstractmethods__", None)
        and c.__module__ == __name__ # Only local results adapters
    ]

    adapters = [cls() for cls in adapter_classes]
    try:
        yield adapters
    finally:
        for adapter in adapters:
            try:
                await adapter.close()
            except Exception:
                pass
        try:
            await fortuna.GlobalResourceManager.cleanup()
        except Exception:
            pass


async def run_analytics(target_dates: List[str]) -> None:
    """Main analytics orchestration function."""
    logger = structlog.get_logger("run_analytics")
    logger.info("Starting Analytics Audit", dates=target_dates)

    # Validate dates
    valid_dates = [d for d in target_dates if validate_date_format(d)]
    if not valid_dates:
        logger.error("No valid dates provided", input_dates=target_dates)
        return

    auditor = AuditorEngine()
    try:
        unverified = await auditor.get_unverified_tips()

        if not unverified:
            logger.info("No unverified tips found in history. Skipping harvest, showing lifetime report.")
        else:
            logger.info("Tips to audit", count=len(unverified))

            all_results: List[ResultRace] = []

            harvest_summary: Dict[str, int] = {}
            async with managed_adapters() as adapters:
                # Create fetch tasks for all date/adapter combinations
                async def fetch_with_adapter(adapter: fortuna.BaseAdapterV3, date_str: str) -> Tuple[str, List[ResultRace]]:
                    try:
                        races = await adapter.get_races(date_str)
                        logger.debug(
                            "Fetched results",
                            adapter=adapter.source_name,
                            date=date_str,
                            count=len(races)
                        )
                        return adapter.source_name, races
                    except Exception as e:
                        logger.warning(
                            "Adapter fetch failed",
                            adapter=adapter.source_name,
                            date=date_str,
                            error=str(e)
                        )
                        return adapter.source_name, []

                tasks = [
                    fetch_with_adapter(adapter, date_str)
                    for date_str in valid_dates
                    for adapter in adapters
                ]

                fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for res in fetch_results:
                    if isinstance(res, tuple):
                        adapter_name, races = res
                        all_results.extend(races)
                        harvest_summary[adapter_name] = harvest_summary.get(adapter_name, 0) + len(races)
                    elif isinstance(res, Exception):
                        logger.warning("Task raised exception", error=str(res))

            logger.info("Total results harvested", count=len(all_results))

            if not all_results:
                logger.warning("No results harvested from any source")
                # We continue to show report if we have previous audits
            else:
                # Perform audit
                await auditor.audit_races(all_results)

        # Generate and save comprehensive report
        all_audited = await auditor.get_all_audited_tips()
        recent_tips = await auditor.get_recent_tips(limit=20)

        report = generate_analytics_report(
            audited_tips=all_audited,
            recent_tips=recent_tips,
            harvest_summary=harvest_summary if 'harvest_summary' in locals() else None
        )
        print(report)

        report_path = Path("analytics_report.txt")
        try:
            report_path.write_text(report, encoding="utf-8")
            logger.info("Report saved", path=str(report_path))
        except IOError as e:
            logger.error("Failed to save report", error=str(e))

        # Summary
        if all_audited:
            logger.info("Analytics audit summary generated", total_audited=len(all_audited))
        else:
            logger.info("No audited tips found in history")
    finally:
        await auditor.close()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fortuna Analytics Engine - Race result auditing and performance analysis"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Target date (YYYY-MM-DD format)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=2,
        help="Number of days to look back (default: 2)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to tip history database (default: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Migrate data from legacy JSON to SQLite"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(log_level)
    )

    # Set DB path
    if args.db_path != DEFAULT_DB_PATH:
        os.environ["FORTUNA_DB_PATH"] = args.db_path

    if args.migrate:
        async def do_migrate():
            db = fortuna.FortunaDB(args.db_path)
            try:
                await db.migrate_from_json()
                print("Migration complete.")
            except Exception as e:
                print(f"Migration failed: {e}")

        asyncio.run(do_migrate())
        return

    # Build target dates list
    target_dates = []
    if args.date:
        if not validate_date_format(args.date):
            print(f"Error: Invalid date format '{args.date}'. Use YYYY-MM-DD.")
            return
        target_dates = [args.date]
    else:
        now = datetime.now(timezone.utc)
        for i in range(args.days):
            d = now - timedelta(days=i)
            target_dates.append(d.strftime("%Y-%m-%d"))

    # Run
    asyncio.run(run_analytics(target_dates))


if __name__ == "__main__":
    main()
