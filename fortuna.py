from __future__ import annotations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CRITICAL: Fix for Playwright + PyInstaller + Windows
# Must be at the very top, before any other imports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import sys
import platform
import os

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CRITICAL: Monkeypatch playwright with patchright
# Scrapling and Camoufox expect playwright internals
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
try:
    import patchright
    sys.modules['playwright'] = patchright
except ImportError:
    pass

if platform.system() == 'Windows' and getattr(sys, 'frozen', False):
    # Running as frozen EXE on Windows
    import asyncio
    try:
        # Check if Playwright is likely to be available
        playwright_path = os.path.expanduser("~\\AppData\\Local\\ms-playwright")
        has_playwright = os.path.exists(playwright_path)

        # Hardening Fix: Default to Selector loop if Playwright is missing to satisfy curl_cffi
        if os.getenv("FORTUNA_USE_SELECTOR_EVENT_LOOP") == "1" or not has_playwright:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        else:
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except AttributeError:
        pass
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# fortuna_discovery_engine.py
# Aggregated monolithic discovery adapters for Fortuna
# This engine serves as a high-reliability fallback for the Fortuna discovery system.

"""
Fortuna Discovery Engine - Production-grade racing data aggregation.

This module provides a unified collection of adapters for fetching racecard data
from various racing websites. It serves as a high-reliability fallback system.
"""
import argparse
import asyncio
import functools
from functools import lru_cache
import html
import json
import logging
import os
import random
import weakref
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
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
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    TypedDict,
    Literal,
)

import httpx
import pandas as pd
import sqlite3
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import structlog
import subprocess
import sys
import threading
import webbrowser
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
except Exception:
    curl_requests = None

try:
    import tomli
    HAS_TOML = True
except ImportError:
    HAS_TOML = False

try:
    from scrapling import AsyncFetcher, Fetcher
    from scrapling.parser import Selector
    ASYNC_SESSIONS_AVAILABLE = True
except Exception:
    ASYNC_SESSIONS_AVAILABLE = False
    Selector = None  # type: ignore

try:
    from scrapling.fetchers import AsyncDynamicSession, AsyncStealthySession
except Exception:
    ASYNC_SESSIONS_AVAILABLE = False

try:
    from scrapling.core.custom_types import StealthMode
except Exception:
    class StealthMode:  # type: ignore
        FAST = "fast"
        CAMOUFLAGE = "camouflage"

try:
    import winsound
except (ImportError, RuntimeError):
    winsound = None


def get_resp_status(resp: Any) -> Union[int, str]:
    if hasattr(resp, "status_code"): return resp.status_code
    return getattr(resp, "status", "unknown")

def is_frozen() -> bool:
    """Check if running as a frozen executable (PyInstaller, cx_Freeze, etc.)"""
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')

def get_base_path() -> Path:
    """Returns the base path of the application (frozen or source)."""
    if is_frozen():
        return Path(sys._MEIPASS)
    return Path(__file__).parent

def load_config() -> Dict[str, Any]:
    """Loads configuration from config.toml with intelligent fallback."""
    config = {
        "analysis": {"simply_success_trust_min": 0.25, "max_field_size": 11},
        "region": {"default": "GLOBAL"},
        "ui": {"auto_open_report": True, "show_status_card": True},
        "logging": {"level": "INFO", "save_to_file": True}
    }

    config_paths = [Path("config.toml")]
    if is_frozen():
        config_paths.insert(0, Path(sys.executable).parent / "config.toml")
        config_paths.append(Path(sys._MEIPASS) / "config.toml")

    selected_config = None
    for cp in config_paths:
        if cp.exists():
            selected_config = cp
            break

    if selected_config and HAS_TOML:
        try:
            with open(selected_config, "rb") as f:
                toml_data = tomli.load(f)
                # Deep merge simple dict
                for section, values in toml_data.items():
                    if section in config and isinstance(values, dict):
                        config[section].update(values)
                    else:
                        config[section] = values

                # Deprecation bridge for trustworthy_ratio_min (BUG-2)
                analysis_cfg = config.get("analysis", {})
                legacy_val = analysis_cfg.get("trustworthy_ratio_min")
                if legacy_val is not None:
                    structlog.get_logger().warning("config key analysis.trustworthy_ratio_min is deprecated; use analysis.simply_success_trust_min")
                    if "simply_success_trust_min" not in toml_data.get("analysis", {}):
                        analysis_cfg["simply_success_trust_min"] = legacy_val

        except Exception as e:
            print(f"Warning: Failed to load config.toml: {e} - using default configuration")
    else:
        # Explicitly log if we are falling back to defaults due to missing config or parser
        if not selected_config:
            structlog.get_logger().debug("No config.toml found, using default configuration")
        elif not HAS_TOML:
            structlog.get_logger().warning("tomli not installed, using default configuration")

    return config

def print_status_card(config: Dict[str, Any]):
    """Prints a friendly status card with application health and latest metrics."""
    if not config.get("ui", {}).get("show_status_card", True):
        return

    version = "Unknown"
    version_file = get_base_path() / "VERSION"
    if version_file.exists():
        version = version_file.read_text().strip()

    try:
        from rich.console import Console
        console = Console()
        print_func = console.print
    except ImportError:
        # Fallback to structlog for telemetry (Capability Improvement)
        sl = structlog.get_logger()
        print_func = lambda msg: sl.info(msg)

    print_func("\n" + "🌟 " + "═" * 54 + " 🌟")
    print_func(f" ✨ FORTUNA FAUCET INTELLIGENCE - v{version} ✨ ".center(60, "═"))
    print_func("🌟 " + "═" * 54 + " 🌟")

    # Friendly Greeting
    print_func(f"\n [bold yellow]Hello JB![/] 🚀")
    print_func(f" Ready to discover some incredible racing opportunities today!\n")

    # Region and active mode
    region = config.get("region", {}).get("default", "GLOBAL")
    print_func(f" 📍 Region: [bold cyan]{region}[/] | 🔍 Status: [bold green]POWERED UP & READY[/]")

    # Database status
    db = FortunaDB()
    # We'll use a sync helper or just run it
    try:
        # Simple sqlite check
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM tips")
        total_tips = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM tips WHERE audit_completed = 1")
        audited = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM tips WHERE is_goldmine = 1")
        goldmines = cursor.fetchone()[0]
        conn.close()

        print_func(f" 📊 Database: [bold white]{total_tips}[/] tips | ✅ [bold green]{audited}[/] audited | 💎 [bold gold1]{goldmines}[/] goldmines")
    except Exception:
        print_func(" 📊 Database: WARMING UP...")

    # Odds Hygiene
    trust_min = config.get("analysis", {}).get("simply_success_trust_min", 0.25)
    print_func(f" 🛡️  Odds Hygiene: >{int(trust_min*100)}% trust ratio (Pure Signal Only!)")

    # Golden Tip
    tips = [
        "Goldmines are the purest signal – trust the gap!",
        "Handicap races with large fields often yield the best place value.",
        "Consistency is key: 24/7 global fetching keeps the faucet running.",
        "The Council of Superbrains recommends checking the Goldmine report first!",
        "A 70% trust ratio ensures we only bet on high-fidelity data."
    ]
    print_func(f" 💡 [bold blue]Golden Tip:[/] {random.choice(tips)}")

    # Reports
    reports = []
    if get_writable_path("summary_grid.txt").exists(): reports.append("Summary")
    if get_writable_path("fortuna_report.html").exists(): reports.append("HTML")
    if reports:
        print_func(f" 📁 Latest Intel: [bold magenta]{', '.join(reports)}[/]")

    print_func("\n" + "🌟 " + "═" * 54 + " 🌟" + "\n")

async def print_goldmine_intelligence(db: FortunaDB) -> None:
    try:
        stats = await db.get_goldmine_stats()
    except Exception:
        return
    if not stats.get('total'):
        return

    from rich.console import Console
    console = Console()

    print('\n' + '💎' * 36)
    print('  GOLDMINE HISTORICAL INTELLIGENCE')
    print('💎' * 36)
    print(f"  Lifetime: {stats['cashed']}/{stats['total']} cashed ({stats['strike_rate']:.1f}%) | Net: ${stats['profit']:+.2f}")
    print(f"  Avg Gap at Selection: {stats['avg_gap']:.2f}")
    if stats.get('superfecta_total'):
        print(f"  Superfecta: {stats['superfecta_hits']}/{stats['superfecta_total']} hits | Avg Payout: ${stats['avg_sf_payout']:.2f}")
    if stats.get('gap_tiers'):
        print(f"\n  {'Gap':>6} {'Strike':>8} {'Profit':>10} {'N':>5}")
        for label, t in stats['gap_tiers'].items():
            print(f"  {label:>6} {t['strike_rate']:>7.1f}% ${t['profit']:>+9.2f} {t['total']:>5}")
    if stats.get('tier_stats'):
        emoji_map = {'Diamond': '💎💎💎', 'Platinum': '💎💎', 'Gold': '💎'}
        for name, t in stats['tier_stats'].items():
            print(f"  {emoji_map.get(name,'')} {name}: {t['cashed']}/{t['total']} ({t['strike_rate']:.1f}%) ${t['profit']:+.2f}")
    print('')

def print_quick_help():
    """Prints a friendly onboarding guide for new users."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        print_func = console.print
    except ImportError:
        # Fallback to structlog for telemetry (Capability Improvement)
        sl = structlog.get_logger()
        print_func = lambda msg: sl.info(msg)

    help_text = """
    [bold yellow]Welcome to Fortuna Faucet Intelligence![/] ✨

    Hello! I'm here to help you discover "Goldmine" racing opportunities where the
    second favorite has strong odds and a significant gap from the favorite.

    [bold]Common Commands:[/]
    • [cyan]Discovery:[/]  Just run the app! It will fetch latest races and find goldmines.
    • [cyan]Monitor:[/]    Run with [green]--monitor[/] for a live-updating dashboard.
    • [cyan]Analytics:[/]  Run [green]fortuna_analytics.py[/] to see how past predictions performed.

    [bold]Useful Flags:[/]
    • [green]--status:[/]    See your database stats and application health.
    • [green]--show-log:[/]  See highlights from recent fetching and auditing.
    • [green]--region:[/]    Force a region (USA, INT, or GLOBAL).

    [italic]Predictions are saved to fortuna_report.html and summary_grid.txt[/]
    """
    if 'Console' in globals() or 'console' in locals():
        print_func(Panel(help_text, title="🚀 Quick Start Guide", border_style="yellow"))
    else:
        print_func(help_text)

async def print_recent_logs():
    """Prints recent fetch and audit highlights from the database."""
    db = FortunaDB()
    try:
        # We need to use sync connection here as it's called from main which is not in loop yet
        # Actually main_all_in_one is async and called via asyncio.run
        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row

        print("\n" + "─" * 60)
        print(" 🔍 RECENT ACTIVITY LOG ".center(60, "─"))
        print("─" * 60)

        # Recent Harvests
        cursor = conn.execute("SELECT timestamp, adapter_name, race_count, region FROM harvest_logs ORDER BY id DESC LIMIT 5")
        print("\n Latest Fetches:")
        for row in cursor.fetchall():
            ts = row['timestamp'][:16].replace('T', ' ')
            print(f"  • {ts} | {row['adapter_name']:<20} | {row['race_count']} races ({row['region']})")

        # Recent Audits
        cursor = conn.execute("SELECT audit_timestamp, venue, race_number, verdict, net_profit FROM tips WHERE audit_completed = 1 ORDER BY audit_timestamp DESC LIMIT 5")
        rows = cursor.fetchall()
        if rows:
            print("\n Latest Audits:")
            for row in rows:
                ts = row['audit_timestamp'][:16].replace('T', ' ')
                emoji = "✅" if row['verdict'] == "CASHED" else "❌"
                print(f"  • {ts} | {row['venue']:<15} R{row['race_number']} | {emoji} {row['verdict']} (${row['net_profit']:+.2f})")

        conn.close()
        print("\n" + "─" * 60 + "\n")
    except Exception as e:
        print(f"Error reading activity log: {e}")

def open_report_in_browser():
    """Opens the HTML report in the default system browser."""
    html_path = get_writable_path("fortuna_report.html")
    if html_path.exists():
        print(f"Opening {html_path} in your browser...")
        try:
            abs_path = html_path.absolute()
            if sys.platform == "win32":
                os.startfile(abs_path)
            else:
                import webbrowser
                webbrowser.open(f"file://{abs_path}")
        except Exception as e:
            print(f"Failed to open report: {e}")
    else:
        print("No report found. Run discovery first!")

try:
    from notifications import DesktopNotifier
    HAS_NOTIFICATIONS = True
except Exception:
    HAS_NOTIFICATIONS = False

try:
    from browserforge.headers import HeaderGenerator
    from browserforge.fingerprints import FingerprintGenerator
    # Smoke test: HeaderGenerator often fails if data files are missing (frozen app issue)
    _hg = HeaderGenerator()
    BROWSERFORGE_AVAILABLE = True
except Exception:
    BROWSERFORGE_AVAILABLE = False


# --- TYPE VARIABLES ---
T = TypeVar("T")
RaceT = TypeVar("RaceT", bound="Race")

# --- CONSTANTS ---
from fortuna_utils import (
    EASTERN, DATE_FORMAT, DATE_FORMAT_OLD, MAX_VALID_ODDS, MIN_VALID_ODDS,
    DEFAULT_ODDS_FALLBACK, COMMON_PLACEHOLDERS,
    VENUE_MAP, RACING_KEYWORDS, BET_TYPE_KEYWORDS, DISCIPLINE_KEYWORDS,
    clean_text, node_text, get_canonical_venue, normalize_venue_name,
    parse_odds_to_decimal, SmartOddsExtractor, is_placeholder_odds,
    is_valid_odds, scrape_available_bets, detect_discipline,
    now_eastern, to_eastern, ensure_eastern, get_places_paid,
    parse_date_string, to_storage_format, from_storage_format, STORAGE_FORMAT,
    DayPart, resolve_daypart, resolve_daypart_from_dt, get_daypart_tag
)
DEFAULT_REGION: Final[str] = "GLOBAL"

# Region-based adapter lists (Refined by Council of Superbrains Directive)
# Single-continent adapters remain in USA/INT jobs.
# Multi-continental adapters move to the GLOBAL parallel fetch job.
# AtTheRaces is duplicated into USA as per explicit request.
USA_DISCOVERY_ADAPTERS: Final[set] = {
    "Equibase",
    "TwinSpires", "RacingPostB2B", "StandardbredCanada", "AtTheRaces", "NYRABets",
    "Official_DelMar", "Official_GulfstreamPark", "Official_TampaBayDowns",
    "Official_OaklawnPark", "Official_SantaAnita", "Official_MonmouthPark",
    "Official_TheMeadowlands", "Official_YonkersRaceway", "Official_Woodbine",
    "Official_LaurelPark", "Official_Pimlico", "Official_FairGrounds",
    "Official_ParxRacing", "Official_PennNational", "Official_CharlesTown",
    "Official_Mountaineer", "Official_TurfParadise", "Official_EmeraldDowns",
    "Official_LoneStarPark", "Official_SamHouston", "Official_RemingtonPark",
    "Official_SunlandPark", "Official_ZiaPark", "Official_FingerLakes",
    "Official_Thistledown", "Official_MahoningValley", "Official_BelterraPark",
    "Official_SaratogaHarness", "Official_HoosierPark", "Official_NorthfieldPark",
    "Official_SciotoDowns", "Official_FortErie", "Official_Hastings"
}
INT_DISCOVERY_ADAPTERS: Final[set] = {
    "TAB", "BetfairDataScientist", "HKJC", "JRA",
    "Official_JRAJapan",
    "Official_Ascot", "Official_Cheltenham", "Official_Flemington"
}
OFFICIAL_DISCOVERY_ADAPTERS: Final[set] = {
    "Official_DelMar", "Official_GulfstreamPark", "Official_TampaBayDowns",
    "Official_OaklawnPark", "Official_SantaAnita", "Official_MonmouthPark",
    "Official_Woodbine", "Official_TheMeadowlands", "Official_YonkersRaceway",
    "Official_JRAJapan", "Official_LaurelPark", "Official_Pimlico",
    "Official_FairGrounds", "Official_ParxRacing", "Official_PennNational",
    "Official_CharlesTown", "Official_Mountaineer", "Official_TurfParadise",
    "Official_EmeraldDowns", "Official_LoneStarPark", "Official_SamHouston",
    "Official_RemingtonPark", "Official_SunlandPark", "Official_ZiaPark",
    "Official_FingerLakes", "Official_Thistledown", "Official_MahoningValley",
    "Official_BelterraPark", "Official_SaratogaHarness", "Official_HoosierPark",
    "Official_NorthfieldPark", "Official_SciotoDowns", "Official_FortErie",
    "Official_Hastings", "Official_Ascot", "Official_Cheltenham", "Official_Flemington",
    "Official_BHA", "Official_Cork", "Official_Dubai", "Official_TheValley",
    "Official_AjaxDowns", "Official_BataviaDowns", "Official_CanterburyPark",
    "Official_Clonmel", "Official_ColonialDowns", "Official_DownRoyal"
}
GLOBAL_DISCOVERY_ADAPTERS: Final[set] = {
    "SkyRacingWorld", "AtTheRaces", "AtTheRacesGreyhound", "RacingPost",
    "Oddschecker", "Timeform", "SportingLife", "SkySports",
    "RacingAndSports", "HKJC", "JRA", "DRF",
    "TwinSpires", "NYRABets", "RacingPostB2B" # US sources for 24/7 global coverage
} | OFFICIAL_DISCOVERY_ADAPTERS

USA_RESULTS_ADAPTERS: Final[set] = {
    "EquibaseResults",
    "SportingLifeResults",
    "StandardbredCanadaResults",
    "RacingPostUSAResults",
    "DRFResults",
    "NYRABetsResults",
}
INT_RESULTS_ADAPTERS: Final[set] = {
    "RacingPostResults", "RacingPostTote",
    "AtTheRacesResults",
    "AtTheRacesGreyhoundResults", "SportingLifeResults", "SkySportsResults",
    "RacingAndSportsResults",
    "TimeformResults"
}

# Quality-based Partitioning (JB/Council Strategy)
SOLID_DISCOVERY_ADAPTERS: Final[set] = {"TwinSpires", "SkyRacingWorld", "RacingPost"}

# Scoring Loop Infrastructure (Phase B1)
SCORING_MTP_MAX: Final[int] = 15
ODDS_REFRESH_ADAPTERS: Final[List[str]] = ["TwinSpires", "NYRABets"]
SNAPSHOT_DIR: Final[str] = "snapshots"

# DayPart regional mapping - Phase A2 fix (adding INT/GLOBAL to Q4)
DAYPART_ACTIVE_REGIONS: Final[Dict[DayPart, frozenset]] = {
    DayPart.Q1: frozenset({'INT', 'GLOBAL'}), # AU/NZ focus
    DayPart.Q2: frozenset({'INT', 'GLOBAL'}), # UK/SA focus
    DayPart.Q3: frozenset({'USA', 'INT', 'GLOBAL'}), # US/UK afternoon
    DayPart.Q4: frozenset({'USA', 'INT', 'GLOBAL'}), # US evening, UK all-weather
}

def get_daypart_discovery_adapters(daypart: DayPart) -> Set[str]:
    """Returns the set of discovery adapter names for the specified DayPart."""
    regions = DAYPART_ACTIVE_REGIONS.get(daypart, frozenset())
    adapters = set()
    if 'USA' in regions:
        adapters |= USA_DISCOVERY_ADAPTERS
    if 'INT' in regions:
        adapters |= INT_DISCOVERY_ADAPTERS
    if 'GLOBAL' in regions:
        adapters |= GLOBAL_DISCOVERY_ADAPTERS
    return adapters

def get_daypart_results_adapters(daypart: DayPart) -> Set[str]:
    """Returns the set of results adapter names for auditing."""
    # We audit results from the *previous* quarter
    # Q1 audits Q4, Q2 audits Q1, etc.
    prev_idx = (list(DayPart).index(daypart) - 1) % 4
    prev_daypart = list(DayPart)[prev_idx]
    prev_regions = DAYPART_ACTIVE_REGIONS.get(prev_daypart, frozenset())

    adapters = set()
    # Results adapters always include INT for broad coverage (Phase A1 fix)
    # (UK/IRE/AUS results are needed for catch-up regardless of quarter)
    adapters |= INT_RESULTS_ADAPTERS
    if 'USA' in prev_regions:
        adapters |= set(USA_RESULTS_ADAPTERS)
    return adapters
SOLID_RESULTS_ADAPTERS: Final[set] = {
    "StandardbredCanadaResults",
    "RacingPostResults",
    "SportingLifeResults",
    "AtTheRacesGreyhoundResults",
    "TimeformResults",
    "SkySportsResults",
    "NYRABetsResults",
}

DEFAULT_CONCURRENT_REQUESTS: Final[int] = 5
DEFAULT_REQUEST_TIMEOUT: Final[int] = 30

DEFAULT_BROWSER_HEADERS: Final[Dict[str, str]] = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Pragma": "no-cache",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

CHROME_USER_AGENT: Final[str] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
)

CHROME_SEC_CH_UA: Final[str] = (
    '"Google Chrome";v="133", "Chromium";v="133", "Not.A/Brand";v="24"'
)

MOBILE_USER_AGENT: Final[str] = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_3 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/18.3 Mobile/15E148 Safari/604.1"
)

MOBILE_SEC_CH_UA: Final[str] = (
    '"Safari";v="18", "Mobile";v="18.3"'
)

# Bet type keywords mapping (lowercase key -> display name)


# --- EXCEPTIONS ---
class FortunaException(Exception):
    """Base exception for all Fortuna-related errors."""
    pass


class ErrorCategory(Enum):
    """Categories for classifying adapter errors."""
    BOT_DETECTION = "bot_detection"
    NETWORK = "network"
    STRUCTURE_CHANGE = "structure_change"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    CONFIGURATION = "configuration"
    PARSING = "parsing"
    UNKNOWN = "unknown"


class AdapterError(FortunaException):
    """Base error for adapter-specific issues."""
    def __init__(self, adapter_name: str, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN):
        self.adapter_name = adapter_name
        self.category = category
        super().__init__(f"[{adapter_name}] {message}")


class AdapterRequestError(AdapterError):
    def __init__(self, adapter_name: str, message: str):
        super().__init__(adapter_name, message, ErrorCategory.NETWORK)


class AdapterHttpError(AdapterRequestError):
    def __init__(self, adapter_name: str, status_code: int, url: str):
        self.status_code = status_code
        self.url = url
        super().__init__(adapter_name, f"Received HTTP {status_code} from {url}")


class AdapterParsingError(AdapterError):
    def __init__(self, adapter_name: str, message: str):
        super().__init__(adapter_name, message, ErrorCategory.PARSING)


class FetchError(Exception):
    def __init__(self, message: str, response: Optional[Any] = None, category: ErrorCategory = ErrorCategory.UNKNOWN):
        super().__init__(message)
        self.response = response
        self.category = category


# --- MODELS ---
def decimal_serializer(value: Any, handler: Callable[[Any], Any]) -> Any:
    if value is None: return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return handler(value)


JsonDecimal = Annotated[Any, WrapSerializer(decimal_serializer, when_used="json")]


class FortunaBaseModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )


class OddsData(FortunaBaseModel):
    win: Optional[JsonDecimal] = None
    place: Optional[JsonDecimal] = None
    source: str
    last_updated: datetime = Field(default_factory=lambda: datetime.now(EASTERN))

    @field_validator("last_updated", mode="after")
    @classmethod
    def validate_eastern(cls, v: datetime) -> datetime:
        return ensure_eastern(v)


def create_odds_data(source: str, win_odds: Optional[float]) -> Optional[OddsData]:
    """Helper to create an OddsData object for a given source and win odds."""
    if win_odds is None:
        return None
    try:
        return OddsData(source=source, win=Decimal(str(win_odds)))
    except Exception:
        return None


class Runner(FortunaBaseModel):
    id: Optional[str] = None
    name: str
    number: Optional[int] = Field(None, alias="saddleClothNumber")
    scratched: bool = False
    odds: Dict[str, OddsData] = Field(default_factory=dict)
    win_odds: Optional[float] = Field(None, alias="winOdds")
    odds_source: Optional[str] = Field(None, description="How win_odds was obtained: 'extracted', 'smart_extractor', 'default', or the source adapter name")
    trainer: Optional[str] = None
    jockey: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", mode="before")
    @classmethod
    def clean_name(cls, v: Any) -> str:
        if not v:
            return "Unknown"
        name = str(v).strip()
        # Handle non-breaking spaces
        name = name.replace('\xa0', ' ')
        # Remove country suffixes in parentheses, e.g., "Jay Bee (IRE)" -> "Jay Bee"
        name = re.sub(r"\s*\([^)]*\)\s*$", "", name)
        # Remove leading numbers followed by a dot and space, e.g., "1. Horse" -> "Horse"
        name = re.sub(r"^\d+\.\s*", "", name)
        # Remove unwanted punctuation/marks that might break parsing or Excel
        # Keep letters, numbers, spaces, hyphens, and apostrophes.
        name = re.sub(r"[^a-zA-Z0-9\s\-\'\\\"]", "", name)
        # Collapse multiple spaces
        name = re.sub(r"\s+", " ", name)
        # Remove explicit "Favorite" or "2nd Favorite" prefixes if they leaked from scraper labels
        name = re.sub(r"^(?:1st|2nd)?\s*Fav(?:orite)?\s*", "", name, flags=re.I)
        return name.strip() or "Unknown"


class Race(FortunaBaseModel):
    id: str
    venue: str
    race_number: int = Field(..., alias="raceNumber", ge=1, le=100)
    start_time: datetime = Field(..., alias="startTime")
    runners: List[Runner] = Field(default_factory=list)
    race_type: Optional[str] = None
    is_handicap: Optional[bool] = None

    @field_validator("venue", mode="after")
    @classmethod
    def normalize_venue(cls, v: str) -> str:
        """Ensure venue is normalized through VENUE_MAP."""
        if not v or v == "Unknown":
            return v
        normalized = normalize_venue_name(v)
        return normalized if normalized != "Unknown" else v

    @field_validator("start_time", mode="after")
    @classmethod
    def validate_eastern(cls, v: datetime) -> datetime:
        """Ensures all race start times are in US Eastern Time."""
        return ensure_eastern(v)

    source: str
    discipline: str = "Thoroughbred"
    surface: Optional[str] = None
    distance: Optional[str] = None
    field_size: Optional[int] = None
    available_bets: List[str] = Field(default_factory=list, alias="availableBets")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    qualification_score: Optional[float] = None
    is_error_placeholder: bool = False
    top_five_numbers: Optional[str] = None
    error_message: Optional[str] = None

# --- UTILITIES ---
async def fetch_json(url: str, *, client: httpx.AsyncClient, adapter_name: str, **kwargs) -> dict:
    """Centralized helper for fetching JSON with strict status validation (Hardening Fix)."""
    response = await client.get(url, **kwargs)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        structlog.get_logger('adapter').error(
            'adapter_http_error', adapter=adapter_name, url=url,
            status=exc.response.status_code, body_snippet=exc.response.text[:200]
        )
        raise
    return response.json()


class ScoringMetadata(TypedDict, total=False):
    """Strongly typed scoring metadata (IMP-CR-02)."""
    composite_score: float
    qualification_grade: str  # A+, A, B+, D, B+ (Override)
    is_goldmine: bool
    is_best_bet: bool
    is_superfecta_key: bool
    tip_tier: Literal['best_bet', 'you_might_like']
    selection_number: Optional[int]
    selection_name: str
    predicted_fav_odds: float
    predicted_2nd_fav_odds: float
    gap_abs: float  # rename from 'gap12'
    place_prob: float
    predicted_ev: float
    market_depth: float
    condition_modifier: float
    superfecta_key_number: Optional[int]
    superfecta_key_name: Optional[str]
    superfecta_box_numbers: Optional[List[str]]
    superfecta_box_valid: bool
    superfecta_scratched_runners: List[str]
    goldmine_tier: str
    goldmine_sources: List[str]
    goldmine_gap_confirmed: bool
    success_tier: str
    goldmine_confidence: str

def build_track_categories(races: List[Any]) -> Dict[str, str]:
    """Shared utility to build track categories from a list of races (IMP-CR-03)."""
    races_by_track = defaultdict(list)
    for r in races:
        v = get_field(r, 'venue')
        track = normalize_venue_name(v)
        races_by_track[track].append(r)
    return {track: get_track_category(tr) for track, tr in races_by_track.items()}

def get_field(obj: Any, field_name: str, default: Any = None) -> Any:
    """Helper to get a field from either an object or a dictionary."""
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    return getattr(obj, field_name, default)


def _safe_int(text: str, default: int = 0) -> int:
    """Extract leading digits from text, return *default* on failure."""
    if not text: return default
    cleaned = re.sub(r"\D", "", str(text))
    try:
        return int(cleaned) if cleaned else default
    except ValueError:
        return default










def generate_race_id(
    prefix: str,
    venue: str,
    start_time: datetime,
    race_number: int,
    discipline: Optional[str] = None,
) -> str:
    venue_slug = get_canonical_venue(venue)

    # Defense: warn on suspiciously long venue slugs (likely race title contamination)
    if len(venue_slug) > 25:
        _log = structlog.get_logger("generate_race_id")
        _log.warning(
            "suspiciously_long_venue_slug",
            raw_venue=venue,
            slug=venue_slug,
            prefix=prefix,
        )
        # Attempt recovery: try first word only
        first_word = venue.split()[0] if venue else venue
        recovered = get_canonical_venue(first_word)
        if recovered != "unknown":
            venue_slug = recovered

    date_str = start_time.strftime(DATE_FORMAT)
    time_str = start_time.strftime("%H%M")

    # Discipline suffix: _t (thoroughbred) | _h (harness) | _g (greyhound) | _q (quarter)
    dl = (discipline or "Thoroughbred").lower()
    if "harness" in dl:
        disc_suffix = "_h"
    elif "greyhound" in dl:
        disc_suffix = "_g"
    elif "quarter" in dl:
        disc_suffix = "_q"
    else:
        disc_suffix = "_t"

    return f"{prefix}_{venue_slug}_{date_str}_{time_str}_R{race_number}{disc_suffix}"


def get_scorable_races(
    cached_races: List[Race],
    already_scored: Optional[Set[str]] = None,
) -> List[Race]:
    """Filter to races within SCORING_MTP_MAX minutes of post that haven't been scored yet."""
    now = now_eastern()
    already = already_scored or set()
    scorable = []
    for race in cached_races:
        if race.id in already:
            continue
        mtp_seconds = (race.start_time - now).total_seconds()
        mtp_minutes = mtp_seconds / 60
        # This is the ONLY timing gate in the scoring path.
        if 0 < mtp_minutes <= SCORING_MTP_MAX:
            scorable.append(race)
    return scorable


class FortunaJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle Fortuna-specific types (BUG-CR-07)."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, datetime):
            return to_storage_format(obj)
        if isinstance(obj, set):
            return sorted(list(obj))
        # Handle Pydantic models if they somehow slip into the leaf nodes
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        return super().default(obj)

def save_quarter_snapshot(
    daypart_tag: str,
    races: List[Race],
    output_dir: str = SNAPSHOT_DIR,
) -> str:
    """Save structural race data for later scoring runs."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{daypart_tag}_races.json"
    filepath = os.path.join(output_dir, filename)
    tmp_path = filepath + ".tmp"

    # Serialize races — reuse existing race serialization pattern
    race_dicts = []
    for race in races:
        rd = race.model_dump(mode="python")
        # Convert datetimes to STORAGE_FORMAT strings
        if isinstance(rd.get("start_time"), datetime):
            rd["start_time"] = to_storage_format(rd["start_time"])
        # Convert runner odds datetimes
        for runner in rd.get("runners", []):
            if not isinstance(runner.get("odds"), dict):
                continue
            for odds_data in runner.get("odds", {}).values():
                if isinstance(odds_data.get("last_updated"), datetime):
                    odds_data["last_updated"] = to_storage_format(odds_data["last_updated"])
        race_dicts.append(rd)

    with open(tmp_path, "w") as f:
        json.dump(race_dicts, f, cls=FortunaJSONEncoder)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, filepath)

    logger = structlog.get_logger("save_quarter_snapshot")
    logger.info("quarter_snapshot_saved",
        daypart_tag=daypart_tag, race_count=len(races), path=filepath)
    return filepath


def load_quarter_snapshot(
    daypart_tag: str,
    output_dir: str = SNAPSHOT_DIR,
) -> Optional[List[Race]]:
    """Load structural race data from a quarter snapshot."""
    filepath = os.path.join(output_dir, f"{daypart_tag}_races.json")
    if not os.path.exists(filepath):
        return None

    logger = structlog.get_logger("load_quarter_snapshot")
    try:
        with open(filepath) as f:
            race_dicts = json.load(f)

        races = []
        for rd in race_dicts:
            # Parse start_time back to datetime
            if isinstance(rd.get("start_time"), str):
                rd["start_time"] = from_storage_format(rd["start_time"])
            # Parse runner odds datetimes
            for runner in rd.get("runners", []):
                if not isinstance(runner.get("odds"), dict):
                    continue
                for odds_data in runner.get("odds", {}).values():
                    if isinstance(odds_data.get("last_updated"), str):
                        odds_data["last_updated"] = from_storage_format(odds_data["last_updated"])
            races.append(Race(**rd))

        logger.info("quarter_snapshot_loaded",
            daypart_tag=daypart_tag, race_count=len(races))
        return races
    except Exception as e:
        logger.error("quarter_snapshot_load_failed",
            daypart_tag=daypart_tag, error=str(e))
        return None


def _find_adapter_class(source_name: str) -> Optional[Type[BaseAdapterV3]]:
    """Find adapter class by SOURCE_NAME."""
    def get_all_subclasses(cls):
        return set(cls.__subclasses__()).union(
            [s for c in cls.__subclasses__() for s in get_all_subclasses(c)]
        )

    for cls in get_all_subclasses(BaseAdapterV3):
        if getattr(cls, "SOURCE_NAME", None) == source_name:
            return cls
    return None


async def refresh_odds_for_races(
    scorable_races: List[Race],
    config: Dict,
) -> List[Race]:
    """Fetch fresh odds from fast adapters and merge into scorable races."""
    if not scorable_races:
        return scorable_races

    logger = structlog.get_logger("refresh_odds")
    date_str = now_eastern().strftime(DATE_FORMAT)

    # Build lookup for matching: (canonical_venue, race_number, date) -> Race
    race_lookup: Dict[tuple, Race] = {}
    for race in scorable_races:
        date_component = race.start_time.strftime(DATE_FORMAT) if isinstance(race.start_time, datetime) else ''
        key = (get_canonical_venue(race.venue), race.race_number, date_component)
        race_lookup[key] = race

    # Fetch from HTTPX-only adapters
    fresh_count = 0
    for adapter_name in ODDS_REFRESH_ADAPTERS:
        adapter_cls = _find_adapter_class(adapter_name)
        if not adapter_cls:
            continue
        adapter = adapter_cls(config=config)
        try:
            fresh_races = await adapter.get_races(date_str)
            for fresh_race in fresh_races:
                fresh_date = fresh_race.start_time.strftime(DATE_FORMAT) if isinstance(fresh_race.start_time, datetime) else ''
                key = (get_canonical_venue(fresh_race.venue), fresh_race.race_number, fresh_date)
                target = race_lookup.get(key)
                if not target:
                    continue
                # Merge runners: match by number, then by name
                for new_runner in fresh_race.runners:
                    matched = None
                    for existing_runner in target.runners:
                        if (new_runner.number and existing_runner.number
                                and new_runner.number == existing_runner.number):
                            matched = existing_runner
                            break
                        if new_runner.name.lower() == existing_runner.name.lower():
                            matched = existing_runner
                            break
                    if matched:
                        # DO NOT CLEAR: merge to preserve multi-source depth (Hardening Fix)
                        matched.odds.update(new_runner.odds)
                        if new_runner.win_odds is not None:
                            matched.win_odds = new_runner.win_odds
                        fresh_count += 1
        except Exception as e:
            logger.warning("odds_refresh_failed",
                adapter=adapter_name, error=str(e))
        finally:
            try:
                await adapter.close()
            except Exception:
                pass

    logger.info("odds_refresh_complete",
        adapters_used=len(ODDS_REFRESH_ADAPTERS),
        runners_updated=fresh_count,
        races_targeted=len(scorable_races))

    # Validate superfecta box runners are not scratched
    for race in scorable_races:
        box_nums = race.metadata.get('superfecta_box_numbers', [])
        if not box_nums:
            continue
        scratched = [str(r.number) for r in race.runners if str(r.number) in box_nums and r.scratched]
        race.metadata['superfecta_box_valid'] = len(scratched) == 0
        if scratched:
            race.metadata['superfecta_scratched_runners'] = scratched
            logger.warning(
                'superfecta_box_runner_scratched',
                venue=race.venue, race=race.race_number, scratched=scratched
            )

    return scorable_races


# --- VALIDATORS ---
class RaceValidator(BaseModel):
    venue: str = Field(..., min_length=1)
    race_number: int = Field(..., ge=1, le=100)
    start_time: datetime
    runners: List[Runner] = Field(..., min_length=2)


class DataValidationPipeline:
    @staticmethod
    def validate_raw_response(adapter_name: str, raw_data: Any) -> tuple[bool, str]:
        if raw_data is None: return False, "Null response"
        return True, "OK"
    @staticmethod
    def validate_parsed_races(races: List[Race], adapter_name: str = "Unknown") -> tuple[List[Race], List[str]]:
        valid_races: List[Race] = []
        warnings: List[str] = []
        for i, race in enumerate(races):
            try:
                data = race.model_dump() if hasattr(race, "model_dump") else race.dict()
                RaceValidator(**data)
                valid_races.append(race)
            except Exception as e:
                err_msg = f"[{adapter_name}] Race {i} ({getattr(race, 'venue', 'Unknown')} R{getattr(race, 'race_number', '?')}) validation failed: {str(e)}"
                warnings.append(err_msg)
                structlog.get_logger().error("race_validation_failed", adapter=adapter_name, error=str(e), race_index=i, venue=getattr(race, 'venue', 'Unknown'))
                continue
        return valid_races, warnings


# --- CORE INFRASTRUCTURE ---
@dataclass
class RateLimiter:
    requests_per_second: float = 10.0
    _tokens: float = field(default=10.0, init=False)
    _last_update: float = field(default_factory=time.time, init=False)
    _locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = field(default_factory=weakref.WeakKeyDictionary, init=False)
    _lock_sentinel: ClassVar[threading.Lock] = threading.Lock()

    def __post_init__(self):
        self._tokens = self.requests_per_second

    def _get_lock(self) -> asyncio.Lock:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.Lock()

        if loop not in self._locks:
            with self._lock_sentinel:
                if loop not in self._locks:
                    self._locks[loop] = asyncio.Lock()
        return self._locks[loop]

    async def acquire(self) -> None:
        lock = self._get_lock()

        for _ in range(1000): # Iteration limit to prevent potential hangs
            wait_time = 0
            async with lock:
                now = time.time()
                elapsed = now - self._last_update
                self._tokens = min(self.requests_per_second, self._tokens + (elapsed * self.requests_per_second))
                self._last_update = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                wait_time = (1 - self._tokens) / self.requests_per_second

            if wait_time >= 0:
                await asyncio.sleep(max(wait_time, 0.01))


class GlobalResourceManager:
    """Manages shared resources like HTTP clients and semaphores."""
    _clients: ClassVar[weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient]] = weakref.WeakKeyDictionary()
    _semaphores: ClassVar[weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]] = weakref.WeakKeyDictionary()
    _playwright_semaphores: ClassVar[weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]] = weakref.WeakKeyDictionary()
    _locks: ClassVar[weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]] = weakref.WeakKeyDictionary()
    _host_limiters: ClassVar[Dict[str, RateLimiter]] = {}
    _lock_initialized: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    async def get_host_limiter(cls, host: str) -> RateLimiter:
        """Returns a per-host rate limiter."""
        if host not in cls._host_limiters:
            with cls._lock_initialized:
                if host not in cls._host_limiters:
                    # Default to 2 requests per second per host to avoid 429s (Fix 13)
                    limit = 2.0
                    if "racingpost" in host: limit = 1.5 # Extra conservative for RP
                    cls._host_limiters[host] = RateLimiter(requests_per_second=limit)
        return cls._host_limiters[host]

    @classmethod
    async def _get_lock(cls) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if loop not in cls._locks:
            with cls._lock_initialized:
                if loop not in cls._locks:
                    cls._locks[loop] = asyncio.Lock()
        return cls._locks[loop]

    @classmethod
    async def get_httpx_client(cls, timeout: Optional[int] = None) -> httpx.AsyncClient:
        """
        Returns a shared httpx client for the current event loop.
        If timeout is provided and differs from current client, the client is recreated.
        """
        loop = asyncio.get_running_loop()
        lock = await cls._get_lock()
        async with lock:
            client = cls._clients.get(loop)
            if client is not None:
                # Guard against None in timeout comparison
                current_timeout = getattr(client.timeout, "read", None)
                if timeout is not None and current_timeout is not None and abs(current_timeout - timeout) > 0.001:
                    try:
                        await client.aclose()
                    except Exception:
                        pass
                    client = None

            if client is None:
                use_timeout = timeout or DEFAULT_REQUEST_TIMEOUT
                client = httpx.AsyncClient(
                    follow_redirects=True,
                    timeout=httpx.Timeout(use_timeout),
                    headers={**DEFAULT_BROWSER_HEADERS, "User-Agent": CHROME_USER_AGENT},
                    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
                )
                cls._clients[loop] = client
        return client

    @classmethod
    def get_global_semaphore(cls) -> asyncio.Semaphore:
        """Returns a shared semaphore for the current event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # If called outside a loop, we create a temporary semaphore
            return asyncio.Semaphore(DEFAULT_CONCURRENT_REQUESTS * 2)

        if loop not in cls._semaphores:
            with cls._lock_initialized:
                if loop not in cls._semaphores:
                    cls._semaphores[loop] = asyncio.Semaphore(DEFAULT_CONCURRENT_REQUESTS * 2)
        return cls._semaphores[loop]

    @classmethod
    def get_playwright_semaphore(cls) -> asyncio.Semaphore:
        """Returns a shared playwright semaphore for the current event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.Semaphore(1) # More conservative when outside loop

        if loop not in cls._playwright_semaphores:
            with cls._lock_initialized:
                if loop not in cls._playwright_semaphores:
                    # Capability Improvement: Increased to 4 for better throughput while maintaining stability
                    cls._playwright_semaphores[loop] = asyncio.Semaphore(4)
        return cls._playwright_semaphores[loop]

    @classmethod
    async def cleanup(cls):
        """Closes all clients for all event loops."""
        clients_to_close = []
        with cls._lock_initialized:
            clients_to_close = list(cls._clients.values())
            cls._clients.clear()
            cls._semaphores.clear()
            cls._playwright_semaphores.clear()
            cls._locks.clear()

        for client in clients_to_close:
            try:
                await client.aclose()
            except (AttributeError, RuntimeError):
                pass


class BrowserEngine(Enum):
    CAMOUFOX = "camoufox"
    PLAYWRIGHT = "playwright"
    CURL_CFFI = "curl_cffi"
    PLAYWRIGHT_LEGACY = "playwright_legacy"
    HTTPX = "httpx"


@dataclass
class UnifiedResponse:
    """Unified response object to normalize data across different fetch engines."""
    text: str
    status: int
    status_code: int
    url: str
    headers: Dict[str, str] = field(default_factory=dict)

    def json(self) -> Any:
        return json.loads(self.text)


class FetchStrategy(FortunaBaseModel):
    primary_engine: BrowserEngine = BrowserEngine.PLAYWRIGHT
    allowed_engines: List[BrowserEngine] = Field(
        default_factory=lambda: [
            BrowserEngine.CAMOUFOX, BrowserEngine.CURL_CFFI,
            BrowserEngine.PLAYWRIGHT, BrowserEngine.HTTPX
        ]
    )
    max_engine_attempts: int = Field(3, ge=1, le=5)
    enable_js: bool = True
    stealth_mode: str = "fast"
    block_resources: bool = False
    max_retries: int = Field(3, ge=0, le=10)
    timeout: int = Field(DEFAULT_REQUEST_TIMEOUT, ge=1, le=300)
    page_load_strategy: str = "domcontentloaded"
    wait_until: Optional[str] = None
    network_idle: bool = False
    wait_for_selector: Optional[str] = None
    impersonate: Optional[str] = "chrome124"  # BUG-D5: chrome133 not supported by curl_cffi


def api_fetch_strategy(**overrides) -> FetchStrategy:
    """For adapters hitting JSON APIs. No browser needed."""
    defaults = dict(
        primary_engine=BrowserEngine.CURL_CFFI,
        allowed_engines=[BrowserEngine.CURL_CFFI, BrowserEngine.HTTPX],
        max_engine_attempts=2,
        enable_js=False,
        timeout=30,
        max_retries=2,
    )
    defaults.update(overrides)
    return FetchStrategy(**defaults)


def scraping_fetch_strategy(**overrides) -> FetchStrategy:
    """For adapters that scrape rendered HTML behind anti-bot protection."""
    defaults = dict(
        primary_engine=BrowserEngine.CAMOUFOX,
        allowed_engines=[
            BrowserEngine.CAMOUFOX, BrowserEngine.PLAYWRIGHT,
            BrowserEngine.CURL_CFFI,
        ],
        max_engine_attempts=3,
        enable_js=True,
        stealth_mode="camouflage",
        timeout=45,
        max_retries=2,
    )
    defaults.update(overrides)
    return FetchStrategy(**defaults)


def lightweight_fetch_strategy(**overrides) -> FetchStrategy:
    """For health checks and simple GETs. HTTPX only, fast timeout."""
    defaults = dict(
        primary_engine=BrowserEngine.HTTPX,
        allowed_engines=[BrowserEngine.HTTPX],
        max_engine_attempts=1,
        enable_js=False,
        timeout=15,
        max_retries=1,
    )
    defaults.update(overrides)
    return FetchStrategy(**defaults)


class GlobalEngineHealthRegistry:
    """Shared state for tracking engine health across all fetcher instances."""
    _health_scores: Dict[BrowserEngine, float] = {
        BrowserEngine.CAMOUFOX: 0.9,
        BrowserEngine.CURL_CFFI: 0.8,
        BrowserEngine.PLAYWRIGHT: 0.7,
        BrowserEngine.PLAYWRIGHT_LEGACY: 0.6,
        BrowserEngine.HTTPX: 0.5
    }
    _last_decay_time: float = time.time()
    _lock = threading.Lock()

    @classmethod
    def get_scores(cls) -> Dict[BrowserEngine, float]:
        with cls._lock:
            # Automatic recovery over time
            now = time.time()
            elapsed = now - cls._last_decay_time
            if elapsed > 60: # Every minute, recover slightly
                for engine in cls._health_scores:
                    cls._health_scores[engine] = min(1.0, cls._health_scores[engine] + 0.2)
                cls._last_decay_time = now
            return cls._health_scores.copy()

    @classmethod
    def record_failure(cls, engine: BrowserEngine, penalty: float = 0.2):
        with cls._lock:
            cls._health_scores[engine] = max(0.0, cls._health_scores[engine] - penalty)

    @classmethod
    def record_success(cls, engine: BrowserEngine, reward: float = 0.05):
        with cls._lock:
            cls._health_scores[engine] = min(1.0, cls._health_scores[engine] + reward)

class SmartFetcher:
    BOT_DETECTION_KEYWORDS: ClassVar[List[str]] = [
        "datadome", "perimeterx", "access denied", "captcha", "cloudflare",
        "please verify", "client challenge", "javascript is disabled",
        "just a moment", "enable cookies", "sucuri", "unsupported browser"
    ]
    def __init__(self, strategy: Optional[FetchStrategy] = None):
        self.strategy = strategy or FetchStrategy()
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.last_engine: str = "unknown"
        self._sessions: Dict[Union[BrowserEngine, str], Any] = {}
        self._curl_sessions: Dict[str, Any] = {} # Hardening Fix: Initialize session tracking (Fix AttributeError)
        self._session_lock = asyncio.Lock()
        if BROWSERFORGE_AVAILABLE:
            self.header_gen = HeaderGenerator()
            self.fingerprint_gen = FingerprintGenerator()
        else:
            self.header_gen = None
            self.fingerprint_gen = None

    async def _get_persistent_session(self, engine: BrowserEngine) -> Any:
        """Gate ALL browser creation behind the global playwright semaphore."""
        if os.getenv('FORTUNA_NO_BROWSER') == '1':
            return None

        # Fast path: session already exists
        async with self._session_lock:
            if engine in self._sessions:
                return self._sessions[engine]

        # Slow path: acquire global browser slot before creating
        pw_sem = GlobalResourceManager.get_playwright_semaphore()
        async with pw_sem:
            async with self._session_lock:
                # Double-check after acquiring semaphore
                if engine in self._sessions:
                    return self._sessions[engine]

                try:
                    if engine == BrowserEngine.CAMOUFOX and ASYNC_SESSIONS_AVAILABLE:
                        session = AsyncStealthySession(headless=True)
                        await asyncio.wait_for(session.__aenter__(), timeout=30)
                        self._sessions[engine] = session
                    elif engine in (BrowserEngine.PLAYWRIGHT, BrowserEngine.PLAYWRIGHT_LEGACY) \
                         and ASYNC_SESSIONS_AVAILABLE:
                        session = AsyncDynamicSession(headless=True)
                        await asyncio.wait_for(session.__aenter__(), timeout=30)
                        self._sessions[engine] = session
                except (BrokenPipeError, ConnectionError, ConnectionResetError, OSError, asyncio.TimeoutError) as e:
                    self.logger.error('browser_session_creation_failed',
                                     engine=engine.value, error=str(e))
                    GlobalEngineHealthRegistry.record_failure(engine, penalty=0.5)
                    raise FetchError(f'Browser launch failed: {e}',
                                     category=ErrorCategory.NETWORK)

                return self._sessions.get(engine)

    async def _invalidate_session(self, engine: BrowserEngine) -> None:
        """Safely tear down a dead browser session."""
        async with self._session_lock:
            session = self._sessions.pop(engine, None)
            if session is not None:
                try:
                    await asyncio.wait_for(session.__aexit__(None, None, None), timeout=10)
                except Exception:
                    pass  # Session is already dead

    async def _get_curl_session(self) -> Any:
        """Returns a persistent curl_cffi session (Fix 07 / BUG-CR-06)."""
        async with self._session_lock:
            if "_curl" not in self._sessions:
                if not curl_requests:
                    raise ImportError("curl_cffi is not available")
                self._sessions["_curl"] = curl_requests.AsyncSession()
            return self._sessions["_curl"]

    async def fetch(self, url: str, **kwargs: Any) -> Any:
        method = kwargs.get("method", "GET").upper()
        kwargs.pop("url", None)

        # Get latest health scores from global registry
        current_health = GlobalEngineHealthRegistry.get_scores()
        available_engines = list(current_health.keys())

        if not curl_requests and BrowserEngine.CURL_CFFI in available_engines:
            available_engines.remove(BrowserEngine.CURL_CFFI)
        if not ASYNC_SESSIONS_AVAILABLE:
            for e in [BrowserEngine.CAMOUFOX, BrowserEngine.PLAYWRIGHT]:
                if e in available_engines: available_engines.remove(e)

        if not available_engines:
            self.logger.error("no_fetch_engines_available", url=url)
            raise FetchError("No fetch engines available (install curl_cffi or scrapling)")

        strategy = kwargs.get("strategy", self.strategy)

        # Domain-specific engine prioritization (Hardening Fix / P0-FIX-B2)
        # Some domains are better handled by specific engines; re-rank locally without mutation
        protected_domains = [
            "attheraces.com", "equibase.com", "nyrabets.com", "oddschecker.com",
            "skyracingworld.com", "cnty.com", "hollywoodmahoningvalley.com",
            "hastingsracecourse.com", "mgmresorts.com", "saratogacasino.com",
            "britishhorseracing.com", "clonmelraces.ie", "ajaxdowns.com",
            "batavia-downs.com", "standardbredcanada.ca", "fanduel.com", "twinspires.com"
        ]
        if any(d in url for d in protected_domains):
            self.logger.debug("Prioritizing browser engines for protected domain", url=url)
            if BrowserEngine.CAMOUFOX in available_engines:
                current_health[BrowserEngine.CAMOUFOX] = max(current_health[BrowserEngine.CAMOUFOX], 1.0)
            if BrowserEngine.PLAYWRIGHT in available_engines:
                current_health[BrowserEngine.PLAYWRIGHT] = max(current_health[BrowserEngine.PLAYWRIGHT], 0.95)
            # Discourage HTTPX and CURL_CFFI locally for these domains to prefer browser automation
            if BrowserEngine.HTTPX in available_engines:
                current_health[BrowserEngine.HTTPX] = 0.05
            if BrowserEngine.CURL_CFFI in available_engines:
                current_health[BrowserEngine.CURL_CFFI] = min(current_health[BrowserEngine.CURL_CFFI], 0.1)

        candidates = [
            eng for eng, score in sorted(
                current_health.items(), key=lambda x: -x[1]
            )
            if eng in strategy.allowed_engines and eng in available_engines and score > 0.05
        ]

        # Primary preference is already expressed through initial health values,
        # so we stay sorted by health score to handle degraded engines.

        # Primary strategy engine should always be tried if allowed
        if strategy.primary_engine in candidates:
            # Move primary to the front
            candidates.remove(strategy.primary_engine)
            candidates.insert(0, strategy.primary_engine)

        engines = candidates[:strategy.max_engine_attempts]
        if not engines:
            raise FetchError(f"No viable engines for {url}", category=ErrorCategory.NETWORK)

        self.logger.debug("Fetch engines ordered", url=url, engines=[e.value for e in engines], primary=strategy.primary_engine.value)
        last_error: Optional[Exception] = None
        for engine in engines:
            try:
                # Hardening Fix: Create a shallow copy of kwargs to prevent multiple values for 'method' during recursion
                fetch_kwargs = kwargs.copy()
                fetch_kwargs["method"] = method
                response = await self._fetch_with_engine(engine, url, **fetch_kwargs)

                # Check for bot detection in response body and status code (Hardening Fix / BUG-CR-05)
                if response and hasattr(response, "status_code"):
                    sc = response.status_code
                    if sc == 429:
                        self.logger.warning("rate_limited", engine=engine.value, url=url)
                        # Penalty for 429 is higher to force engine switch or backoff
                        GlobalEngineHealthRegistry.record_failure(engine, penalty=0.4)
                        raise FetchError("Rate limited (429)", response=response, category=ErrorCategory.RATE_LIMIT)
                    if sc in (403, 503):
                        # Don't raise yet, check body for specific challenge keywords first (P0 Resilience)
                        self.logger.warning("http_block_status", engine=engine.value, status=sc, url=url)

                if response and hasattr(response, "text") and response.text:
                    body_lower = response.text.lower()
                    challenge_solved = False
                    for kw in self.BOT_DETECTION_KEYWORDS:
                        if kw in body_lower:
                            self.logger.warning("bot_challenge_detected", engine=engine.value, keyword=kw, url=url)
                            # Hardening Fix: If using Playwright, wait and retry once to allow automated solving
                            if engine in (BrowserEngine.PLAYWRIGHT, BrowserEngine.CAMOUFOX):
                                self.logger.info("Waiting for automated challenge solution...", engine=engine.value)
                                # Increased wait to 30s to handle complex Cloudflare/Datadome challenges (P0 Improvement)
                                await asyncio.sleep(30)
                                # Hardening Fix: Use copy for internal wait-retry too
                                retry_kwargs = kwargs.copy()
                                retry_kwargs["method"] = method
                                # Ensure we use the same browser engine for retry to leverage solving
                                response = await self._fetch_with_engine(engine, url, **retry_kwargs)
                                if response and hasattr(response, "text") and not any(k in response.text.lower() for k in self.BOT_DETECTION_KEYWORDS):
                                    self.logger.info("Challenge solved successfully!", engine=engine.value)
                                    challenge_solved = True
                                    break # Success in the kw loop

                            if not challenge_solved:
                                # Invalidate poisoned session
                                if engine in (BrowserEngine.PLAYWRIGHT, BrowserEngine.CAMOUFOX):
                                    await self._invalidate_session(engine)
                                raise FetchError(f"Bot challenge detected ({kw})", response=response, category=ErrorCategory.BOT_DETECTION)

                # Final check: if status code was 403/503 and we didn't find a keyword or solve it, we must fail
                if response and hasattr(response, "status_code") and response.status_code in (403, 503):
                    GlobalEngineHealthRegistry.record_failure(engine, penalty=0.3)
                    raise FetchError(f"HTTP {response.status_code}", response=response, category=ErrorCategory.BOT_DETECTION)

                GlobalEngineHealthRegistry.record_success(engine)
                self.last_engine = engine.value
                return response
            except Exception as e:
                self.logger.debug(f"Engine {engine.value} failed", error=str(e))
                GlobalEngineHealthRegistry.record_failure(engine)
                last_error = e
                continue
        err_msg = repr(last_error) if last_error else "All fetch engines failed"
        self.logger.error("all_engines_failed", url=url, error=err_msg)
        raise last_error or FetchError("All fetch engines failed")

    
    async def _fetch_with_engine(self, engine: BrowserEngine, url: str, **kwargs: Any) -> Any:
        method = kwargs.pop("method", "GET").upper()
        # Generate browserforge headers if available
        if BROWSERFORGE_AVAILABLE:
            try:
                # Generate headers and a corresponding user agent
                fingerprint = self.fingerprint_gen.generate()
                bf_headers = self.header_gen.generate()
                # Ensure User-Agent is consistent between fingerprint and headers
                ua = getattr(fingerprint.navigator, 'userAgent', getattr(fingerprint.navigator, 'user_agent', CHROME_USER_AGENT))
                bf_headers['User-Agent'] = ua

                # Copy headers before mutation to avoid leaking state across requests
                headers = dict(kwargs.get("headers", {}))
                # Merge - browserforge headers complement provided ones
                for k, v in bf_headers.items():
                    if k not in headers:
                        headers[k] = v
                kwargs["headers"] = headers
                self.logger.debug("Applied browserforge headers", engine=engine.value)
            except Exception as e:
                self.logger.warning("Failed to generate browserforge headers", error=str(e))

        # Define browser-specific arguments to strip for non-browser engines
        BROWSER_SPECIFIC_KWARGS = [
            "network_idle", "wait_selector", "wait_until", "impersonate",
            "stealth", "block_resources", "wait_for_selector", "stealth_mode",
            "strategy", "update_status", "follow_redirects", "allow_redirects"
        ]

        # Extract redirect settings for normalization
        follow_redirects = kwargs.get("follow_redirects", True)
        allow_redirects = kwargs.get("allow_redirects", follow_redirects)

        strategy = kwargs.get("strategy", self.strategy)
        if engine == BrowserEngine.HTTPX:
            # Pass strategy timeout if present in kwargs or use default
            timeout = kwargs.get("timeout", strategy.timeout)
            client = await GlobalResourceManager.get_httpx_client(timeout=timeout)

            # Remove timeout and browser-specific keys from kwargs
            req_kwargs = {
                k: v for k, v in kwargs.items()
                if k != "timeout" and k not in BROWSER_SPECIFIC_KWARGS
            }
            req_kwargs["follow_redirects"] = allow_redirects
            resp = await client.request(method, url, timeout=timeout, **req_kwargs)
            return UnifiedResponse(resp.text, resp.status_code, resp.status_code, str(resp.url), resp.headers)
        
        if engine == BrowserEngine.CURL_CFFI:
            if not curl_requests:
                raise ImportError("curl_cffi is not available")
            
            self.logger.debug(f"Using curl_cffi for {url}")
            timeout = kwargs.get("timeout", strategy.timeout)

            # Default headers if still not present after browserforge attempt
            headers = kwargs.get("headers", {**DEFAULT_BROWSER_HEADERS, "User-Agent": CHROME_USER_AGENT})

            # BUG-14: Impersonation fallback chain to handle unsupported versions
            requested_impersonate = kwargs.get("impersonate") or getattr(strategy, "impersonate", None) or "chrome124"
            impersonate_chain = [requested_impersonate, "chrome124", "chrome120", "chrome110"]
            # Filter out duplicates while preserving order
            impersonate_chain = list(dict.fromkeys(impersonate_chain))
            
            # Remove keys that curl_requests.AsyncSession.request doesn't like
            clean_kwargs = {
                k: v for k, v in kwargs.items()
                if k not in ["timeout", "headers", "impersonate"] + BROWSER_SPECIFIC_KWARGS
            }
            clean_kwargs["allow_redirects"] = allow_redirects
            
            last_err = None
            session = await self._get_curl_session()
            for imp_version in impersonate_chain:
                try:
                    resp = await session.request(
                        method,
                        url,
                        timeout=timeout,
                        headers=headers,
                        impersonate=imp_version,
                        **clean_kwargs
                    )
                    return UnifiedResponse(resp.text, resp.status_code, resp.status_code, resp.url, resp.headers)
                except Exception as e:
                    err_lower = str(e).lower()
                    if ("impersonat" in err_lower or "supported" in err_lower) and "chrome" in err_lower:
                        self.logger.debug("curl_cffi impersonation not supported, trying next", version=imp_version)
                        # Discard the poisoned session from the main cache (Hardening Fix)
                        async with self._session_lock:
                            self._sessions.pop('_curl', None)
                        last_err = e
                        continue
                    raise

            raise last_err or FetchError(f"All curl_cffi impersonations failed for {url}")

        if not ASYNC_SESSIONS_AVAILABLE:
            raise ImportError("scrapling not available")

        # 1. Broaden supported kwargs explicitly!
        SCRAPLING_KWARGS = [
            "network_idle", "wait_selector", "wait_until", "stealth_mode",
            "block_resources", "timeout", "headers", "extra_headers", "proxy", "data", "json", "params",
            "follow_redirects", "allow_redirects"
        ]

        scrapling_kwargs = {k: v for k, v in kwargs.items() if k in SCRAPLING_KWARGS}

        # Enforce essential mapping fallback for custom header names & HTTP formats.
        if "headers" in kwargs and "headers" not in scrapling_kwargs:
            scrapling_kwargs["headers"] = kwargs["headers"]

        timeout_val = scrapling_kwargs.get("timeout") or kwargs.get("timeout") or strategy.timeout
        is_browser = engine in (BrowserEngine.CAMOUFOX, BrowserEngine.PLAYWRIGHT)
        # Browsers process times typically in miliseconds inside underlying libs (Fix timeout units)
        if is_browser and timeout_val < 1000:
            scrapling_kwargs["timeout"] = int(timeout_val * 1000)
        else:
            scrapling_kwargs["timeout"] = timeout_val

        if "wait_until" not in scrapling_kwargs:
            scrapling_kwargs["wait_until"] = strategy.wait_until or strategy.page_load_strategy
        if "network_idle" not in scrapling_kwargs:
            scrapling_kwargs["network_idle"] = strategy.network_idle
        if "stealth_mode" not in scrapling_kwargs:
            scrapling_kwargs["stealth_mode"] = strategy.stealth_mode
        if "block_resources" not in scrapling_kwargs:
            scrapling_kwargs["block_resources"] = strategy.block_resources

        # Helper method: Safely and completely unpack Adapters regardless of structure
        def _get_unified_resp(r) -> UnifiedResponse:
            st = getattr(r, "status", getattr(r, "status_code", 200))
            url_str = str(getattr(r, "url", url))
            hdrs = getattr(r, "headers", getattr(r, "response_headers", getattr(r, "extra_headers", {})))

            # Smart html fallback block avoiding bytes => string mutation corruption (`b"<html>..."` literal representation string failures)
            body = getattr(r, "html_content", getattr(r, "body", None))
            if isinstance(body, (bytes, bytearray)):
                encoding = getattr(r, "encoding", "utf-8") or "utf-8"
                cont = body.decode(encoding, errors="replace")
            elif isinstance(body, str) and body:
                cont = body
            else:
                alt = getattr(r, "raw_html", getattr(r, "html", getattr(r, "text", "")))
                if isinstance(alt, (bytes, bytearray)):
                    cont = alt.decode("utf-8", errors="replace")
                else:
                    cont = str(alt)

            # Map Unified responses effectively
            return UnifiedResponse(cont, st, st, url_str, dict(hdrs))

        if engine in (BrowserEngine.CAMOUFOX, BrowserEngine.PLAYWRIGHT):
            try:
                s = await self._get_persistent_session(engine)
                if s is None:
                    raise FetchError(f"No session for {engine.value}")
                if method.upper() == "POST":
                    # Ensure the POST configuration navigations execute
                    action = getattr(s, "post", s.fetch)
                    # Mapping data/json to body for scrapling
                    if "data" in scrapling_kwargs:
                        scrapling_kwargs["body"] = scrapling_kwargs.pop("data")
                    if "json" in scrapling_kwargs:
                        scrapling_kwargs["body"] = json.dumps(scrapling_kwargs.pop("json"))
                    resp = await action(url, **scrapling_kwargs)
                else:
                    # Direct fetching or mapping
                    action = getattr(s, "get", getattr(s, "fetch"))
                    try:
                        # Map params to url for scrapling if needed
                        if "params" in scrapling_kwargs:
                            from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl
                            params = scrapling_kwargs.pop("params")
                            url_parts = list(urlparse(url))
                            query = dict(parse_qsl(url_parts[4]))
                            query.update(params)
                            url_parts[4] = urlencode(query)
                            url = urlunparse(url_parts)

                        resp = await action(url, **scrapling_kwargs)
                    except TypeError as te:
                        # In instances older version underlying calls mandate missing standard default keywords safely back out
                        if ("method" in str(te) or "redirect" in str(te)) and getattr(s, "fetch", None) == action:
                            # Strip incompatible kwargs and retry
                            for k in ["follow_redirects", "allow_redirects", "method"]:
                                scrapling_kwargs.pop(k, None)
                            resp = await action(url, method="GET", **scrapling_kwargs)
                        else:
                            raise te

                return _get_unified_resp(resp)
            except (BrokenPipeError, ConnectionResetError, OSError, asyncio.TimeoutError) as e:
                self.logger.warning("browser_pipe_error", engine=engine.value, url=url, error=str(e))
                await self._invalidate_session(engine)
                # Penalize browser engine for timeout or pipe error
                GlobalEngineHealthRegistry.record_failure(engine, penalty=0.4 if isinstance(e, asyncio.TimeoutError) else 0.5)
                raise FetchError(f"Browser error: {e}", category=ErrorCategory.NETWORK)

        elif engine == BrowserEngine.PLAYWRIGHT_LEGACY:
            try:
                # Direct Playwright usage for cases where scrapling/camoufox fail
                from playwright.async_api import async_playwright
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True)
                    # Apply impersonation via context
                    ua = kwargs.get("headers", {}).get("User-Agent", CHROME_USER_AGENT)
                    context = await browser.new_context(user_agent=ua)
                    page = await context.new_page()

                    timeout = kwargs.get("timeout", strategy.timeout) * 1000
                    wait_until = "networkidle" if strategy.network_idle else "domcontentloaded"

                    # Apply headers
                    if "headers" in kwargs:
                        await context.set_extra_http_headers(kwargs["headers"])

                    resp_obj = await page.goto(url, wait_until=wait_until, timeout=timeout)
                    content = await page.content()
                    status = resp_obj.status if resp_obj else 0
                    headers = resp_obj.headers if resp_obj else {}

                    await browser.close()
                    return UnifiedResponse(content, status, status, url, headers)
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                self.logger.warning("browser_pipe_error", engine=engine.value, url=url, error=str(e))
                GlobalEngineHealthRegistry.record_failure(engine, penalty=0.5)
                raise FetchError(f"Browser process died: {e}", category=ErrorCategory.NETWORK)

        else:
            try:
                # Fallback bare Fetcher block without the specific session configuration elements
                async with AsyncFetcher() as fetcher:
                    allowed_http = {"timeout", "headers", "extra_headers", "proxy", "data", "json", "params"}
                    safe_fetch_kwargs = {k: v for k, v in scrapling_kwargs.items() if k in allowed_http}

                    if method.upper() == "GET":
                        resp = await fetcher.get(url, **safe_fetch_kwargs)
                    else:
                        action = getattr(fetcher, "post", getattr(fetcher, "request"))
                        resp = await action(url, **safe_fetch_kwargs)

                    return _get_unified_resp(resp)
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                self.logger.warning("browser_pipe_error", engine=engine.value, url=url, error=str(e))
                GlobalEngineHealthRegistry.record_failure(engine, penalty=0.5)
                raise FetchError(f"Browser process died: {e}", category=ErrorCategory.NETWORK)


    async def close(self) -> None:
        """Tear down all sessions, tolerating dead browser processes."""
        async with self._session_lock:
            for engine, session in list(self._sessions.items()):
                try:
                    if engine == "_curl":
                        await session.close()
                    else:
                        await asyncio.wait_for(
                            session.__aexit__(None, None, None), timeout=5
                        )
                except Exception:
                    pass  # Browser already dead
            self._sessions.clear()

            for key, session in list(self._curl_sessions.items()):
                try:
                    await session.close()
                except Exception:
                    pass
            self._curl_sessions.clear()


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    state: str = "closed"
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    async def record_success(self) -> None:
        self.failure_count = 0
        self.state = "closed"
    async def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold: self.state = "open"
    async def allow_request(self) -> bool:
        if self.state == "closed": return True
        if self.state == "open" and self.last_failure_time:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                return True
        return self.state == "half-open"


class AdapterMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency_ms = 0.0
        self.consecutive_failures = 0
        self.last_failure_reason: Optional[str] = None
        self.parse_warnings = 0
        self.parse_errors = 0

    @property
    def success_rate(self) -> float:
        return self.successful_requests / self.total_requests if self.total_requests > 0 else 1.0

    async def record_success(self, latency_ms: float) -> None:
        with self._lock:
            self.total_requests += 1
            self.successful_requests += 1
            self.total_latency_ms += latency_ms
            self.consecutive_failures = 0
            self.last_failure_reason = None

    async def record_failure(self, error: str) -> None:
        with self._lock:
            self.total_requests += 1
            self.failed_requests += 1
            self.consecutive_failures += 1
            self.last_failure_reason = error

    def record_parse_warning(self) -> None:
        with self._lock:
            self.parse_warnings += 1

    def record_parse_error(self) -> None:
        with self._lock:
            self.parse_errors += 1

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "success_rate": self.success_rate,
            "failed_requests": self.failed_requests,
            "consecutive_failures": self.consecutive_failures,
            "last_failure_reason": getattr(self, "last_failure_reason", None),
            "parse_warnings": self.parse_warnings,
            "parse_errors": self.parse_errors
        }


# --- MIXINS ---
class JSONParsingMixin:
    """Mixin for safe JSON extraction from HTML and scripts."""
    def _parse_json_from_script(self, parser: HTMLParser, selector: str, context: str = "script") -> Optional[Any]:
        script = parser.css_first(selector)
        if not script:
            return None
        try:
            return json.loads(node_text(script))
        except json.JSONDecodeError as e:
            if hasattr(self, 'logger'):
                self.logger.error("failed_parsing_json", context=context, selector=selector, error=str(e))
            return None

    def _parse_json_from_attribute(self, parser: HTMLParser, selector: str, attribute: str, context: str = "attribute") -> Optional[Any]:
        el = parser.css_first(selector)
        if not el:
            return None
        raw = el.attributes.get(attribute)
        if not raw:
            return None
        try:
            return json.loads(html.unescape(raw))
        except json.JSONDecodeError as e:
            if hasattr(self, 'logger'):
                self.logger.error("failed_parsing_json", context=context, selector=selector, attribute=attribute, error=str(e))
            return None

    def _parse_all_jsons_from_scripts(self, parser: HTMLParser, selector: str, context: str = "scripts") -> List[Any]:
        results = []
        for script in parser.css(selector):
            try:
                results.append(json.loads(node_text(script)))
            except json.JSONDecodeError as e:
                if hasattr(self, 'logger'):
                    self.logger.error("failed_parsing_json_in_list", context=context, selector=selector, error=str(e))
        return results


class BrowserHeadersMixin:
    def _get_browser_headers(self, host: Optional[str] = None, referer: Optional[str] = None, **extra: str) -> Dict[str, str]:
        is_mobile = getattr(self, "config", {}).get("mobile", False)
        ua = MOBILE_USER_AGENT if is_mobile else CHROME_USER_AGENT
        sec_ua = MOBILE_SEC_CH_UA if is_mobile else CHROME_SEC_CH_UA
        mob = "?1" if is_mobile else "?0"
        plat = '"iOS"' if is_mobile else '"Windows"'

        h = {
            **DEFAULT_BROWSER_HEADERS,
            "User-Agent": ua,
            "sec-ch-ua": sec_ua,
            "sec-ch-ua-mobile": mob,
            "sec-ch-ua-platform": plat
        }
        if host: h["Host"] = host
        if referer: h["Referer"] = referer
        h.update(extra)
        return h


class DebugMixin:
    def _save_debug_snapshot(self, content: str, context: str, url: Optional[str] = None) -> None:
        if not content or not os.getenv("DEBUG_SNAPSHOTS"): return
        try:
            d = get_writable_path("debug_snapshots")
            d.mkdir(parents=True, exist_ok=True)
            f = d / f"{context}_{datetime.now(EASTERN).strftime('%y%m%d_%H%M%S')}.html"
            with open(f, "w", encoding="utf-8") as out:
                if url: out.write(f"<!-- URL: {url} -->\n")
                out.write(content)
        except Exception: pass
    def _save_debug_html(self, content: str, filename: str, **kwargs) -> None:
        self._save_debug_snapshot(content, filename)


class RacePageFetcherMixin:
    async def _fetch_race_pages_concurrent(self, metadata: List[Dict[str, Any]], headers: Dict[str, str], semaphore_limit: int = 5, delay_range: tuple[float, float] = (0.5, 1.5), **kwargs: Any) -> List[Dict[str, Any]]:
        local_sem = asyncio.Semaphore(semaphore_limit)
        async def fetch_single(item):
            url = item.get("url")
            if not url: return None

            async with local_sem:
                    # Stagger requests by sleeping inside the semaphore (Project Convention)
                    await asyncio.sleep(delay_range[0] + random.random() * (delay_range[1] - delay_range[0]))
                    try:
                        if hasattr(self, 'logger'):
                            self.logger.debug("fetching_race_page", url=url)
                        # make_request handles global_sem internally
                        resp = None
                        for attempt in range(2): # 1 retry
                            resp = await self.make_request("GET", url, headers=headers, **kwargs)
                            # Lowered threshold to 100 to avoid unnecessary retries for small valid data files
                            if resp and hasattr(resp, "text") and resp.text and len(resp.text) > 100:
                                break
                            await asyncio.sleep(1 * (attempt + 1))

                        if resp and hasattr(resp, "text") and resp.text:
                            if hasattr(self, 'logger'):
                                self.logger.debug("fetched_race_page", url=url, status=getattr(resp, 'status', 'unknown'))
                            return {**item, "html": resp.text}
                        elif resp:
                            if hasattr(self, 'logger'):
                                self.logger.warning("failed_fetching_race_page_unexpected_status", url=url, status=getattr(resp, 'status', 'unknown'))
                    except Exception as e:
                        if hasattr(self, 'logger'):
                            self.logger.error("failed_fetching_race_page", url=url, error=str(e))
                    return None
        tasks = [fetch_single(m) for m in metadata]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception) and r is not None]


# --- BASE ADAPTER ---
class BaseAdapterV3(ABC):
    ADAPTER_TYPE: ClassVar[str] = "discovery"
    # Default to False to ensure races with partial odds data are analyzed
    PROVIDES_ODDS: ClassVar[bool] = False

    def __init__(self, source_name: str, base_url: str, rate_limit: float = 10.0, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        self.source_name = source_name
        self.base_url = base_url.rstrip("/")
        self.config = config or {}
        # Merge kwargs into config
        self.config.update(kwargs)
        self.headers: Dict[str, str] = {}
        self.trust_ratio = 0.0 # Tracking odds quality ratio (0.0 to 1.0)

        # Override rate_limit from config if present
        actual_rate_limit = float(self.config.get("rate_limit", rate_limit))

        self.logger = structlog.get_logger(adapter_name=self.source_name)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=int(self.config.get("failure_threshold", 5)),
            recovery_timeout=float(self.config.get("recovery_timeout", 60.0))
        )
        self.rate_limiter = RateLimiter(requests_per_second=actual_rate_limit)
        self.metrics = AdapterMetrics()
        self.smart_fetcher = SmartFetcher(strategy=self._configure_fetch_strategy())
        self.last_race_count = 0
        self.last_duration_s = 0.0
        self.last_response_status: Optional[Union[int, str]] = None

    @abstractmethod
    def _configure_fetch_strategy(self) -> FetchStrategy: pass
    @abstractmethod
    async def _fetch_data(self, date: str) -> Optional[Any]: pass
    @abstractmethod
    def _parse_races(self, raw_data: Any) -> List[Race]: pass

    async def get_races(self, date: str) -> List[Race]:
        start = time.time()
        try:
            # Check for browser requirement in monolith mode
            strategy = self.smart_fetcher.strategy
            if strategy.primary_engine in [BrowserEngine.PLAYWRIGHT, BrowserEngine.CAMOUFOX]:
                if is_frozen():
                    self.logger.info("Skipping browser-dependent adapter in monolith mode")
                    return []
                # FIX_06: Gracefully skip if Browser support is required but missing (GHA check)
                if not ASYNC_SESSIONS_AVAILABLE:
                    self.logger.warning("Browser support (scrapling/patchright) not available, skipping browser-based adapter", source=self.source_name)
                    return []

            if not await self.circuit_breaker.allow_request(): return []

            raw = None
            # Hardening Fix: Implement retries with strict status validation
            for attempt in range(3):
                try:
                    await self.rate_limiter.acquire()
                    raw = await self._fetch_data(date)
                    if raw:
                        break
                except Exception as e:
                    if attempt == 2:
                        raise
                    self.logger.warning("Fetch attempt failed, retrying", attempt=attempt+1, error=str(e))
                    await asyncio.sleep(1)

            if not raw:
                await self.circuit_breaker.record_failure()
                return []
            races = self._validate_and_parse_races(raw)
            self.last_race_count = len(races)
            self.last_duration_s = time.time() - start
            await self.circuit_breaker.record_success()
            await self.metrics.record_success(self.last_duration_s * 1000)
            return races
        except Exception as e:
            self.logger.error("Adapter failed", error=str(e))
            await self.circuit_breaker.record_failure()
            await self.metrics.record_failure(str(e))
            return []

    def _validate_and_parse_races(self, raw_data: Any) -> List[Race]:
        races = self._parse_races(raw_data)

        # BUG-D1: Drop races with impossible race numbers before any other processing.
        # Real meetings rarely exceed 15 races. Numbers like R19, R37 indicate
        # the adapter scraped a card index or meeting-level counter, not the real race number.
        MAX_RACE_NUM_TB = 15   # Thoroughbred / Harness
        MAX_RACE_NUM_GH = 14   # Greyhound (some UK cards run 13-14)
        filtered_races: List[Race] = []
        for r in races:
            disc_lower = (r.discipline or 'thoroughbred').lower()
            limit = MAX_RACE_NUM_GH if 'greyhound' in disc_lower else MAX_RACE_NUM_TB
            if r.race_number > limit:
                self.logger.warning(
                    'impossible_race_number_dropped',
                    venue=r.venue, race_number=r.race_number, limit=limit,
                    discipline=r.discipline, source=self.source_name,
                )
                continue
            filtered_races.append(r)
        races = filtered_races

        total_runners = 0
        trustworthy_runners = 0

        # Propagate adapter capability flag to race metadata
        for r in races:
            r.metadata["provides_odds"] = self.PROVIDES_ODDS
            r.source = self.source_name # Phase 2: Ensure source is always set (PIPE-5 Fix)

        for r in races:
            # BUG-D7: Infer race_type from venue name, race metadata, or available text.
            if not r.race_type:
                # Check metadata for any race_type hints
                meta_text = ' '.join(str(v) for v in r.metadata.values() if isinstance(v, str))
                combined_text = f"{r.venue} {meta_text}".upper()
                rt_match = re.search(
                    r'(MAIDEN|CLAIMING|ALLOWANCE|GRADED\s+STAKES|STAKES|HANDICAP|'
                    r'NOVICE|GROUP\s+\d|GRADE\s+\d|LISTED|HURDLE|CHASE|BUMPER|'
                    r'NATIONAL\s+HUNT|STEEPLECHASE)',
                    combined_text, re.I
                )
                if rt_match:
                    r.race_type = rt_match.group(1).title()

            # Also infer is_handicap if not already set
            if r.is_handicap is None:
                rt_upper = (r.race_type or '').upper()
                if any(kw in rt_upper for kw in ['HANDICAP', "H'CAP", 'HCAP']):
                    r.is_handicap = True

            # Global heuristic for runner numbers (addressing "impossible" high numbers)
            active_runners = [run for run in r.runners if not run.scratched]
            field_size = len(active_runners)

            # If any runner has a number > 20 and it's also > field_size + 10 (buffer)
            # or if it's extremely high (> 100), re-index everything as it's likely a parsing error (horse IDs).
            # Also re-index if all numbers are missing/zero.
            suspicious = all(run.number == 0 or run.number is None for run in r.runners)
            if not suspicious:
                for run in r.runners:
                    if run.number:
                        if run.number > 100 or (run.number > 20 and run.number > field_size + 10):
                            suspicious = True
                            break

            if suspicious:
                self.logger.warning(
                    "suspicious_runner_numbers",
                    venue=r.venue,
                    field_size=field_size,
                    original_numbers=[run.number for run in r.runners[:5]],
                )
                for i, run in enumerate(r.runners):
                    # Preserve original number in metadata for audit debugging
                    run.metadata['original_number'] = run.number
                    # Only reindex active (non-scratched) runners
                    if not run.scratched:
                        run.number = i + 1
                    else:
                        # Scratched runners keep their original number (won't affect scoring)
                        pass

            for runner in r.runners:
                if not runner.scratched:
                    # Explicitly enrich win_odds using all available sources (including fallbacks)
                    best = _get_best_win_odds(runner)
                    # Untrustworthy odds should be flagged
                    is_trustworthy = best is not None
                    runner.metadata["odds_source_trustworthy"] = is_trustworthy
                    if best:
                        runner.win_odds = float(best)
                        trustworthy_runners += 1
                    else:
                        # Clear invalid or missing odds to maintain hygiene
                        runner.win_odds = None
                    total_runners += 1

        if total_runners > 0:
            self.trust_ratio = round(trustworthy_runners / total_runners, 2)
            self.logger.info("adapter_odds_quality", ratio=self.trust_ratio, source=self.source_name)

        # FIX_03: Duplicate race data detection (content fingerprinting)
        deduped_races = []
        fingerprints = {}
        for r in races:
            active = [(run.name, str(run.win_odds)) for run in r.runners if not run.scratched]
            fp = (r.venue, frozenset(active))
            if fp in fingerprints:
                fingerprints[fp] += 1
                if fingerprints[fp] >= 3:
                    self.logger.warning("Duplicate race content detected at venue, skipping", venue=r.venue, race=r.race_number)
                    continue
            else:
                fingerprints[fp] = 1
            deduped_races.append(r)

        valid, warnings = DataValidationPipeline.validate_parsed_races(deduped_races, adapter_name=self.source_name)
        return valid

    async def make_request(self, method: str, url: str, **kwargs: Any) -> Any:
        full_url = url if url.startswith("http") else f"{self.base_url}/{url.lstrip('/')}"
        raise_for_status = kwargs.pop("raise_for_status", True)
        update_status = kwargs.pop("update_status", True)

        # Capability Improvement: Interactive / Manual Ingest Mode (Phase A1)
        # Check if manual_fetch directory has a file matching this URL slug
        from urllib.parse import urlparse
        p = urlparse(full_url)
        domain = p.netloc.lower()
        # Include query string in slug to differentiate indices (Fix-Manual-Ingest)
        raw_slug = domain + p.path + (f"?{p.query}" if p.query else "")
        slug = re.sub(r'[^a-z0-9]', '_', raw_slug.lower()).strip('_')
        if len(slug) > 180: slug = slug[:180]
        filepath = Path("manual_fetch") / f"{slug}.html"
        if filepath.exists():
            content = filepath.read_text(encoding="utf-8", errors="replace")
            # Log hit for audit transparency
            self.logger.info("manual_fetch_hit", file=str(filepath))
            if update_status: self.last_response_status = 200
            return UnifiedResponse(text=content, status=200, status_code=200, url=full_url)

        # Apply host-based rate limiting to prevent 429s (Fix 13)
        host = urlparse(full_url).netloc
        if host:
            limiter = await GlobalResourceManager.get_host_limiter(host)
            await limiter.acquire()

        self.logger.debug("Requesting", method=method, url=full_url)

        # Merge adapter-level headers if defined
        if hasattr(self, 'headers') and self.headers:
            current_headers = kwargs.get("headers", {})
            # Passed headers take precedence over adapter defaults
            merged_headers = {**self.headers, **current_headers}
            kwargs["headers"] = merged_headers

        # Apply global concurrency limit
        async with GlobalResourceManager.get_global_semaphore():
            try:
                # Use adapter-specific strategy
                kwargs.setdefault("strategy", self.smart_fetcher.strategy)
                resp = await self.smart_fetcher.fetch(full_url, method=method, **kwargs)
                status = get_resp_status(resp)
                if update_status:
                    self.last_response_status = status
                self.logger.debug("Response received", method=method, url=full_url, status=status)

                # Hardening Fix: Raise for status if not 200
                if raise_for_status and status != 200:
                    self.logger.error("adapter_http_error", adapter=self.source_name, url=full_url, status=status)
                    if hasattr(resp, "raise_for_status"):
                        try:
                            resp.raise_for_status()
                        except Exception:
                            raise httpx.HTTPStatusError(f"HTTP {status}", request=None, response=resp)
                    else:
                        raise Exception(f"HTTP {status}")

                return resp
            except Exception as e:
                self.logger.error("Request failed", method=method, url=full_url, error=str(e))
                raise

    async def close(self) -> None: await self.smart_fetcher.close()
    async def shutdown(self) -> None: await self.close()

# ============================================================================
# ADAPTER IMPLEMENTATIONS
# ============================================================================

# ----------------------------------------
# EquibaseAdapter
# ----------------------------------------
class HKJCAdapter(JSONParsingMixin, BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
    """
    Adapter for Hong Kong Jockey Club (HKJC).
    Extremely reliable data source for Hong Kong racing.
    """
    SOURCE_NAME: ClassVar[str] = "HKJC"
    BASE_URL: ClassVar[str] = "https://racing.hkjc.com"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        # Capability Improvement: Upgrade to scraping strategy to bypass persistent 403s on aggregate discovery
        return scraping_fetch_strategy()

    def _get_headers(self) -> Dict[str, str]:
        return self._get_browser_headers(host="racing.hkjc.com")

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        # date is YYMMDD, HKJC results/entries often use YYYY/MM/DD
        dt = parse_date_string(date)
        date_hk = dt.strftime("%Y/%m/%d")

        # FIX-CR-HKJC: Establish session first
        try:
            await self.make_request("GET", "/racing/information/English/racing/RaceCard.aspx", headers=self._get_headers(), raise_for_status=False)
        except Exception: pass

        # Try RaceCard first (Discovery)
        url = f"/racing/information/English/racing/RaceCard.aspx?RaceDate={date_hk}"
        resp = await self.make_request("GET", url, headers=self._get_headers(), follow_redirects=True)

        if not resp or not resp.text or "Information will be released shortly" in resp.text:
            # Try Results page if RaceCard is not available (maybe it just finished)
            url = f"/racing/information/English/Racing/LocalResults.aspx?RaceDate={date_hk}"
            resp = await self.make_request("GET", url, headers=self._get_headers())

        if not resp or not resp.text:
            return None

        self._save_debug_snapshot(resp.text, f"hkjc_index_{date}")
        parser = HTMLParser(resp.text)

        # If still no info, try the general entries page
        if "Information will be released shortly" in resp.text:
            entries_url = "/racing/information/English/racing/Entries.aspx"
            resp = await self.make_request("GET", entries_url, headers=self._get_headers())
            if not resp or not resp.text:
                return None
            parser = HTMLParser(resp.text)

        # Find race links
        # HKJC uses specific icons or text for race numbers
        metadata = []
        # Case-insensitive attribute match for RaceNo (Fix 16)
        for a in parser.css("a"):
            href = a.attributes.get("href", "")
            if "RaceNo=" in href or "raceno=" in href:
                metadata.append({"url": href})

        if not metadata:
            # Maybe it's a single race page or all-races page
            if "Race Card" in resp.text:
                return {"html": resp.text, "url": url, "date": date}
            return None

        # Fetch all races
        pages = await self._fetch_race_pages_concurrent(metadata, self._get_headers())
        return {"pages": pages, "date": date}

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data: return []
        races = []
        date_str = raw_data["date"]
        try:
            race_date = parse_date_string(date_str).date()
        except Exception:
            race_date = datetime.now(EASTERN).date()

        if "pages" in raw_data:
            for p in raw_data["pages"]:
                if p and p.get("html"):
                    race = self._parse_single_race(p["html"], p.get("url", ""), race_date)
                    if race: races.append(race)
        elif "html" in raw_data:
            race = self._parse_single_race(raw_data["html"], raw_data.get("url", ""), race_date)
            if race: races.append(race)

        return races

    def _parse_single_race(self, html_content: str, url: str, race_date: date) -> Optional[Race]:
        parser = HTMLParser(html_content)

        # Venue is usually Sha Tin or Happy Valley
        venue = "Hong Kong"
        if "Sha Tin" in html_content: venue = "Sha Tin"
        elif "Happy Valley" in html_content: venue = "Happy Valley"

        # Race number
        race_num = 1
        num_match = re.search(r"RaceNo=(\d+)", url)
        if num_match:
            race_num = int(num_match.group(1))
        else:
            # Try to find in text "Race 1"
            txt_match = re.search(r"Race\s+(\d+)", html_content, re.I)
            if txt_match: race_num = int(txt_match.group(1))

        # Runners
        runners = []
        # HKJC uses a table with class 'performance'
        for row in parser.css("table.performance tr"):
            cols = row.css("td")
            if len(cols) < 5: continue

            # Saddle cloth number
            try:
                num = int(clean_text(node_text(cols[0])))
            except Exception: continue

            # Horse Name
            name_node = cols[2].css_first("a")
            name = clean_text(node_text(name_node or cols[2]))
            if not name or name.upper() in ["HORSE", "NAME"]: continue

            # Odds
            win_odds = None
            # HKJC odds are usually in a specific column or can be found in text
            # For now, we'll use SmartOddsExtractor as HKJC layout is complex
            win_odds = SmartOddsExtractor.extract_from_node(row)

            odds_data = {}
            if ov := create_odds_data(self.SOURCE_NAME, win_odds):
                odds_data[self.SOURCE_NAME] = ov

            runners.append(Runner(name=name, number=num, odds=odds_data, win_odds=win_odds))

        if not runners: return None

        # Start time - HKJC usually lists it
        start_time = datetime.combine(race_date, datetime.min.time())
        time_match = re.search(r"(\d{1,2}:\d{2})", html_content)
        if time_match:
            try:
                start_time = datetime.combine(race_date, datetime.strptime(time_match.group(1), "%H:%M").time())
            except Exception: pass

        return Race(
            id=generate_race_id("hkjc", venue, start_time, race_num),
            venue=venue,
            race_number=race_num,
            start_time=ensure_eastern(start_time),
            runners=runners,
            source=self.SOURCE_NAME,
            discipline="Thoroughbred"
        )

class OfficialTrackAdapter(BaseAdapterV3):
    """
    Adapter that verifies the availability of an official racetrack website.
    Supports a '200 OK' health check as requested by JB.
    """
    IS_HEALTH_CHECK_ONLY: ClassVar[bool] = True
    ADAPTER_TYPE = "discovery"
    PROVIDES_ODDS = False
    DISCIPLINE = "Thoroughbred"

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return FetchStrategy(
            primary_engine=BrowserEngine.HTTPX,
            allowed_engines=[BrowserEngine.HTTPX],
            max_engine_attempts=1,
            max_retries=1,
            timeout=15,
            enable_js=False,
        )

    def __init__(self, track_name: str, url: str, config: Optional[Dict[str, Any]] = None):
        self.track_name = track_name
        self.official_url = url
        # Use a safe name for the source
        source = f"Official_{track_name.replace(' ', '').replace('/', '')}"
        super().__init__(source_name=source, base_url=url, config=config)

    async def _fetch_data(self, date: str) -> Optional[str]:
        # Perform a GET to check status
        try:
            # Hardening Fix: Defensive check to strip trailing slash if URL ends in .html or .php
            url_to_fetch = self.official_url
            if any(url_to_fetch.lower().endswith(ext) for ext in [".html", ".htm", ".php", ".aspx"]):
                url_to_fetch = url_to_fetch.rstrip("/")

            resp = await self.make_request("GET", url_to_fetch)
            if resp and get_resp_status(resp) == 200:
                return "ALIVE"
        except Exception as e:
            if hasattr(e, "response") and e.response:
                self.last_response_status = get_resp_status(e.response)
            self.logger.debug("Health check request failed", error=str(e))
        return None

    def _parse_races(self, raw_data: Any) -> List[Race]:
        # Return empty list; health check success is tracked via BaseAdapterV3 metrics/logs
        # (Fix GEMINI_4: Remove dummy race pollution)
        return []

class OfficialDelMarAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_DelMar"
    def __init__(self, config=None): super().__init__("Del Mar", "https://www.dmtc.com/racing/entries", config=config)

class OfficialGulfstreamAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_GulfstreamPark"
    def __init__(self, config=None): super().__init__("Gulfstream Park", "https://www.gulfstreampark.com/racing/entries", config=config)

class OfficialTampaBayAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_TampaBayDowns"
    def __init__(self, config=None): super().__init__("Tampa Bay Downs", "https://www.tampabaydowns.com/betting/entries/", config=config)

class OfficialOaklawnAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_OaklawnPark"
    def __init__(self, config=None): super().__init__("Oaklawn Park", "https://www.oaklawn.com/racing/", config=config)

class OfficialSantaAnitaAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_SantaAnita"
    def __init__(self, config=None): super().__init__("Santa Anita", "https://www.santaanita.com/racing/entries", config=config)

class OfficialMonmouthAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_MonmouthPark"
    def __init__(self, config=None): super().__init__("Monmouth Park", "https://www.monmouthpark.com/racing-info/entries/", config=config)

class OfficialWoodbineAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_Woodbine"
    def __init__(self, config=None): super().__init__("Woodbine", "https://woodbine.com/horse-racing/", config=config)

class OfficialMeadowlandsAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_TheMeadowlands"
    DISCIPLINE = "Harness"
    def __init__(self, config=None): super().__init__("The Meadowlands", "https://playmeadowlands.com/", config=config)

class OfficialYonkersAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_YonkersRaceway"
    DISCIPLINE = "Harness"
    def __init__(self, config=None): super().__init__("Yonkers Raceway", "https://empirecitycasino.mgmresorts.com/en/racing.html", config=config)

class OfficialJRAAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_JRAJapan"
    def __init__(self, config=None): super().__init__("JRA Japan", "https://japanracing.jp/", config=config)

class OfficialLaurelParkAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_LaurelPark"
    def __init__(self, config=None): super().__init__("Laurel Park", "https://www.laurelpark.com/", config=config)

class OfficialPimlicoAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_Pimlico"
    def __init__(self, config=None): super().__init__("Pimlico", "https://www.pimlico.com/", config=config)

class OfficialFairGroundsAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_FairGrounds"
    def __init__(self, config=None): super().__init__("Fair Grounds", "https://www.fairgroundsracecourse.com/", config=config)

class OfficialParxRacingAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_ParxRacing"
    def __init__(self, config=None): super().__init__("Parx Racing", "https://www.parxracing.com/overnights.php", config=config)

class OfficialPennNationalAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_PennNational"
    def __init__(self, config=None): super().__init__("Penn National", "https://www.pennnational.com/racing/entries", config=config)

class OfficialCharlesTownAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_CharlesTown"
    DECOMMISSIONED: ClassVar[bool] = True  # Persistent 403
    def __init__(self, config=None): super().__init__("Charles Town", "https://www.hollywoodcasinocharlestown.com/racing/entries", config=config)

class OfficialMountaineerAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_Mountaineer"
    def __init__(self, config=None): super().__init__("Mountaineer", "https://www.cnty.com/mountaineer/racing/entries-results/", config=config)

class OfficialTurfParadiseAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_TurfParadise"
    def __init__(self, config=None): super().__init__("Turf Paradise", "https://www.turfparadise.com/racing/", config=config)

class OfficialEmeraldDownsAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_EmeraldDowns"
    def __init__(self, config=None): super().__init__("Emerald Downs", "https://emeralddowns.com/", config=config)

class OfficialLoneStarParkAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_LoneStarPark"
    def __init__(self, config=None): super().__init__("Lone Star Park", "https://www.lonestarpark.com/racing/entries/", config=config)

class OfficialSamHoustonAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_SamHouston"
    DISCIPLINE = "Harness"
    def __init__(self, config=None): super().__init__("Sam Houston", "https://www.shrp.com/", config=config)

class OfficialRemingtonParkAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_RemingtonPark"
    def __init__(self, config=None): super().__init__("Remington Park", "https://www.remingtonpark.com/", config=config)

class OfficialSunlandParkAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_SunlandPark"
    def __init__(self, config=None): super().__init__("Sunland Park", "https://www.sunlandpark.com/", config=config)

class OfficialZiaParkAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_ZiaPark"
    def __init__(self, config=None): super().__init__("Zia Park", "https://www.ziapark.com/racing/entries/", config=config)

class OfficialFingerLakesAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_FingerLakes"
    def __init__(self, config=None): super().__init__("Finger Lakes", "https://www.fingerlakesracing.com/racing/entries/", config=config)

class OfficialThistledownAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_Thistledown"
    def __init__(self, config=None): super().__init__("Thistledown", "https://www.thistledown.com/racing/entries/", config=config)

class OfficialMahoningValleyAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_MahoningValley"
    def __init__(self, config=None): super().__init__("Mahoning Valley", "https://www.hollywoodmahoningvalley.com/racing/entries-results/", config=config)

class OfficialBelterraParkAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_BelterraPark"
    DECOMMISSIONED: ClassVar[bool] = True  # Redirect loop to 404 page
    def __init__(self, config=None): super().__init__("Belterra Park", "https://www.belterrapark.com/racing/entries/", config=config)

class OfficialSaratogaHarnessAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_SaratogaHarness"
    DISCIPLINE = "Harness"
    def __init__(self, config=None): super().__init__("Saratoga Harness", "https://saratogacasino.com/racing/", config=config)

class OfficialHoosierParkAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_HoosierPark"
    DISCIPLINE = "Harness"
    def __init__(self, config=None): super().__init__("Hoosier Park", "https://www.hoosierpark.com/racing/entries/", config=config)

class OfficialNorthfieldParkAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_NorthfieldPark"
    DISCIPLINE = "Harness"
    def __init__(self, config=None): super().__init__("Northfield Park", "https://mgmnorthfieldpark.mgmresorts.com/en/racing.html", config=config)

class OfficialSciotoDownsAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_SciotoDowns"
    DISCIPLINE = "Harness"
    def __init__(self, config=None): super().__init__("Scioto Downs", "https://www.eldoradoscioto.com/racing/entries/", config=config)

class OfficialFortErieAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_FortErie"
    def __init__(self, config=None): super().__init__("Fort Erie", "https://www.forterieracing.com/", config=config)

class OfficialHastingsAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_Hastings"
    def __init__(self, config=None): super().__init__("Hastings Racecourse", "https://www.hastingsracecourse.com/racing/", config=config)

class OfficialAscotAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_Ascot"
    def __init__(self, config=None): super().__init__("Ascot", "https://www.ascot.com/", config=config)

class OfficialCheltenhamAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_Cheltenham"
    def __init__(self, config=None): super().__init__("Cheltenham", "https://www.cheltenham.co.uk/", config=config)

class OfficialFlemingtonAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_Flemington"
    def __init__(self, config=None): super().__init__("Flemington", "https://www.vrc.com.au/", config=config)

class OfficialBHAAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_BHA"
    def __init__(self, config=None): super().__init__("BHA", "https://www.britishhorseracing.com", config=config)

class OfficialCorkAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_Cork"
    def __init__(self, config=None): super().__init__("Cork", "https://www.corkracecourse.ie", config=config)

class OfficialDubaiAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_Dubai"
    def __init__(self, config=None): super().__init__("Dubai", "https://dubairacingclub.com/", config=config)

class OfficialTheValleyAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_TheValley"
    def __init__(self, config=None): super().__init__("The Valley", "https://www.thevalley.com.au", config=config)

class OfficialAjaxDownsAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_AjaxDowns"
    def __init__(self, config=None): super().__init__("Ajax Downs", "https://www.ajaxdowns.com", config=config)

class OfficialBataviaDownsAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_BataviaDowns"
    def __init__(self, config=None): super().__init__("Batavia Downs", "https://www.batavia-downs.com", config=config)

class OfficialCanterburyParkAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_CanterburyPark"
    def __init__(self, config=None): super().__init__("Canterbury Park", "https://www.canterburypark.com", config=config)

class OfficialClonmelAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_Clonmel"
    def __init__(self, config=None): super().__init__("Clonmel", "https://www.clonmelraces.ie", config=config)

class OfficialColonialDownsAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_ColonialDowns"
    def __init__(self, config=None): super().__init__("Colonial Downs", "https://www.colonialdowns.com", config=config)

class OfficialDownRoyalAdapter(OfficialTrackAdapter):
    SOURCE_NAME = "Official_DownRoyal"
    def __init__(self, config=None): super().__init__("Down Royal", "https://www.downroyal.com", config=config)

class DRFAdapter(BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
    """
    Adapter for Daily Racing Form (DRF) entries.
    Provides canonical US thoroughbred data.
    """
    SOURCE_NAME: ClassVar[str] = "DRF"
    PROVIDES_ODDS: ClassVar[bool] = True
    BASE_URL: ClassVar[str] = "https://www.drf.com"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return scraping_fetch_strategy(network_idle=True)

    def _get_headers(self) -> Dict[str, str]:
        return self._get_browser_headers(host="www.drf.com")

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        # Hardening Fix: Establish session on the home page first
        try:
            await self.make_request("GET", "https://www.drf.com/", timeout=20, raise_for_status=False, update_status=False)
            await asyncio.sleep(1)
        except Exception: pass

        url = "/race-entries"
        resp = await self.make_request("GET", url, headers=self._get_headers())
        if not resp or not resp.text: return None

        self._save_debug_snapshot(resp.text, f"drf_entries_{date}")
        return {"html": resp.text, "date": date}

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data or not raw_data.get("html"): return []
        return []

class JRAAdapter(BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
    """
    Adapter for Japan Racing Association (JRA).
    Provides high-quality data for Japanese racing.
    """
    SOURCE_NAME: ClassVar[str] = "JRA"
    BASE_URL: ClassVar[str] = "https://japanracing.jp"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        # Capability Improvement: Upgrade to scraping strategy to bypass persistent 403s on aggregate discovery
        return scraping_fetch_strategy()

    def _get_headers(self) -> Dict[str, str]:
        return self._get_browser_headers(host="japanracing.jp")

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        # JRA uses /racing/calendar/{YYYY}/{MM}/{DD}.html or similar
        dt = parse_date_string(date)
        url = f"/racing/calendar/{dt.year}/{dt.month}/{dt.day}.html"

        # Actually JRA has a simpler entries page
        # https://japanracing.jp/en/racing/go_racing/jra_racecourses/
        # For now we'll check the calendar
        resp = await self.make_request("GET", url, headers=self._get_headers())
        if not resp or not resp.text:
            # Fallback to current entries
            resp = await self.make_request("GET", "/en/racing/go_racing/", headers=self._get_headers())
            if not resp or not resp.text: return None

        self._save_debug_snapshot(resp.text, f"jra_index_{date}")
        parser = HTMLParser(resp.text)

        metadata = []
        # JRA layout is very structured. Look for race links.
        for a in parser.css("a[href*='/racing/calendar/']"):
            href = a.attributes.get("href")
            if href and "index.html" not in href:
                metadata.append({"url": href})

        if not metadata:
             return None

        pages = await self._fetch_race_pages_concurrent(metadata[:20], self._get_headers())
        return {"pages": pages, "date": date}

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data or not raw_data.get("pages"): return []
        races = []
        date_str = raw_data["date"]
        try:
            race_date = parse_date_string(date_str).date()
        except Exception:
            race_date = datetime.now(EASTERN).date()

        for p in raw_data["pages"]:
            if p and p.get("html"):
                race = self._parse_single_race(p["html"], p.get("url", ""), race_date)
                if race: races.append(race)

        return races

    def _parse_single_race(self, html_content: str, url: str, race_date: date) -> Optional[Race]:
        parser = HTMLParser(html_content)

        # Extract venue from header or URL
        venue = "Japan"
        header = parser.css_first("h1") or parser.css_first("h2")
        if header:
            venue = normalize_venue_name(node_text(header))

        # Race number
        race_num = 1
        num_match = re.search(r"race(\d+)", url)
        if num_match: race_num = int(num_match.group(1))

        # Runners
        runners = []
        for row in parser.css("table.race_table tr"):
            cols = row.css("td")
            if len(cols) < 5: continue

            try:
                num = int(clean_text(node_text(cols[0])))
                name = clean_text(node_text(cols[2]))
                if not name or name.upper() in ["HORSE", "NAME"]: continue

                win_odds = SmartOddsExtractor.extract_from_node(row)
                odds_data = {}
                if ov := create_odds_data(self.SOURCE_NAME, win_odds):
                    odds_data[self.SOURCE_NAME] = ov

                runners.append(Runner(name=name, number=num, odds=odds_data, win_odds=win_odds))
            except Exception: continue

        if not runners: return None

        # Start time
        start_time = datetime.combine(race_date, datetime.min.time())
        time_match = re.search(r"(\d{1,2}:\d{2})", html_content)
        if time_match:
            try:
                start_time = datetime.combine(race_date, datetime.strptime(time_match.group(1), "%H:%M").time())
            except Exception: pass

        return Race(
            id=generate_race_id("jra", venue, start_time, race_num),
            venue=venue,
            race_number=race_num,
            start_time=ensure_eastern(start_time),
            runners=runners,
            source=self.SOURCE_NAME,
            discipline="Thoroughbred"
        )

class RacingAndSportsAdapter(BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
    """
    Adapter for Racing & Sports (RAS).
    Note: Highly protected by Cloudflare; requires advanced impersonation.
    """
    SOURCE_NAME: ClassVar[str] = "RacingAndSports"
    PROVIDES_ODDS: ClassVar[bool] = True
    BASE_URL: ClassVar[str] = "https://www.racingandsports.com.au"
    DECOMMISSIONED: ClassVar[bool] = False

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _get_headers(self) -> Dict[str, str]:
        return self._get_browser_headers(host="www.racingandsports.com.au")

    def _configure_fetch_strategy(self) -> FetchStrategy:
        # RAS is heavily protected; use CAMOUFOX for the form-guide sweep
        return scraping_fetch_strategy(network_idle=True)

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        # Success Strategy: Always bootstrap session on the homepage
        try:
            await self.make_request("GET", "https://www.racingandsports.com.au", timeout=30, raise_for_status=False)
            await asyncio.sleep(1)
        except Exception: pass

        # 1. Primary: Use the structured JSON endpoint for speed and stability
        # Note: This often returns a minimal list of meetings with FormGuideUrls.
        url = "https://www.racingandsports.com.au/todays-racing-json-v2"
        try:
            resp = await self.make_request("GET", url, headers=self._get_headers(), raise_for_status=False)
            if resp and resp.text:
                try:
                    data = json.loads(resp.text)
                    if isinstance(data, (dict, list)) and data:
                        # Extract race links from JSON to fetch full data
                        metadata = []
                        if isinstance(data, list):
                            for disc_data in data:
                                for country_data in disc_data.get("Countries", []):
                                    for meeting in country_data.get("Meetings", []):
                                        if url := meeting.get("FormGuideUrl"):
                                            metadata.append({"url": url})
                        elif isinstance(data, dict):
                            for m in data.get("meetings", []):
                                for r in m.get("races", []):
                                    if url := r.get("formGuideUrl") or m.get("formGuideUrl"):
                                        metadata.append({"url": url})

                        if metadata:
                            metadata = list({m['url']: m for m in metadata}.values())
                            self.logger.info("ras_json_links_discovered", count=len(metadata))
                            pages = await self._fetch_race_pages_concurrent(metadata[:40], self._get_headers(), semaphore_limit=5)
                            if pages:
                                return {"pages": pages, "date": date}

                        self.logger.info("ras_json_fetch_success", meeting_count=len(data.get('meetings', [])) if isinstance(data, dict) else len(data))
                        return {"json_data": data, "date": date}
                except Exception as e:
                    self.logger.warning("ras_json_parse_failed", error=str(e))
        except Exception as e:
            self.logger.warning("ras_json_request_failed", error=str(e))

        # 2. Secondary (EXTRA CREDIT): Fetch from /form-guide as requested in AGENTS.md
        # This provides deeper coverage for Australian/NZ regions
        dt = parse_date_string(date)
        date_iso = dt.strftime("%Y-%m-%d")
        # RAS form-guide index
        fg_url = f"/form-guide/{date_iso}"

        try:
            resp = await self.make_request("GET", fg_url, headers=self._get_headers())
            if not resp or not resp.text:
                return None

            self._save_debug_snapshot(resp.text, f"ras_formguide_{date}")
            parser = HTMLParser(resp.text)

            # Discover race links in form-guide
            metadata = []
            for a in parser.css('a[href*="/form-guide/"]'):
                href = a.attributes.get("href")
                # Look for race links like /form-guide/australia/track/date/R1
                if href and re.search(r'/R\d+$', href):
                    metadata.append({"url": href})

            if metadata:
                # Deduplicate and limit to capture a broad sample
                metadata = list({m['url']: m for m in metadata}.values())
                self.logger.info("found_ras_formguide_links", count=len(metadata))
                pages = await self._fetch_race_pages_concurrent(metadata[:40], self._get_headers(), semaphore_limit=5)
                return {"pages": pages, "date": date}
        except Exception as e:
            self.logger.error("failed_fetching_ras_formguide", error=str(e))

        return None

    def _parse_races(self, raw_data: Any) -> List[Race]:
        self.logger.debug("parsing_ras_races", keys=list(raw_data.keys()) if isinstance(raw_data, dict) else "not_dict")
        if not raw_data: return []

        # Handle form-guide pages fetch result
        if "pages" in raw_data:
            races = []
            try:
                race_date = parse_date_string(raw_data["date"]).date()
            except Exception:
                race_date = now_eastern().date()

            for p in raw_data["pages"]:
                if p and p.get("html"):
                    # Extract venue/race from URL if possible
                    # /form-guide/australia/wyong/2026-03-17/R1
                    url = p.get("url", "")
                    venue = "Unknown"
                    race_num = 1
                    parts = url.rstrip("/").split("/")
                    if len(parts) >= 6:
                        venue = parts[-3] # /form-guide/country/track/date/R1 -> track
                        r_match = re.search(r'R(\d+)', parts[-1])
                        if r_match:
                            race_num = int(r_match.group(1))

                    race = self._parse_single_race(p["html"], url, race_date, normalize_venue_name(venue), race_num)
                    if race: races.append(race)
            return races

        if not raw_data.get("json_data"): return []
        data = raw_data["json_data"]
        try:
            race_date = parse_date_string(raw_data["date"]).date()
        except Exception:
            race_date = now_eastern().date()

        races = []

        # RAS JSON v2 structure: data['meetings'] -> list of meetings
        # OR data as a list: list of Disciplines -> Countries -> Meetings
        meetings = []
        if isinstance(data, dict):
            meetings = data.get('meetings', [])
        elif isinstance(data, list):
            # Complex nested structure: list of disciplines
            for disc_data in data:
                assigned_discipline = disc_data.get("DisciplineFullText")
                for country_data in disc_data.get("Countries", []):
                    country_name = country_data.get("CountryName")
                    for meeting in country_data.get("Meetings", []):
                        # Ensure meeting has discipline and country for downstream filters
                        meeting["assigned_discipline"] = assigned_discipline
                        meeting["country"] = country_name
                        meetings.append(meeting)

        self.logger.info("parsing_ras_json", meeting_count=len(meetings))
        for m in meetings:
            venue_raw = m.get('venueName') or m.get('venue') or m.get('Course')
            if not venue_raw: continue
            venue = normalize_venue_name(venue_raw)

            # Filter by region if set in config
            target_region = self.config.get('region')
            # Normalize country name for filtering (Standardize to ISO-ish or common names)
            country = (m.get('country') or m.get('countryCode') or '').upper()
            if target_region == 'USA' and country not in ['US', 'USA', 'CAN']: continue
            if target_region == 'INT' and country in ['US', 'USA', 'CAN']: continue

            # If JSON provides race list, parse them directly
            ras_races = m.get('races', [])
            if not ras_races and m.get('RaceNumber'):
                # Single race summary, often with links
                ras_races = [m]

            for r_idx, r in enumerate(ras_races, 1):
                try:
                    race_num = int(r.get('raceNumber') or r.get('RaceNumber') or r_idx)

                    # Start time
                    time_str = r.get('raceTime') # e.g. "12:30"
                    if time_str:
                        start_time = datetime.combine(race_date, datetime.strptime(time_str, "%H:%M").time())
                    else:
                        start_time = datetime.combine(race_date, datetime.min.time())

                    runners = []
                    for run_data in r.get('runners', []):
                        name = run_data.get('horseName') or run_data.get('name')
                        if not name: continue
                        num = int(run_data.get('tabNo') or run_data.get('number') or 0)
                        win_odds = parse_odds_to_decimal(run_data.get('winOdds') or run_data.get('fixedOdds'))
                        odds_data = {}
                        if ov := create_odds_data(self.SOURCE_NAME, win_odds):
                            odds_data[self.SOURCE_NAME] = ov
                        runners.append(Runner(name=name, number=num, odds=odds_data, win_odds=win_odds, odds_source="extracted" if win_odds else None))

                    if not runners: continue

                    # Standardize discipline names (e.g. THOROUGHBRED -> Thoroughbred)
                    assigned_disc = m.get("assigned_discipline")
                    if assigned_disc:
                        assigned_disc = assigned_disc.title()

                    races.append(Race(
                        id=generate_race_id("ras", venue, start_time, race_num),
                        venue=venue,
                        race_number=race_num,
                        start_time=ensure_eastern(start_time),
                        runners=runners,
                        source=self.SOURCE_NAME,
                        discipline=assigned_disc or detect_discipline(str(r))
                    ))
                except Exception:
                    continue
        return races

    def _parse_single_race(self, html_content: str, url: str, race_date: date, venue: str, race_num: int) -> Optional[Race]:
        tree = HTMLParser(html_content)

        runners = []
        for row in tree.css("tr.runner-row"):
            name_node = row.css_first(".runner-name")
            if not name_node: continue
            name = clean_text(node_text(name_node))

            num_node = row.css_first(".runner-number")
            number = int("".join(filter(str.isdigit, node_text(num_node)))) if num_node else 0

            odds_node = row.css_first(".odds-win")
            win_odds = parse_odds_to_decimal(clean_text(node_text(odds_node))) if odds_node else None
            odds_source = "extracted" if win_odds is not None else None

            # Advanced heuristic fallback
            if win_odds is None:
                win_odds = SmartOddsExtractor.extract_from_node(row)
                odds_source = "smart_extractor" if win_odds is not None else None

            odds_data = {}
            if ov := create_odds_data(self.SOURCE_NAME, win_odds):
                odds_data[self.SOURCE_NAME] = ov

            runners.append(Runner(name=name, number=number, odds=odds_data, win_odds=win_odds, odds_source=odds_source))

        if not runners: return None

        # Start time from page if available, else guess
        start_time = datetime.combine(race_date, datetime.min.time())
        # Try to find time in text
        time_match = re.search(r"(\d{1,2}:\d{2})", html_content)
        if time_match:
            try:
                start_time = datetime.combine(race_date, datetime.strptime(time_match.group(1), "%H:%M").time())
            except Exception: pass

        return Race(
            id=generate_race_id("ras", venue, start_time, race_num),
            venue=venue,
            race_number=race_num,
            start_time=ensure_eastern(start_time),
            runners=runners,
            source=self.SOURCE_NAME,
            available_bets=scrape_available_bets(html_content)
        )

class SkyRacingWorldAdapter(BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
    SOURCE_NAME: ClassVar[str] = "SkyRacingWorld"
    PROVIDES_ODDS: ClassVar[bool] = True
    BASE_URL: ClassVar[str] = "https://www.skyracingworld.com"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        # Capability Improvement: Upgrade to scraping strategy to bypass persistent 403s on aggregate discovery
        return scraping_fetch_strategy()

    def _get_headers(self) -> Dict[str, str]:
        return self._get_browser_headers(host="www.skyracingworld.com")

    async def make_request(self, method: str, url: str, **kwargs: Any) -> Any:
        kwargs.setdefault("impersonate", "chrome124")
        return await super().make_request(method, url, **kwargs)

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        # Index for the day
        dt = parse_date_string(date)
        date_iso = dt.strftime("%Y-%m-%d")
        # Standard URL for SRW form guide
        index_url = f"/form-guide/thoroughbred/{date_iso}"

        # Success Strategy: Bootstrap session on the form guide root first
        try:
            # FIX-CR-SRW: Establish session on the home page first
            await self.make_request("GET", "https://www.skyracingworld.com/", timeout=20, raise_for_status=False, update_status=False)
            await self.make_request("GET", "/form-guide", timeout=20, raise_for_status=False, update_status=False)
            await self.make_request("GET", "/form-guide/thoroughbred", timeout=20, raise_for_status=False, update_status=False)
            # FIX: Additional delay to satisfy SRW's rate limiters during bootstrap
            await asyncio.sleep(2)
        except Exception: pass

        # Success Strategy: Try both dated and generic index (Fix redirects)
        index_urls = [index_url, "/form-guide/thoroughbred", "/form-guide"]
        resp = None
        for u in index_urls:
            try:
                # FIX-CR-SRW: Follow redirects explicitly
                resp = await self.make_request("GET", u, headers=self._get_headers(), follow_redirects=True)
                if resp and resp.text:
                    # Check if the page actually contains race links for the target date
                    if date_iso in resp.text or "/R" in resp.text:
                        break
            except Exception:
                continue

        if not resp or not resp.text:
            if resp: self.logger.warning("Unexpected status", status=getattr(resp, 'status', 'unknown'), url=index_url)
            return None
        self._save_debug_snapshot(resp.text, f"skyracing_index_{date}")

        parser = HTMLParser(resp.text)
        track_links = defaultdict(list)
        now = now_eastern()
        today_str = now.strftime(DATE_FORMAT)

        # Optimization: If it's late in ET, skip countries that are finished
        # Europe/Turkey/SA usually finished by 18:00 ET
        skip_finished_countries = (now.hour >= 18 or now.hour < 6) and (date == today_str)
        finished_keywords = ["turkey", "south-africa", "united-kingdom", "france", "germany", "dubai", "bahrain"]

        # Broaden selectors for race links (Fix 15)
        for link in parser.css("a.fg-race-link, a[href*='/form-guide/'][href*='/R']"):
            url = link.attributes.get("href")
            if url:
                if not url.startswith("http"):
                    url = self.BASE_URL + url

                if skip_finished_countries:
                    if any(kw in url.lower() for kw in finished_keywords):
                        continue

                # Group by track (everything before R#)
                track_key = re.sub(r'/R\d+$', '', url)
                track_links[track_key].append(url)

        metadata = []
        for t_url in track_links:
            # For discovery, we usually only care about upcoming races.
            # Without times in index, we pick R1 as a guess, but if we have multiple,
            # R1 might be in the past. However, picking R1 is the safest if we want "one per track".
            if track_links[t_url]:
                metadata.append({"url": track_links[t_url][0]})

        if not metadata:
            self.logger.warning("No metadata found", context="SRW Index Parsing", url=index_url)
            self.metrics.record_parse_warning()
            return None
        # Limit to first 50 to avoid hammering
        pages = await self._fetch_race_pages_concurrent(metadata[:50], self._get_headers(), semaphore_limit=5)
        return {"pages": pages, "date": date}

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data or not raw_data.get("pages"): return []
        try: race_date = parse_date_string(raw_data["date"]).date()
        except Exception: return []
        races: List[Race] = []
        for item in raw_data["pages"]:
            html_content = item.get("html")
            if not html_content: continue
            try:
                race = self._parse_single_race(html_content, item.get("url", ""), race_date)
                if race: races.append(race)
            except Exception:
                self.metrics.record_parse_error()
        return races

    def _parse_single_race(self, html_content: str, url: str, race_date: date) -> Optional[Race]:
        parser = HTMLParser(html_content)

        # Extract venue and time from header
        # Format usually: "14:30 LINGFIELD" or similar
        header = parser.css_first(".sdc-site-racing-header__name") or parser.css_first("h1") or parser.css_first("h2")
        if not header: return None

        header_text = clean_text(node_text(header))

        # Strategy 0: Extract track name from URL if possible (most reliable)
        # URL usually /form-guide/australia/wyong/2026-02-17/R1
        venue = None
        url_parts = url.lower().split("/")
        if "form-guide" in url_parts:
            idx = url_parts.index("form-guide")
            # Skip discipline if present (thoroughbred, harness, greyhound)
            if len(url_parts) > idx + 1 and url_parts[idx+1] in ["thoroughbred", "harness", "greyhound"]:
                idx += 1
            if len(url_parts) > idx + 2:
                # idx+1 is country, idx+2 is track
                venue = normalize_venue_name(url_parts[idx+2])

        match = re.search(r"(\d{1,2}:\d{2})\s+(.+)", header_text)
        if match:
            time_str = match.group(1)
            if not venue:
                venue = normalize_venue_name(match.group(2))
        else:
            venue = normalize_venue_name(header_text)
            time_str = "12:00" # Fallback

        try:
            start_time = datetime.combine(race_date, datetime.strptime(time_str, "%H:%M").time())
        except Exception:
            start_time = datetime.combine(race_date, datetime.min.time())

        # Race number from URL
        race_num = 1
        num_match = re.search(r'/R(\d+)$', url)
        if num_match:
            race_num = int(num_match.group(1))

        runners = []
        # Try different selectors for runners
        # Broadened selectors to catch dynamic/mobile-first layouts
        runner_rows = parser.css(".runner_row") or parser.css(".mobile-runner") or parser.css("div[class*='runner']")
        for row in runner_rows:
            try:
                # Broaden name selectors
                name_node = (
                    row.css_first(".horseName")
                    or row.css_first("a[href*='/horse/']")
                    or row.css_first(".name")
                )
                if not name_node: continue
                name = clean_text(node_text(name_node))

                num_node = row.css_first(".tdContent b") or row.css_first("[data-tab-no]") or row.css_first(".number")
                number = 0
                if num_node:
                    if num_node.attributes.get("data-tab-no"):
                        number = int(num_node.attributes.get("data-tab-no"))
                    else:
                        digits = "".join(filter(str.isdigit, node_text(num_node)))
                        if digits: number = int(digits)

                scratched = "strikeout" in (row.attributes.get("class") or "").lower() or row.attributes.get("data-scratched") == "True"

                win_odds = None
                # Broaden odds selectors
                odds_node = row.css_first(".pa_odds") or row.css_first(".odds") or row.css_first(".win-odds")
                win_odds = parse_odds_to_decimal(clean_text(node_text(odds_node))) if odds_node else None
                odds_source = "extracted" if win_odds is not None else None

                if win_odds is None:
                    win_odds = SmartOddsExtractor.extract_from_node(row)
                    odds_source = "smart_extractor" if win_odds is not None else None

                od = {}
                if ov := create_odds_data(self.SOURCE_NAME, win_odds):
                    od[self.SOURCE_NAME] = ov

                runners.append(Runner(name=name, number=number, odds=od, win_odds=win_odds, odds_source=odds_source))
            except Exception: continue

        if not runners: return None

        disc = detect_discipline(html_content)

        # S5 — extract race type (independent review item)
        # Hardening Fix: Broaden race type detection to improve scoring population
        race_type = None
        is_handicap = None
        header_node = parser.css_first(".sdc-site-racing-header__name") or parser.css_first("h1") or parser.css_first("h2")
        if header_node:
            header_text = node_text(header_node)
            rt_match = re.search(r'(Maiden\s+\w+|Claiming|Allowance|Graded\s+Stakes|Stakes|Handicap|Novice|Group\s+\d|Grade\s+\d|Listed|Condition|Hurdle|Chase|Bumper|National\s+Hunt|Steeplechase)', header_text, re.I)
            if rt_match: race_type = rt_match.group(1)
            if "HANDICAP" in header_text.upper():
                is_handicap = True

        return Race(
            id=generate_race_id("srw", venue, start_time, race_num, disc),
            venue=venue,
            race_number=race_num,
            start_time=start_time,
            runners=runners,
            discipline=disc,
            race_type=race_type,
            is_handicap=is_handicap,
            source=self.SOURCE_NAME,
            available_bets=scrape_available_bets(html_content)
        )

# ----------------------------------------
# AtTheRacesAdapter
# ----------------------------------------
class AtTheRacesAdapter(BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
    SOURCE_NAME: ClassVar[str] = "AtTheRaces"
    PROVIDES_ODDS: ClassVar[bool] = True
    BASE_URL: ClassVar[str] = "https://www.attheraces.com"

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return scraping_fetch_strategy()

    async def make_request(self, method: str, url: str, **kwargs: Any) -> Any:
        kwargs.setdefault("impersonate", "chrome124")
        return await super().make_request(method, url, **kwargs)

    SELECTORS: ClassVar[Dict[str, List[str]]] = {
        "race_links": ['a.race-navigation-link', 'a.sidebar-racecards-navigation-link', 'a[href^="/racecard/"]', 'a[href*="/racecard/"]'],
        "details_container": [".race-header__details--primary", "atr-racecard-race-header .container", ".racecard-header .container"],
        "track_name": ["h2", "h1 a", "h1"],
        "race_time": ["h2 b", "h1 span", ".race-time"],
        "distance": [".race-header__details--secondary .p--large", ".race-header__details--secondary div"],
        "runners": [".card-cell--horse", ".odds-grid-horse", "atr-horse-in-racecard", ".horse-in-racecard"],
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _get_headers(self) -> Dict[str, str]:
        h = self._get_browser_headers(host="www.attheraces.com", referer="https://www.attheraces.com/racecards")
        h["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
        h["Upgrade-Insecure-Requests"] = "1"
        return h

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        # Success Strategy: Bootstrap session if using browser
        # For ATR, always try to establish a session on the main site first
        try:
            # Hardening Fix: ATR is extremely sensitive; bootstrapping to the homepage is mandatory
            await self.make_request("GET", "https://www.attheraces.com/", wait_until="networkidle", timeout=30, raise_for_status=False, update_status=False)
            # Establish session on racecards index too
            await self.make_request("GET", "https://www.attheraces.com/racecards", wait_until="networkidle", timeout=30, raise_for_status=False, update_status=False)
            # Boot movers index specifically for AJAX calls
            await self.make_request("GET", "https://www.attheraces.com/market-movers", wait_until="networkidle", timeout=30, raise_for_status=False, update_status=False)
            # Also try hitting the international tab specifically to set its cookies
            await self.make_request("GET", "https://www.attheraces.com/market-movers/international", wait_until="networkidle", timeout=30, raise_for_status=False, update_status=False)
            # FIX: Additional delay to allow browser scripts/moat/cloudflare to settle
            await asyncio.sleep(5)
        except Exception: pass

        # Success Strategy: Use Market Movers AJAX for deterministic top-tier odds (Council Intelligence)
        dt = parse_date_string(date)
        date_iso = dt.strftime("%Y-%m-%d")

        # We fetch for both UK/IRE and International to ensure global coverage
        regions = ["uk-ire", "international"]
        all_movers = []

        for region in regions:
            ajax_url = f"/ajax/marketmovers/tabs/{region}/{date_iso}"
            try:
                resp = await self.make_request("GET", ajax_url, headers=self._get_headers())
                if resp and resp.text:
                    parser = HTMLParser(resp.text)
                    # AJAX returns HTML with mover rows
                    for row in parser.css(".market-mover-row"):
                        all_movers.append(row.html)
            except Exception as e:
                self.logger.error("failed_fetching_atr_movers", region=region, error=str(e))

        if all_movers:
            return {"mover_rows": all_movers, "date": date}

        # Fallback to standard index scraping if movers endpoint is dry
        index_url = f"/racecards/{date_iso}"
        # Hardening Fix: Use wait_until and timeout to handle slow page loads/bot challenge solving
        resp = await self.make_request("GET", index_url, headers=self._get_headers(), wait_until="networkidle", timeout=60)
        metadata = []
        if resp and resp.text:
            parser = HTMLParser(resp.text)
            metadata.extend(self._extract_race_metadata(parser, date))

        if not metadata:
            return None

        pages = await self._fetch_race_pages_concurrent(metadata, self._get_headers(), semaphore_limit=5)
        return {"pages": pages, "date": date}

    def _extract_race_metadata(self, parser: HTMLParser, date_str: str) -> List[Dict[str, Any]]:
        meta: List[Dict[str, Any]] = []
        track_map = defaultdict(list)

        try:
            target_date = parse_date_string(date_str).date()
        except Exception:
            target_date = datetime.now(EASTERN).date()

        for link in parser.css('a[href*="/racecard/"]'):
            url = link.attributes.get("href")
            if not url:
                continue
            time_match = re.search(r"/(\d{4})$", url)
            if not time_match:
                if not re.search(r"/\d{1,2}$", url):
                    continue

            parts = url.rstrip("/").split("/")
            if len(parts) >= 3:
                # Handle absolute (parts[4]) or relative (parts[2]) URLs
                raw_slug = parts[4] if url.startswith("http") and len(parts) >= 5 else parts[2]

                # Normalize venue from URL slug using word-boundary matching
                slug_words = raw_slug.replace('-', ' ').upper().split()
                track_name = None
                for end in range(len(slug_words), 0, -1):
                    candidate = " ".join(slug_words[:end])
                    if candidate in VENUE_MAP:
                        track_name = VENUE_MAP[candidate]
                        break
                if not track_name:
                    track_name = normalize_venue_name(raw_slug)

                time_str = time_match.group(1) if time_match else None
                track_map[track_name].append({"url": url, "time_str": time_str})

        site_tz = ZoneInfo("Europe/London")
        now_site = datetime.now(site_tz)

        # After building track_map, assign sequential race numbers per track (Fix 2)
        for track, race_infos in track_map.items():
            # Sort by time to assign correct sequential race numbers
            race_infos_sorted = sorted(
                race_infos,
                key=lambda r: r["time_str"] or "0000",
            )
            for race_idx, r in enumerate(race_infos_sorted, start=1):
                if r["time_str"]:
                    try:
                        rt = datetime.strptime(r["time_str"], "%H%M").replace(
                            year=target_date.year,
                            month=target_date.month,
                            day=target_date.day,
                            tzinfo=site_tz,
                        )
                        diff = (rt - now_site).total_seconds() / 60
                        if not (-45 < diff <= 1080):
                            continue
                        meta.append({
                            "url": r["url"],
                            "race_number": race_idx,
                            "venue_raw": track,
                        })
                    except Exception:
                        pass

        if not meta:
            for meeting in (parser.css(".meeting-summary") or parser.css(".p-meetings__item")):
                for link in meeting.css('a[href*="/racecard/"]'):
                    if url := link.attributes.get("href"):
                        meta.append({"url": url, "race_number": 1})
        return meta

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data: return []
        date_str = raw_data.get("date")
        try: race_date = parse_date_string(date_str).date()
        except Exception: return []

        races_map = {}

        # 1. Process movers (High-quality AJAX data)
        if "mover_rows" in raw_data:
            for row_html in raw_data["mover_rows"]:
                parser = HTMLParser(row_html)
                # Mover row contains: Race details, Horse, and Price
                # Format varies, but often has a link to the racecard
                link = parser.css_first("a[href*='/racecard/']")
                if not link: continue

                url = link.attributes.get("href", "")
                venue_node = parser.css_first(".track-name") or parser.css_first(".track")
                venue = normalize_venue_name(node_text(venue_node)) if venue_node else "Unknown"

                # Deduplicate by URL (unique race)
                if url not in races_map:
                    # Create a skeleton race
                    races_map[url] = {
                        "venue": venue,
                        "url": url,
                        "runners": {}
                    }

                # Extract runner from this mover row
                name_node = parser.css_first(".horse-name") or parser.css_first(".name")
                name = clean_text(node_text(name_node))
                if not name: continue

                odds_node = parser.css_first(".price") or parser.css_first(".odds")
                win_odds = parse_odds_to_decimal(node_text(odds_node))

                if win_odds:
                    races_map[url]["runners"][name] = win_odds

        # 2. Process standard pages if provided
        races: List[Race] = []
        if "pages" in raw_data:
            for item in raw_data["pages"]:
                html_content = item.get("html")
                if not html_content: continue
                try:
                    race = self._parse_single_race(html_content, item.get("url", ""), race_date, item.get("race_number"))
                    if race: races.append(race)
                except Exception: continue

        # 3. Convert skeletal mover races to objects (only if not already parsed from pages)
        for url, r_data in races_map.items():
            # Check if this race was already parsed (comparing normalized venue and race number)
            # Actually simplest to just build them and let master deduplication handle it.
            runners = []
            for name, odds in r_data["runners"].items():
                od = {}
                if ov := create_odds_data(self.SOURCE_NAME, odds):
                    od[self.SOURCE_NAME] = ov
                runners.append(Runner(name=name, odds=od, win_odds=odds))

            if runners:
                # Guess race number and time from URL
                # e.g. /racecard/ludlow/18-February-2026/1330
                time_match = re.search(r"/(\d{4})$", url)
                time_str = time_match.group(1) if time_match else "1200"
                start_time = datetime.combine(race_date, datetime.strptime(time_str, "%H%M").time())

                races.append(Race(
                    id=generate_race_id("atr", r_data["venue"], start_time, 1),
                    venue=r_data["venue"],
                    race_number=1,
                    start_time=ensure_eastern(start_time),
                    runners=runners,
                    source=self.SOURCE_NAME
                ))

        return races

    def _parse_single_race(self, html_content: str, url_path: str, race_date: date, race_number_fallback: Optional[int]) -> Optional[Race]:
        parser = HTMLParser(html_content)
        track_name, time_str, header_text = None, None, ""

        # Strategy 0: Extract track name from URL (most reliable for UK tracks)
        # ATR URLs: /racecard/[race-title-slug]/date/time
        # e.g., /racecard/ludlow-suzuki-king-quad/2026-02-18/1705
        # We need "Ludlow" from "ludlow-suzuki-king-quad"
        url_parts = url_path.lower().split("/")
        for marker in ["racecard", "racecards"]:
            if marker in url_parts:
                idx = url_parts.index(marker)
                for candidate in url_parts[idx+1:]:
                    if (candidate
                        and candidate not in ["international", "uk-ire", "usa"]
                        and not re.match(r"\d{4}-\d{2}-\d{2}", candidate)
                        and not re.match(r"^\d{4}$", candidate)):

                        # Word-boundary venue matching against VENUE_MAP
                        slug_words = candidate.replace('-', ' ').upper().split()
                        for end in range(len(slug_words), 0, -1):
                            test = " ".join(slug_words[:end])
                            if test in VENUE_MAP:
                                track_name = VENUE_MAP[test]
                                break
                        else:
                            # No known venue found — use first word as fallback
                            # (venue names are 1-3 words; race titles are 4+)
                            if len(slug_words) >= 4:
                                track_name = normalize_venue_name(slug_words[0])
                            else:
                                track_name = normalize_venue_name(candidate)
                        break
                if track_name:
                    break

        header = parser.css_first(".race-header__details") or parser.css_first(".racecard-header")
        if header:
            header_text = clean_text(node_text(header)) or ""
            time_match = re.search(r"(\d{1,2}:\d{2})", header_text)
            if time_match:
                time_str = time_match.group(1)
                if not track_name:
                    # More aggressive stripping of race titles from venue
                    # We use the VENUE_MAP to try and find a known track name in the header.
                    upper_header = header_text.upper()
                    found_track = None
                    for known_track in sorted(VENUE_MAP.keys(), key=len, reverse=True):
                        if known_track in upper_header:
                            found_track = VENUE_MAP[known_track]
                            break

                    if found_track:
                        track_name = found_track
                    else:
                        track_raw = re.sub(r"\d{1,2}\s+[A-Za-z]{3}\s+\d{4}", "", header_text.replace(time_str, "")).strip()
                        track_raw = re.split(r"\s+Race\s+\d+", track_raw, flags=re.I)[0]
                        track_raw = re.sub(r"^\d+\s+", "", track_raw).split(" - ")[0].split("|")[0].strip()
                        track_name = normalize_venue_name(track_raw)
        if not track_name:
            details = parser.css_first(".race-header__details--primary")
            if details:
                track_node = details.css_first("h2") or details.css_first("h1 a") or details.css_first("h1")
                if track_node: track_name = normalize_venue_name(clean_text(node_text(track_node)))
                if not time_str:
                    time_node = details.css_first("h2 b") or details.css_first(".race-time")
                    if time_node: time_str = clean_text(node_text(time_node)).replace(" ATR", "")
        if not track_name:
            parts = url_path.split("/")
            if len(parts) >= 3: track_name = normalize_venue_name(parts[2])
        if not time_str:
            parts = url_path.split("/")
            if len(parts) >= 5 and re.match(r"\d{4}", parts[-1]):
                raw_time = parts[-1]
                time_str = f"{raw_time[:2]}:{raw_time[2:]}"
        if not track_name or not time_str: return None
        try: start_time = datetime.combine(race_date, datetime.strptime(time_str, "%H:%M").time())
        except Exception: return None

        # Extract correct race number from header or URL
        race_number = race_number_fallback or 1
        rn_match = re.search(r"Race\s+(\d+)", header_text, re.I)
        if rn_match:
            race_number = int(rn_match.group(1))
        else:
            # Fallback to URL if it ends in a small number
            url_rn_match = re.search(r"/(\d{1,2})$", url_path.rstrip("/"))
            if url_rn_match:
                race_number = int(url_rn_match.group(1))

        distance = None
        dist_match = re.search(r"\|\s*(\d+[mfy].*)", header_text, re.I)
        if dist_match: distance = dist_match.group(1).strip()

        # S5 — extract race type (independent review item)
        # Hardening Fix: Broaden race type detection to improve scoring population
        race_type = None
        is_handicap = None
        rt_match = re.search(r'(Maiden\s+\w+|Claiming|Allowance|Graded\s+Stakes|Stakes|Handicap|Novice|Group\s+\d|Grade\s+\d|Listed|Condition|Hurdle|Chase|Bumper|National\s+Hunt|Steeplechase)', header_text, re.I)
        if rt_match: race_type = rt_match.group(1)
        if "HANDICAP" in header_text.upper():
            is_handicap = True

        runners = self._parse_runners(parser)
        if not runners: return None
        return Race(discipline="Thoroughbred", id=generate_race_id("atr", track_name, start_time, race_number), venue=track_name, race_number=race_number, start_time=start_time, runners=runners, distance=distance, race_type=race_type, is_handicap=is_handicap, source=self.source_name, available_bets=scrape_available_bets(html_content))

    def _parse_runners(self, parser: HTMLParser) -> List[Runner]:
        odds_map: Dict[str, float] = {}
        for row in parser.css(".odds-grid__row--horse"):
            if m := re.search(r"row-(\d+)", row.attributes.get("id", "")):
                if price := row.attributes.get("data-bestprice"):
                    try:
                        p_val = float(price)
                        if is_valid_odds(p_val): odds_map[m.group(1)] = p_val
                    except Exception: pass
        runners: List[Runner] = []
        for selector in self.SELECTORS["runners"]:
            nodes = parser.css(selector)
            if nodes:
                for i, node in enumerate(nodes):
                    runner = self._parse_runner(node, odds_map, i + 1)
                    if runner: runners.append(runner)
                break
        return runners

    def _parse_runner(self, row: Node, odds_map: Dict[str, float], fallback_number: int = 0) -> Optional[Runner]:
        try:
            name_node = row.css_first("h3") or row.css_first("a.horse__link") or row.css_first('a[href*="/form/horse/"]')
            if not name_node: return None
            name = clean_text(node_text(name_node))
            if not name: return None
            num_node = row.css_first(".horse-in-racecard__saddle-cloth-number") or row.css_first(".odds-grid-horse__no")
            number = 0
            if num_node:
                ns = clean_text(node_text(num_node))
                if ns:
                    digits = "".join(filter(str.isdigit, ns))
                    if digits: number = int(digits)

            if number == 0 or number > 40:
                number = fallback_number
            win_odds = None
            odds_source = None
            if horse_link := row.css_first('a[href*="/form/horse/"]'):
                if m := re.search(r"/(\d+)(\?|$)", horse_link.attributes.get("href", "")):
                    win_odds = odds_map.get(m.group(1))
                    if win_odds is not None:
                        odds_source = "extracted"
            if win_odds is None:
                if odds_node := row.css_first(".horse-in-racecard__odds"):
                    win_odds = parse_odds_to_decimal(clean_text(node_text(odds_node)))
                    if win_odds is not None:
                        odds_source = "extracted"

            # Advanced heuristic fallback
            if win_odds is None:
                win_odds = SmartOddsExtractor.extract_from_node(row)
                if win_odds is not None:
                    odds_source = "smart_extractor"

            odds: Dict[str, OddsData] = {}
            if od := create_odds_data(self.source_name, win_odds): odds[self.source_name] = od
            return Runner(number=number, name=name, odds=odds, win_odds=win_odds, odds_source=odds_source)
        except Exception: return None

# ----------------------------------------
# AtTheRacesGreyhoundAdapter
# ----------------------------------------
class AtTheRacesGreyhoundAdapter(JSONParsingMixin, BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
    SOURCE_NAME: ClassVar[str] = "AtTheRacesGreyhound"
    PROVIDES_ODDS: ClassVar[bool] = True
    BASE_URL: ClassVar[str] = "https://greyhounds.attheraces.com"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return scraping_fetch_strategy()

    def _get_headers(self) -> Dict[str, str]:
        return self._get_browser_headers(host="greyhounds.attheraces.com", referer="https://greyhounds.attheraces.com/racecards")

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        dt = parse_date_string(date)
        date_iso = dt.strftime("%Y-%m-%d")
        index_url = f"/racecards/{date_iso}" if date else "/racecards"
        resp = await self.make_request("GET", index_url, headers=self._get_headers())
        if not resp or not resp.text:
            if resp: self.logger.warning("Unexpected status", status=resp.status, url=index_url)
            return None
        self._save_debug_snapshot(resp.text, f"atr_grey_index_{date}")
        parser = HTMLParser(resp.text)
        metadata = self._extract_race_metadata(parser, date)
        if not metadata:
            links = []
            scripts = self._parse_all_jsons_from_scripts(parser, 'script[type="application/ld+json"]', context="ATR Greyhound Index")
            for d in scripts:
                items = d.get("@graph", [d]) if isinstance(d, dict) else []
                for item in items:
                    if item.get("@type") == "SportsEvent":
                        loc = item.get("location")
                        if isinstance(loc, list):
                            for l in loc:
                                if u := l.get("url"): links.append(u)
                        elif isinstance(loc, dict):
                            if u := loc.get("url"): links.append(u)
            metadata = [{"url": l, "race_number": 0} for l in set(links)]
        if not metadata:
            self.logger.warning("No metadata found", context="ATR Greyhound Index Parsing", url=index_url)
            self.metrics.record_parse_warning()
            return None
        pages = await self._fetch_race_pages_concurrent(metadata, self._get_headers(), semaphore_limit=5)
        return {"pages": pages, "date": date}

    def _extract_race_metadata(self, parser: HTMLParser, date_str: str) -> List[Dict[str, Any]]:
        meta: List[Dict[str, Any]] = []
        pc = parser.css_first("page-content")
        if not pc: return []
        items_raw = pc.attributes.get(":items") or pc.attributes.get(":modules")
        if not items_raw: return []

        try:
            target_date = parse_date_string(date_str).date()
        except Exception:
            target_date = datetime.now(EASTERN).date()

        # Usually UK time
        site_tz = ZoneInfo("Europe/London")
        now_site = datetime.now(site_tz)

        try:
            modules = json.loads(html.unescape(items_raw))
            for module in modules:
                for meeting in module.get("data", {}).get("items", []):
                    # Broaden window to capture multiple races
                    races = [r for r in meeting.get("items", []) if r.get("type") == "racecard"]

                    for race in races:
                        r_time_str = race.get("time") # Usually HH:MM
                        if r_time_str:
                            try:
                                rt = datetime.strptime(r_time_str, "%H:%M").replace(
                                    year=target_date.year, month=target_date.month, day=target_date.day, tzinfo=site_tz
                                )
                                diff = (rt - now_site).total_seconds() / 60
                                if not (-45 < diff <= 1080):
                                    continue

                                r_num = race.get("raceNumber") or race.get("number") or 1
                                if u := race.get("cta", {}).get("href"):
                                    if "/racecard/" in u:
                                        meta.append({"url": u, "race_number": r_num})
                            except Exception: pass
        except Exception: pass
        return meta

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data or not raw_data.get("pages"): return []
        try: race_date = parse_date_string(raw_data.get("date", "")).date()
        except Exception: race_date = datetime.now(EASTERN).date()
        races: List[Race] = []
        for item in raw_data["pages"]:
            if not item or not item.get("html"): continue
            try:
                race = self._parse_single_race(item["html"], item.get("url", ""), race_date, item.get("race_number"))
                if race: races.append(race)
            except Exception: pass
        return races

    def _parse_single_race(self, html_content: str, url_path: str, race_date: date, race_number: Optional[int]) -> Optional[Race]:
        parser = HTMLParser(html_content)
        pc = parser.css_first("page-content")
        if not pc: return None
        items_raw = pc.attributes.get(":items") or pc.attributes.get(":modules")
        if not items_raw: return None
        try: modules = json.loads(html.unescape(items_raw))
        except Exception: return None
        venue, race_time_str, distance, runners, odds_map = "", "", "", [], {}

        # Try to extract venue from title as high-priority fallback
        title_node = parser.css_first("title")
        if title_node:
            title_text = node_text(title_node).strip()
            # Title: "14:26 Oxford Greyhound Racecard..."
            tm = re.search(r'\d{1,2}:\d{2}\s+(.+?)\s+Greyhound', title_text)
            if tm:
                venue = normalize_venue_name(tm.group(1))
        for module in modules:
            m_type, m_data = module.get("type"), module.get("data", {})
            if m_type == "RacecardHero":
                venue = normalize_venue_name(m_data.get("track", ""))
                race_time_str = m_data.get("time", "")
                distance = m_data.get("distance", "")
                if not race_number: race_number = m_data.get("raceNumber") or m_data.get("number")
            elif m_type == "OddsGrid":
                odds_grid = m_data.get("oddsGrid", {})

                # If venue still empty, try to get it from OddsGrid data
                if not venue:
                    venue = normalize_venue_name(odds_grid.get("track", ""))
                if not race_time_str:
                    race_time_str = odds_grid.get("time", "")
                if not distance:
                    distance = odds_grid.get("distance", "")

                partners = odds_grid.get("partners", {})
                all_partners = []
                if isinstance(partners, dict):
                    for p_list in partners.values(): all_partners.extend(p_list)
                elif isinstance(partners, list): all_partners = partners
                for partner in all_partners:
                    for o in partner.get("odds", []):
                        g_id = o.get("betParams", {}).get("greyhoundId")
                        price = o.get("value", {}).get("decimal")
                        if g_id and price:
                            p_val = parse_odds_to_decimal(price)
                            if p_val and is_valid_odds(p_val): odds_map[str(g_id)] = p_val
                for t in odds_grid.get("traps", []):
                    trap_num = t.get("trap", 0)
                    name = clean_text(t.get("name", "")) or ""
                    g_id_match = re.search(r"/greyhound/(\d+)", t.get("href", ""))
                    g_id = g_id_match.group(1) if g_id_match else None
                    win_odds = odds_map.get(str(g_id)) if g_id else None
                    odds_source = "extracted" if win_odds is not None else None

                    # Advanced heuristic fallback
                    if win_odds is None:
                        win_odds = SmartOddsExtractor.extract_from_text(str(t))
                        if win_odds is not None:
                            odds_source = "smart_extractor"


                    odds_data = {}
                    if ov := create_odds_data(self.source_name, win_odds): odds_data[self.source_name] = ov
                    runners.append(Runner(number=trap_num or 0, name=name, odds=odds_data, win_odds=win_odds, odds_source=odds_source))

        url_parts = url_path.split("/")
        if not venue:
             # /racecard/GB/oxford/10-February-2026/1426
             m = re.search(r'/(?:racecard|result)/[A-Z]{2,3}/([^/]+)', url_path)
             if m:
                 venue = normalize_venue_name(m.group(1))
        if not race_time_str and len(url_parts) >= 5:
             race_time_str = url_parts[-1]
        if not venue or not runners: return None
        try:
            if ":" not in race_time_str and len(race_time_str) == 4: race_time_str = f"{race_time_str[:2]}:{race_time_str[2:]}"
            start_time = datetime.combine(race_date, datetime.strptime(race_time_str, "%H:%M").time())
        except Exception: return None
        return Race(discipline="Greyhound", id=generate_race_id("atrg", venue, start_time, race_number or 0, "Greyhound"), venue=venue, race_number=race_number or 0, start_time=start_time, runners=runners, distance=str(distance) if distance else None, source=self.source_name, available_bets=scrape_available_bets(html_content))



# ----------------------------------------
# SportingLifeAdapter
# ----------------------------------------
class SportingLifeAdapter(JSONParsingMixin, BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
    SOURCE_NAME: ClassVar[str] = "SportingLife"
    PROVIDES_ODDS: ClassVar[bool] = True
    BASE_URL: ClassVar[str] = "https://www.sportinglife.com"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return FetchStrategy(primary_engine=BrowserEngine.CURL_CFFI, enable_js=False, stealth_mode="camouflage", timeout=30, impersonate="chrome124")

    def _get_headers(self) -> Dict[str, str]:
        return self._get_browser_headers(host="www.sportinglife.com", referer="https://www.sportinglife.com/racing/racecards")

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        dt = parse_date_string(date)
        date_iso = dt.strftime("%Y-%m-%d")
        index_url = f"/racing/racecards/{date_iso}/" if date else "/racing/racecards/"
        resp = await self.make_request("GET", index_url, headers=self._get_headers(), follow_redirects=True)
        if not resp or not resp.text:
            if resp: self.logger.warning("Unexpected status", status=resp.status, url=index_url)
            raise AdapterHttpError(self.source_name, getattr(resp, 'status', 500), index_url)
        self._save_debug_snapshot(resp.text, f"sportinglife_index_{date}")
        parser = HTMLParser(resp.text)
        metadata = self._extract_race_metadata(parser, date)
        if not metadata:
            self.logger.warning("No metadata found", context="SportingLife Index Parsing", url=index_url)
            self.metrics.record_parse_warning()
            return None
        pages = await self._fetch_race_pages_concurrent(metadata, self._get_headers(), semaphore_limit=8)
        return {"pages": pages, "date": date}

    def _extract_race_metadata(self, parser: HTMLParser, date_str: str) -> List[Dict[str, Any]]:
        meta: List[Dict[str, Any]] = []
        data = self._parse_json_from_script(parser, "script#__NEXT_DATA__", context="SportingLife Index")

        try:
            target_date = parse_date_string(date_str).date()
        except Exception:
            target_date = datetime.now(EASTERN).date()

        site_tz = ZoneInfo("Europe/London")
        now_site = datetime.now(site_tz)

        if data:
            for meeting in data.get("props", {}).get("pageProps", {}).get("meetings", []):
                # Broaden window to capture multiple races
                races = meeting.get("races", [])
                for i, race in enumerate(races):
                    r_time_str = race.get("time") # Usually HH:MM
                    if r_time_str:
                        try:
                            rt = datetime.strptime(r_time_str, "%H:%M").replace(
                                year=target_date.year, month=target_date.month, day=target_date.day, tzinfo=site_tz
                            )
                            diff = (rt - now_site).total_seconds() / 60
                            if not (-45 < diff <= 1080):
                                continue

                            if url := race.get("racecard_url"):
                                meta.append({"url": url, "race_number": i + 1})
                        except Exception: pass
        if not meta:
            meetings = parser.css('section[class^="MeetingSummary"]') or parser.css(".meeting-summary")
            for meeting in meetings:
                # In HTML fallback, just take the first upcoming link we find
                for link in meeting.css('a[href*="/racecard/"]'):
                    if url := link.attributes.get("href"):
                        # Try to see if time is in link text
                        txt = node_text(link)
                        if re.match(r"\d{1,2}:\d{2}", txt):
                            try:
                                rt = datetime.strptime(txt, "%H:%M").replace(
                                    year=target_date.year, month=target_date.month, day=target_date.day, tzinfo=site_tz
                                )
                                # Skip if in past (Today only)
                                if target_date == now_site.date() and rt < now_site - timedelta(minutes=5):
                                    continue
                            except Exception: pass

                        meta.append({"url": url, "race_number": 1})
                        break
        return meta

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data or not raw_data.get("pages"): return []
        try: race_date = parse_date_string(raw_data["date"]).date()
        except Exception: return []
        races: List[Race] = []
        for item in raw_data["pages"]:
            html_content = item.get("html")
            if not html_content: continue
            try:
                parser = HTMLParser(html_content)
                race = self._parse_from_next_data(parser, race_date, item.get("race_number"), html_content)
                if not race:
                    race = self._parse_from_html(parser, race_date, item.get("race_number"), html_content, item.get("url", ""))
                if race: races.append(race)
            except Exception: pass
        return races

    def _parse_from_next_data(self, parser: HTMLParser, race_date: date, race_number_fallback: Optional[int], html_content: str) -> Optional[Race]:
        data = self._parse_json_from_script(parser, "script#__NEXT_DATA__", context="SportingLife Race")
        if not data: return None
        race_info = data.get("props", {}).get("pageProps", {}).get("race")
        if not race_info: return None
        summary = race_info.get("race_summary") or {}

        # Skip completed races (Insight 4)
        stage = (summary.get("race_stage") or "").upper()
        if stage in ["WEIGHEDIN", "RESULT", "OFF", "FINISHED", "ABANDONED"]:
            self.logger.debug("Skipping completed race", stage=stage, venue=summary.get("course_name"))
            return None

        # Strategy 0: Extract track name from URL if possible (most reliable)
        # /racing/racecards/2026-02-18/punchestown/1340/
        track_name = None
        current_url = data.get("query", {}).get("url", "")
        url_parts = current_url.lower().split("/")
        if len(url_parts) >= 5:
            # 0: '', 1: 'racing', 2: 'racecards', 3: 'date', 4: 'venue'
            track_name = normalize_venue_name(url_parts[4])

        if not track_name:
            track_name = normalize_venue_name(race_info.get("meeting_name") or summary.get("course_name") or "Unknown")
        rt = race_info.get("time") or summary.get("time") or race_info.get("off_time") or race_info.get("start_time")
        if not rt:
            def f(o):
                if isinstance(o, str) and re.match(r"^\d{1,2}:\d{2}$", o): return o
                if isinstance(o, dict):
                    for v in o.values():
                        if t := f(v): return t
                if isinstance(o, list):
                    for v in o:
                        if t := f(v): return t
                return None
            rt = f(race_info)
        if not rt: return None
        try: start_time = datetime.combine(race_date, datetime.strptime(rt, "%H:%M").time())
        except Exception: return None
        runners = []
        for rd in (race_info.get("runners") or race_info.get("rides") or []):
            name = clean_text(rd.get("horse_name") or rd.get("horse", {}).get("name", ""))
            if not name: continue
            num = rd.get("saddle_cloth_number") or rd.get("cloth_number") or 0
            wo = parse_odds_to_decimal(rd.get("betting", {}).get("current_odds") or rd.get("betting", {}).get("current_price") or rd.get("forecast_price") or rd.get("forecast_odds") or rd.get("betting_forecast_price") or rd.get("odds") or rd.get("bookmakerOdds") or "")
            odds_source = "extracted" if wo is not None else None

            # Advanced heuristic fallback
            if wo is None:
                wo = SmartOddsExtractor.extract_from_text(str(rd))
                odds_source = "smart_extractor" if wo is not None else None

            odds_data = {}
            if ov := create_odds_data(self.source_name, wo): odds_data[self.source_name] = ov
            runners.append(Runner(number=num, name=name, scratched=rd.get("is_non_runner") or rd.get("ride_status") == "NON_RUNNER", odds=odds_data, win_odds=wo, odds_source=odds_source))
        if not runners: return None

        # S5 — extract race type (independent review item)
        # Hardening Fix: Broaden race type detection to improve scoring population
        is_handicap = summary.get("has_handicap")
        race_type_raw = summary.get("race_title") or summary.get("race_name") or ""
        rt_match = re.search(r'(Maiden\s+\w+|Claiming|Allowance|Graded\s+Stakes|Stakes|Handicap|Novice|Group\s+\d|Grade\s+\d|Listed|Condition)', race_type_raw, re.I)
        race_type = rt_match.group(1) if rt_match else ("Handicap" if is_handicap else None)
        return Race(id=generate_race_id("sl", track_name or "Unknown", start_time, race_info.get("race_number") or race_number_fallback or 1), venue=track_name or "Unknown", race_number=race_info.get("race_number") or race_number_fallback or 1, start_time=start_time, runners=runners, distance=summary.get("distance") or race_info.get("distance"), race_type=race_type, is_handicap=is_handicap, source=self.source_name, discipline="Thoroughbred", available_bets=scrape_available_bets(html_content))

    def _parse_from_html(self, parser: HTMLParser, race_date: date, race_number_fallback: Optional[int], html_content: str, url: str = "") -> Optional[Race]:
        h1 = parser.css_first('h1[class*="RacingRacecardHeader__Title"]')
        if not h1: return None
        ht = clean_text(node_text(h1))
        if not ht: return None
        parts = ht.split()
        if not parts: return None
        try: start_time = datetime.combine(race_date, datetime.strptime(parts[0], "%H:%M").time())
        except Exception: return None

        # Strategy 0: Extract track name from URL if possible (most reliable)
        track_name = None
        url_parts = url.lower().split("/")
        if len(url_parts) >= 5:
            # 0: '', 1: 'racing', 2: 'racecards', 3: 'date', 4: 'venue'
            track_name = normalize_venue_name(url_parts[4])

        if not track_name:
            track_name = normalize_venue_name(" ".join(parts[1:]))
        runners = []
        for row in parser.css('div[class*="RunnerCard"]'):
            try:
                nn = row.css_first('a[href*="/racing/profiles/horse/"]')
                if not nn: continue
                name = clean_text(node_text(nn)).splitlines()[0].strip()
                num_node = row.css_first('span[class*="SaddleCloth__Number"]')
                number = int("".join(filter(str.isdigit, clean_text(node_text(num_node))))) if num_node else 0
                on = row.css_first('span[class*="Odds__Price"]')
                wo = parse_odds_to_decimal(clean_text(node_text(on)) if on else "")
                odds_source = "extracted" if wo is not None else None

                # Advanced heuristic fallback
                if wo is None:
                    wo = SmartOddsExtractor.extract_from_node(row)
                    odds_source = "smart_extractor" if wo is not None else None

                od = {}
                if ov := create_odds_data(self.source_name, wo): od[self.source_name] = ov
                runners.append(Runner(number=number, name=name, odds=od, win_odds=wo, odds_source=odds_source))
            except Exception: continue
        if not runners: return None

        # S5 — extract race type (independent review item)
        race_type = None
        ht_node = parser.css_first('h1[class*="RacingRacecardHeader__Title"]')
        if ht_node:
            rt_match = re.search(r'(Maiden\s+\w+|Claiming|Allowance|Graded\s+Stakes|Stakes|Handicap|Novice|Group\s+\d|Grade\s+\d|Listed|Condition|Hurdle|Chase|Bumper|National\s+Hunt|Steeplechase)', node_text(ht_node), re.I)
            if rt_match: race_type = rt_match.group(1)

        dn = parser.css_first('span[class*="RacecardHeader__Distance"]') or parser.css_first(".race-distance")
        return Race(id=generate_race_id("sl", track_name or "Unknown", start_time, race_number_fallback or 1), venue=track_name or "Unknown", race_number=race_number_fallback or 1, start_time=start_time, runners=runners, distance=clean_text(node_text(dn)) if dn else None, race_type=race_type, source=self.source_name, available_bets=scrape_available_bets(html_content))

# ----------------------------------------
# SkySportsAdapter
# ----------------------------------------
class SkySportsAdapter(JSONParsingMixin, BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
    SOURCE_NAME: ClassVar[str] = "SkySports"
    PROVIDES_ODDS: ClassVar[bool] = True
    BASE_URL: ClassVar[str] = "https://www.skysports.com"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return FetchStrategy(primary_engine=BrowserEngine.CURL_CFFI, enable_js=False, stealth_mode="fast", timeout=30, impersonate="chrome124")

    def _get_headers(self) -> Dict[str, str]:
        return self._get_browser_headers(host="www.skysports.com", referer="https://www.skysports.com/racing")

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        dt = parse_date_string(date)
        index_url = f"/racing/racecards/{dt.strftime('%d-%m-%Y')}"
        resp = await self.make_request("GET", index_url, headers=self._get_headers())
        if not resp or not resp.text:
            if resp: self.logger.warning("Unexpected status", status=resp.status, url=index_url)
            raise AdapterHttpError(self.source_name, getattr(resp, 'status', 500), index_url)
        self._save_debug_snapshot(resp.text, f"skysports_index_{date}")
        parser = HTMLParser(resp.text)
        metadata = []

        try:
            target_date = parse_date_string(date).date()
        except Exception:
            target_date = datetime.now(EASTERN).date()

        site_tz = ZoneInfo("Europe/London")
        now_site = datetime.now(site_tz)

        meetings = parser.css(".sdc-site-concertina-block") or parser.css(".page-details__section") or parser.css(".racing-meetings__meeting")
        for meeting in meetings:
            hn = meeting.css_first(".sdc-site-concertina-block__title") or meeting.css_first(".racing-meetings__meeting-title")
            if not hn:
                continue
            vr = clean_text(node_text(hn)) or ""
            if "ABD:" in vr:
                continue

            # Normalize meeting name to strip session qualifiers (Fix 6)
            vr_words = vr.upper().split()
            for end in range(len(vr_words), 0, -1):
                test = " ".join(vr_words[:end])
                if test in VENUE_MAP:
                    vr = VENUE_MAP[test]
                    break

            # Updated Sky Sports event discovery logic
            events = meeting.css(".sdc-site-racing-meetings__event") or meeting.css(".racing-meetings__event")
            if events:
                for i, event in enumerate(events):
                    tn = event.css_first(".sdc-site-racing-meetings__event-time") or event.css_first(".racing-meetings__event-time")
                    ln = event.css_first(".sdc-site-racing-meetings__event-link") or event.css_first(".racing-meetings__event-link")
                    if tn and ln:
                        txt, h = clean_text(node_text(tn)), ln.attributes.get("href")
                        if h and re.match(r"\d{1,2}:\d{2}", txt):
                            try:
                                rt = datetime.strptime(txt, "%H:%M").replace(
                                    year=target_date.year, month=target_date.month, day=target_date.day, tzinfo=site_tz
                                )
                                diff = (rt - now_site).total_seconds() / 60
                                if not (-45 < diff <= 1080):
                                    continue
                                metadata.append({"url": h, "venue_raw": vr, "race_number": i + 1})
                            except Exception: pass
            else:
                # Fallback to older anchor-based discovery
                for i, link in enumerate(meeting.css('a[href*="/racecards/"]')):
                    if h := link.attributes.get("href"):
                        txt = node_text(link)
                        if re.match(r"\d{1,2}:\d{2}", txt):
                            try:
                                rt = datetime.strptime(txt, "%H:%M").replace(
                                    year=target_date.year, month=target_date.month, day=target_date.day, tzinfo=site_tz
                                )
                                diff = (rt - now_site).total_seconds() / 60
                                if not (-45 < diff <= 1080):
                                    continue
                                metadata.append({"url": h, "venue_raw": vr, "race_number": i + 1})
                            except Exception: pass

        if not metadata:
            self.logger.warning("No metadata found", context="SkySports Index Parsing", url=index_url)
            self.metrics.record_parse_warning()
            return None
        pages = await self._fetch_race_pages_concurrent(metadata, self._get_headers(), semaphore_limit=10)
        return {"pages": pages, "date": date}

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data or not raw_data.get("pages"): return []
        try: race_date = parse_date_string(raw_data.get("date", "")).date()
        except Exception: race_date = datetime.now(EASTERN).date()
        races: List[Race] = []
        for item in raw_data["pages"]:
            html_content = item.get("html")
            if not html_content: continue
            parser = HTMLParser(html_content)
            h = parser.css_first(".sdc-site-racing-header__name")
            if not h: continue
            ht = clean_text(node_text(h)) or ""
            m = re.match(r"(\d{1,2}:\d{2})\s+(.+)", ht)
            if not m:
                tn, cn = parser.css_first(".sdc-site-racing-header__time"), parser.css_first(".sdc-site-racing-header__course")
                if tn and cn: rts, tnr = clean_text(node_text(tn)) or "", clean_text(node_text(cn)) or ""
                else: continue
            else: rts, tnr = m.group(1), m.group(2)

            # Strategy 0: Extract track name from URL with word-boundary matching (Fix 6)
            track_name = None
            url_parts = item.get("url", "").lower().split("/")
            if "racecards" in url_parts:
                idx = url_parts.index("racecards")
                if len(url_parts) > idx + 1:
                    slug = url_parts[idx + 1]
                    slug_words = slug.replace('-', ' ').upper().split()
                    for end in range(len(slug_words), 0, -1):
                        test = " ".join(slug_words[:end])
                        if test in VENUE_MAP:
                            track_name = VENUE_MAP[test]
                            break
                    if not track_name:
                        track_name = normalize_venue_name(slug)

            if not track_name:
                track_name = normalize_venue_name(tnr)
            if not track_name: continue
            try: start_time = datetime.combine(race_date, datetime.strptime(rts, "%H:%M").time())
            except Exception: continue
            dist = None
            for d in parser.css(".sdc-site-racing-header__detail-item"):
                dt = clean_text(node_text(d)) or ""
                if "Distance:" in dt: dist = dt.replace("Distance:", "").strip(); break

            # BUG-16: Improved discipline detection for SkySports
            disc = detect_discipline(html_content)
            harness_venues = {'le croise laroche', 'vincennes', 'enghien', 'laval', 'cabourg', 'caen', 'graignes', 'mohawk', 'meadowlands', 'woodbine mohawk'}
            if get_canonical_venue(track_name).lower() in harness_venues:
                disc = "Harness"
            elif any(k in html_content.lower() for k in ['trot', 'harness', 'pacer']):
                disc = "Harness"
            else:
                disc = "Thoroughbred"

            runners = []
            for i, node in enumerate(parser.css(".sdc-site-racing-card__item")):
                nn = node.css_first(".sdc-site-racing-card__name a")
                if not nn: continue
                name = clean_text(node_text(nn))
                if not name: continue
                nnode = node.css_first(".sdc-site-racing-card__number strong")
                number = i + 1
                if nnode:
                    nt = clean_text(node_text(nnode))
                    if nt:
                        try: number = int(nt)
                        except Exception: pass
                onode = (
                    node.css_first(".sdc-site-racing-card__betting-odds")
                    or node.css_first(".sdc-site-racing-card__odds")
                    or node.css_first(".odds")
                    or node.css_first("[class*='odds']")
                    or node.css_first("[class*='price']")
                )
                wo = parse_odds_to_decimal(clean_text(node_text(onode)) if onode else "")
                odds_source = "extracted" if wo is not None else None

                # Advanced heuristic fallback
                if wo is None:
                    wo = SmartOddsExtractor.extract_from_node(node)
                    odds_source = "smart_extractor" if wo is not None else None

                ntxt = clean_text(node_text(node)) or ""
                scratched = "NR" in ntxt or "Non-runner" in ntxt
                od = {}
                if ov := create_odds_data(self.source_name, wo): od[self.source_name] = ov
                runners.append(Runner(number=number, name=name, scratched=scratched, odds=od, win_odds=wo, odds_source=odds_source))
            if not runners: continue
            ab = scrape_available_bets(html_content)
            if not ab and (disc == "Harness" or "(us)" in tnr.lower()) and len([r for r in runners if not r.scratched]) >= 6: ab.append("Superfecta")

            # S5 — extract race type (independent review item)
            race_type = None
            if h:
                rt_match = re.search(r'(Maiden\s+\w+|Claiming|Allowance|Graded\s+Stakes|Stakes|Handicap|Novice|Group\s+\d|Grade\s+\d|Listed|Condition|Hurdle|Chase|Bumper|National\s+Hunt|Steeplechase)', node_text(h), re.I)
                if rt_match: race_type = rt_match.group(1)

            races.append(Race(id=generate_race_id("sky", track_name, start_time, item.get("race_number", 0), disc), venue=track_name, race_number=item.get("race_number", 0), start_time=start_time, runners=runners, distance=dist, discipline=disc, race_type=race_type, source=self.source_name, available_bets=ab))
        return races

# ----------------------------------------
# RacingPostB2BAdapter
# ----------------------------------------
class RacingPostB2BAdapter(BaseAdapterV3):
    SOURCE_NAME: ClassVar[str] = "RacingPostB2B"
    PROVIDES_ODDS: ClassVar[bool] = True
    BASE_URL: ClassVar[str] = "https://backend-us-racecards.widget.rpb2b.com"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config, enable_cache=True, cache_ttl=300.0, rate_limit=5.0)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return FetchStrategy(primary_engine=BrowserEngine.HTTPX, enable_js=False, max_retries=3, timeout=20)

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        dt = parse_date_string(date)
        date_iso = dt.strftime("%Y-%m-%d")
        endpoint = f"/v2/racecards/daily/{date_iso}"
        resp = await self.make_request("GET", endpoint)
        if not resp: return None
        try: data = resp.json()
        except Exception: return None
        if not isinstance(data, list): return None
        return {"venues": data, "date": date, "fetched_at": to_storage_format(datetime.now(EASTERN))}

    def _parse_races(self, raw_data: Optional[Dict[str, Any]]) -> List[Race]:
        if not raw_data or not raw_data.get("venues"): return []
        races: List[Race] = []
        target_countries = {"USA", "CAN", "AUS", "NZL", "GBR", "IRL", "ZAF"}
        for vd in raw_data["venues"]:
            if vd.get("isAbandoned"): continue
            vn, cc, rd = vd.get("name", "Unknown"), vd.get("countryCode", "USA"), vd.get("races", [])
            if cc not in target_countries: continue
            for r in rd:
                if r.get("raceStatusCode") == "ABD": continue
                parsed = self._parse_single_race(r, vn, cc)
                if parsed: races.append(parsed)
        return races

    def _parse_single_race(self, rd: Dict[str, Any], vn: str, cc: str) -> Optional[Race]:
        rid, rnum, dts, nr = rd.get("id"), rd.get("raceNumber"), rd.get("datetimeUtc"), rd.get("numberOfRunners", 0)
        if not all([rid, rnum, dts]): return None
        try: st = from_storage_format(dts.replace("Z", "+00:00"))
        except Exception: return None
        # Only return race if we have real runners (avoid placeholder generic runners)
        runners = []
        if runners_raw := rd.get("runners"):
            for i, run_data in enumerate(runners_raw):
                name = run_data.get("name") or f"Runner {i+1}"
                num = run_data.get("number") or i + 1
                runners.append(Runner(number=num, name=name))

        if not runners:
            return None

        return Race(discipline="Thoroughbred", id=f"rpb2b_{rid.replace('-', '')[:16]}", venue=normalize_venue_name(vn), race_number=rnum, start_time=st, runners=runners, source=self.source_name, metadata={"original_race_id": rid, "country_code": cc, "num_runners": nr})


# ----------------------------------------
# StandardbredCanadaAdapter
# ----------------------------------------
class StandardbredCanadaAdapter(BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
    SOURCE_NAME: ClassVar[str] = "StandardbredCanada"
    BASE_URL: ClassVar[str] = "https://standardbredcanada.ca"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return scraping_fetch_strategy(timeout=60)

    def _get_headers(self) -> Dict[str, str]:
        return self._get_browser_headers(host="standardbredcanada.ca", referer="https://standardbredcanada.ca/racing")

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        dt = parse_date_string(date)
        date_label = dt.strftime(f"%A %b {dt.day}, %Y")
        date_short = dt.strftime("%m%d") # e.g. 0208

        index_html = None

        # 1. Try browser-based fetch if available
        # FIX-CR-SC: Gate browser launch behind playwright semaphore to prevent unregulated spawns
        try:
            pw_sem = GlobalResourceManager.get_playwright_semaphore()
            async with pw_sem:
                from playwright.async_api import async_playwright
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True)
                    page = await browser.new_page()
                    try:
                        await page.goto(f"{self.base_url}/entries", wait_until="networkidle")
                        await page.evaluate("() => { document.querySelectorAll('details').forEach(d => d.open = true); }")
                        try: await page.select_option("#edit-entries-track", label="View All Tracks")
                        except Exception: pass
                        try: await page.select_option("#edit-entries-date", label=date_label)
                        except Exception: pass
                        try: await page.click("#edit-custom-submit-entries", force=True, timeout=5000)
                        except Exception: pass
                        try: await page.wait_for_selector("#entries-results-container a[href*='/entries/']", timeout=10000)
                        except Exception: pass
                        index_html = await page.content()
                    finally:
                        await page.close()
                        await browser.close()
        except Exception as e:
            self.logger.debug("Playwright index fetch failed, trying fallback", error=str(e))

        # 2. Fallback: Try to guess the data URL pattern if index fetch failed
        if not index_html:
            # Common tracks and their codes (heuristic)
            tracks = [
                ("Western Fair", f"e{date_short}lonn.dat"),
                ("Mohawk", f"e{date_short}wbsbsn.dat"),
                ("Flamboro", f"e{date_short}flmn.dat"),
                ("Rideau", f"e{date_short}ridcn.dat"),
            ]
            metadata = []
            for track_name, filename in tracks:
                url = f"/racing/entries/data/{filename}"
                metadata.append({"url": url, "venue": track_name, "finalized": True})

            pages = await self._fetch_race_pages_concurrent(metadata, self._get_headers())
            return {"pages": pages, "date": date}

        if not index_html:
            self.logger.warning("No index HTML found", context="StandardbredCanada Index Fetch")
            return None
        self._save_debug_snapshot(index_html, f"sc_index_{date}")
        parser = HTMLParser(index_html)
        metadata = []
        for container in parser.css("#entries-results-container .racing-results-ex-wrap > div"):
            tnn = container.css_first("h4.track-name")
            if not tnn: continue
            tn = clean_text(node_text(tnn)) or ""
            isf = "*" in tn or "*" in (clean_text(node_text(container)) or "")
            for link in container.css('a[href*="/entries/"]'):
                if u := link.attributes.get("href"):
                    metadata.append({"url": u, "venue": tn.replace("*", "").strip(), "finalized": isf})
        if not metadata:
            self.logger.warning("No metadata found", context="StandardbredCanada Index Parsing")
            self.metrics.record_parse_warning()
            return None
        pages = await self._fetch_race_pages_concurrent(metadata, self._get_headers(), semaphore_limit=3)
        return {"pages": pages, "date": date}

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data or not raw_data.get("pages"): return []
        try: race_date = parse_date_string(raw_data.get("date", "")).date()
        except Exception: race_date = datetime.now(EASTERN).date()
        races: List[Race] = []
        for item in raw_data["pages"]:
            html_content = item.get("html")
            # Relaxed check: allow if "Changes Made" or "Track:" exists
            valid_content = html_content and any(x in html_content for x in ["Final Changes Made", "Changes Made", "Track:", "Post Time:"])
            if not html_content or (not valid_content and not item.get("finalized")):
                continue
            track_name = normalize_venue_name(item["venue"])
            for pre in HTMLParser(html_content).css("pre"):
                text = node_text(pre)
                race_chunks = re.split(r"(\d+)\s+--\s+", text)
                for i in range(1, len(race_chunks), 2):
                    try:
                        r = self._parse_single_race(race_chunks[i+1], int(race_chunks[i]), race_date, track_name)
                        if r: races.append(r)
                    except Exception: continue
        return races

    def _parse_single_race(self, content: str, race_num: int, race_date: date, track_name: str) -> Optional[Race]:
        tm = re.search(r"Post\s+Time:\s*(\d{1,2}:\d{2}\s*[APM]{2})", content, re.I)
        st = None
        if tm:
            try: st = datetime.combine(race_date, datetime.strptime(tm.group(1), "%I:%M %p").time())
            except Exception: pass
        if not st: st = datetime.combine(race_date, datetime.min.time())
        ab = scrape_available_bets(content)
        dist = "1 Mile"
        dm = re.search(r"(\d+(?:/\d+)?\s+(?:MILE|MILES|KM|F))", content, re.I)
        if dm: dist = dm.group(1)
        runners = []
        for line in content.split("\n"):
            # Robust runner detection: starts with number, then name.
            # Stops at multiple spaces or common odds markers to prevent swallowing odds into the name.
            m = re.search(r"^\s*(\d+)\s+([A-Z0-9'\-. ]+?)(?:\s{2,}|ML|M/L|Morning Line|$)", line, re.I)
            if m:
                num, name = int(m.group(1)), m.group(2).strip()
                # If name is followed by (L), (B), (AE) etc, strip it
                name = re.sub(r"\s*\([A-Z/]+\)\s*$", "", name).strip()
                sc = "SCR" in line or "Scratched" in line
                # Try smarter odds extraction from the line
                # Harness entries often have ML odds like 5/2 or 5-2 near the end or after 'ML', 'M/L', or 'Morning Line'
                wo = None
                odds_source = None
                ml_match = re.search(r"(?:ML|M/L|Morning Line)\s*(\d+[/-]\d+|[0-9.]+)", line, re.I)
                if ml_match:
                    wo = parse_odds_to_decimal(ml_match.group(1))
                    if wo is not None:
                        odds_source = "morning_line"

                if wo is None:
                    wo = SmartOddsExtractor.extract_from_text(line)
                    if wo is not None:
                        odds_source = "smart_extractor"

                if wo is None:
                    # Look for anything that looks like odds at the end of the line
                    om = re.search(r"(\d+-\d+|\d+/\d+|[0-9.]+)\s*$", line)
                    if om:
                        wo = parse_odds_to_decimal(om.group(1))
                        if wo is not None:
                            odds_source = "extracted"

                odds_data = {}
                if ov := create_odds_data(self.source_name, wo): odds_data[self.source_name] = ov
                runners.append(Runner(number=num, name=name, scratched=sc, odds=odds_data, win_odds=wo, odds_source=odds_source))
        if not runners: return None
        return Race(discipline="Harness", id=generate_race_id("sc", track_name, st, race_num, "Harness"), venue=track_name, race_number=race_num, start_time=st, runners=runners, distance=dist, source=self.source_name, available_bets=ab)

# ----------------------------------------
# TabAdapter
# ----------------------------------------
class TabAdapter(BaseAdapterV3):
    SOURCE_NAME: ClassVar[str] = "TAB"
    PROVIDES_ODDS: ClassVar[bool] = True
    # Note: api.tab.com.au often has DNS resolution issues in some environments.
    # api.beta.tab.com.au is more reliable.
    BASE_URL: ClassVar[str] = "https://api.beta.tab.com.au/v1/tab-info-service/racing"
    BASE_URL_STABLE: ClassVar[str] = "https://api.tab.com.au/v1/tab-info-service/racing"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config, rate_limit=2.0)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return api_fetch_strategy(timeout=45)

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        dt = parse_date_string(date)
        date_iso = dt.strftime("%Y-%m-%d")
        url = f"{self.base_url}/dates/{date_iso}/meetings"
        resp = await self.make_request("GET", url, headers={"Accept": "application/json", "User-Agent": CHROME_USER_AGENT})

        if not resp or resp.status != 200:
            self.logger.info("Falling back to STABLE TAB API")
            url = f"{self.BASE_URL_STABLE}/dates/{date_iso}/meetings"
            resp = await self.make_request("GET", url, headers={"Accept": "application/json", "User-Agent": CHROME_USER_AGENT})

        if not resp: return None
        try: data = resp.json() if hasattr(resp, "json") else json.loads(resp.text)
        except Exception: return None
        if not data or "meetings" not in data:
            self.metrics.record_parse_warning()
            return None

        # TAB meetings often only have race headers. We need to fetch each meeting's details
        # to get runners and odds.
        all_meetings = []
        for m in data["meetings"]:
            try:
                vn = m.get("meetingName")
                mt = m.get("meetingType")
                if vn and mt:
                    # Endpoint for meeting details (includes races and runners)
                    m_url = f"{self.base_url}/dates/{date}/meetings/{mt}/{vn}?jurisdiction=VIC"
                    m_resp = await self.make_request("GET", m_url, headers={"Accept": "application/json", "User-Agent": CHROME_USER_AGENT})
                    if m_resp:
                        try:
                            m_data = m_resp.json() if hasattr(m_resp, "json") else json.loads(m_resp.text)
                            if m_data:
                                all_meetings.append(m_data)
                                continue
                        except Exception: pass
                # Fallback to the summary data if detail fetch fails
                all_meetings.append(m)
            except Exception:
                all_meetings.append(m)

        return {"meetings": all_meetings, "date": date}

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data or "meetings" not in raw_data: return []
        races: List[Race] = []
        for m in raw_data["meetings"]:
            vn = normalize_venue_name(m.get("meetingName"))
            mt = m.get("meetingType", "R")
            disc = {"R": "Thoroughbred", "H": "Harness", "G": "Greyhound"}.get(mt, "Thoroughbred")

            for rd in m.get("races", []):
                rn = rd.get("raceNumber")
                rst = rd.get("raceStartTime")
                if not rst or not rn: continue

                try: st = from_storage_format(rst.replace("Z", "+00:00"))
                except Exception: continue

                runners = []
                # If detail data was fetched, extract runners
                for runner_data in rd.get("runners", []):
                    name = runner_data.get("runnerName", "Unknown")
                    num = runner_data.get("runnerNumber")

                    # Try to get win odds
                    win_odds = None
                    odds_source = None
                    fixed_odds = runner_data.get("fixedOdds", {})
                    if fixed_odds:
                        win_odds = fixed_odds.get("returnWin") or fixed_odds.get("win")
                        if win_odds is not None:
                            odds_source = "extracted"

                    odds_dict = {}
                    if win_odds:
                        if ov := create_odds_data(self.source_name, win_odds):
                            odds_dict[self.source_name] = ov

                    runners.append(Runner(
                        name=name,
                        number=num,
                        win_odds=win_odds,
                        odds=odds_dict,
                        odds_source=odds_source,
                        scratched=runner_data.get("scratched", False)
                    ))

                races.append(Race(
                    id=generate_race_id("tab", vn, st, rn, disc),
                    venue=vn,
                    race_number=rn,
                    start_time=st,
                    runners=runners,
                    discipline=disc,
                    source=self.source_name,
                    available_bets=scrape_available_bets(str(rd))
                ))
        return races

# ----------------------------------------
# BetfairDataScientistAdapter
# ----------------------------------------
class BetfairDataScientistAdapter(JSONParsingMixin, BaseAdapterV3):
    ADAPTER_NAME: ClassVar[str] = "BetfairDataScientist"
    PROVIDES_ODDS: ClassVar[bool] = True

    def __init__(self, model_name: str = "Ratings", url: str = "https://www.betfair.com.au/hub/ratings/model/horse-racing/", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=f"{self.ADAPTER_NAME}_{model_name}", base_url=url, config=config)
        self.model_name = model_name

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return api_fetch_strategy(impersonate='chrome124')

    async def _fetch_data(self, date: str) -> Optional[StringIO]:
        dt = parse_date_string(date)
        date_iso = dt.strftime("%Y-%m-%d")
        endpoint = f"?date={date_iso}&presenter=RatingsPresenter&csv=true"
        resp = await self.make_request("GET", endpoint)
        if not resp or not resp.text:
            self.metrics.record_parse_warning()
            return None
        return StringIO(resp.text)

    def _parse_races(self, raw_data: Optional[StringIO]) -> List[Race]:
        if not raw_data: return []
        try:
            df = pd.read_csv(raw_data)
            if df.empty: return []
            df = df.rename(columns={"meetings.races.bfExchangeMarketId": "market_id", "meetings.name": "meeting_name", "meetings.races.raceNumber": "race_number", "meetings.races.runners.runnerName": "runner_name", "meetings.races.runners.clothNumber": "saddle_cloth", "meetings.races.runners.ratedPrice": "rated_price"})
            races: List[Race] = []
            for mid, group in df.groupby("market_id"):
                ri = group.iloc[0]
                runners = []
                for _, row in group.iterrows():
                    rp, od = row.get("rated_price"), {}
                    if pd.notna(rp):
                        if ov := create_odds_data(self.source_name, float(rp)): od[self.source_name] = ov
                    runners.append(Runner(name=str(row.get("runner_name", "Unknown")), number=int(row.get("saddle_cloth", 0)), odds=od))

                vn = normalize_venue_name(str(ri.get("meeting_name", "")))

                # Try to find a start time in the CSV
                start_time = datetime.now(EASTERN)
                for col in ["meetings.races.startTime", "startTime", "start_time", "time"]:
                    if col in ri and pd.notna(ri[col]):
                        try:
                            # Assume UTC and convert to Eastern if it looks like ISO
                            st_val = str(ri[col])
                            if "T" in st_val:
                                start_time = to_eastern(from_storage_format(st_val.replace("Z", "+00:00")))
                            break
                        except Exception: pass

                races.append(Race(id=str(mid), venue=vn, race_number=int(ri.get("race_number", 0)), start_time=start_time, runners=runners, source=self.source_name, discipline="Thoroughbred"))
            return races
        except Exception: return []

# ----------------------------------------
# NYRABetsAdapter
# ----------------------------------------
class NYRABetsAdapter(BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
    """
    Adapter for NYRABets.com - an aggregate ADW source.
    Uses the internal JSON API for fast discovery and detailed runner info.
    """
    SOURCE_NAME: ClassVar[str] = "NYRABets"
    PROVIDES_ODDS: ClassVar[bool] = True
    BASE_URL: ClassVar[str] = "https://api.nyrabets.com"
    API_URL: ClassVar[str] = "https://api.nyrabets.com"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return FetchStrategy(
            primary_engine=BrowserEngine.CURL_CFFI,
            timeout=45,
            impersonate="chrome124"
        )

    def _get_headers(self) -> Dict[str, str]:
        # Using the base domain as host to avoid internal API 403s (Fix 3)
        # Content-Type aligned with Results adapter for reliable POST (Hardening Fix)
        h = self._get_browser_headers(host="api.nyrabets.com")
        h["Origin"] = "https://www.nyrabets.com"
        h["Referer"] = "https://www.nyrabets.com/"
        h["X-Requested-With"] = "XMLHttpRequest"
        h["Content-Type"] = "application/x-www-form-urlencoded"
        return h

    async def _fetch_data(self, date_str: str) -> Optional[Dict[str, Any]]:
        # FIX-24: Bootstrap session cookies by hitting the main site first to prevent 403 Forbidden on API
        try:
            # Hardening Fix: Establish session on the homepage before API calls
            # NYRABets often challenges standard HTTPX; use scraping strategy
            await self.make_request(
                "GET",
                "https://www.nyrabets.com/",
                headers=self._get_headers(),
                timeout=30,
                raise_for_status=False,
                update_status=False,
                strategy=scraping_fetch_strategy()
            )
            # Also hit a form page to get deeper cookies
            await self.make_request(
                "GET",
                "https://www.nyrabets.com/betting",
                headers=self._get_headers(),
                timeout=30,
                raise_for_status=False,
                update_status=False,
                strategy=scraping_fetch_strategy()
            )
            await asyncio.sleep(5)  # Allow cookies and bot scripts to settle
        except Exception as e:
            self.logger.debug("NYRABets bootstrap failed", error=str(e))

        # 1. Get Cards (Meetings)
        # Modern NYRA backend requires 8-digit years (YYYY-MM-DD)
        dt = parse_date_string(date_str)
        nyra_date = dt.strftime("%Y-%m-%dT00:00:00.000")

        header = {
            "version": 2, "fragmentLanguage": "Javascript", "fragmentVersion": "", "clientIdentifier": "nyra.1b"
        }
        cards_payload = {
            "header": header, "cohort": "A--", "wageringCohort": "NBI",
            "cardDate": nyra_date, "wantFeaturedContent": True
        }

        # Strategy: Ensure form-data payload is correctly stringified for ListCards (Hardening Fix)
        try:
            resp = await self.make_request(
                "POST",
                f"{self.API_URL}/ListCards.ashx",
                data={"request": json.dumps(cards_payload)},
                headers=self._get_headers(),
                timeout=40
            )
            if not resp or not resp.text: return None
            cards_data = json.loads(resp.text)
            card_ids = [c["cardId"] for c in cards_data.get("cards", [])]
            if not card_ids: return None

            # 2. List Races
            races_payload = {
                "header": header, "cohort": "A--", "wageringCohort": "NBI", "cardIds": card_ids
            }
            resp = await self.make_request(
                "POST",
                f"{self.API_URL}/ListRaces.ashx",
                data={"request": json.dumps(races_payload)},
                headers=self._get_headers()
            )
            if not resp or not resp.text: return None
            list_races_data = json.loads(resp.text)
            all_races = list_races_data.get("races", [])
            # Filter US/AU/NZ races for discovery efficiency (Fix: Include AUS/NZ for coverage)
            target_countries = {"US", "AU", "NZ"}
            target_race_ids = [r["raceId"] for r in all_races if r.get("countryCode") in target_countries]
            if not target_race_ids:
                self.metrics.record_parse_warning()
                return {"races": [], "details": {}}

            # 3. Get Details (Runners) - chunked
            details = {}
            for i in range(0, len(target_race_ids), 50):
                chunk = target_race_ids[i:i+50]
                get_races_payload = {
                    "header": header, "cohort": "A--", "wageringCohort": "NBI", "raceIds": chunk, "wantContents": True
                }
                # FIX-CR-NYRA: Use make_request instead of direct smart_fetcher.fetch to respect rate limits
                resp = await self.make_request(
                    "POST",
                    f"{self.API_URL}/GetRaces.ashx",
                    data={"request": json.dumps(get_races_payload)},
                    headers=self._get_headers()
                )
                if resp and resp.text:
                    chunk_data = json.loads(resp.text)
                    for race_detail in chunk_data.get("races", []):
                        details[race_detail["raceId"]] = race_detail
            return {"races": all_races, "details": details}
        except Exception as e:
            self.logger.error("NYRABets fetch failed", error=str(e))
            return None

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data: return []
        races_list = raw_data.get("races", [])
        details = raw_data.get("details", {})
        parsed_races = []
        for r in races_list:
            race_id_num = r["raceId"]
            if race_id_num not in details: continue
            detail = details[race_id_num]

            # Filter for Thoroughbreds (Success Playbook Item)
            breed = detail.get("breedType") or r.get("breedCode")
            if breed and breed != "TB":
                continue

            venue = normalize_venue_name(r["raceMeetingName"])
            race_num = r["raceNumber"]
            start_time_str = r["postTime"]
            try:
                # ISO format example: 2026-02-24T14:35:00Z
                start_time = datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M:%SZ")
            except Exception: continue
            runners = []
            for runner in detail.get("runners", []):
                number_str = "".join(filter(str.isdigit, str(runner.get("programNumber", "0"))))
                number = int(number_str) if number_str else 0
                name = runner.get("runnerName", "Unknown")
                win_odds = runner.get("currentWinPrice")
                odds_source = "extracted" if win_odds and win_odds > 1.0 else None
                if not win_odds or win_odds <= 1.0:
                    win_odds = runner.get("morningLineOdds")
                    if win_odds and win_odds > 1.0:
                        odds_source = "morning_line"
                wo = float(win_odds) if win_odds else None
                od = {}
                if ov := create_odds_data(self.source_name, wo): od[self.source_name] = ov
                runners.append(Runner(
                    number=number, name=name, odds=od, win_odds=wo, odds_source=odds_source,
                    trainer=runner.get("trainer"), jockey=runner.get("jockey")
                ))
            if not runners: continue
            race_type = r.get("raceType")
            is_handicap = None
            if race_type and "HANDICAP" in race_type.upper():
                is_handicap = True

            parsed_races.append(Race(
                id=generate_race_id("nyrab", venue, start_time, race_num),
                venue=venue, race_number=race_num, start_time=start_time,
                runners=runners, distance=r.get("distance"), surface=r.get("surface"),
                race_type=race_type, is_handicap=is_handicap, source=self.source_name,
                discipline="Thoroughbred"
            ))
        return parsed_races

class EquibaseAdapter(BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
    SOURCE_NAME: ClassVar[str] = "Equibase"
    DECOMMISSIONED = False
    PROVIDES_ODDS: ClassVar[bool] = False
    BASE_URL: ClassVar[str] = "https://www.equibase.com"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return scraping_fetch_strategy()

    async def make_request(self, method: str, url: str, **kwargs: Any) -> Any:
        # Force chrome124 for Equibase as it's the most reliable impersonation for Imperva/Cloudflare
        kwargs.setdefault("impersonate", "chrome124")
        # Let SmartFetcher/curl_cffi handle headers mostly, but provide minimal essentials if not already set
        h = kwargs.get("headers", {})
        if "Referer" not in h: h["Referer"] = "https://www.equibase.com/"
        kwargs["headers"] = h
        return await super().make_request(method, url, **kwargs)

    def _get_headers(self) -> Dict[str, str]:
        return self._get_browser_headers(host="www.equibase.com")

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        dt = parse_date_string(date)
        date_str = dt.strftime("%m%d%y")

        # Try different possible index URLs
        index_urls = [
            f"/static/entry/index.html?SAP=TN",
            f"/static/entry/index.html",
            f"/entries/{date}",
            f"/entries/index.cfm?date={dt.strftime('%m/%d/%Y')}",
        ]

        resp = None
        for url in index_urls:
            # Try multiple impersonations to bypass block
            for imp in ["chrome124", "chrome128", "safari17_0"]:
                try:
                    resp = await self.make_request("GET", url, impersonate=imp)
                    if resp and resp.status == 200 and resp.text and len(resp.text) > 1000 and "Pardon Our Interruption" not in resp.text:
                        self.logger.info("Found Equibase index", url=url, impersonate=imp)
                        break
                    else:
                        text_len = len(resp.text) if resp and resp.text else 0
                        has_pardon = "Pardon Our Interruption" in resp.text if resp and resp.text else False
                        self.logger.debug("Equibase candidate blocked or invalid", url=url, impersonate=imp, len=text_len, has_pardon=has_pardon)
                        resp = None
                except Exception as e:
                    self.logger.debug("Equibase request exception", url=url, impersonate=imp, error=str(e))
                    resp = None
            if resp: break

        if not resp or not resp.text or resp.status != 200:
            if resp: self.logger.warning("Unexpected status", status=resp.status, url=getattr(resp, 'url', 'Unknown'))
            return None

        self._save_debug_snapshot(resp.text, f"equibase_index_{date}")
        parser, links = HTMLParser(resp.text), []

        # New: Look for links in JSON data within scripts (Common on Equibase)
        # Handles escaped slashes and different path separators
        script_json_matches = re.findall(r'"URL":"([^"]+)"', resp.text)
        for url in script_json_matches:
            # Normalizing backslashes and escaped slashes in found URLs
            url_norm = url.replace("\\/", "/").replace("\\", "/")
            # Restrict lookahead: ensure link is for the targeted date_str
            if "/static/entry/" in url_norm and (date_str in url_norm or "RaceCardIndex" in url_norm):
                links.append(url_norm)

        for a in parser.css("a"):
            h = a.attributes.get("href") or ""
            c = a.attributes.get("class") or ""
            txt = node_text(a).lower()
            # Normalize backslashes (Project fix for Equibase path separators)
            h_norm = h.replace("\\", "/")

            # Restrict lookahead: ensure link strictly belongs to targeted date_str (Project Hardening)
            if "/static/entry/" in h_norm and (date_str in h_norm or "RaceCardIndex" in h_norm):
                self.logger.debug("Equibase link matched", href=h_norm)
                links.append(h_norm)
            elif "entry-race-level" in c and date_str in h_norm:
                links.append(h_norm)
            elif ("race-link" in c or "track-link" in c) and date_str in h_norm:
                links.append(h_norm)
            elif "entries" in txt and "/static/entry/" in h_norm and date_str in h_norm:
                links.append(h_norm)

        if not links:
            self.logger.warning("No links found", context="Equibase Index Parsing", date=date)
            self.metrics.record_parse_warning()
            return None

        # Fetch initial set of pages
        # Hardening Fix: Clean and deduplicate links to avoid net::ERR_INVALID_ARGUMENT
        clean_links = [l.strip() for l in set(links) if l and l.strip()]
        pages = await self._fetch_race_pages_concurrent([{"url": l} for l in clean_links], self._get_headers(), semaphore_limit=5)

        all_htmls = []
        extra_links = []
        try:
            target_date = parse_date_string(date).date()
        except Exception:
            target_date = datetime.now(EASTERN).date()

        now = now_eastern()
        for p in pages:
            html_content = p.get("html")
            if not html_content: continue

            # If it's an index page for a track, we need to extract individual race links
            if "RaceCardIndex" in p.get("url", ""):
                sub_parser = HTMLParser(html_content)
                # Only take the "next" race link for this track
                track_races = []
                for a in sub_parser.css("a"):
                    sh = (a.attributes.get("href") or "").replace("\\", "/")
                    if "/static/entry/" in sh and date_str in sh and "RaceCardIndex" not in sh:
                        # Try to find time in text nearby
                        time_txt = ""
                        parent = a.parent
                        if parent:
                            time_txt = node_text(parent)
                        track_races.append({"url": sh, "time_txt": time_txt})

                next_race = None
                for r in track_races:
                    # Look for 1:00 PM etc
                    tm = re.search(r"(\d{1,2}:\d{2}\s*[APM]{2})", r["time_txt"], re.I)
                    if tm:
                        try:
                            rt = datetime.strptime(tm.group(1).upper(), "%I:%M %p").replace(
                                year=target_date.year, month=target_date.month, day=target_date.day, tzinfo=EASTERN
                            )
                            # Skip if in past (Today only)
                            if target_date == now.date() and rt < now - timedelta(minutes=5):
                                continue
                            next_race = r
                            break
                        except Exception: pass

                if next_race:
                    extra_links.append(next_race["url"])
            else:
                all_htmls.append(html_content)

        if extra_links:
            self.logger.info("Fetching extra race pages from track index", count=len(extra_links))
            # Hardening Fix: Clean and deduplicate links to avoid net::ERR_INVALID_ARGUMENT
            clean_extra = [l.strip() for l in set(extra_links) if l and l.strip()]
            extra_pages = await self._fetch_race_pages_concurrent([{"url": l} for l in clean_extra], self._get_headers(), semaphore_limit=5)
            all_htmls.extend([p.get("html") for p in extra_pages if p and p.get("html")])

        return {"pages": all_htmls, "date": date}

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data or not raw_data.get("pages"): return []
        ds, races = raw_data.get("date", ""), []
        for html_content in raw_data["pages"]:
            if not html_content: continue
            try:
                p = HTMLParser(html_content)
                vn = p.css_first("div.track-information strong")
                rn = p.css_first("div.race-information strong")
                pt = p.css_first("p.post-time span")
                if not vn or not rn or not pt: continue
                venue = clean_text(node_text(vn))
                rnum_txt = node_text(rn).replace("Race", "").strip()
                if not venue or not rnum_txt.isdigit(): continue
                st = self._parse_post_time(ds, node_text(pt).strip())
                ab = scrape_available_bets(html_content)

                # S5 — extract race type (independent review item)
                race_type = None
                header_text = node_text(p.css_first("div.race-information")) or html_content[:2000]
                rt_match = re.search(r'(Maiden\s+\w+|Claiming|Allowance|Graded\s+Stakes|Stakes|Handicap|Novice|Group\s+\d|Grade\s+\d|Listed|Condition|Hurdle|Chase|Bumper|National\s+Hunt|Steeplechase)', header_text, re.I)
                if rt_match: race_type = rt_match.group(1)

                runners = [r for node in p.css("table.entries-table tbody tr") if (r := self._parse_runner(node))]
                if not runners: continue
                races.append(Race(id=f"eqb_{venue.lower().replace(' ', '')}_{ds}_{rnum_txt}", venue=venue, race_number=int(rnum_txt), start_time=st, runners=runners, race_type=race_type, source=self.source_name, discipline="Thoroughbred", available_bets=ab))
            except Exception: continue
        return races

    def _parse_runner(self, node: Node) -> Optional[Runner]:
        try:
            cols = node.css("td")
            if len(cols) < 3: return None

            # P1: Try to find number in first col
            number = 0
            num_text = clean_text(node_text(cols[0]))
            if num_text.isdigit():
                number = int(num_text)

            # P2: Horse name usually in 3rd col, but can vary
            name = None
            for idx in [2, 1, 3]:
                if len(cols) > idx:
                    n_text = clean_text(node_text(cols[idx]))
                    if n_text and not n_text.isdigit() and len(n_text) > 2:
                        name = n_text
                        break

            if not name: return None

            sc = "scratched" in node.attributes.get("class", "").lower() or "SCR" in (clean_text(node_text(node)) or "")

            odds, wo = {}, None
            odds_source = None
            if not sc:
                # Odds column can be 9 or 10 (blind indexing fallback)
                for idx in [9, 8, 10]:
                    if len(cols) > idx:
                        o_text = clean_text(node_text(cols[idx]))
                        if o_text:
                            wo = parse_odds_to_decimal(o_text)
                            if wo:
                                odds_source = "extracted"
                                break

                if wo is None:
                    wo = SmartOddsExtractor.extract_from_node(node)
                    if wo is not None:
                        odds_source = "smart_extractor"

                if od := create_odds_data(self.source_name, wo): odds[self.source_name] = od

            return Runner(number=number, name=name, odds=odds, win_odds=wo, odds_source=odds_source, scratched=sc)
        except Exception as e:
            self.logger.debug("equibase_runner_parse_failed", error=str(e))
            return None

    def _parse_post_time(self, ds: str, ts: str) -> datetime:
        try:
            parts = ts.replace("Post Time:", "").strip().split()
            if len(parts) >= 2:
                dt = datetime.strptime(f"{ds} {parts[0]} {parts[1]}", f"{DATE_FORMAT} %I:%M %p")
                return dt.replace(tzinfo=EASTERN)
        except Exception: pass
        # Fallback to noon UTC for the given date if time parsing fails
        try:
            dt = parse_date_string(ds)
            return dt.replace(hour=12, minute=0, tzinfo=EASTERN)
        except Exception:
            return datetime.now(EASTERN)

# ----------------------------------------
# TwinSpiresAdapter
# ----------------------------------------
class TwinSpiresAdapter(JSONParsingMixin, DebugMixin, BaseAdapterV3):
    SOURCE_NAME: ClassVar[str] = "TwinSpires"
    PROVIDES_ODDS: ClassVar[bool] = True
    BASE_URL: ClassVar[str] = "https://www.twinspires.com"

    RACE_CONTAINER_SELECTORS: ClassVar[List[str]] = ['div[class*="RaceCard"]', 'div[class*="race-card"]', 'div[data-testid*="race"]', 'div[data-race-id]', 'section[class*="race"]', 'article[class*="race"]', ".race-container", "[data-race]", 'div[class*="card"][class*="race" i]', 'div[class*="event"]']
    TRACK_NAME_SELECTORS: ClassVar[List[str]] = ['[class*="track-name"]', '[class*="trackName"]', '[data-track-name]', 'h2[class*="track"]', 'h3[class*="track"]', ".track-title", '[class*="venue"]']
    RACE_NUMBER_SELECTORS: ClassVar[List[str]] = ['[class*="race-number"]', '[class*="raceNumber"]', '[class*="race-num"]', '[data-race-number]', 'span[class*="number"]']
    POST_TIME_SELECTORS: ClassVar[List[str]] = ["time[datetime]", '[class*="post-time"]', '[class*="postTime"]', '[class*="mtp"]', "[data-post-time]", '[class*="race-time"]']
    RUNNER_ROW_SELECTORS: ClassVar[List[str]] = ['tr[class*="runner"]', 'div[class*="runner"]', 'li[class*="runner"]', "[data-runner-id]", 'div[class*="horse-row"]', 'tr[class*="horse"]', 'div[class*="entry"]', ".runner-row", ".horse-entry"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config, enable_cache=True, cache_ttl=180.0, rate_limit=1.5)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        # Success Strategy: Switch TwinSpires to browser-free API strategy for robustness (P1-Audit)
        return api_fetch_strategy(timeout=40)

    async def make_request(self, method: str, url: str, **kwargs: Any) -> Any:
        # Force chrome124 for TwinSpires to bypass basic bot checks
        kwargs.setdefault("impersonate", "chrome124")
        # Provide common browser-like headers for TwinSpires
        h = kwargs.get("headers", {})
        if "Referer" not in h: h["Referer"] = "https://www.google.com/"
        kwargs["headers"] = h
        return await super().make_request(method, url, **kwargs)

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        ard = []
        last_err = None

        # Respect region from config if provided
        target_region = self.config.get("region") # "USA", "INT", or None for both

        async def fetch_disc(disc, region="USA"):
            suffix = "" if region == "USA" else "?region=INT"
            # Try date-specific URL first, fallback to todays-races
            # Modern TwinSpires backend requires 8-digit years (YYYYMMDD) for API stability
            dt = parse_date_string(date)
            date8 = dt.strftime("%Y%m%d")

            if date == datetime.now(EASTERN).strftime(DATE_FORMAT):
                url = f"{self.BASE_URL}/bet/todays-races/{disc}{suffix}"
            else:
                url = f"{self.BASE_URL}/bet/races/{date8}/{disc}{suffix}"
            try:
                resp = await self.make_request("GET", url, network_idle=True, wait_selector='div[class*="race"], [class*="RaceCard"], [class*="track"]')
                if resp and resp.status == 200:
                    self._save_debug_snapshot(resp.text, f"ts_{disc}_{region}_{date}")
                    dr = self._extract_races_from_page(resp, date)
                    for r in dr: r["assigned_discipline"] = disc.capitalize()
                    return dr
            except Exception as e:
                self.logger.error("TwinSpires fetch failed", discipline=disc, region=region, error=str(e))
            return []

        # Fetch both USA and International for all disciplines
        # Handle 'GLOBAL' region mapping (fetch both)
        tasks = []
        for d in ["thoroughbred", "harness", "greyhound"]:
            if target_region in [None, "USA", "GLOBAL"]:
                tasks.append(fetch_disc(d, "USA"))
            if target_region in [None, "INT", "GLOBAL"]:
                tasks.append(fetch_disc(d, "INT"))
        results = await asyncio.gather(*tasks)
        for r_list in results:
            ard.extend(r_list)

        if not ard:
            try:
                resp = await self.make_request("GET", f"{self.BASE_URL}/bet/todays-races/time", network_idle=True)
                if resp and resp.status == 200: ard = self._extract_races_from_page(resp, date)
            except Exception as e: last_err = last_err or e
        if not ard and last_err: raise last_err
        return {"races": ard, "date": date, "source": self.source_name} if ard else None

    def _extract_races_from_page(self, resp, date: str) -> List[Dict[str, Any]]:
        if Selector is not None:
            page = Selector(resp.text)
        else:
            self.logger.warning("Scrapling Selector not available, falling back to selectolax")
            page = HTMLParser(resp.text)

        rd = []
        relems, used = [], None
        for s in self.RACE_CONTAINER_SELECTORS:
            try:
                el = page.css(s)
                if el:
                    relems, used = el, s
                    break
            except Exception: continue

        if not relems:
            return [{"html": resp.text, "selector": page, "track": "Unknown", "race_number": 0, "date": date, "full_page": True}]

        track_counters = defaultdict(int)
        last_track = "Unknown"

        for i, relem in enumerate(relems, 1):
            try:
                # Handle both Scrapling Selector and Selectolax Node
                if hasattr(relem, 'html'):
                    html_str = str(relem.html)
                elif hasattr(relem, 'raw_html'):
                     html_str = relem.raw_html.decode('utf-8', 'ignore') if isinstance(relem.raw_html, bytes) else str(relem.raw_html)
                else:
                    # Last resort for selectolax: reconstruct HTML or use text
                    html_str = str(relem)

                # Try to find track name in the card, but fallback to the last seen track
                # (addressing grouped race cards)
                tn = self._find_with_selectors(relem, self.TRACK_NAME_SELECTORS)
                if tn:
                    last_track = tn.strip()

                venue = last_track

                track_counters[venue] += 1
                rnum = track_counters[venue] # Track-specific index as default (Fixes Race 20 issue)

                rn_txt = self._find_with_selectors(relem, self.RACE_NUMBER_SELECTORS)
                if rn_txt:
                    digits = "".join(filter(str.isdigit, rn_txt))
                    if digits: rnum = int(digits)

                rd.append({
                    "html": html_str,
                    "selector": relem,
                    "track": venue,
                    "race_number": rnum,
                    "post_time_text": self._find_with_selectors(relem, self.POST_TIME_SELECTORS),
                    "distance": self._find_with_selectors(relem, ['[class*="distance"]', '[class*="Distance"]', '[data-distance]', ".race-distance"]),
                    "date": date,
                    "full_page": False,
                    "available_bets": scrape_available_bets(html_str)
                })
            except Exception: continue
        return rd

    def _find_with_selectors(self, el, selectors: List[str]) -> Optional[str]:
        for s in selectors:
            try:
                f = el.css_first(s)
                if f:
                    t = node_text(f)
                    if t: return t
            except Exception: continue
        return None

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data or "races" not in raw_data: return []
        rl, ds, parsed = raw_data["races"], raw_data.get("date", datetime.now(EASTERN).strftime(DATE_FORMAT)), []
        for rd in rl:
            try:
                r = self._parse_single_race(rd, ds)
                if r and r.runners: parsed.append(r)
            except Exception: continue
        return parsed

    def _parse_single_race(self, rd: dict, ds: str) -> Optional[Race]:
        page = rd.get("selector")
        hc = rd.get("html", "")
        if not page:
            if not hc: return None
            if Selector is not None:
                page = Selector(hc)
            else:
                page = HTMLParser(hc)
        tn, rnum = rd.get("track", "Unknown"), rd.get("race_number", 1)
        st = self._parse_post_time(rd.get("post_time_text"), page, ds)
        runners = self._parse_runners(page)
        disc = rd.get("assigned_discipline") or detect_discipline(hc)
        ab = scrape_available_bets(hc)
        return Race(discipline=disc, id=generate_race_id("ts", tn, st, rnum, disc), venue=tn, race_number=rnum, start_time=st, runners=runners, distance=rd.get("distance"), source=self.source_name, available_bets=ab)

    def _parse_post_time(self, tt: Optional[str], page, ds: str) -> datetime:
        bd = parse_date_string(ds).date()
        if tt:
            p = self._parse_time_string(tt, bd)
            if p: return p
        for s in self.POST_TIME_SELECTORS:
            try:
                e = page.css_first(s)
                if e:
                    # Scrapling attrib vs Selectolax attributes
                    da = getattr(e, 'attrib', getattr(e, 'attributes', {})).get('datetime')
                    if da:
                        try:
                            dt = from_storage_format(da.replace('Z', '+00:00'))
                            # Only trust the date from HTML if it's within 1 day of what we expected
                            if abs((dt.date() - bd).days) <= 1:
                                return dt
                            else:
                                self.logger.debug("Suspicious date in HTML datetime attribute", html_dt=da, expected_date=bd)
                        except Exception: pass
                    p = self._parse_time_string(node_text(e), bd)
                    if p: return p
            except Exception: continue
        return datetime.combine(bd, datetime.now(EASTERN).time()) + timedelta(hours=1)

    def _parse_time_string(self, ts: str, bd) -> Optional[datetime]:
        if not ts: return None
        tc = re.sub(r"\s+(EST|EDT|CST|CDT|MST|MDT|PST|PDT|ET|PT|CT|MT)$", "", ts, flags=re.I).strip()
        m = re.search(r"(\d+)\s*(?:min|mtp)", tc, re.I)
        if m: return now_eastern() + timedelta(minutes=int(m.group(1)))

        for f in ['%I:%M %p', '%I:%M%p', '%H:%M', '%I:%M:%S %p']:
            try:
                t = datetime.strptime(tc, f).time()
                # Heuristic: If time is between 1:00 and 7:00 and no AM/PM was explicitly in the format
                # (or even if it was, but we are suspicious), for US night tracks like Turfway,
                # it's likely PM. But %I requires %p. If %H was used and gave < 12, check if it should be PM.
                if f == '%H:%M' and 1 <= t.hour <= 7:
                    # In US horse racing, 1-7 AM is rare, 1-7 PM is common.
                    t = t.replace(hour=t.hour + 12)

                return datetime.combine(bd, t)
            except Exception: continue
        return None

    def _parse_runners(self, page) -> List[Runner]:
        runners = []
        relems = []
        for s in self.RUNNER_ROW_SELECTORS:
            try:
                el = page.css(s)
                if el: relems = el; break
            except Exception: continue
        for i, e in enumerate(relems):
            try:
                r = self._parse_single_runner(e, i + 1)
                if r: runners.append(r)
            except Exception: continue
        return runners

    def _parse_single_runner(self, e, dn: int) -> Optional[Runner]:
        # Scrapling Selector has .html property
        es = str(getattr(e, 'html', e))
        sc = any(s in es.lower() for s in ['scratched', 'scr', 'scratch'])
        num = None
        for s in ['[class*="program"]', '[class*="saddle"]', '[class*="post"]', '[class*="number"]', '[data-program-number]', 'td:first-child']:
            try:
                ne = e.css_first(s)
                if ne:
                    nt = node_text(ne)
                    dig = "".join(filter(str.isdigit, nt))
                    if dig:
                        val = int(dig)
                        if val <= 40:
                            num = val
                            break
            except Exception: continue
        name = None
        for s in ['[class*="horse-name"]', '[class*="horseName"]', '[class*="runner-name"]', 'a[class*="name"]', '[data-horse-name]', 'td:nth-child(2)']:
            try:
                ne = e.css_first(s)
                if ne:
                    nt = node_text(ne)
                    if nt and len(nt) > 1: name = re.sub(r"\(.*\)", "", nt).strip(); break
            except Exception: continue
        if not name: return None
        odds, wo = {}, None
        odds_source = None
        if not sc:
            for s in ['[class*="odds"]', '[class*="ml"]', '[class*="morning-line"]', '[data-odds]']:
                try:
                    oe = e.css_first(s)
                    if oe:
                        ot = node_text(oe)
                        if ot and ot.upper() not in ['SCR', 'SCRATCHED', '--', 'N/A']:
                            wo = parse_odds_to_decimal(ot)
                            if wo is not None:
                                odds_source = "extracted"
                                if od := create_odds_data(self.source_name, wo): odds[self.source_name] = od; break
                except Exception: continue

            # Advanced heuristic fallback
            if wo is None:
                wo = SmartOddsExtractor.extract_from_node(e)
                if wo is not None:
                    odds_source = "smart_extractor"
                    if od := create_odds_data(self.source_name, wo): odds[self.source_name] = od

        return Runner(number=num or dn, name=name, scratched=sc, odds=odds, win_odds=wo, odds_source=odds_source)

    async def cleanup(self):
        await self.close()
        self.logger.info("TwinSpires adapter cleaned up")


# ----------------------------------------
# ANALYZER LOGIC
# ----------------------------------------
# FanDuelRacingAdapter (TVG)
# ----------------------------------------
class FanDuelRacingAdapter(BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
    """
    Adapter for FanDuel Racing (formerly TVG).
    High-fidelity source for US and International racecards.
    """
    SOURCE_NAME: ClassVar[str] = "FanDuelRacing"
    PROVIDES_ODDS: ClassVar[bool] = True
    DECOMMISSIONED: ClassVar[bool] = True  # 404 since 2026-03
    BASE_URL: ClassVar[str] = "https://www.fanduel.com/racing"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return FetchStrategy(
            primary_engine=BrowserEngine.CAMOUFOX,
            enable_js=True,
            stealth_mode="camouflage",
            timeout=90,
            network_idle=True
        )

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        # Index for today's races
        url = f"{self.BASE_URL}/race-schedule"
        try:
            resp = await self.make_request("GET", url, network_idle=True)
            if not resp or not resp.text:
                return None

            self._save_debug_snapshot(resp.text, f"fanduel_schedule_{date}")
            parser = HTMLParser(resp.text)
            links = []
            for link in parser.css('a[href*="/racing/track/"]'):
                href = link.attributes.get("href")
                if href and "/race/" in href:
                    links.append({"url": href if href.startswith("http") else self.BASE_URL + href})

            if not links:
                return None

            # Fetch a sample of pages to verify structure
            pages = await self._fetch_race_pages_concurrent(links[:20], {}, semaphore_limit=3)
            return {"pages": pages, "date": date}
        except Exception as e:
            self.logger.error("FanDuel fetch failed", error=str(e))
            return None

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data or not raw_data.get("pages"): return []
        races = []
        target_date = parse_date_string(raw_data["date"]).date()

        for item in raw_data["pages"]:
            html = item.get("html")
            if not html: continue
            try:
                parser = HTMLParser(html)
                # Structure: Venue Name usually in h1 or specific class
                venue_node = parser.css_first("h1") or parser.css_first('[class*="TrackName"]')
                venue = normalize_venue_name(node_text(venue_node))

                # Race number
                rnum_node = parser.css_first('[class*="RaceNumber"]')
                rnum = 1
                if rnum_node:
                    digits = "".join(filter(str.isdigit, node_text(rnum_node)))
                    if digits: rnum = int(digits)

                # Time
                st = datetime.combine(target_date, datetime.now(EASTERN).time())
                time_node = parser.css_first('[class*="PostTime"]')
                if time_node:
                    # Simple parse for now
                    txt = node_text(time_node)
                    time_match = re.search(r"(\d{1,2}:\d{2})\s*(AM|PM)?", txt, re.I)
                    if time_match:
                        try:
                            tm = datetime.strptime(time_match.group(1), "%H:%M")
                            if time_match.group(2) and time_match.group(2).upper() == "PM" and tm.hour < 12:
                                tm = tm.replace(hour=tm.hour + 12)
                            st = datetime.combine(target_date, tm.time())
                        except Exception: pass

                runners = []
                for row in parser.css('[class*="RunnerRow"], [class*="EntryRow"]'):
                    name_node = row.css_first('[class*="HorseName"]')
                    if not name_node: continue
                    name = clean_text(node_text(name_node))

                    num_node = row.css_first('[class*="ProgramNumber"]')
                    number = int("".join(filter(str.isdigit, node_text(num_node)))) if num_node else 0

                    odds_node = row.css_first('[class*="Odds"]')
                    win_odds = parse_odds_to_decimal(node_text(odds_node)) if odds_node else None

                    odds_data = {}
                    if ov := create_odds_data(self.SOURCE_NAME, win_odds):
                        odds_data[self.SOURCE_NAME] = ov

                    runners.append(Runner(name=name, number=number, odds=odds_data, win_odds=win_odds))

                if runners:
                    races.append(Race(
                        id=generate_race_id("fd", venue, st, rnum),
                        venue=venue,
                        race_number=rnum,
                        start_time=ensure_eastern(st),
                        runners=runners,
                        source=self.SOURCE_NAME
                    ))
            except Exception: continue
        return races


# ----------------------------------------

log = structlog.get_logger(__name__)


def _get_best_win_odds(runner: Runner) -> Optional[Decimal]:
    """Gets the best win odds for a runner, filtering out invalid or placeholder values."""
    if not runner.odds:
        # Fallback to win_odds if available
        if runner.win_odds and is_valid_odds(runner.win_odds):
            return Decimal(str(runner.win_odds))

    valid_odds = []
    for source_data in runner.odds.values():
        # Handle both dict and primitive formats
        if isinstance(source_data, dict):
            win = source_data.get('win')
        elif hasattr(source_data, 'win'):
            win = source_data.win
        else:
            win = source_data

        if is_valid_odds(win):
            valid_odds.append(Decimal(str(win)))

    if valid_odds:
        return min(valid_odds)

    # Final fallback to win_odds if present
    if runner.win_odds and is_valid_odds(runner.win_odds):
        return Decimal(str(runner.win_odds))

    return None


class BaseAnalyzer(ABC):
    """The abstract interface for all future analyzer plugins."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.config = config or {}

    @abstractmethod
    def qualify_races(self, races: List[Race], now: Optional[datetime] = None) -> Dict[str, Any]:
        """The core method every analyzer must implement."""
        pass


class TrifectaAnalyzer(BaseAnalyzer):
    """Analyzes races and assigns a qualification score based on the 'Trifecta of Factors'."""

    @property
    def name(self) -> str:
        return "trifecta_analyzer"

    def __init__(
        self,
        max_field_size: Optional[int] = None,
        min_favorite_odds: float = 0.01,
        min_second_favorite_odds: float = 0.01,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Use config value if provided and no explicit override (Capability Improvement)
        self.max_field_size = max_field_size or self.config.get("analysis", {}).get("max_field_size", 11)
        self.min_favorite_odds = Decimal(str(min_favorite_odds))
        self.min_second_favorite_odds = Decimal(str(min_second_favorite_odds))
        self.notifier = RaceNotifier()

    def is_race_qualified(self, race: Race, now: Optional[datetime] = None) -> bool:
        """A race is qualified for a trifecta if it has at least 3 non-scratched runners."""
        if not race or not race.runners:
            return False

        # Apply global timing cutoff (45m ago, 120m future)
        if now is None:
            now = now_eastern()
        past_cutoff = now - timedelta(minutes=45)
        future_cutoff = now + timedelta(minutes=120)
        st = race.start_time
        if st.tzinfo is None:
            st = st.replace(tzinfo=EASTERN)
        if st < past_cutoff or st > future_cutoff:
            return False

        active_runners = sum(1 for r in race.runners if not r.scratched)
        return active_runners >= 3

    def qualify_races(self, races: List[Race], now: Optional[datetime] = None) -> Dict[str, Any]:
        """Scores all races and returns a dictionary with criteria and a sorted list."""
        qualified_races = []
        TRUSTWORTHY_RATIO_MIN = self.config.get("analysis", {}).get("simply_success_trust_min", 0.25)

        for race in races:
            if not self.is_race_qualified(race, now=now):
                continue

            active_runners = [r for r in race.runners if not r.scratched]
            total_active = len(active_runners)

            # Handicap Inference (Insight 1)
            if race.is_handicap is None:
                rt = (race.race_type or "").upper()
                if any(kw in rt for kw in ["HANDICAP", "H'CAP", "HCAP", "(H)"]):
                    race.is_handicap = True

            # Trustworthiness Airlock (Success Playbook Item)
            # Skip airlock for sources known to not provide odds (discovery-only adapters)
            skip_trust_check = race.metadata.get("provides_odds") is False
            if skip_trust_check:
                valid_odds_count = sum(
                    1 for r in active_runners
                    if isinstance(r.win_odds, (int, float)) and r.win_odds > 0
                )
                if valid_odds_count < 2:
                    self.logger.debug("Skipping race: provides_odds=False and fewer than 2 runners with valid odds", race_id=race.id)
                    continue

            if total_active > 0 and not skip_trust_check:
                trustworthy_count = sum(1 for r in active_runners if r.metadata.get("odds_source_trustworthy"))
                if trustworthy_count / total_active < TRUSTWORTHY_RATIO_MIN:
                    log.warning("Not enough trustworthy odds for Trifecta; skipping", venue=race.venue, race=race.race_number, ratio=round(trustworthy_count/total_active, 2))
                    continue

            # Uniform Odds Check
            all_odds = []
            for runner in active_runners:
                odds = _get_best_win_odds(runner)
                if odds: all_odds.append(odds)

            if len(all_odds) >= 3 and len(set(all_odds)) == 1:
                log.warning("Race contains uniform odds; likely placeholder. Skipping Trifecta.", venue=race.venue, race=race.race_number)
                continue

            score = self._evaluate_race(race)
            if score > 0:
                race.qualification_score = score
                qualified_races.append(race)

        qualified_races.sort(key=lambda r: r.qualification_score, reverse=True)

        criteria = {
            "max_field_size": self.max_field_size,
            "min_favorite_odds": float(self.min_favorite_odds),
            "min_second_favorite_odds": float(self.min_second_favorite_odds),
        }

        log.info(
            "Universal scoring complete",
            total_races_scored=len(qualified_races),
            criteria=criteria,
        )

        for race in qualified_races:
            if race.qualification_score and race.qualification_score >= 85:
                self.notifier.notify_qualified_race(race)

        return {"criteria": criteria, "races": qualified_races}

    def _evaluate_race(self, race: Race) -> float:
        """Evaluates a single race and returns a qualification score."""
        # --- Constants for Scoring Logic ---
        FAV_ODDS_NORMALIZATION = 10.0
        SEC_FAV_ODDS_NORMALIZATION = 15.0
        FAV_ODDS_WEIGHT = 0.6
        SEC_FAV_ODDS_WEIGHT = 0.4
        FIELD_SIZE_SCORE_WEIGHT = 0.3
        ODDS_SCORE_WEIGHT = 0.7

        active_runners = [r for r in race.runners if not r.scratched]

        runners_with_odds = []
        for runner in active_runners:
            best_odds = _get_best_win_odds(runner)
            if best_odds is not None:
                runners_with_odds.append((runner, best_odds))

        if len(runners_with_odds) < 2:
            if len(active_runners) >= 2:
                # If we have runners but no odds, use fallbacks
                favorite_odds = Decimal(str(DEFAULT_ODDS_FALLBACK))
                second_favorite_odds = Decimal(str(DEFAULT_ODDS_FALLBACK))
            else:
                return 0.0
        else:
            runners_with_odds.sort(key=lambda x: x[1])
            favorite_odds = runners_with_odds[0][1]
            second_favorite_odds = runners_with_odds[1][1]

        # --- Calculate Qualification Score (as inspired by the TypeScript Genesis) ---
        # --- Apply hard filters before scoring ---
        if (
            len(active_runners) > self.max_field_size
            or favorite_odds < Decimal("2.0")
            or favorite_odds < self.min_favorite_odds
            or second_favorite_odds < self.min_second_favorite_odds
        ):
            return 0.0

        field_score = (self.max_field_size - len(active_runners)) / self.max_field_size

        # Normalize odds scores - cap influence of extremely high odds
        fav_odds_score = min(float(favorite_odds) / FAV_ODDS_NORMALIZATION, 1.0)
        sec_fav_odds_score = min(float(second_favorite_odds) / SEC_FAV_ODDS_NORMALIZATION, 1.0)

        # Weighted average
        odds_score = (fav_odds_score * FAV_ODDS_WEIGHT) + (sec_fav_odds_score * SEC_FAV_ODDS_WEIGHT)
        field_score = max(0.0, field_score)
        final_score = (field_score * FIELD_SIZE_SCORE_WEIGHT) + (odds_score * ODDS_SCORE_WEIGHT)
        # To be safe:
        score = round(final_score * 100, 2)
        # BUG-CR-09: Removed redundant assignment to race.qualification_score
        return score


class TinyFieldTrifectaAnalyzer(TrifectaAnalyzer):
    """A specialized TrifectaAnalyzer that only considers races with 6 or fewer runners."""

    def __init__(self, **kwargs):
        # Override the max_field_size to 6 for "tiny field" analysis
        # Set low odds thresholds to "let them through" as per user request
        super().__init__(max_field_size=6, min_favorite_odds=0.01, min_second_favorite_odds=0.01, **kwargs)

    @property
    def name(self) -> str:
        return "tiny_field_trifecta_analyzer"


# Discipline-specific qualification thresholds (Phase J: Strategic Option B)
DISCIPLINE_THRESHOLDS: Final[Dict[str, Dict[str, float]]] = {
    "Thoroughbred": {
        "min_gap12": 0.75,
        "min_odds": 3.0,
        "max_odds": 15.0,
        "min_field_size": 5,
        "max_field_size": 14,
    },
    "Greyhound": {
        "min_gap12": 1.00,  # Stricter requirement for dominant plays
        "min_odds": 3.5,
        "max_odds": 7.0,    # Narrow value window
        "min_field_size": 5,
        "max_field_size": 6,
    },
    "Harness": {
        "min_gap12": 0.75,
        "min_odds": 3.0,
        "max_odds": 10.0,
        "min_field_size": 5,
        "max_field_size": 12,
    },
}

# Regional exclusion list for scoring (Council of Superbrains Directive)
INVALID_REGION_PREFIXES: Final[Tuple[str, ...]] = ('fr', 'uae')
BLOCKED_VENUES: Final[Set[str]] = {
    'fontainebleau', 'cagnessurmer', 'longchamp', 'chantilly', 'deauville',
    'parislongchamp', 'saintcloud', 'compiegne', 'vichy', 'clairefontaine',
    'marseilleborely', 'toulouse', 'lyon', 'strasbourg', 'amiens',
    'meydan', 'abudhabi', 'jebelali',
    'milan', 'sanrossore', 'capannelle',
}


def get_discipline_threshold(discipline: str, key: str) -> float:
    """Helper to retrieve threshold based on normalized discipline name."""
    d = (discipline or "Thoroughbred").title()
    if "Grey" in d or "Hound" in d:
        norm_d = "Greyhound"
    elif "Harness" in d:
        norm_d = "Harness"
    else:
        norm_d = "Thoroughbred"
    return DISCIPLINE_THRESHOLDS.get(norm_d, DISCIPLINE_THRESHOLDS["Thoroughbred"]).get(key, 0.0)


def _classify_race_profile(
    total_active: int,
    sec_fav_odds: float,
    fav_odds: float,
    distance: str | None,
    race_type: str | None,
) -> dict:
    """
    Encode validated dimensional conditions from historical analysis.
    Returns a dict of boolean flags and a human-readable profile tag.

    Validated slices (PowerQuery_For_Hobby.csv, n=8,708):

    ┌─────────────────────────────────────┬───────┬──────────┬─────────┐
    │ Slice                               │  n    │ FvP ROI  │ Hit%    │
    ├─────────────────────────────────────┼───────┼──────────┼─────────┤
    │ r9 × 6f × low-purse  (v0)           │  152  │  +2.5 %  │  74.3 % │
    │ r8 × mid-purse × route              │   74  │  –       │  79.5 % │
    │ r10 × 6f × mid-purse (v3)           │   50  │  +4.0 %  │  62.0 % │
    │ r10 × sprint × mid-purse            │   91  │  +2.3 %  │  59.3 % │
    │ Fav2Group 6.0–6.5  (all runners)    │  695  │  varies  │  varies │
    └─────────────────────────────────────┴───────┴──────────┴─────────┘
    """
    # ── Distance parsing ──────────────────────────────────────────────
    dist_str   = (distance or "").lower().strip()
    is_sprint  = False   # Under 7 furlongs
    is_six_fur = False   # Exactly ~6f (0.70–0.80 miles)

    # Accept common formats: "6f", "6 furlongs", "1200m", "6.0f", "0.75m"
    fur_match = None
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*f(?:url)?", dist_str)
    if m:
        fur_match = float(m.group(1))
    elif re.search(r"(\d+(?:\.\d+)?)\s*m(?:i|le)?", dist_str):
        # Extract miles value specifically from the pattern (Hardening Fix)
        mile_match = re.search(r"(\d+(?:\.\d+)?)\s*m", dist_str)
        if mile_match:
            miles = float(mile_match.group(1))
            fur_match = miles * 8.0
    elif re.search(r"(\d+)\s*m\b", dist_str):          # metres
        metre_match = re.search(r"(\d+)\s*m\b", dist_str)
        if metre_match:
            metres = float(metre_match.group(1))
            fur_match = metres / 201.168

    if fur_match is not None:
        is_sprint  = fur_match < 7.0
        is_six_fur = 5.5 <= fur_match <= 6.5   # 6f ±0.5f tolerance

    # ── Fav2Group tier ────────────────────────────────────────────────
    fg_high    = sec_fav_odds >= 6.0            # dominant positive zone
    fg_mid     = 5.0 <= sec_fav_odds < 6.0     # selective positives
    fg_low     = sec_fav_odds < 5.0            # mostly negative

    # ── Field-size groups ─────────────────────────────────────────────
    in_goldmine_field = 7 <= total_active <= 11   # validated Goldmine range
    in_small_field    = 5 <= total_active <= 6    # 55-family sweet spot
    in_mid_field      = 8 <= total_active <= 10   # r8/r9/r10 focused slices

    # ── Validated sweet-spot combos ───────────────────────────────────
    #   r9 × 6f × sec_fav >= 4.5  (FvP hit 74 %, FAV_PS hit 76 %)
    r9_six_fur = (total_active == 9 and is_six_fur and sec_fav_odds >= 4.5)

    #   r10 × 6f × sec_fav >= 4.5  (FvP +4 %, TopTwoW +11 %)
    r10_six_fur = (total_active == 10 and is_six_fur and sec_fav_odds >= 4.5)

    #   r8 × route × sec_fav >= 4.5  (FvW +5.4 %, hit 60 %)
    r8_route = (total_active == 8 and not is_sprint and sec_fav_odds >= 4.5)

    #   55-family / Sup1x55 fires hard: small field + sec_fav >= 6.0
    sup55_zone = (in_small_field and sec_fav_odds >= 6.0)

    # ── Profile tag ───────────────────────────────────────────────────
    if sec_fav_odds >= 6.0 and in_small_field:
        profile = "sup55_prime"          # best 55-family superfecta zone
    elif r9_six_fur:
        profile = "r9_6f_sweet_spot"     # highest validated hit rate
    elif r10_six_fur:
        profile = "r10_6f_sweet_spot"
    elif r8_route:
        profile = "r8_route_sweet_spot"
    elif sec_fav_odds >= 6.0 and in_goldmine_field:
        profile = "goldmine_high_fav2"   # Fav2Group 6+ standard Goldmine
    elif sec_fav_odds >= 6.0:
        profile = "high_fav2"
    elif in_goldmine_field and sec_fav_odds >= 4.5:
        profile = "goldmine_standard"
    else:
        profile = "standard"

    return {
        "is_sprint":         is_sprint,
        "is_six_fur":        is_six_fur,
        "furlongs":          round(fur_match, 1) if fur_match is not None else None,
        "fg_high":           fg_high,
        "fg_mid":            fg_mid,
        "fg_low":            fg_low,
        "in_goldmine_field": in_goldmine_field,
        "in_small_field":    in_small_field,
        "in_mid_field":      in_mid_field,
        "r9_six_fur":        r9_six_fur,
        "r10_six_fur":       r10_six_fur,
        "r8_route":          r8_route,
        "sup55_zone":        sup55_zone,
        "profile":           profile,
    }


class SimplySuccessAnalyzer(BaseAnalyzer):
    """
    Core qualification engine for Fortuna.
    Implements the 'Simply Success' playbook strategy:
    - Target favourite (optimal risk/reward for Place betting)
    - Require significant odds gap (gap_abs)
    - Detect 'Goldmines' where favourite has dominant value
    """

    def qualify_races(self, races, now=None):
        """
        Core qualification engine — Simply Success playbook.
        Recalibrated against 8,708-race historical validation (Mar 2024–Jul 2025).
        """
        qualified = []
        if now is None:
            now = now_eastern()

        analysis_cfg = self.config.get("analysis", {})
        TRUSTWORTHY_RATIO_MIN = analysis_cfg.get("simply_success_trust_min", 0.25)
        NON_CHALK_MIN         = analysis_cfg.get("non_chalk_min", 0.90)
        SEC_FAV_FLOOR         = analysis_cfg.get("second_fav_floor", 4.50)
        GAP_ABS_MIN           = analysis_cfg.get("gap_abs_min", 0.75)

        # Goldmine configuration
        GOLDMINE_FAV_MIN      = analysis_cfg.get("goldmine_fav_min", 1.10)
        GOLDMINE_SEC_FAV_MIN  = analysis_cfg.get("goldmine_sec_fav_min", 4.50)
        GOLDMINE_GAP_MIN      = analysis_cfg.get("goldmine_gap_min", 2.00)
        GOLDMINE_FIELD_MIN    = analysis_cfg.get("goldmine_field_min", 5)
        # [CHANGE 1] raised from 10 → 11: r11 fires well in all validated slices
        GOLDMINE_FIELD_MAX    = analysis_cfg.get("goldmine_field_max", 11)

        # [CHANGE 5] Superfecta trigger: sec_fav ≥ 6.0 (not gap-only).
        # Data shows the superfecta edge lives almost entirely in Fav2Group 6.0+.
        SUPERFECTA_SEC_FAV_MIN = analysis_cfg.get("superfecta_sec_fav_min", 6.00)

        # [CHANGE 3] Sprint / 6-furlong bonus thresholds
        R9_SIX_FUR_BONUS   = analysis_cfg.get("r9_six_fur_bonus", 4.0)
        R10_SIX_FUR_BONUS  = analysis_cfg.get("r10_six_fur_bonus", 2.5)
        R8_ROUTE_BONUS     = analysis_cfg.get("r8_route_bonus", 2.0)

        # [CHANGE 2] Graduated second-favourite composite bonuses
        SEC_FAV_BONUS_HIGH = analysis_cfg.get("sec_fav_bonus_high", 6.0)
        SEC_FAV_BONUS_MID  = analysis_cfg.get("sec_fav_bonus_mid", 3.0)
        SEC_FAV_BONUS_LOW  = analysis_cfg.get("sec_fav_bonus_low", 1.0)

        fingerprints = {}

        for race in races:
            canonical_venue = get_canonical_venue(race.venue)
            if any(canonical_venue.startswith(p) for p in INVALID_REGION_PREFIXES) \
                    or canonical_venue in BLOCKED_VENUES:
                self.logger.info("Skipping race in untested region",
                                 venue=race.venue, canonical=canonical_venue)
                continue

            active_runners = [r for r in race.runners if not r.scratched]
            total_active   = len(active_runners)

            min_field = get_discipline_threshold(race.discipline, "min_field_size")
            max_field = get_discipline_threshold(race.discipline, "max_field_size")
            if total_active < min_field or (max_field > 0 and total_active > max_field):
                self.logger.debug("Skipping race: field size outside discipline limits",
                                  venue=race.venue, size=total_active, disc=race.discipline)
                continue

            # Trustworthiness airlock
            skip_trust_check = race.metadata.get("provides_odds") is False
            if skip_trust_check:
                valid_odds_count = sum(
                    1 for r in active_runners
                    if isinstance(r.win_odds, (int, float)) and r.win_odds > 0
                )
                if valid_odds_count < 2:
                    continue
            elif total_active > 0:
                trustworthy_count = sum(
                    1 for r in active_runners if r.metadata.get("odds_source_trustworthy")
                )
                if trustworthy_count / total_active < TRUSTWORTHY_RATIO_MIN:
                    continue

            all_valid_with_odds = sorted(
                [(r, odds) for r in active_runners
                 if (odds := _get_best_win_odds(r)) is not None],
                key=lambda x: x[1],
            )

            seen_runner_names = set()
            valid_r_with_odds = []
            for r, odds in all_valid_with_odds:
                name_key = (r.name or "").lower().strip()
                if name_key not in seen_runner_names:
                    seen_runner_names.add(name_key)
                    valid_r_with_odds.append((r, odds))

            if len(valid_r_with_odds) < 2:
                if 4 <= total_active <= 5:
                    race.metadata["qualification_grade"] = "B+ (Override)"
                    race.metadata["composite_score"]     = 45.0
                    race.metadata["tip_tier"]            = "you_might_like"
                    race.metadata["is_best_bet"]         = False
                    race.metadata["is_goldmine"]         = False
                    race.metadata["gap_abs"]             = 0.0
                    qualified.append(race)
                continue

            seen_nums  = set()
            top_nums   = []
            for r, o in valid_r_with_odds:
                n = r.number
                if n and n not in seen_nums:
                    seen_nums.add(n)
                    top_nums.append(str(n))
                if len(top_nums) >= 5:
                    break
            race.top_five_numbers = ", ".join(top_nums)

            fav          = valid_r_with_odds[0][0]
            fav_odds     = float(valid_r_with_odds[0][1])
            sec_fav_odds = float(valid_r_with_odds[1][1])

            if fav_odds < 0.10:        continue
            if fav_odds < NON_CHALK_MIN: continue
            if sec_fav_odds < SEC_FAV_FLOOR: continue

            race.metadata["selection_number"]    = fav.number
            race.metadata["selection_name"]      = fav.name
            race.metadata["predicted_fav_odds"]  = fav_odds

            if (len(valid_r_with_odds) >= 3
                    and len(set(o for r, o in valid_r_with_odds[:3])) == 1):
                continue

            active_content = [(r.name, str(r.win_odds)) for r in race.runners if not r.scratched]
            content_fp     = (canonical_venue, frozenset(active_content))
            if content_fp in fingerprints:
                continue
            fingerprints[content_fp] = 1

            gap_abs = sec_fav_odds - fav_odds
            if gap_abs <= GAP_ABS_MIN:
                continue

            # ── Validated race profile (new helper) ───────────────────
            profile = _classify_race_profile(
                total_active=total_active,
                sec_fav_odds=sec_fav_odds,
                fav_odds=fav_odds,
                distance=race.distance,
                race_type=race.race_type,
            )
            race.metadata["race_profile"]   = profile["profile"]
            race.metadata["race_furlongs"]  = profile["furlongs"]
            race.metadata["is_sprint"]      = profile["is_sprint"]

            # ── Goldmine detection ────────────────────────────────────
            is_goldmine = (
                fav_odds     >= GOLDMINE_FAV_MIN     and
                sec_fav_odds >= GOLDMINE_SEC_FAV_MIN and
                gap_abs      >= GOLDMINE_GAP_MIN     and
                GOLDMINE_FIELD_MIN <= total_active <= GOLDMINE_FIELD_MAX  # [CHANGE 1]
            )

            is_best_bet       = False
            is_superfecta_key = False
            tip_tier          = "best_bet"

            if is_goldmine:
                fav_sources = set()
                if fav.odds:
                    for source, data in fav.odds.items():
                        win = (data.get("win") if isinstance(data, dict)
                               else getattr(data, "win", data))
                        if is_valid_odds(win):
                            fav_sources.add(source)

                is_multi_source = len(fav_sources) >= 2
                race.metadata["is_goldmine_multi_source"] = is_multi_source

                if is_multi_source:
                    self.logger.info(
                        "🌟 High-Confidence Goldmine discovered! 🌟",
                        venue=race.venue, race=race.race_number,
                        selection=fav.name, sources=list(fav_sources),
                        profile=profile["profile"],
                    )
                    race.metadata["goldmine_confidence"] = "high"
                    race.metadata["goldmine_sources"]    = list(fav_sources)
                    race.metadata["is_goldmine"]         = True
                    # [CHANGE 8] Diamond: multi-source AND runners 7–11 (was ≥ 8)
                    if 7 <= total_active <= 11:
                        race.metadata["success_tier"] = "Diamond"
                    else:
                        race.metadata["success_tier"] = "Platinum"
                else:
                    self.logger.info(
                        "✨ Emerging Goldmine (single-source).",
                        venue=race.venue, race=race.race_number,
                        selection=fav.name, sources=list(fav_sources),
                    )
                    race.metadata["goldmine_confidence"] = "low"
                    race.metadata["goldmine_sources"]    = list(fav_sources)
                    race.metadata["success_tier"]        = "Gold"

            # ── Composite scoring ─────────────────────────────────────
            composite = 45.0

            # Gap contribution
            gap_weight = (3.0 if is_goldmine
                          and race.metadata.get("goldmine_confidence") == "high"
                          else 2.5)
            composite += min(gap_abs, 8.0) * gap_weight

            # [CHANGE 4] Field-size tiers — r11 is not neutral, it's in the
            # same validated band as r9/r10. Removed the old neutral/penalty
            # for 11-12 and the sub-minimum penalty for <5.
            if total_active < 5:
                composite -= 10.0              # too small for exotic payouts
            elif 5 <= total_active <= 8:
                composite += 8.0               # small-to-mid sweet spot
            elif 9 <= total_active <= 11:
                composite += 4.0               # [CHANGE 4] r9/r10/r11 all equal
            # 12+ gets no bonus and no penalty — insufficient data

            # Favourite odds quality
            if 2.00 <= fav_odds <= 4.00:
                composite += 5.0
            elif 1.50 <= fav_odds < 2.00 or 4.00 < fav_odds <= 6.00:
                composite += 2.0
            elif fav_odds < 1.50:
                composite -= 3.0

            # [CHANGE 2] Graduated second-favourite bonus.
            # Old code: flat +3 if sec_fav >= 7.0.
            # New: stepped bonuses aligned with the validated performance tiers.
            # Data shows the cliff is at 6.0, not 7.0.
            if sec_fav_odds >= 6.0:
                composite += SEC_FAV_BONUS_HIGH  # dominant positive zone in data
            elif sec_fav_odds >= 5.0:
                composite += SEC_FAV_BONUS_MID   # selective positive zone
            elif sec_fav_odds >= 4.5:
                composite += SEC_FAV_BONUS_LOW   # mostly negative but SEC_FAV_FLOOR passes

            # [CHANGE 3] Sprint / 6-furlong sweet-spot bonus.
            # r9 × 6f achieves 74 % FvP hit rate; r10 × 6f achieves 62 %.
            # Only applies where sec_fav_odds >= 4.5 (data gate).
            if profile["r9_six_fur"]:
                composite += R9_SIX_FUR_BONUS    # strongest validated single slice
            elif profile["r10_six_fur"]:
                composite += R10_SIX_FUR_BONUS
            elif profile["r8_route"] and not profile["is_sprint"]:
                composite += R8_ROUTE_BONUS      # r8 route FvW +5.4 %

            # Discipline-specific gap floor
            min_gap = get_discipline_threshold(race.discipline, "min_gap12")
            if gap_abs < min_gap:
                composite -= 20.0
                is_goldmine = False

            # Grade → tip tier
            if composite >= 60:
                race.metadata["qualification_grade"] = "A+"
                is_best_bet = True
                tip_tier    = "best_bet"
            elif composite >= 52:
                race.metadata["qualification_grade"] = "A"
                is_best_bet = True
                tip_tier    = "best_bet"
            elif composite >= 45:
                race.metadata["qualification_grade"] = "B+"
                is_best_bet = False
                tip_tier    = "you_might_like"
            else:
                race.metadata["qualification_grade"] = "D"
                is_goldmine = False
                is_best_bet = False
                continue

            # [CHANGE 5] Superfecta key: triggered by sec_fav ≥ 6.0 (not gap).
            # Historical data: superfecta ROI positive almost exclusively when
            # Fav2Group ≥ 6.0.  Gap alone is insufficient — tight-fav/long-field
            # races look similar by gap but have no superfecta edge.
            if sec_fav_odds >= SUPERFECTA_SEC_FAV_MIN:
                is_superfecta_key = True
                race.metadata["superfecta_key_number"]  = fav.number
                race.metadata["superfecta_key_name"]    = fav.name
                race.metadata["superfecta_box_numbers"] = [
                    str(r[0].number) for r in valid_r_with_odds[1:4]
                ]

            # Final metadata
            race.metadata["composite_score"]       = round(composite, 2)
            race.metadata["is_goldmine"]           = is_goldmine
            race.metadata["is_best_bet"]           = is_best_bet
            race.metadata["tip_tier"]              = tip_tier
            race.metadata["gap_abs"]               = round(gap_abs, 4)
            race.metadata["is_superfecta_key"]     = is_superfecta_key
            race.metadata["predicted_2nd_fav_odds"] = sec_fav_odds
            race.metadata["market_depth"]          = float(len(valid_r_with_odds))

            cond_mod = 1.0
            if race.is_handicap:               cond_mod += 0.10
            if total_active <= 6:              cond_mod -= 0.10
            elif total_active >= 10:           cond_mod += 0.05
            if gap_abs >= 3.0:                 cond_mod += 0.05
            # [CHANGE 3] Extra reliability for validated 6f profiles
            if profile["r9_six_fur"] or profile["r10_six_fur"]:
                                               cond_mod += 0.05
            race.metadata["condition_modifier"] = round(cond_mod, 2)

            # Place probability (field-size-aware)
            if total_active <= 7:
                p_place = round(min(0.95, 1.8 / max(1.0, fav_odds)), 4)
            else:
                p_place = round(min(0.95, 2.5 / max(1.0, fav_odds)), 4)
            # [CHANGE 3] Validated hit-rate uplift for r9 × 6f × low-purse
            if profile["r9_six_fur"]:
                p_place = min(0.95, p_place * 1.10)   # +10 % relative to base
            race.metadata["place_prob"] = p_place

            place_divisor    = 4.0 if total_active >= 8 else 3.0
            est_place_return = ((fav_odds - 1.0) / place_divisor) + 1.0
            race.metadata["predicted_ev"] = round(
                (p_place * est_place_return) - 1.0, 4
            )

            race.qualification_score = round(composite, 2)
            qualified.append(race)

        return {
            "criteria": {
                "mode":                   "simply_success",
                "non_chalk_min":          NON_CHALK_MIN,
                "sec_fav_floor":          SEC_FAV_FLOOR,
                "gap_abs_min":            GAP_ABS_MIN,
                "goldmine_fav_min":       GOLDMINE_FAV_MIN,
                "goldmine_sec_fav_min":   GOLDMINE_SEC_FAV_MIN,
                "goldmine_gap_min":       GOLDMINE_GAP_MIN,
                "goldmine_field_max":     GOLDMINE_FIELD_MAX,          # [CHANGE 1]
                "superfecta_sec_fav_min": SUPERFECTA_SEC_FAV_MIN,      # [CHANGE 5]
                "r9_six_fur_bonus":       R9_SIX_FUR_BONUS,
                "r10_six_fur_bonus":      R10_SIX_FUR_BONUS,
                "r8_route_bonus":         R8_ROUTE_BONUS,
                "sec_fav_bonus_high":     SEC_FAV_BONUS_HIGH,
                "sec_fav_bonus_mid":      SEC_FAV_BONUS_MID,
                "sec_fav_bonus_low":      SEC_FAV_BONUS_LOW,
            },
            "races": qualified,
        }

class AnalyzerEngine:
    """Discovers and manages all available analyzer plugins."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.analyzers: Dict[str, Type[BaseAnalyzer]] = {}
        self.config = config or {}
        self._discover_analyzers()

    def _discover_analyzers(self):
        # In a real plugin system, this would inspect a folder.
        # For now, we register them manually.
        self.register_analyzer("trifecta", TrifectaAnalyzer)
        self.register_analyzer("tiny_field_trifecta", TinyFieldTrifectaAnalyzer)
        self.register_analyzer("simply_success", SimplySuccessAnalyzer)
        log.info(
            "AnalyzerEngine discovered plugins",
            available_analyzers=list(self.analyzers.keys()),
        )

    def register_analyzer(self, name: str, analyzer_class: Type[BaseAnalyzer]):
        self.analyzers[name] = analyzer_class

    def get_analyzer(self, name: str, **kwargs) -> BaseAnalyzer:
        analyzer_class = self.analyzers.get(name)
        if not analyzer_class:
            log.error("Requested analyzer not found", requested_analyzer=name)
            raise ValueError(f"Analyzer '{name}' not found.")
        return analyzer_class(config=self.config, **kwargs)


class AudioAlertSystem:
    """Plays sound alerts for important events."""

    def __init__(self):
        self.sounds = {
            "high_value": Path(__file__).resolve().parent / "assets" / "sounds" / "alert_premium.wav",
        }
        self.enabled = winsound is not None

    def play(self, sound_type: str):
        if not self.enabled:
            return

        sound_file = self.sounds.get(sound_type)
        if sound_file and sound_file.exists():
            try:
                winsound.PlaySound(str(sound_file), winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception as e:
                log.warning("Could not play sound", file=sound_file, error=e)


class RaceNotifier:
    """Handles sending native notifications and audio alerts for high-value races."""

    def __init__(self):
        self.notifier = DesktopNotifier() if HAS_NOTIFICATIONS else None
        self.audio_system = AudioAlertSystem()
        self.notified_races = set()
        self.notifications_enabled = self.notifier is not None
        if not self.notifications_enabled:
            log.debug("Native notifications disabled (platform not supported or library missing)")

    def notify_qualified_race(self, race):
        if race.id in self.notified_races:
            return

        # Always log the high-value opportunity regardless of notification setting
        log.info(
            "High-value opportunity identified",
            venue=race.venue,
            race=race.race_number,
            score=race.qualification_score
        )

        if not self.notifications_enabled or self.notifier is None:
            return

        title = "🐎 High-Value Opportunity!"
        # Guard against None start_time
        time_str = race.start_time.strftime('%I:%M %p') if race.start_time else "TBD"
        message = f"{race.venue} - Race {race.race_number}\nScore: {race.qualification_score:.0f}%\nPost Time: {time_str}"

        try:
            # Use keyword arguments for better compatibility (AI Review Fix)
            self.notifier.send(
                title=title,
                message=message,
                urgency="high" if race.qualification_score >= 80 else "normal"
            )
            self.notified_races.add(race.id)
            self.audio_system.play("high_value")
            log.info("Notification and audio alert sent for high-value race", race_id=race.id)
        except Exception as e:
            log.error("Failed to send notification", error=str(e))


# ----------------------------------------
def get_track_category(races_at_track: List[Any]) -> str:
    """Categorize the track as T (Thoroughbred), H (Harness), or G (Greyhounds)."""
    if not races_at_track:
        return 'T'

    # Never allow any track with a field size above 7 to be G
    has_large_field = False
    for r in races_at_track:
        runners = get_field(r, 'runners', [])
        active_runners = len([run for run in runners if not get_field(run, 'scratched', False)])
        if active_runners > 7:
            has_large_field = True
            break

    for race in races_at_track:
        source = get_field(race, 'source', '') or ""
        race_id = (get_field(race, 'id', '') or "").lower()
        discipline = get_field(race, 'discipline', '') or ""

        if discipline == "Harness" or '_h' in race_id: return 'H'
        if (discipline == "Greyhound" or '_g' in race_id) and not has_large_field:
            return 'G'

        source_lower = source.lower()
        if ("greyhound" in source_lower or source in ["GBGB", "Greyhound", "AtTheRacesGreyhound"]) and not has_large_field:
            return 'G'
        if source in ["USTrotting", "StandardbredCanada", "Harness"] or any(kw in source_lower for kw in ['harness', 'standardbred', 'trot', 'pace']):
            return 'H'

    # Distance consistency check (Disabled - was mis-identifying Thoroughbred tracks)
    # dist_counts = defaultdict(int)
    # for r in races_at_track:
    #     dist = get_field(r, 'distance')
    #     if dist:
    #         dist_counts[dist] += 1
    # if dist_counts and max(dist_counts.values()) >= 4:
    #     return 'H'

    return 'T'


def generate_fortuna_fives(races: List[Any], all_races: Optional[List[Any]] = None) -> str:
    """Generate the FORTUNA FIVES appendix."""
    lines = ["", "", "FORTUNA FIVES", "-------------"]
    fives = []
    for race in (all_races or races):
        runners = get_field(race, 'runners', [])
        field_size = len([r for r in runners if not get_field(r, 'scratched', False)])
        if field_size == 5:
            fives.append(race)

    if not fives:
        lines.append("No qualifying races.")
        return "\n".join(lines)

    track_odds_sums = defaultdict(float)
    track_odds_counts = defaultdict(int)
    stats_races = all_races if all_races is not None else races
    for race in stats_races:
        v = get_field(race, 'venue')
        track = normalize_venue_name(v)
        for runner in get_field(race, 'runners', []):
            win_odds = get_field(runner, 'win_odds')
            if not get_field(runner, 'scratched') and win_odds:
                track_odds_sums[track] += float(win_odds)
                track_odds_counts[track] += 1

    track_avgs = {}
    for track, total in track_odds_sums.items():
        count = track_odds_counts[track]
        if count > 0:
            track_avgs[track] = str(int(total / count))

    track_to_nums = defaultdict(list)
    for r in fives:
        v = get_field(r, 'venue')
        if v:
            track_to_nums[normalize_venue_name(v)].append(get_field(r, 'race_number'))

    for track in sorted(track_to_nums.keys()):
        nums = sorted(list(set(track_to_nums[track])))
        avg_str = f" [{track_avgs[track]}]" if track in track_avgs else ""
        lines.append(f"{track}{avg_str}: {', '.join(map(str, nums))}")

    return "\n".join(lines)


def generate_field_matrix(races: List[Any]) -> str:
    """
    Generates a Markdown table matrix of races by Track and Field Size.
    Cells contain alphabetic race codes (lowercase=normal, uppercase=goldmine).
    """
    if not races:
        return "No races available for field matrix."

    # Group races by Track and Field Size
    matrix = defaultdict(lambda: defaultdict(list))

    for r in races:
        track = normalize_venue_name(get_field(r, 'venue'))
        field_size = len([run for run in get_field(r, 'runners', []) if not get_field(run, 'scratched', False)])

        # Only interested in field sizes 3-14 for this report
        if 3 <= field_size <= 14:
            is_gold = get_field(r, 'metadata', {}).get('is_goldmine', False)
            race_num = get_field(r, 'race_number')
            matrix[track][field_size].append((race_num, is_gold))

    if not matrix:
        return "No qualifying races for field matrix (3-14 runners)."

    # Header: Display sizes 3 to 14
    display_sizes = range(3, 15)

    header = "| TRACK / FIELD | " + " | ".join(map(str, display_sizes)) + " |"
    separator = "| :--- | " + " | ".join([":---:"] * len(display_sizes)) + " |"
    lines = [header, separator]

    for track in sorted(matrix.keys()):
        row = [track]
        for size in display_sizes:
            race_list = matrix[track].get(size, [])
            if race_list:
                # Standardize formatting of race codes
                code_parts = format_grid_code(race_list, wrap_width=12)
                row.append("<br>".join(code_parts))
            else:
                row.append(" ")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def generate_goldmines(races: List[Any], all_races: Optional[List[Any]] = None) -> str:
    """Generate the GOLDMINE RACES appendix, filtered to Superfecta races."""
    lines = ["", "", "GOLDMINE RACES", "--------------"]

    # Pre-calculate track categories (IMP-CR-03)
    source_races_for_cat = all_races if all_races is not None else races
    track_categories = build_track_categories(source_races_for_cat)

    def is_superfecta_effective(r):
        if get_field(r, 'metadata', {}).get('is_superfecta_key'):
            return True
        available_bets = get_field(r, 'available_bets', [])
        metadata_bets = get_field(r, 'metadata', {}).get('available_bets', [])
        if 'Superfecta' in available_bets or 'Superfecta' in metadata_bets:
            return True

        track = normalize_venue_name(get_field(r, 'venue'))
        cat = track_categories.get(track, 'T')
        runners = get_field(r, 'runners', [])
        field_size = len([run for run in runners if not get_field(run, 'scratched', False)])
        if cat == 'T' and field_size >= 6:
            return True
        return False

    qualified_races = [
        r for r in races
        if (get_field(r, 'metadata', {}).get('is_goldmine') or get_field(r, 'metadata', {}).get('is_superfecta_key'))
        and is_superfecta_effective(r)
    ]

    if not qualified_races:
        lines.append("No qualifying races.")
        return "\n".join(lines)

    track_to_formatted = defaultdict(list)
    for r in qualified_races:
        v = get_field(r, 'venue')
        if v:
            track = normalize_venue_name(v)
            num = get_field(r, 'race_number')
            is_key = get_field(r, 'metadata', {}).get('is_superfecta_key', False)
            label = f"{num}[K]" if is_key else str(num)
            track_to_formatted[track].append((num, label))

    # Sort tracks descending by category (T > H > G)
    cat_map = {'T': 3, 'H': 2, 'G': 1}

    formatted_tracks = []
    for track in track_to_formatted.keys():
        cat = track_categories.get(track, 'T')
        display_name = f"{cat}~{track}"
        formatted_tracks.append((cat, track, display_name))

    # Sort: Category Descending, then Track Name Ascending
    formatted_tracks.sort(key=lambda x: (-cat_map.get(x[0], 0), x[1]))

    for cat, track, display_name in formatted_tracks:
        # Sort by race number then join labels
        entries = sorted(track_to_formatted[track], key=lambda x: x[0])
        labels = [e[1] for e in entries]
        lines.append(f"{display_name}: {', '.join(labels)}")
    return "\n".join(lines)


def generate_goldmine_report(races: List[Any], all_races: Optional[List[Any]] = None) -> str:
    """Generate a detailed report for Goldmine races."""
    # 1. Reuse category logic (IMP-CR-03)
    source_races_for_cat = all_races if all_races is not None else races
    track_categories = build_track_categories(source_races_for_cat)

    def is_superfecta_available(r):
        available_bets = get_field(r, 'available_bets', [])
        metadata_bets = get_field(r, 'metadata', {}).get('available_bets', [])
        if 'Superfecta' in available_bets or 'Superfecta' in metadata_bets:
            return True
        track = normalize_venue_name(get_field(r, 'venue'))
        cat = track_categories.get(track, 'T')
        runners = get_field(r, 'runners', [])
        field_size = len([run for run in runners if not get_field(run, 'scratched', False)])
        return cat == 'T' and field_size >= 6

    # Include all goldmines (favourite >= 4.5)
    # Deduplicate to prevent double-reporting (e.g. from multiple sources)
    goldmines = []
    seen_gold = set()
    for r in races:
        if get_field(r, 'metadata', {}).get('is_goldmine'):
            track = get_canonical_venue(get_field(r, 'venue'))
            num = get_field(r, 'race_number')
            st = get_field(r, 'start_time')
            st_str = st.strftime('%y%m%d') if isinstance(st, datetime) else str(st)
            # Use canonical key for cross-adapter deduplication
            key = (track, num, st_str)
            if key not in seen_gold:
                seen_gold.add(key)
                goldmines.append(r)

    if not goldmines:
        return "No Goldmine races found."

    # Sort goldmines: Cat descending, Track asc, Race num asc
    cat_map = {'T': 3, 'H': 2, 'G': 1}
    def goldmine_sort_key(r):
        track = normalize_venue_name(get_field(r, 'venue'))
        cat = track_categories.get(track, 'T')
        return (-cat_map.get(cat, 0), track, get_field(r, 'race_number', 0))

    goldmines.sort(key=goldmine_sort_key)

    now = datetime.now(EASTERN)
    high_conf_gold = []     # Multi-source validated
    emerging_gold = []      # Single-source only
    remaining_gold = []

    for r in goldmines:
        start_time = get_field(r, 'start_time')
        if isinstance(start_time, str):
            try:
                start_time = from_storage_format(start_time.replace('Z', '+00:00'))
            except ValueError:
                remaining_gold.append(r)
                continue

        if start_time:
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=EASTERN)

            # Classify by multi-source confidence (P2-ENH-2)
            is_multi = get_field(r, 'metadata', {}).get('is_goldmine_multi_source', False)
            if is_multi:
                high_conf_gold.append(r)
            else:
                emerging_gold.append(r)
        else:
            remaining_gold.append(r)

    report_lines = ["LIST OF BEST BETS - GOLDMINE REPORT", "===================================", ""]

    # Tiered rendering (P2-ENH-2)
    def render_races(races_to_render, label, description=None):
        if not races_to_render:
            return
        report_lines.append(f"--- {label.upper()} ---")
        if description:
            report_lines.append(f"({description})")
        report_lines.append("-" * (len(label) + 8))
        report_lines.append("")

        for r in races_to_render:
            track = normalize_venue_name(get_field(r, 'venue'))
            cat = track_categories.get(track, 'T')
            race_num = get_field(r, 'race_number')
            start_time = get_field(r, 'start_time')
            if isinstance(start_time, datetime):
                # Ensure it's in Eastern for the display
                st_eastern = to_eastern(start_time)
                time_str = st_eastern.strftime("%H:%M ET")
            else:
                time_str = str(start_time)

            # Identify Top 5
            runners = get_field(r, 'runners', [])
            active_with_odds = []
            for run in runners:
                if get_field(run, 'scratched'): continue
                wo = _get_best_win_odds(run)
                if wo: active_with_odds.append((run, wo))

            sorted_by_odds = sorted(active_with_odds, key=lambda x: x[1])
            top_5_nums = ", ".join([str(get_field(run[0], 'number') or '?') for run in sorted_by_odds[:5]])
            if hasattr(r, 'top_five_numbers'):
                r.top_five_numbers = top_5_nums

            gap_abs = get_field(r, 'metadata', {}).get('gap_abs', 0.0)
            conf = get_field(r, 'metadata', {}).get('goldmine_confidence', 'low')
            tier = get_field(r, 'metadata', {}).get('success_tier', 'Gold')
            tier_emoji = {'Diamond': '💎💎💎', 'Platinum': '💎💎', 'Gold': '💎'}.get(tier, '💎')
            conf_icon = "💎" if conf == 'high' else "🔍"

            report_lines.append(f"{tier_emoji} {tier.upper()} | {cat}~{track} - Race {race_num} ({time_str})")
            report_lines.append(f"{conf_icon} PREDICTED TOP 5: [{top_5_nums}] | gap_abs: {gap_abs:.2f}")
            # Superfecta Keybox annotation
            if get_field(r, 'metadata', {}).get('is_superfecta_key'):
                key_num  = get_field(r, 'metadata', {}).get('superfecta_key_number', '?')
                box_nums = get_field(r, 'metadata', {}).get('superfecta_box_numbers', [])
                box_str  = ", ".join(str(n) for n in box_nums) if box_nums else "?"
                report_lines.append(f"🗝️  SUPERFECTA KEYBOX: #{key_num} [KEY] → #{box_str} [BOX 2-3-4]")
            report_lines.append("-" * 40)

            # Sort runners by number
            sorted_runners = sorted(runners, key=lambda x: get_field(x, 'number') or 0)

            for run in sorted_runners:
                if get_field(run, 'scratched'):
                    continue
                name = get_field(run, 'name')
                num = get_field(run, 'number')
                odds = get_field(run, 'win_odds')
                odds_str = f"{odds:.2f}" if odds else "N/A"
                report_lines.append(f"  #{num:<2} {name:<25}  ~ {odds_str}")

            report_lines.append("")

    render_races(high_conf_gold, "HIGH-CONFIDENCE GOLDMINES",
                 "Multi-source validated - Strong consensus on favorite value")
    render_races(emerging_gold, "EMERGING GOLDMINES",
                 "Single-source only - Monitor for odds confirmation")
    render_races(remaining_gold, "OTHER GOLDMINES")

    return "\n".join(report_lines)


def generate_historical_goldmine_report(audited_tips: List[Dict[str, Any]]) -> str:
    """Generate a report for recently audited Goldmine races."""
    if not audited_tips:
        return ""

    lines = ["", "RECENT AUDITED GOLDMINES", "------------------------"]

    # Calculate simple stats
    total = len(audited_tips)
    cashed = sum(1 for t in audited_tips if t.get("verdict") == "CASHED")
    total_profit = sum((t.get("net_profit") or 0.0) for t in audited_tips)
    sr = (cashed / total * 100) if total > 0 else 0

    lines.append(f"Performance Summary (Last {total} Goldmines):")
    lines.append(f"  Strike Rate: {sr:.1f}% | Total Net Profit: ${total_profit:+.2f}")
    lines.append("")

    for tip in audited_tips:
        venue = tip.get("venue", "Unknown")
        race_num = tip.get("race_number", "?")
        verdict = tip.get("verdict", "?")
        profit = tip.get("net_profit", 0.0)
        start_time_raw = tip.get("start_time", "")

        try:
            st = from_storage_format(start_time_raw.replace('Z', '+00:00'))
            # Use YYMMDD format as per system-wide overhaul
            time_str = to_eastern(st).strftime("%y%m%dT%H:%M ET")
        except Exception:
            time_str = str(start_time_raw)[:16]

        emoji = "✅" if verdict == "CASHED" else "❌" if verdict == "BURNED" else "⚪"

        line = f"{emoji} {time_str} | {venue} R{race_num} | {verdict:<6} | Profit: ${profit:+.2f}"

        # Add top place payouts for proof
        p1 = tip.get("top1_place_payout")
        p2 = tip.get("top2_place_payout")
        if p1 or p2:
            line += f" | Place: {p1 or 0:.2f}/{p2 or 0:.2f}"

        # Prioritize Superfecta info to "prove" with payouts
        super_payout = tip.get("superfecta_payout")
        tri_payout = tip.get("trifecta_payout")

        if super_payout:
            line += f" | Super: ${super_payout:.2f}"
        elif tri_payout:
            line += f" | Tri: ${tri_payout:.2f}"

        lines.append(line)

    return "\n".join(lines)


def generate_next_to_jump(races: List[Any]) -> str:
    """Generate the NEXT TO JUMP section."""
    lines = ["", "", "NEXT TO JUMP", "------------"]
    now = datetime.now(EASTERN)
    upcoming = []
    for r in races:
        r_time = get_field(r, 'start_time')
        if isinstance(r_time, str):
            try:
                r_time = from_storage_format(r_time.replace('Z', '+00:00'))
            except ValueError:
                continue

        if r_time:
            if r_time.tzinfo is None:
                r_time = r_time.replace(tzinfo=EASTERN)
            if r_time > now:
                upcoming.append((r, r_time))

    if upcoming:
        next_r, next_r_time = min(upcoming, key=lambda x: x[1])
        diff = next_r_time - now
        minutes = int(diff.total_seconds() / 60)
        lines.append(f"{normalize_venue_name(get_field(next_r, 'venue'))} Race {get_field(next_r, 'race_number')} in {minutes}m")
    else:
        lines.append("All races complete for today.")

    return "\n".join(lines)


def generate_adapter_health_report(harvest_summary: Dict[str, Dict]) -> str:
    """Generate a human-readable adapter health dashboard."""
    lines = []
    lines.append('')
    lines.append('═' * 72)
    lines.append('  🚀 ADAPTER HEALTH DASHBOARD — READY FOR SUCCESS!')
    lines.append('═' * 72)

    succeeded = []
    failed = []
    blocked = []

    for name, stats in sorted(harvest_summary.items()):
        count = stats.get('count', 0)
        max_odds = stats.get('max_odds', 0)
        error = str(stats.get('error', ''))
        status = str(stats.get('status', ''))

        if count > 0:
            succeeded.append((name, count, max_odds))
        elif any(kw in error.lower() + status.lower() for kw in
                 ['bot', 'captcha', 'cloudflare', '403', 'challenge', 'blocked']):
            blocked.append((name, error or status))
        else:
            failed.append((name, error or status or 'no data'))

    if succeeded:
        lines.append(f'\n  \u2705 PRODUCING DATA ({len(succeeded)} adapters)')
        lines.append(f'  {"Adapter":<30} {"Races":>6} {"Max Odds":>10}')
        u_line1 = "\u2500" * 30
        u_line2 = "\u2500" * 6
        u_line3 = "\u2500" * 10
        lines.append(f'  {u_line1} {u_line2} {u_line3}')
        total_races = 0
        for name, count, odds in sorted(succeeded, key=lambda x: -x[1]):
            lines.append(f'  {name:<30} {count:>6} {odds:>10.2f}')
            total_races += count
        lines.append(f'  {"TOTAL":<30} {total_races:>6}')

    if blocked:
        lines.append(f'\n  \U0001F6AB BLOCKED ({len(blocked)} adapters)')
        for name, err in blocked:
            lines.append(f'  {name:<30} {err[:40]}')

    if failed:
        lines.append(f'\n  \u274C FAILED ({len(failed)} adapters)')
        for name, err in failed:
            lines.append(f'  {name:<30} {err[:40]}')

    total = len(harvest_summary)
    lines.append(f'\n  \U0001F4CA {len(succeeded)}/{total} adapters producing data '
                 f'({len(blocked)} blocked, {len(failed)} failed)')
    lines.append('=' * 72)

    return '\n'.join(lines)


async def generate_friendly_html_report(races: List[Any], stats: Dict[str, Any]) -> str:
    """Generates a high-impact, friendly HTML report for the Fortuna Faucet."""
    now_str = datetime.now(EASTERN).strftime(' %H:%M:%S')

    # 1. Best Bet Opportunities
    rows = []
    for r in sorted(races, key=lambda x: getattr(x, 'start_time', '')):
        # Get selection (favourite)
        runners = getattr(r, 'runners', [])
        active = [run for run in runners if not getattr(run, 'scratched', False)]
        if len(active) < 2: continue

        # IMP-CR-02: Sort runners using _get_best_win_odds to handle multi-source data
        # Mapping back to float for sorting and metadata access
        with_best_odds = []
        for run in active:
            best = _get_best_win_odds(run)
            with_best_odds.append((run, float(best) if best else 999.0))

        with_best_odds.sort(key=lambda x: x[1])
        sel = with_best_odds[0][0] # Favourite is index 0 (Target for Place betting)

        st = getattr(r, 'start_time', '')
        if isinstance(st, datetime):
            # Ensure it's in Eastern for display (Capability Improvement)
            st_str = to_eastern(st).strftime('%H:%M')
        elif isinstance(st, str):
            try:
                dt = from_storage_format(st.replace('Z', '+00:00'))
                st_str = to_eastern(dt).strftime('%H:%M')
            except Exception:
                s_st = str(st)
                st_str = s_st[11:16] if len(s_st) >= 16 else "??"
        else:
            s_st = str(st)
            st_str = s_st[11:16] if len(s_st) >= 16 else "??"

        is_gold = getattr(r, 'metadata', {}).get('is_goldmine', False)
        gold_badge = '<span class="badge gold">GOLD</span>' if is_gold else ''
        is_superfecta_key = getattr(r, 'metadata', {}).get('is_superfecta_key', False)
        key_badge = '<span class="badge key">KEY</span>' if is_superfecta_key else ''

        d_str = '??/??'
        if isinstance(st, datetime):
            d_str = st.strftime(DATE_FORMAT)
        elif isinstance(st, str):
            try:
                dt = from_storage_format(st.replace('Z', '+00:00'))
                d_str = dt.strftime(DATE_FORMAT)
            except Exception: pass

        rows.append(f"""
            <tr>
                <td>{st_str} ({d_str})</td>
                <td>{getattr(r, 'venue', 'Unknown')}</td>
                <td>R{getattr(r, 'race_number', '?')}</td>
                <td>#{getattr(sel, 'number', '?')} {getattr(sel, 'name', 'Unknown')}</td>
                <td>{ float(_get_best_win_odds(sel) or 0.0):.2f}</td>
                <td>{gold_badge}{key_badge}</td>
            </tr>
        """)

    tips_count = stats.get('tips', 0)
    cashed_count = stats.get('cashed', 0)
    profit = stats.get('profit', 0.0)

    # 2. Goldmine Spotlight Cards (P2-ENH-3)
    goldmine_cards = []
    for r in sorted(races, key=lambda x: getattr(x, 'start_time', '')):
        if not getattr(r, 'metadata', {}).get('is_goldmine'):
            continue

        is_high_conf = getattr(r, 'metadata', {}).get('is_goldmine_multi_source', False)
        conf_class = "high-conf" if is_high_conf else ""
        conf_label = "HIGH CONFIDENCE" if is_high_conf else "EMERGING"

        fav_odds = getattr(r, 'metadata', {}).get('predicted_fav_odds', 0.0)
        sec_odds = getattr(r, 'metadata', {}).get('predicted_2nd_fav_odds', 0.0)
        gap = getattr(r, 'metadata', {}).get('gap_abs', 0.0)

        # Get favorite name
        runners = getattr(r, 'runners', [])
        active = [run for run in runners if not getattr(run, 'scratched', False)]
        # IMP-CR-02: Sort active runners using best win odds
        with_best_odds = []
        for run in active:
            best = _get_best_win_odds(run)
            with_best_odds.append((run, float(best) if best else 999.0))
        with_best_odds.sort(key=lambda x: x[1])

        sel_name = with_best_odds[0][0].name if with_best_odds else "Unknown"
        sel_num = with_best_odds[0][0].number if with_best_odds else "?"

        st = getattr(r, 'start_time', '')
        if isinstance(st, datetime):
            st_str = to_eastern(st).strftime('%H:%M')
        else:
            st_str = "??:??"

        tier = r.metadata.get('success_tier', 'Gold')
        tier_emoji = {'Diamond': '💎💎💎', 'Platinum': '💎💎', 'Gold': '💎'}.get(tier, '💎')
        tier_color = {'Diamond': '#00ffff', 'Platinum': '#e5e4e2', 'Gold': '#ffd700'}.get(tier, '#ffd700')
        tier_badge = f'<span class="badge" style="background-color: {tier_color}; color: #0f172a; margin-right: 5px;">{tier_emoji} {tier.upper()}</span>'

        goldmine_cards.append(f"""
            <div class="goldmine-card {conf_class}">
                <div style="float:right;">{tier_badge}<span class="badge gold">{conf_label}</span></div>
                <div class="goldmine-title">💎 #{sel_num} {sel_name}</div>
                <div class="goldmine-venue">{getattr(r, 'venue', 'Unknown')} R{getattr(r, 'race_number', '?')} @ {st_str}</div>
                <div style="display:flex; justify-content:space-between;">
                    <div>
                        <div class="goldmine-odds">{fav_odds:.2f}</div>
                        <div class="goldmine-label">Fav Odds</div>
                    </div>
                    <div>
                        <div class="goldmine-odds">{sec_odds:.2f}</div>
                        <div class="goldmine-label">2nd Fav</div>
                    </div>
                    <div>
                        <div class="goldmine-odds" style="color:#fbbf24;">+{gap:.2f}</div>
                        <div class="goldmine-label">Gap</div>
                    </div>
                </div>
            </div>
        """)

    goldmine_section = ""
    if goldmine_cards:
        goldmine_section = f"""
            <h2>💎 Goldmine Spotlights</h2>
            <div class="goldmine-grid">
                {''.join(goldmine_cards)}
            </div>
        """

    # Build keybox rows
    keybox_rows = []
    for r in sorted(races, key=lambda x: getattr(x, 'start_time', '')):
        if not getattr(r, 'metadata', {}).get('is_superfecta_key'):
            continue
        st = getattr(r, 'start_time', '')
        if isinstance(st, datetime):
            st_str = to_eastern(st).strftime('%H:%M')
        elif isinstance(st, str):
            try:
                dt = from_storage_format(st.replace('Z', '+00:00'))
                st_str = to_eastern(dt).strftime('%H:%M')
            except Exception:
                s_st = str(st)
                st_str = s_st[11:16] if len(s_st) >= 16 else "??"
        else:
            s_st = str(st)
            st_str = s_st[11:16] if len(s_st) >= 16 else "??"

        key_num  = r.metadata.get('superfecta_key_number', '?')
        key_name = r.metadata.get('superfecta_key_name', 'Unknown')
        box_nums = r.metadata.get('superfecta_box_numbers', [])
        box_str  = " / ".join(f"#{n}" for n in box_nums) if box_nums else "?"
        gap_abs    = r.metadata.get('gap_abs', 0.0)
        keybox_rows.append(f"""
            <tr>
                <td>{st_str}</td>
                <td>{getattr(r, 'venue', 'Unknown')}</td>
                <td>R{getattr(r, 'race_number', '?')}</td>
                <td>#{key_num} {key_name}</td>
                <td>{box_str}</td>
                <td>{gap_abs:.2f}</td>
            </tr>
        """)

    keybox_section = ""
    if keybox_rows:
        keybox_section = f"""
            <h2>🗝️ Superfecta Keybox Plays</h2>
            <p style="color:#94a3b8;font-size:13px;">
                Key the favourite in 1st. Box the next 3 runners in 2nd–3rd–4th.
                Triggered when gap_abs &gt; 0.75.
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Time</th><th>Venue</th><th>Race</th>
                        <th>Key (1st)</th><th>Box (2-3-4)</th><th>gap_abs</th>
                    </tr>
                </thead>
                <tbody>{''.join(keybox_rows)}</tbody>
            </table>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fortuna Faucet Intelligence Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; background-color: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }}
            .container {{ max-width: 1000px; margin: 0 auto; background-color: #1e293b; padding: 30px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); border: 1px solid #334155; }}
            h1 {{ color: #fbbf24; text-align: center; text-transform: uppercase; letter-spacing: 3px; border-bottom: 2px solid #fbbf24; padding-bottom: 15px; margin-bottom: 10px; }}
            .welcome-msg {{ text-align: center; color: #4ade80; font-style: italic; margin-bottom: 25px; font-size: 1.1em; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 30px 0; }}
            .stat-card {{ background-color: #334155; padding: 20px; border-radius: 8px; text-align: center; }}
            .stat-value {{ font-size: 24px; font-weight: bold; color: #fbbf24; }}
            .stat-label {{ font-size: 14px; color: #94a3b8; text-transform: uppercase; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th {{ background-color: #334155; color: #fbbf24; text-align: left; padding: 12px; }}
            td {{ padding: 12px; border-bottom: 1px solid #334155; }}
            tr:hover {{ background-color: #334155; }}
            .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
            .gold {{ background-color: #fbbf24; color: #0f172a; }}
            .key {{ background-color: #7c3aed; color: #fff; margin-left: 4px; }}
            .goldmine-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; margin: 30px 0; }}
            .goldmine-card {{ background: linear-gradient(135deg, #1e293b 0%, #334155 100%); border-left: 4px solid #fbbf24; padding: 20px; border-radius: 8px; position: relative; }}
            .goldmine-card.high-conf {{ border-left-color: #4ade80; }}
            .goldmine-title {{ color: #fbbf24; font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
            .goldmine-venue {{ font-size: 14px; color: #94a3b8; margin-bottom: 15px; }}
            .goldmine-odds {{ font-size: 20px; font-weight: bold; color: #f8fafc; }}
            .goldmine-label {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; margin-top: 5px; }}
            .footer {{ margin-top: 40px; text-align: center; font-size: 12px; color: #64748b; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Fortuna Faucet Intelligence</h1>
            <div class="welcome-msg">Hello JB! Ready to discover some incredible racing opportunities together! ✨</div>
            <p style="text-align:center;">Real-time global racing analysis generated at {now_str} ET</p>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{tips_count}</div>
                    <div class="stat-label">Total Selections</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{cashed_count}</div>
                    <div class="stat-label">Recently Audited Wins</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${profit:+.2f}</div>
                    <div class="stat-label">Estimated Profit</div>
                </div>
            </div>

            {goldmine_section}

            <h2>🔥 Best Bet Opportunities</h2>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Venue</th>
                        <th>Race</th>
                        <th>Selection</th>
                        <th>Odds</th>
                        <th>Type</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows) if rows else '<tr><td colspan="6" style="text-align:center;">No immediate opportunities identified.</td></tr>'}
                </tbody>
            </table>

            {keybox_section}

            {await _generate_audit_history_html()}

            <div class="footer">
                <p>Fortuna Faucet Portable App - Sci-Fi Intelligence Edition</p>
                <p style="color: #4ade80;">"Every day is a new opportunity for success. Let's make it a great one!" ✨</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html


async def _generate_audit_history_html() -> str:
    """Generates HTML for recent audited results."""
    db = FortunaDB()
    history = await db.get_all_audited_tips()
    if not history:
        return ""

    # Take latest 15
    history = sorted(history, key=lambda x: x.get('audit_timestamp', ''), reverse=True)[:15]

    rows = []
    for t in history:
        verdict = t.get("verdict", "?")
        emoji = "✅" if verdict == "CASHED" else "❌" if verdict == "BURNED" else "⚪"
        profit = t.get("net_profit", 0.0)
        p_class = "profit-pos" if profit > 0 else "profit-neg" if profit < 0 else ""

        po = t.get("predicted_2nd_fav_odds")
        ao = t.get("actual_2nd_fav_odds")
        odds_str = f"{po or '?':.1f} → {ao or '?':.1f}"

        rows.append(f"""
            <tr>
                <td>{emoji} {verdict}</td>
                <td>{t.get('venue', 'Unknown')}</td>
                <td>R{t.get('race_number', '?')}</td>
                <td>{odds_str}</td>
                <td class="{p_class}">${profit:+.2f}</td>
            </tr>
        """)

    return f"""
        <style>
            .profit-pos {{ color: #4ade80; font-weight: bold; }}
            .profit-neg {{ color: #f87171; }}
        </style>
        <h2 style="margin-top: 40px;">💰 Recent Audit Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Verdict</th>
                    <th>Venue</th>
                    <th>Race</th>
                    <th>Odds (Pred → Act)</th>
                    <th>Net Profit</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    """


def generate_summary_grid(races: List[Any], all_races: Optional[List[Any]] = None) -> str:
    """
    Generates a Markdown table summary of upcoming races.
    Sorted by MTP, ceiling of 18 hours from now.
    """
    now = datetime.now(EASTERN)
    cutoff = now + timedelta(hours=18)

    # 1. Pre-calculate track categories (IMP-CR-03)
    source_races = all_races if all_races is not None else races
    track_categories = build_track_categories(source_races)

    table_races = []
    seen = set()
    for race in (all_races or races):
        st = get_field(race, 'start_time')
        if isinstance(st, str):
            try: st = from_storage_format(st.replace('Z', '+00:00'))
            except Exception: continue
        if st and st.tzinfo is None: st = st.replace(tzinfo=EASTERN)

        # Ceiling of 18 hours, ignore races more than 10 mins past
        if not st or st < now - timedelta(minutes=10) or st > cutoff:
            continue

        track = normalize_venue_name(get_field(race, 'venue')).replace("|", " ")
        canonical_track = get_canonical_venue(get_field(race, 'venue'))
        num = get_field(race, 'race_number')
        # Deduplication key: Use canonical track/num/date
        key = (canonical_track, num, st.strftime('%y%m%d'))
        if key in seen: continue
        seen.add(key)

        mtp = int((st - now).total_seconds() / 60)
        runners = get_field(race, 'runners', [])
        field_size = len([run for run in runners if not get_field(run, 'scratched', False)])
        top5 = getattr(race, 'top_five_numbers', 'N/A')
        gap_abs = get_field(race, 'metadata', {}).get('gap_abs', 0.0)
        is_gold = get_field(race, 'metadata', {}).get('is_goldmine', False)

        table_races.append({
            'mtp': mtp,
            'cat': track_categories.get(track, 'T'),
            'track': track,
            'num': num,
            'field': field_size,
            'top5': top5,
            'gap': gap_abs,
            'gold': '[G]' if is_gold else '',
            'key': '[K]' if get_field(race, 'metadata', {}).get('is_superfecta_key') else ''
        })

    # Sort by MTP
    table_races.sort(key=lambda x: x['mtp'])

    if not table_races:
        return "No upcoming races in the next 18 hours."

    lines = [
        "| MTP | CAT | TRACK | R# | FLD | TOP 5 | GAP | | |",
        "|:---:|:---:|:---|:---:|:---:|:---|:---:|:---:|:---:|"
    ]
    for tr in table_races:
        # Better alignment: leading zero for single digits
        mtp_val = tr['mtp']
        mtp_str = f"{mtp_val:02d}" if 0 <= mtp_val < 10 else str(mtp_val)
        lines.append(f"| {mtp_str}m | {tr['cat']} | {tr['track'][:20]} | {tr['num']} | {tr['field']} | `{tr['top5']}` | {tr['gap']:.2f} | {tr['gold']} | {tr['key']} |")

    return "\n".join(lines)


def normalize_course_name(name: str) -> str:
    if not name:
        return ""
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9\s-]", "", name)
    name = re.sub(r"[\s-]+", "_", name)
    return name


def num_to_alpha(n, is_goldmine=False):
    """Convert race number to alphabetic code. Goldmines are uppercase."""
    if not isinstance(n, int) or n < 1:
        return '?'
    letter = chr(ord('a') + n - 1) if n <= 26 else str(n)
    return letter.upper() if is_goldmine else letter


def wrap_text(text, width):
    """Wrap string into a list of fixed-width segments."""
    if not text:
        return [""]
    return [text[i:i+width] for i in range(0, len(text), width)]


def format_grid_code(race_info_list, wrap_width=4):
    """
    Standardizes the formatting of race code strings for the grid.
    Includes midpoint space for readability if length exceeds 5.

    Args:
        race_info_list: List of (race_num, is_goldmine) tuples
        wrap_width: Width to wrap at
    """
    if not race_info_list:
        return [""]

    code = "".join([num_to_alpha(n, gm) for n, gm in sorted(list(set(race_info_list)))])

    # Midpoint space logic for readability (Project Convention)
    if len(code) > 5:
        mid = len(code) // 2
        code = code[:mid] + " " + code[mid:]

    return wrap_text(code, wrap_width)


def format_predictions_section(qualified_races: List[Race]) -> str:
    """Generates the Predictions & Proof section for the GHA Job Summary (Monospace Grid)."""
    lines = ["### 🔮 Fortuna Predictions & Proof", ""]
    if not qualified_races:
        lines.append("No Goldmine predictions available for this run.")
        return "\n".join(lines)

    now = datetime.now(EASTERN)

    def get_mtp(r):
        st = r.start_time
        if isinstance(st, str):
            try:
                st = from_storage_format(st.replace('Z', '+00:00'))
            except Exception:
                return 9999
        if st and st.tzinfo is None:
            st = st.replace(tzinfo=EASTERN)
        return (st - now).total_seconds() / 60 if st else 9999

    # Sort by MTP ascending
    sorted_races = sorted(qualified_races, key=get_mtp)
    # Take top 10 opportunities
    top_10 = sorted_races[:10]

    lines.append("```text")
    header = f"  {'DATE':<5}  {'VENUE':<18}  {'R#':>2}   {'PICK':<21}  {'ODDS':>6}  {'GAP':>5}  {'GOLD':<4}  {'TOP 5':<5}  PAYOUT PROOF"
    underline = f"  {'─────':<5}  {'──────────────────':<18}  {'──':>2}   {'─────────────────────':<21}  {'──────':>6}  {'─────':>5}  {'────':<4}  {'─────':<5}  {'────────────':<12}"
    lines.append(header)
    lines.append(underline)

    for r in top_10:
        metadata = getattr(r, 'metadata', {})
        st = r.start_time
        if isinstance(st, str):
            try: st = from_storage_format(st.replace('Z', '+00:00'))
            except Exception: st = None
        date_str = st.strftime(DATE_FORMAT) if st else '??/??'

        venue = (r.venue or 'Unknown')[:18]
        rn = str(r.race_number or '?')

        sel_name = metadata.get('selection_name') or "Unknown"
        sel_num = metadata.get('selection_number', '?')
        pick = f"#{sel_num} {sel_name}"[:21]

        odds = metadata.get('predicted_2nd_fav_odds')
        odds_str = f"{odds:>6.2f}" if odds else '   N/A'

        gap = metadata.get('gap_abs', 0.0)
        gap_str = f"{gap:>5.2f}"

        gold = 'GOLD' if metadata.get('is_goldmine') else ' —  '
        top5 = str(getattr(r, 'top_five_numbers', 'TBD'))[:5]

        payouts = []
        # Check both metadata and attributes for payouts
        for label in ('top1_place_payout', 'trifecta_payout', 'superfecta_payout'):
            val = metadata.get(label) or getattr(r, label, None)
            if val:
                display_label = label.replace('_', ' ').title().replace('Top1 ', '')
                payouts.append(f"{display_label}: ${float(val):.2f}")

        payout_text = ' | '.join(payouts) or 'Awaiting Results'
        lines.append(f"  {date_str:<5}  {venue:<18}  {rn:>2}   {pick:<21}  {odds_str}  {gap_str}  {gold}  {top5:<5}  {payout_text}")

    lines.append("```")
    return "\n".join(lines)


async def format_proof_section(db: FortunaDB) -> str:
    """Generates the Recent Audited Proof subsection for the GHA Job Summary."""
    lines = ["", "#### 💰 Recent Audited Proof", ""]
    try:
        # First attempt to get recent goldmines
        tips = await db.get_recent_audited_goldmines(limit=10)
        # Fallback to any audited tips if no goldmines found
        if not tips:
            tips = await db.get_all_audited_tips()
            tips = tips[:10]

        if not tips:
            lines.append("Awaiting race results; nothing audited yet.")
            return "\n".join(lines)

        lines.append("```text")
        header = f"  {'VERDICT':<13}  {'PROFIT':>8}  {'VENUE':<18}  {'R#':>2}   {'ACTUAL TOP 5':<12}  {'ODDS':>6}  PAYOUT DETAILS"
        underline = f"  {'─────────────':<13}  {'────────':>8}  {'──────────────────':<18}  {'──':>2}   {'────────────':<12}  {'──────':>6}  {'──────────────':<14}"
        lines.append(header)
        lines.append(underline)
        for tip in tips:
            payouts = []
            if tip.get('superfecta_payout'):
                payouts.append(f"Super ${tip['superfecta_payout']:.2f}")
            if tip.get('trifecta_payout'):
                payouts.append(f"Tri ${tip['trifecta_payout']:.2f}")
            if tip.get('top1_place_payout'):
                payouts.append(f"Place ${tip['top1_place_payout']:.2f}")

            payout_text = ' / '.join(payouts) if payouts else 'No payout data'

            verdict = tip.get("verdict", "?")
            emoji = "✅" if verdict in ("CASHED", "CASHED_ESTIMATED") else "❌" if verdict == "BURNED" else "⚪"
            profit = tip.get('net_profit', 0.0)
            actual_odds = tip.get('actual_2nd_fav_odds')
            actual_odds_str = f"{actual_odds:>6.2f}" if actual_odds else "   N/A"
            venue = (tip.get('venue') or 'Unknown')[:18]
            rn = str(tip.get('race_number', '?'))
            top5 = (tip.get('actual_top_5') or 'N/A')[:12]

            lines.append(
                f"  {emoji} {verdict:<10}  ${profit:>7.2f}  {venue:<18}  {rn:>2}   {top5:<12}  {actual_odds_str}  {payout_text}"
            )
        lines.append("```")
    except Exception as e:
        lines.append(f"Error generating audited proof: {e}")

    return "\n".join(lines)


def build_harvest_table(summary: Dict[str, Any], title: str) -> str:
    """Generates a harvest performance table for the GHA Job Summary (Monospace)."""
    lines = [f"### {title}", ""]

    lines.append("```text")
    header = f"  {'ADAPTER':<32}  {'RACES':>5}  {'MAX ODDS':>9}  STATUS"
    underline = f"  {'────────────────────────────────':<32}  {'─────':>5}  {'─────────':>9}  ──────────"
    lines.append(header)
    lines.append(underline)

    if not summary:
        lines.append(f"  {'N/A':<32}  {0:>5}  {0.0:>9.1f}  ⚠️ No Data")
        lines.append("```")
        return "\n".join(lines)

    # Sort by Records Found (descending), then alphabetically
    def sort_key(item):
        adapter, data = item
        count = data.get('count', 0) if isinstance(data, dict) else data
        return (-count, adapter)

    sorted_adapters = sorted(summary.items(), key=sort_key)

    for adapter, data in sorted_adapters:
        if isinstance(data, dict):
            count = data.get('count', 0)
            max_odds = data.get('max_odds', 0.0)
        else:
            count = data
            max_odds = 0.0

        status = '✅' if count > 0 else '⚠️ No Data'
        lines.append(f"  {adapter:<32}  {count:>5}  {max_odds:>9.1f}  {status}")

    lines.append("```")
    return "\n".join(lines)


def format_artifact_links() -> str:
    """Generates the report artifacts links for the GHA Job Summary."""
    return '\n'.join([
        "### 📁 Report Artifacts",
        "",
        "- [Summary Grid](summary_grid.txt)",
        "- [Field Matrix](field_matrix.txt)",
        "- [Goldmine Report](goldmine_report.txt)",
        "- [HTML Report](fortuna_report.html)",
        "- [Analytics Log](analytics_report.txt)"
    ])


from contextlib import contextmanager

class SummaryWriter:
    """Stream-based summary writer for file/stdout output. (Fix 15)
    See also: GHASummaryWriter in generate_gha_summary.py for GHA Job Summary output."""
    def __init__(self, stream: TextIO) -> None:
        self._s = stream
    def write(self, text: str = "") -> None:
        self._s.write(text + "\n")
        self._s.flush()
    def lines(self, rows: list[str]) -> None:
        self._s.write("\n".join(rows) + "\n")
        self._s.flush()

@contextmanager
def open_summary():
    """Context manager for writing to GHA Job Summary with fallback to stdout (Optimized)."""
    path = os.environ.get('GITHUB_STEP_SUMMARY')
    if path:
        with open(path, 'a', encoding='utf-8') as f:
            yield SummaryWriter(f)
    else:
        # Fallback to stdout if not in GHA
        yield SummaryWriter(sys.stdout)

def write_job_summary(predictions_md: str, harvest_md: str, proof_md: str, artifacts_md: str) -> None:
    """Writes the consolidated sections to $GITHUB_STEP_SUMMARY using an efficient context manager."""
    with open_summary() as f:
        # Narrate the entire workflow
        summary = '\n'.join([
            predictions_md,
            '',
            harvest_md,
            '',
            proof_md,
            '',
            artifacts_md,
        ])
        try:
            f.write(summary)
        except Exception as e:
            structlog.get_logger().error("job_summary_write_failed", error=str(e))


def get_writable_path(filename: str) -> Path:
    """Returns a writable path for the given filename, using AppData in frozen mode."""
    if is_frozen() and sys.platform == "win32":
        appdata = os.getenv('APPDATA')
        if appdata:
            out_dir = Path(appdata) / "Fortuna"
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / filename
            # Ensure subdirectories within Fortuna folder exist
            target.parent.mkdir(parents=True, exist_ok=True)
            return target
    return Path(filename)


def get_db_path() -> str:
    """Returns the path to the SQLite database, using AppData in frozen mode."""
    return str(get_writable_path("fortuna.db"))


def validate_artifact_freshness(filepath: str, max_age_hours: int = 12) -> bool:
    """Verifies that the given artifact exists and is not too old (Improvement 1)."""
    p = Path(filepath)
    if not p.exists():
        return False
    mtime = p.stat().st_mtime
    age_hours = (time.time() - mtime) / 3600
    return age_hours <= max_age_hours


def _write_github_output(name: str, value: Any) -> None:
    """Writes a value to GitHub Actions output if environment variable is present (Improvement 1)."""
    if 'GITHUB_OUTPUT' in os.environ:
        try:
            with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                f.write(f"{name}={value}\n")
        except Exception:
            pass


class FortunaDB:
    """
    Thread-safe SQLite backend for Fortuna using the standard library.
    Handles persistence for tips, predictions, and audit outcomes.
    """
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or get_db_path()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._conn = None
        self._conn_lock = threading.Lock()

        self._initialized = False
        self.logger = structlog.get_logger(self.__class__.__name__)

    def _get_conn(self):
        """Returns a thread-safe connection using WAL and a thread lock (Requirement)."""
        with self._conn_lock:
            if not self._conn:
                # check_same_thread=False is safe because we use a ThreadPoolExecutor(max_workers=1)
                # and a connection lock for all direct cursor operations.
                self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
                # Enable WAL mode for better concurrency once during initialization
                try:
                    self._conn.execute("PRAGMA journal_mode=WAL")
                except sqlite3.Error:
                    pass
        return self._conn

    @asynccontextmanager
    async def get_connection(self):
        """Returns an async context manager for a database connection."""
        try:
            import aiosqlite
        except ImportError:
            self.logger.error("aiosqlite not installed. Async database features will fail.")
            raise

        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            yield conn

    async def _run_in_executor(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, func, *args)

    async def get_column_population(self, columns: List[str]) -> List[Tuple[str, int, int, Optional[str]]]:
        """Public API to query column usage stats (IMP-CR-05)."""
        def _query():
            conn = self._get_conn()
            total_row = conn.execute("SELECT COUNT(*) FROM tips").fetchone()
            total = total_row[0] if total_row else 0
            db_cols = {row[1] for row in conn.execute("PRAGMA table_info('tips')").fetchall()}
            results = []
            for col in columns:
                if col not in db_cols:
                    results.append((col, 0, total, None))
                    continue
                # Population count
                n_row = conn.execute(f"SELECT COUNT(*) FROM tips WHERE {col} IS NOT NULL AND CAST({col} AS TEXT) != ''").fetchone()
                n = n_row[0] if n_row else 0
                # Recent sample
                sample_row = conn.execute(f"SELECT {col} FROM tips WHERE {col} IS NOT NULL AND CAST({col} AS TEXT) != '' ORDER BY id DESC LIMIT 1").fetchone()
                sample = str(sample_row[0])[:10] if sample_row and sample_row[0] is not None else None
                results.append((col, n, total, sample))
            return results
        return await self._run_in_executor(_query)

    async def initialize(self):
        """Creates the database schema if it doesn't exist."""
        if self._initialized: return

        # Pre-fetch current version to avoid NameError in _init (Hardening Fix)
        def _get_version():
            try:
                cursor = self._get_conn().execute("SELECT MAX(version) FROM schema_version")
                row = cursor.fetchone()
                return row[0] if row and row[0] is not None else 0
            except Exception:
                return 0
        current_version = await self._run_in_executor(_get_version)

        def _init():
            # Force close and reopen to ensure fresh state for migrations (Hardening Fix)
            with self._conn_lock:
                if self._conn:
                    self._conn.close()
                    self._conn = None
            conn = self._get_conn()
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INTEGER PRIMARY KEY,
                        applied_at TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS harvest_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        region TEXT,
                        adapter_name TEXT NOT NULL,
                        race_count INTEGER NOT NULL,
                        max_odds REAL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS tips (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        race_id TEXT NOT NULL,
                        venue TEXT NOT NULL,
                        race_number INTEGER NOT NULL,
                        discipline TEXT,
                        start_time TEXT NOT NULL,
                        report_date TEXT NOT NULL,
                        is_goldmine INTEGER NOT NULL,
                        source TEXT,
                        gap_abs TEXT,
                        top_five TEXT,
                        selection_number INTEGER,
                        selection_name TEXT,
                        audit_completed INTEGER DEFAULT 0,
                        verdict TEXT,
                        net_profit REAL,
                        selection_position INTEGER,
                        actual_top_5 TEXT,
                        actual_2nd_fav_odds REAL,
                        trifecta_payout REAL,
                        trifecta_combination TEXT,
                        superfecta_payout REAL,
                        superfecta_combination TEXT,
                        top1_place_payout REAL,
                        top2_place_payout REAL,
                        predicted_2nd_fav_odds REAL,
                        audit_timestamp TEXT,
                        field_size INTEGER,
                        market_depth REAL,
                        place_prob REAL,
                        predicted_ev REAL,
                        race_type TEXT,
                        condition_modifier REAL,
                        qualification_grade TEXT,
                        composite_score REAL,
                        match_confidence TEXT,
                        is_handicap INTEGER,
                        is_best_bet INTEGER,
                        is_superfecta_key INTEGER DEFAULT 0,
                        superfecta_key_number INTEGER,
                        superfecta_key_name TEXT,
                        daypart TEXT,
                        predicted_fav_odds REAL,
                        tip_tier TEXT,
                        actual_fav_odds REAL
                    )
                """)
                conn.execute('''CREATE TABLE IF NOT EXISTS quarter_harvests (
                    quarter_id TEXT PRIMARY KEY,
                    region TEXT NOT NULL DEFAULT 'GLOBAL',
                    daypart TEXT,
                    card_fetch_started TEXT,
                    card_fetch_completed TEXT,
                    race_count INTEGER DEFAULT 0,
                    snapshot_path TEXT,
                    scoring_runs INTEGER DEFAULT 0,
                    last_scoring_run TEXT,
                    tips_produced INTEGER DEFAULT 0,
                    audit_completed INTEGER DEFAULT 0
                )''')
                # Composite index for deduplication - changed to race_id only for better deduplication
                conn.execute("DROP INDEX IF EXISTS idx_race_report")

                # FIX-01: Dedup migration: run ONCE via schema versioning, not on every init
                if current_version < 8:
                    try:
                        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_race_id ON tips (race_id)")
                    except sqlite3.IntegrityError:
                        # Index creation failed due to existing duplicates — migrate once
                        self.logger.info("Duplicate race_ids detected, running one-time cleanup")
                        conn.execute("""
                            DELETE FROM tips
                            WHERE id NOT IN (
                                SELECT MAX(id)
                                FROM tips
                                GROUP BY race_id
                            )
                        """)
                        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_race_id ON tips (race_id)")
                        self.logger.info("One-time dedup complete, unique index created")
                    except Exception as e:
                        self.logger.error("Failed to create unique index", error=str(e))
                # Declarative column migration (IMP-CR-01)
                EXPECTED_COLUMNS = {
                    "source": "TEXT", "gap_abs": "REAL", "daypart": "TEXT", "top_five": "TEXT",
                    "selection_number": "INTEGER", "selection_name": "TEXT",
                    "audit_completed": "INTEGER DEFAULT 0", "verdict": "TEXT", "net_profit": "REAL",
                    "selection_position": "INTEGER", "actual_top_5": "TEXT", "actual_2nd_fav_odds": "REAL",
                    "trifecta_payout": "REAL", "trifecta_combination": "TEXT",
                    "superfecta_payout": "REAL", "superfecta_combination": "TEXT",
                    "top1_place_payout": "REAL", "top2_place_payout": "REAL",
                    "predicted_2nd_fav_odds": "REAL", "audit_timestamp": "TEXT",
                    "field_size": "INTEGER", "market_depth": "REAL", "place_prob": "REAL",
                    "predicted_ev": "REAL", "race_type": "TEXT", "condition_modifier": "REAL",
                    "qualification_grade": "TEXT", "composite_score": "REAL", "match_confidence": "TEXT",
                    "is_handicap": "INTEGER", "is_best_bet": "INTEGER",
                    "is_superfecta_key": "INTEGER DEFAULT 0", "superfecta_key_number": "INTEGER",
                    "superfecta_key_name": "TEXT", "predicted_fav_odds": "REAL",
                    "tip_tier": "TEXT", "actual_fav_odds": "REAL", "discipline": "TEXT",
                    "goldmine_tier": "TEXT", "goldmine_sources": "INTEGER", "goldmine_gap_confirmed": "INTEGER"
                }
                existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(tips)").fetchall()}
                for col, dtype in EXPECTED_COLUMNS.items():
                    if col not in existing_cols:
                        conn.execute(f"ALTER TABLE tips ADD COLUMN {col} {dtype}")

                # Composite index for audit performance (BUG-CR-10: Added idx_daypart)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_time ON tips (audit_completed, start_time)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_start_time ON tips (start_time)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_venue ON tips (venue)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_discipline ON tips (discipline)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_daypart ON tips (daypart)")

        await self._run_in_executor(_init)

        # Track and execute migrations based on schema version

        if current_version < 2:
            await self.migrate_utc_to_eastern()
            def _update_version():
                with self._get_conn() as conn:
                    conn.execute("INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (2, ?)", (to_storage_format(datetime.now(EASTERN)),))
            await self._run_in_executor(_update_version)
            self.logger.info("Schema migrated to version 2")

        if current_version < 3:
            def _declutter():
                # Delete old records to keep database lean (30-day retention cleanup)
                cutoff = to_storage_format(datetime.now(EASTERN) - timedelta(days=30))
                with self._get_conn() as conn:
                    cursor = conn.execute("DELETE FROM tips WHERE report_date < ?", (cutoff,))
                    self.logger.info("Database decluttered (30-day retention cleanup)", deleted_count=cursor.rowcount)
                    conn.execute("INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (3, ?)", (to_storage_format(datetime.now(EASTERN)),))
            await self._run_in_executor(_declutter)
            self.logger.info("Schema migrated to version 3")

        if current_version < 4:
            # Migration to version 4: Housekeeping & Long-term retention.
            def _housekeeping():
                with self._get_conn() as conn:
                    # v4 was a one-time historical wipe. If we're initializing
                    # a fresh DB, just bump the version without deleting.
                    existing = conn.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
                    if existing > 0 and current_version == 3:
                        self.logger.warning("v4 migration: clearing legacy v3 tips")
                        conn.execute("DELETE FROM tips")
                    conn.execute(
                        "INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (4, ?)",
                        (to_storage_format(datetime.now(EASTERN)),),
                    )
            await self._run_in_executor(_housekeeping)
            self.logger.info("Schema migrated to version 4 (Housekeeping complete, long-term retention enabled)")

        if current_version < 5:
            # Migration to version 5: Scoring signal columns (independent review items)
            def _migrate_v5():
                with self._get_conn() as conn:
                    # Columns already added in initialization PRAGMA check if missing.
                    conn.execute("INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (5, ?)", (to_storage_format(datetime.now(EASTERN)),))
            await self._run_in_executor(_migrate_v5)
            self.logger.info("Schema migrated to version 5 — scoring signal columns added")

        if current_version < 6:
            def _migrate_v6():
                with self._get_conn() as conn:
                    conn.execute("INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (6, ?)", (to_storage_format(datetime.now(EASTERN)),))
            await self._run_in_executor(_migrate_v6)
            self.logger.info("Schema migrated to version 6 — handicap status added")

        if current_version < 7:
            def _migrate_v7():
                with self._get_conn() as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(
                        "SELECT id, race_id, venue, start_time, "
                        "race_number, discipline FROM tips"
                    )
                    rows = cursor.fetchall()
                    updates = []
                    skipped = 0
                    for row in rows:
                        try:
                            old_id = row['race_id']
                            # Extract prefix (e.g., 'RP') or default to 'unk'
                            prefix = old_id.split('_')[0] if '_' in old_id else 'unk'
                            st = from_storage_format(row['start_time'])
                            new_id = generate_race_id(
                                prefix, row['venue'], st,
                                row['race_number'], row['discipline']
                            )
                            if old_id != new_id:
                                updates.append((new_id, row['id']))
                        except Exception as e:
                            skipped += 1
                            self.logger.warning("v7_migration_skip",
                                race_id=row['race_id'] if isinstance(row, sqlite3.Row) else 'unknown',
                                error=str(e))

                    updated = 0
                    deleted = 0
                    for new_id, row_id in updates:
                        try:
                            conn.execute(
                                "UPDATE tips SET race_id = ? WHERE id = ?",
                                (new_id, row_id)
                            )
                            updated += 1
                        except sqlite3.IntegrityError:
                            # If new_id already exists, delete this duplicate record
                            conn.execute(
                                "DELETE FROM tips WHERE id = ?",
                                (row_id,)
                            )
                            deleted += 1

                    self.logger.info("v7_migration_stats",
                        rows_examined=len(rows),
                        updated=updated,
                        deleted_duplicates=deleted,
                        skipped=skipped)

                    conn.execute(
                        "INSERT OR REPLACE INTO schema_version "
                        "(version, applied_at) VALUES (7, ?)",
                        (to_storage_format(datetime.now(EASTERN)),)
                    )
            await self._run_in_executor(_migrate_v7)
            self.logger.info("Schema migrated to version 7 — race_ids re-keyed")

        if current_version < 8:
            def _migrate_v8():
                # v8 handles the unique index migration if not already handled in _init
                with self._get_conn() as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO schema_version "
                        "(version, applied_at) VALUES (8, ?)",
                        (to_storage_format(datetime.now(EASTERN)),)
                    )
            await self._run_in_executor(_migrate_v8)
            self.logger.info("Schema migrated to version 8 — unique index confirmed")

        if current_version < 9:
            def _migrate_v9():
                # v9: Data migration from gap12 to gap_abs (IMP-CR-02)
                with self._get_conn() as conn:
                    # Check if old column exists before migrating
                    existing = {row[1] for row in conn.execute("PRAGMA table_info(tips)").fetchall()}
                    if "gap12" in existing:
                        conn.execute("UPDATE tips SET gap_abs = gap12 WHERE gap_abs IS NULL OR gap_abs = 0.0")
                    conn.execute(
                        "INSERT OR REPLACE INTO schema_version "
                        "(version, applied_at) VALUES (9, ?)",
                        (to_storage_format(datetime.now(EASTERN)),)
                    )
            await self._run_in_executor(_migrate_v9)
            self.logger.info("Schema migrated to version 9 — gap_abs data backfilled")

        self._initialized = True
        self.logger.info("Database initialized", path=self.db_path, schema_version=max(current_version, 9))

    async def is_quarter_fetched(self, quarter_id: str) -> bool:
        def _check():
            conn = self._get_conn()
            row = conn.execute(
                'SELECT 1 FROM quarter_harvests WHERE quarter_id = ? AND card_fetch_completed IS NOT NULL',
                (quarter_id,)
            ).fetchone()
            return row is not None
        return await self._run_in_executor(_check)

    async def log_quarter_fetch(self, quarter_id: str, region: str, daypart: str, race_count: int, snapshot_path: str) -> None:
        def _log():
            conn = self._get_conn()
            now = to_storage_format(now_eastern())
            with conn:
                conn.execute(
                    '''INSERT OR REPLACE INTO quarter_harvests
                       (quarter_id, region, daypart, card_fetch_started, card_fetch_completed,
                        race_count, snapshot_path)
                       VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (quarter_id, region, daypart, now, now, race_count, snapshot_path)
                )
        return await self._run_in_executor(_log)

    async def log_scoring_run(self, quarter_id: str, tips_count: int) -> None:
        def _log():
            conn = self._get_conn()
            now = to_storage_format(now_eastern())
            with conn:
                conn.execute(
                    '''UPDATE quarter_harvests
                       SET scoring_runs = scoring_runs + 1,
                           last_scoring_run = ?,
                           tips_produced = tips_produced + ?
                       WHERE quarter_id = ?''',
                    (now, tips_count, quarter_id)
                )
        return await self._run_in_executor(_log)

    async def merge_databases(self, other_path: str) -> int:
        """Consolidates data from another database file (BUG-CR-11)."""
        def _merge():
            if not os.path.exists(other_path): return 0

            target_conn = self._get_conn()
            source_conn = sqlite3.connect(other_path)

            # Use intersection of columns to be safe against schema drift
            t_cols = {row[1] for row in target_conn.execute("PRAGMA table_info(tips)").fetchall()}
            s_cols = {row[1] for row in source_conn.execute("PRAGMA table_info(tips)").fetchall()}
            shared = sorted(t_cols & s_cols)

            col_str = ", ".join(shared)
            placeholders = ", ".join(["?"] * len(shared))

            rows = source_conn.execute(f"SELECT {col_str} FROM tips").fetchall()
            merged = 0
            with target_conn:
                for row in rows:
                    try:
                        target_conn.execute(f"INSERT OR IGNORE INTO tips ({col_str}) VALUES ({placeholders})", tuple(row))
                        merged += 1
                    except Exception: pass

            source_conn.close()
            return merged

        return await self._run_in_executor(_merge)

    async def get_scored_race_ids(self, daypart_tag: str) -> Set[str]:
        def _get():
            conn = self._get_conn()
            # Primary: Check explicit daypart column
            cursor = conn.execute(
                'SELECT race_id FROM tips WHERE daypart = ?',
                (daypart_tag,)
            )
            results = {row['race_id'] for row in cursor.fetchall()}

            # Fallback: Check report_date range for legacy records (migration period)
            if not results:
                parts = daypart_tag.split('_')
                if len(parts) == 2:
                    daypart_enum, date_str = parts[0], parts[1]
                    try:
                        q_num = int(daypart_enum[1])
                        start_hour = (q_num - 1) * 6
                        q_start = f'{date_str}T{start_hour:02d}:00:00'
                        # FIX-19: Use 23:59:59 as the upper boundary for Q4 queries
                        q_end = f'{date_str}T{min(23, q_num * 6):02d}:59:59' if q_num == 4 else f'{date_str}T{q_num * 6:02d}:00:00'
                        # Use < for upper bound to maintain half-open interval (Fix code review feedback)
                        q_op = '<=' if q_num == 4 else '<'
                        cursor = conn.execute(
                            f'SELECT race_id FROM tips WHERE report_date >= ? AND report_date {q_op} ?',
                            (q_start, q_end)
                        )
                        results.update({row['race_id'] for row in cursor.fetchall()})
                    except (ValueError, IndexError):
                        pass
            return results
        return await self._run_in_executor(_get)

    async def migrate_utc_to_eastern(self) -> None:
        """Migrates existing database records from UTC to US Eastern Time."""
        def _migrate():
            conn = self._get_conn()
            cursor = conn.execute("""
                SELECT id, start_time, report_date, audit_timestamp FROM tips
                WHERE start_time LIKE '%+00:00' OR start_time LIKE '%Z'
                OR report_date LIKE '%+00:00' OR report_date LIKE '%Z'
                OR audit_timestamp LIKE '%+00:00' OR audit_timestamp LIKE '%Z'
            """)
            rows = cursor.fetchall()
            if not rows: return

            total = len(rows)
            self.logger.info("Migrating legacy UTC timestamps to Eastern", count=total)
            converted = 0
            errors = 0

            # Process in chunks of 1000 for safety
            for i in range(0, total, 1000):
                chunk = rows[i:i+1000]
                with conn:
                    for row in chunk:
                        updates = {}
                        for col in ["start_time", "report_date", "audit_timestamp"]:
                            if col not in row.keys(): continue
                            val = row[col]
                            if val:
                                try:
                                    dt = from_storage_format(val.replace("Z", "+00:00"))
                                    dt_eastern = ensure_eastern(dt)
                                    updates[col] = to_storage_format(dt_eastern)
                                except Exception: pass
                        if updates:
                            try:
                                set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
                                conn.execute(f"UPDATE tips SET {set_clause} WHERE id = ?", (*updates.values(), row["id"]))
                                converted += 1
                            except Exception as e:
                                errors += 1
                                self.logger.warning("Failed to migrate row", row_id=row["id"], error=str(e))
                self.logger.info("Migration progress", processed=min(i + 1000, total), total=total)

            self.logger.info("Migration complete", total=total, converted=converted, errors=errors)
        await self._run_in_executor(_migrate)

    async def log_harvest(self, harvest_summary: Dict[str, Any], region: Optional[str] = None):
        """Logs harvest performance metrics to the database."""
        if not self._initialized: await self.initialize()

        def _log():
            conn = self._get_conn()
            now = to_storage_format(datetime.now(EASTERN))
            to_insert = []
            for adapter, data in harvest_summary.items():
                if isinstance(data, dict):
                    count = data.get("count", 0)
                    max_odds = data.get("max_odds", 0.0)
                else:
                    count = data
                    max_odds = 0.0

                to_insert.append((now, region, adapter, count, max_odds))

            if to_insert:
                with conn:
                    conn.executemany("""
                        INSERT INTO harvest_logs (timestamp, region, adapter_name, race_count, max_odds)
                        VALUES (?, ?, ?, ?, ?)
                    """, to_insert)

        await self._run_in_executor(_log)

    async def get_adapter_scores(self, days: int = 30) -> Dict[str, float]:
        """Calculates historical performance scores for each adapter."""
        if not self._initialized: await self.initialize()

        def _get():
            conn = self._get_conn()
            cutoff = to_storage_format(datetime.now(EASTERN) - timedelta(days=days))
            cursor = conn.execute("""
                SELECT adapter_name,
                       AVG(race_count) as avg_count,
                       AVG(max_odds) as avg_max_odds
                FROM harvest_logs
                WHERE timestamp > ?
                GROUP BY adapter_name
            """, (cutoff,))

            scores = {}
            for row in cursor.fetchall():
                # Heuristic: Score = Avg Race Count + (Avg Max Odds * 2)
                # This prioritizes adapters that find races and high longshots
                scores[row["adapter_name"]] = (row["avg_count"] or 0) + ((row["avg_max_odds"] or 0) * 2)
            return scores

        return await self._run_in_executor(_get)

    async def log_tips(self, tips: List[Dict[str, Any]], dedup_window_hours: int = 12):
        """Logs new tips to the database with batch deduplication and scoring updates."""
        if not self._initialized: await self.initialize()

        def _log():
            conn = self._get_conn()
            now = now_eastern()

            # Batch check for recently logged tips to avoid redundant entries
            race_ids = [t.get("race_id") for t in tips if t.get("race_id")]
            if not race_ids: return

            placeholders = ",".join(["?"] * len(race_ids))

            # Use a more absolute check to ensure distinct races across all time
            cursor = conn.execute(
                f"SELECT race_id FROM tips WHERE race_id IN ({placeholders})",
                (*race_ids,)
            )
            already_logged = {row["race_id"] for row in cursor.fetchall()}

            # ── BUG-3 FIX: Cross-race clone detection ────────────────────
            seen_selections = set()
            # FIX-02: Use date-range query instead of fragile LIKE pattern
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1)
            sod_str = to_storage_format(start_of_day)
            eod_str = to_storage_format(end_of_day)
            existing_selections = set()
            try:
                for row in conn.execute(
                    "SELECT venue, race_number, selection_name FROM tips "
                    "WHERE start_time >= ? AND start_time < ? AND selection_name IS NOT NULL",
                    (sod_str, eod_str)
                ).fetchall():
                    v_db = get_canonical_venue(str(row["venue"] or "")).upper()
                    r_db = row["race_number"]
                    s_db = str(row["selection_name"] or "").strip().upper()
                    if v_db and s_db:
                        existing_selections.add((v_db, r_db, s_db))
            except Exception:
                pass

            to_insert = []
            to_update = []
            skipped_clones = 0

            for tip in tips:
                rid = tip.get("race_id")
                if not rid: continue

                # ── BUG-3 FIX: Clone detection ──────────────────────────
                sel_name = str(tip.get("selection_name") or "").strip()
                venue = str(tip.get("venue") or "").strip()
                race_num = tip.get("race_number")
                norm_v = get_canonical_venue(venue).upper()
                if sel_name and norm_v:
                    clone_key = (norm_v, race_num, sel_name.upper())
                    if clone_key in seen_selections or clone_key in existing_selections:
                        self.logger.warning("clone_selection_blocked",
                            race_id=rid, venue=venue,
                            selection=sel_name,
                            msg="Same horse already tipped at this venue/race today")
                        skipped_clones += 1
                        continue
                    seen_selections.add(clone_key)

                # ── BUG-2c FIX: Normalize start_time ────────────────────
                raw_st = tip.get("start_time")
                if isinstance(raw_st, datetime):
                    normalized_st = to_storage_format(ensure_eastern(raw_st))
                elif isinstance(raw_st, str):
                    try:
                        datetime.strptime(raw_st, STORAGE_FORMAT)
                        normalized_st = raw_st
                    except ValueError:
                        try:
                            # Use from_storage_format fallback (Fix 01)
                            dt = from_storage_format(str(raw_st))
                            normalized_st = to_storage_format(ensure_eastern(dt))
                        except (ValueError, TypeError):
                            normalized_st = raw_st
                else:
                    normalized_st = to_storage_format(now) if raw_st is None else str(raw_st)

                # ── BUG-1 Fix: Use passed daypart tag if available
                daypart_val = tip.get("daypart")
                if not daypart_val:
                    try:
                        st_dt = datetime.strptime(normalized_st, STORAGE_FORMAT).replace(tzinfo=EASTERN)
                        daypart_val = resolve_daypart_from_dt(st_dt).value
                    except Exception:
                        daypart_val = None

                report_date = tip.get("report_date") or to_storage_format(now)
                # Prepare elements for INSERT or UPDATE (29 elements)
                data = (
                    rid, tip.get("venue"), tip.get("race_number"),
                    tip.get("discipline"), normalized_st, report_date,
                    1 if tip.get("is_goldmine") else 0,
                    tip.get("source"),
                    str(tip.get("gap_abs", 0.0)),
                    tip.get("top_five"), tip.get("selection_number"), tip.get("selection_name"),
                    float(tip.get("predicted_2nd_fav_odds")) if tip.get("predicted_2nd_fav_odds") is not None else None,
                    tip.get("field_size"),
                    tip.get("market_depth"),
                    tip.get("place_prob"),
                    tip.get("predicted_ev"),
                    tip.get("race_type"),
                    tip.get("condition_modifier"),
                    tip.get("qualification_grade"),
                    tip.get("composite_score"),
                    1 if tip.get("is_handicap") is True else (0 if tip.get("is_handicap") is False else None),
                    1 if tip.get("is_best_bet") else 0,
                    1 if tip.get("is_superfecta_key") else 0,
                    tip.get("superfecta_key_number"),
                    tip.get("superfecta_key_name"),
                    daypart_val,
                    float(tip.get("predicted_fav_odds")) if tip.get("predicted_fav_odds") is not None else None,
                    tip.get("tip_tier"),
                    tip.get("goldmine_tier"),
                    tip.get("goldmine_sources"),
                    tip.get("goldmine_gap_confirmed")
                )

                if rid not in already_logged:
                    to_insert.append(data)
                    already_logged.add(rid) # Avoid duplicates within the same batch
                else:
                    # Update existing record if not audited to refresh scoring/metadata
                    # Shift rid to the end for WHERE clause, exclude rid from SET clause
                    update_tuple = data[1:] + (rid,)
                    to_update.append(update_tuple)

            if to_insert or to_update:
                with conn:
                    if to_insert:
                        conn.executemany("""
                            INSERT INTO tips (
                                race_id, venue, race_number, discipline, start_time, report_date,
                                is_goldmine, source, gap_abs, top_five, selection_number, selection_name, predicted_2nd_fav_odds,
                                field_size, market_depth, place_prob, predicted_ev, race_type,
                                condition_modifier, qualification_grade, composite_score, is_handicap, is_best_bet,
                                is_superfecta_key, superfecta_key_number, superfecta_key_name, daypart,
                                predicted_fav_odds, tip_tier, goldmine_tier, goldmine_sources, goldmine_gap_confirmed
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, to_insert)

                    if to_update:
                        conn.executemany("""
                            UPDATE tips SET
                                venue=?, race_number=?, discipline=?, start_time=?, report_date=?,
                                is_goldmine=?, source=?, gap_abs=?, top_five=?, selection_number=?, selection_name=?,
                                predicted_2nd_fav_odds=?, field_size=?, market_depth=?, place_prob=?,
                                predicted_ev=?, race_type=?, condition_modifier=?, qualification_grade=?,
                                composite_score=?, is_handicap=?, is_best_bet=?,
                                is_superfecta_key=?, superfecta_key_number=?, superfecta_key_name=?, daypart=?,
                                predicted_fav_odds=?, tip_tier=?, goldmine_tier=?, goldmine_sources=?, goldmine_gap_confirmed=?
                            WHERE race_id=? AND audit_completed=0
                        """, to_update)

                self.logger.info("Hot tips processed", inserted=len(to_insert), updated=len(to_update), clones_skipped=skipped_clones)

        await self._run_in_executor(_log)

    async def get_upcoming_tips(self, past_minutes: int = 10, future_hours: int = 18, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns unaudited tips with start_time in the upcoming window (Fix 08)."""
        if not self._initialized:
            await self.initialize()

        def _query():
            conn = self._get_conn()
            now = now_eastern()
            cutoff_past = to_storage_format(now - timedelta(minutes=past_minutes))
            cutoff_future = to_storage_format(now + timedelta(hours=future_hours))
            cursor = conn.execute(
                """SELECT * FROM tips
                   WHERE audit_completed = 0
                   AND start_time > ?
                   AND start_time < ?
                   ORDER BY start_time ASC
                   LIMIT ?""",
                (cutoff_past, cutoff_future, limit)
            )
            return [dict(row) for row in cursor.fetchall()]

        return await self._run_in_executor(_query)

    async def get_unverified_tips(self, lookback_hours: int = 48) -> List[Dict[str, Any]]:
        """Returns tips that haven't been audited yet but have likely finished."""
        if not self._initialized: await self.initialize()

        def _get():
            conn = self._get_conn()
            now = datetime.now(EASTERN)
            cutoff = to_storage_format(now - timedelta(hours=lookback_hours))

            cursor = conn.execute(
                """SELECT * FROM tips
                   WHERE audit_completed = 0
                   AND report_date > ?
                   AND start_time < ?""",
                (cutoff, to_storage_format(now))
            )
            return [dict(row) for row in cursor.fetchall()]
        return await self._run_in_executor(_get)

    async def get_recent_tips(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns the most recent tips regardless of audit status, ordered by discovery time."""
        if not self._initialized: await self.initialize()
        def _get():
            # Use ID DESC to show most recently discovered tips first
            cursor = self._get_conn().execute(
                "SELECT * FROM tips ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
        return await self._run_in_executor(_get)

    async def update_audit_result(self, race_id: str, outcome: Dict[str, Any]):
        """Updates a single tip with its audit outcome."""
        if not self._initialized: await self.initialize()

        def _update():
            conn = self._get_conn()
            with conn:
                conn.execute("""
                    UPDATE tips SET
                        audit_completed = 1,
                        verdict = ?,
                        net_profit = ?,
                        selection_position = ?,
                        actual_top_5 = ?,
                        actual_2nd_fav_odds = ?,
                        trifecta_payout = ?,
                        trifecta_combination = ?,
                        superfecta_payout = ?,
                        superfecta_combination = ?,
                        top1_place_payout = ?,
                        top2_place_payout = ?,
                        audit_timestamp = ?,
                        match_confidence = ?,
                        field_size = COALESCE(field_size, ?),
                        actual_fav_odds = ?
                    WHERE id = (
                        SELECT id FROM tips
                        WHERE race_id = ? AND audit_completed = 0
                        LIMIT 1
                    )
                """, (
                    outcome.get("verdict"), outcome.get("net_profit"),
                    outcome.get("selection_position"), outcome.get("actual_top_5"),
                    outcome.get("actual_2nd_fav_odds"), outcome.get("trifecta_payout"),
                    outcome.get("trifecta_combination"),
                    outcome.get("superfecta_payout"),
                    outcome.get("superfecta_combination"),
                    outcome.get("top1_place_payout"),
                    outcome.get("top2_place_payout"),
                    to_storage_format(now_eastern()),
                    outcome.get("match_confidence") or "none",
                    outcome.get("field_size"),
                    outcome.get("actual_fav_odds"),
                    race_id
                ))
        await self._run_in_executor(_update)

    async def update_audit_results_batch(self, outcomes: List[Tuple[str, Dict[str, Any]]]):
        """Updates multiple tips with their audit outcomes in a single transaction."""
        if not outcomes: return
        if not self._initialized: await self.initialize()

        def _update():
            conn = self._get_conn()
            with conn:
                for race_id, outcome in outcomes:
                    conn.execute("""
                        UPDATE tips SET
                            audit_completed = 1,
                            verdict = ?,
                            net_profit = ?,
                            selection_position = ?,
                            actual_top_5 = ?,
                            actual_2nd_fav_odds = ?,
                            trifecta_payout = ?,
                            trifecta_combination = ?,
                            superfecta_payout = ?,
                            superfecta_combination = ?,
                            top1_place_payout = ?,
                            top2_place_payout = ?,
                            audit_timestamp = ?,
                            match_confidence = ?,
                            field_size = COALESCE(field_size, ?),
                            actual_fav_odds = ?
                        WHERE id = (
                            SELECT id FROM tips
                            WHERE race_id = ? AND audit_completed = 0
                            LIMIT 1
                        )
                    """, (
                        outcome.get("verdict"), outcome.get("net_profit"),
                        outcome.get("selection_position"), outcome.get("actual_top_5"),
                        outcome.get("actual_2nd_fav_odds"), outcome.get("trifecta_payout"),
                        outcome.get("trifecta_combination"),
                        outcome.get("superfecta_payout"),
                        outcome.get("superfecta_combination"),
                        outcome.get("top1_place_payout"),
                        outcome.get("top2_place_payout"),
                        outcome.get("audit_timestamp") or to_storage_format(now_eastern()),
                        outcome.get("match_confidence") or "none",
                        outcome.get("field_size"),
                        outcome.get("actual_fav_odds"),
                        race_id
                    ))
        await self._run_in_executor(_update)

    async def get_all_audited_tips(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Returns audited tips for reporting. Pass limit=N for recent only."""
        if not self._initialized:
            await self.initialize()

        def _get():
            if limit:
                cursor = self._get_conn().execute(
                    "SELECT * FROM tips WHERE audit_completed = 1 ORDER BY audit_timestamp DESC, start_time DESC LIMIT ?",
                    (limit,),
                )
            else:
                cursor = self._get_conn().execute(
                    "SELECT * FROM tips WHERE audit_completed = 1 ORDER BY audit_timestamp DESC, start_time DESC"
                )
            return [dict(row) for row in cursor.fetchall()]

        return await self._run_in_executor(_get)

    async def get_recent_audited_goldmines(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Returns recent successfully audited goldmine tips."""
        if not self._initialized: await self.initialize()
        def _get():
            cursor = self._get_conn().execute(
                "SELECT * FROM tips WHERE audit_completed = 1 AND is_goldmine = 1 ORDER BY start_time DESC LIMIT ?",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
        return await self._run_in_executor(_get)

    async def get_tips(self, audited: Optional[bool] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Returns tips from the database with optional filtering and limiting."""
        if not self._initialized: await self.initialize()
        def _get():
            query = "SELECT * FROM tips"
            params = []
            if audited is not None:
                query += " WHERE audit_completed = ?"
                params.append(1 if audited else 0)

            # Sort audited by audit_timestamp, unaudited by start_time
            if audited is True:
                query += " ORDER BY audit_timestamp DESC"
            else:
                query += " ORDER BY start_time ASC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor = self._get_conn().execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        return await self._run_in_executor(_get)

    async def get_harvest_logs(self, limit: int = 200, hours: Optional[int] = None) -> List[Dict[str, Any]]:
        """Returns recent harvest logs."""
        if not self._initialized: await self.initialize()
        def _get():
            if hours:
                cutoff = to_storage_format(datetime.now(EASTERN) - timedelta(hours=hours))
                cursor = self._get_conn().execute(
                    "SELECT * FROM harvest_logs WHERE timestamp >= ? ORDER BY id DESC LIMIT ?", (cutoff, limit)
                )
            else:
                cursor = self._get_conn().execute(
                    "SELECT * FROM harvest_logs ORDER BY id DESC LIMIT ?", (limit,)
                )
            return [dict(row) for row in cursor.fetchall()]
        return await self._run_in_executor(_get)

    async def get_harvest_metrics(self, hours: int = 24) -> Dict[str, Dict[str, Any]]:
        """Calculates harvest metrics (avg count, trust ratio) per adapter."""
        if not self._initialized: await self.initialize()
        def _get():
            conn = self._get_conn()
            cutoff = to_storage_format(datetime.now(EASTERN) - timedelta(hours=hours))

            cursor = conn.execute("""
                SELECT adapter_name,
                       AVG(race_count) as avg_count,
                       MAX(max_odds) as peak_odds
                FROM harvest_logs
                WHERE timestamp >= ?
                GROUP BY adapter_name
            """, (cutoff,))

            return {row[0]: {"avg_count": row[1], "peak_odds": row[2]} for row in cursor.fetchall()}
        return await self._run_in_executor(_get)

    async def get_stats(self) -> Dict[str, Any]:
        """Returns aggregate statistics for reporting."""
        if not self._initialized: await self.initialize()
        def _get():
            conn = self._get_conn()
            stats = {}
            row = conn.execute("SELECT COUNT(*) FROM tips").fetchone()
            stats['total_tips'] = row[0] if row else 0

            # Count best bets (A/A+)
            row = conn.execute("SELECT COUNT(*) FROM tips WHERE is_best_bet = 1 OR qualification_grade IN ('A', 'A+')").fetchone()
            stats['total_best_bets'] = row[0] if row else 0

            row = conn.execute("SELECT COUNT(*) FROM tips WHERE audit_completed = 1 AND verdict IN ('CASHED', 'CASHED_ESTIMATED')").fetchone()
            stats['cashed'] = row[0] if row else 0

            row = conn.execute("SELECT COUNT(*) FROM tips WHERE audit_completed = 1 AND verdict = 'BURNED'").fetchone()
            stats['burned'] = row[0] if row else 0

            row = conn.execute("SELECT COUNT(*) FROM tips WHERE audit_completed = 1 AND verdict = 'VOID'").fetchone()
            stats['voided'] = row[0] if row else 0

            row = conn.execute("SELECT SUM(net_profit) FROM tips WHERE audit_completed = 1").fetchone()
            stats['total_profit'] = row[0] if row and row[0] is not None else 0.0

            row = conn.execute("SELECT MAX(report_date) FROM tips").fetchone()
            stats['max_report_date'] = row[0] if row and row[0] is not None else None

            # For BUG-10 verification
            row = conn.execute("SELECT COUNT(*) FROM tips WHERE qualification_grade IS NOT NULL AND qualification_grade != ''").fetchone()
            stats['populated_scoring_count'] = row[0] if row and row[0] is not None else 0

            # Get deep builder analytics (Picks, Wins, Profit per source)
            # source can be comma-separated list of names
            cursor = conn.execute("""
                SELECT source, verdict, net_profit, is_best_bet, qualification_grade
                FROM tips
            """)

            builder_stats = {}
            for s_str, verdict, profit, is_bb, grade in cursor.fetchall():
                if not s_str: continue
                # Split and clean sources
                names = [n.strip() for n in s_str.split(",")]
                is_high_qual = bool(is_bb) or (grade in ('A', 'A+'))

                for name in names:
                    if name not in builder_stats:
                        builder_stats[name] = {
                            "total": 0, "cashed": 0, "burned": 0, "profit": 0.0,
                            "bb_total": 0, "bb_cashed": 0, "bb_profit": 0.0
                        }
                    b = builder_stats[name]
                    b["total"] += 1
                    if is_high_qual:
                        b["bb_total"] += 1

                    if verdict in ('CASHED', 'CASHED_ESTIMATED'):
                        b["cashed"] += 1
                        b["profit"] += (profit or 0.0)
                        if is_high_qual:
                            b["bb_cashed"] += 1
                            b["bb_profit"] += (profit or 0.0)
                    elif verdict == 'BURNED':
                        b["burned"] += 1
                        b["profit"] += (profit or 0.0)
                        if is_high_qual:
                            b["bb_profit"] += (profit or 0.0)

            stats['builder_analytics'] = builder_stats
            # Legacy bridge for GHA summary
            stats['best_bet_builders'] = {k: v['bb_total'] for k, v in builder_stats.items() if v['bb_total'] > 0}

            # Lifetime avg payout (used for breakeven)
            row = conn.execute("SELECT AVG(COALESCE(net_profit, 0.0) + 2.0) FROM tips WHERE verdict IN ('CASHED', 'CASHED_ESTIMATED')").fetchone()
            stats['lifetime_avg_payout'] = row[0] if row and row[0] is not None else 0.0

            # Goldmine Performance Stats (P2-ENH-5)
            row = conn.execute("SELECT COUNT(*) FROM tips WHERE is_goldmine = 1").fetchone()
            stats['total_goldmines'] = row[0] if row else 0

            row = conn.execute("SELECT COUNT(*) FROM tips WHERE is_goldmine = 1 AND audit_completed = 1 AND verdict IN ('CASHED', 'CASHED_ESTIMATED')").fetchone()
            stats['goldmines_cashed'] = row[0] if row else 0

            row = conn.execute("SELECT SUM(net_profit) FROM tips WHERE is_goldmine = 1 AND audit_completed = 1").fetchone()
            stats['goldmine_profit'] = row[0] if row and row[0] is not None else 0.0

            return stats

    async def get_goldmine_stats(self) -> Dict[str, Any]:
        """Get goldmine-specific performance statistics."""
        def _get():
            conn = self._get_conn()
            gap_filter = "AND gap_abs IS NOT NULL AND gap_abs != '' AND CAST(gap_abs AS REAL) > 0"

            total  = conn.execute('SELECT COUNT(*) FROM tips WHERE is_goldmine=1 AND audit_completed=1').fetchone()[0]
            cashed = conn.execute("SELECT COUNT(*) FROM tips WHERE is_goldmine=1 AND audit_completed=1 AND verdict IN ('CASHED','CASHED_ESTIMATED')").fetchone()[0]
            profit = conn.execute('SELECT SUM(net_profit) FROM tips WHERE is_goldmine=1 AND audit_completed=1').fetchone()[0] or 0.0
            avg_gap= conn.execute(f'SELECT AVG(CAST(gap_abs AS REAL)) FROM tips WHERE is_goldmine=1 {gap_filter}').fetchone()[0] or 0.0
            sf_total = conn.execute('SELECT COUNT(*) FROM tips WHERE is_goldmine=1 AND is_superfecta_key=1 AND audit_completed=1').fetchone()[0]
            sf_hits  = conn.execute("SELECT COUNT(*) FROM tips WHERE is_goldmine=1 AND is_superfecta_key=1 AND audit_completed=1 AND superfecta_payout > 0").fetchone()[0]
            avg_sf   = conn.execute('SELECT AVG(superfecta_payout) FROM tips WHERE is_goldmine=1 AND superfecta_payout > 0').fetchone()[0] or 0.0

            gap_tiers = {}
            for lo, hi, label in [(2.0, 3.0, '2-3'), (3.0, 5.0, '3-5'), (5.0, 99.0, '5+')]:
                t = conn.execute(f'SELECT COUNT(*) FROM tips WHERE is_goldmine=1 AND audit_completed=1 {gap_filter} AND CAST(gap_abs AS REAL)>=? AND CAST(gap_abs AS REAL)<?', (lo, hi)).fetchone()[0]
                c = conn.execute(f"SELECT COUNT(*) FROM tips WHERE is_goldmine=1 AND audit_completed=1 {gap_filter} AND CAST(gap_abs AS REAL)>=? AND CAST(gap_abs AS REAL)<? AND verdict IN ('CASHED','CASHED_ESTIMATED')", (lo, hi)).fetchone()[0]
                p = conn.execute(f'SELECT SUM(net_profit) FROM tips WHERE is_goldmine=1 AND audit_completed=1 {gap_filter} AND CAST(gap_abs AS REAL)>=? AND CAST(gap_abs AS REAL)<?', (lo, hi)).fetchone()[0] or 0.0
                if t > 0:
                    gap_tiers[label] = {'total': t, 'cashed': c, 'strike_rate': c/t*100, 'profit': p}

            tier_stats = {}
            try:
                for name in ['Diamond', 'Platinum', 'Gold']:
                    t = conn.execute('SELECT COUNT(*) FROM tips WHERE is_goldmine=1 AND audit_completed=1 AND goldmine_tier=?', (name,)).fetchone()[0]
                    c = conn.execute("SELECT COUNT(*) FROM tips WHERE is_goldmine=1 AND audit_completed=1 AND goldmine_tier=? AND verdict IN ('CASHED','CASHED_ESTIMATED')", (name,)).fetchone()[0]
                    p = conn.execute('SELECT SUM(net_profit) FROM tips WHERE is_goldmine=1 AND audit_completed=1 AND goldmine_tier=?', (name,)).fetchone()[0] or 0.0
                    if t > 0:
                        tier_stats[name] = {'total': t, 'cashed': c, 'strike_rate': c/t*100, 'profit': p}
            except Exception:
                pass

            return {
                'total': total, 'cashed': cashed,
                'strike_rate': (cashed/total*100) if total else 0,
                'profit': profit, 'avg_gap': avg_gap,
                'superfecta_total': sf_total, 'superfecta_hits': sf_hits, 'avg_sf_payout': avg_sf,
                'gap_tiers': gap_tiers, 'tier_stats': tier_stats,
            }
        return await self._run_in_executor(_get)

    async def clear_all_tips(self):
        """Wipes all records from the tips table."""
        if not self._initialized: await self.initialize()
        def _clear():
            conn = self._get_conn()
            with conn:
                conn.execute("DELETE FROM tips")
                conn.execute("DELETE FROM harvest_logs")
            conn.execute("VACUUM")
            self.logger.info("Database cleared (all tips and logs deleted)")
        await self._run_in_executor(_clear)

    async def migrate_from_json(self, json_path: str = "hot_tips_db.json"):
        """Migrates data from existing JSON file to SQLite with detailed error logging."""
        path = Path(json_path)
        if not path.exists(): return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if not isinstance(data, list): return
            self.logger.info("Migrating data from JSON", count=len(data))
            if not self._initialized: await self.initialize()

            def _migrate():
                conn = self._get_conn()
                success_count = 0
                for entry in data:
                    try:
                        with conn:
                            conn.execute("""
                                INSERT OR IGNORE INTO tips (
                                    race_id, venue, race_number, start_time, report_date,
                                    is_goldmine, gap_abs, top_five, selection_number,
                                    audit_completed, verdict, net_profit, selection_position,
                                    actual_top_5, actual_2nd_fav_odds, trifecta_payout,
                                    trifecta_combination, superfecta_payout,
                                    superfecta_combination, top1_place_payout,
                                    top2_place_payout, audit_timestamp
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                entry.get("race_id"), entry.get("venue"), entry.get("race_number"),
                                entry.get("start_time"), entry.get("report_date"),
                                1 if entry.get("is_goldmine") else 0, str(entry.get("gap_abs", 0.0)),
                                entry.get("top_five"), entry.get("selection_number"),
                                1 if entry.get("audit_completed") else 0, entry.get("verdict"),
                                entry.get("net_profit"), entry.get("selection_position"),
                                entry.get("actual_top_5"), entry.get("actual_2nd_fav_odds"),
                                entry.get("trifecta_payout"), entry.get("trifecta_combination"),
                                entry.get("superfecta_payout"), entry.get("superfecta_combination"),
                                entry.get("top1_place_payout"), entry.get("top2_place_payout"),
                                entry.get("audit_timestamp")
                            ))
                        success_count += 1
                    except Exception as e:
                        self.logger.error("Failed to migrate entry", race_id=entry.get("race_id"), error=str(e))
                return success_count

            count = await self._run_in_executor(_migrate)
            self.logger.info("Migration complete", successful=count)
        except Exception as e:
            self.logger.error("Migration failed", error=str(e))

    async def close(self):
        def _close():
            if self._conn:
                self._conn.close()
                self._conn = None

        await self._run_in_executor(_close)
        self._executor.shutdown(wait=True)


class HotTipsTracker:
    """Logs reported opportunities to a SQLite database."""
    def __init__(self, db_path: Optional[Union[str, FortunaDB]] = None, config: Optional[Dict[str, Any]] = None):
        if db_path and hasattr(db_path, '_get_conn'): # Duck typing for FortunaDB (handles mixed import versions)
            self.db = db_path
        elif isinstance(db_path, str):
            self.db = FortunaDB(db_path)
        else:
            self.db = db_path if db_path else FortunaDB()
        self.config = config or {}
        self.logger = structlog.get_logger(self.__class__.__name__)

    async def log_tips(self, races: List[Race], daypart_tag: Optional[str] = None):
        if not races:
            return

        # Ensure daypart_tag is passed if available
        await self.db.initialize()
        now = now_eastern()
        report_date = to_storage_format(now)
        new_tips = []
        already_handled_soft_keys = set()

        # Future cutoff relaxed to allow advance tips
        future_limit = now + timedelta(hours=24)

        # BUG-D2: Detect and skip races where the adapter assigned the same
        # start_time to every race at a venue (meeting-level timestamp leak).
        # Group by (canonical_venue, date) and check for duplicate times.
        venue_date_times: Dict[str, List[Tuple[Race, str]]] = defaultdict(list)
        for r in races:
            st = r.start_time
            if isinstance(st, str):
                try: st = from_storage_format(st.replace('Z', '+00:00'))
                except Exception: continue
            if not st: continue
            vd_key = f"{get_canonical_venue(r.venue)}|{st.strftime(DATE_FORMAT)}"
            time_key = st.strftime('%H%M')
            venue_date_times[vd_key].append((r, time_key))

        bad_venue_dates: Set[str] = set()
        for vd_key, entries in venue_date_times.items():
            if len(entries) < 3:
                continue
            time_counts = Counter(t for _, t in entries)
            most_common_time, most_common_count = time_counts.most_common(1)[0]
            # If 60%+ of races share the exact same time, it's a meeting-level timestamp
            if most_common_count / len(entries) >= 0.6 and most_common_count >= 3:
                bad_venue_dates.add(vd_key)
                self.logger.warning(
                    'meeting_level_timestamp_detected',
                    venue_date=vd_key,
                    shared_time=most_common_time,
                    race_count=len(entries),
                    msg='Skipping all races — start times are not per-race',
                )

        if bad_venue_dates:
            original_count = len(races)
            filtered_races = []
            for r in races:
                st = r.start_time
                if isinstance(st, str):
                    try: st = from_storage_format(st.replace('Z', '+00:00'))
                    except Exception: st = None
                if st:
                    vd_key = f"{get_canonical_venue(r.venue)}|{st.strftime(DATE_FORMAT)}"
                    if vd_key not in bad_venue_dates:
                        filtered_races.append(r)
                else:
                    # Keep if we can't determine date (likely to be dropped later anyway)
                    filtered_races.append(r)
            races = filtered_races
            self.logger.info(
                'meeting_timestamp_filter_applied',
                original=original_count,
                remaining=len(races),
                venues_dropped=len(bad_venue_dates),
            )

        for r in races:
            # Only store "Best Bets" (Goldmine, BET NOW, or You Might Like)
            # These are marked in metadata by the analyzer.
            # Phase 2: Explicitly include 'you_might_like' tier in logging gate
            is_best_bet = r.metadata.get('is_best_bet')
            is_goldmine = r.metadata.get('is_goldmine')
            is_superfecta_key = r.metadata.get('is_superfecta_key')
            tip_tier = r.metadata.get('tip_tier', 'best_bet')

            if not is_best_bet and not is_goldmine and not is_superfecta_key and tip_tier != 'you_might_like':
                continue

            # BUG-D3 & BUG-D4: Hardened guards for tip persistence
            # 1. Reject tips from analyzers that produce unrecognized grades
            VALID_GRADES = {'A+', 'A', 'B+', 'B+ (Override)', 'D'}
            grade = r.metadata.get('qualification_grade')
            if not grade or grade not in VALID_GRADES:
                self.logger.debug('rejecting_unrecognized_or_missing_grade', race_id=r.id, grade=grade, venue=r.venue)
                continue

            # 2. Reject Goldmines that lack valid odds (Prevents bogus flagging)
            if is_goldmine and r.metadata.get('predicted_fav_odds') is None:
                self.logger.debug('rejecting_bogus_goldmine_missing_odds', race_id=r.id, venue=r.venue)
                continue

            # Trustworthiness Airlock Safeguard (Council of Superbrains Directive)
            active_runners = [run for run in r.runners if not run.scratched]
            total_active = len(active_runners)

            # Ensure trustworthy odds exist before logging
            if r.metadata.get('predicted_2nd_fav_odds') is None:
                continue

            if total_active > 0:
                trustworthy_count = sum(1 for run in active_runners if run.metadata.get("odds_source_trustworthy"))
                trust_ratio = trustworthy_count / total_active
                # Relaxed to match SimplySuccessAnalyzer config (Alignment)
                # BUG-2 Fix: Align with expected config key
                min_trust = self.config.get("analysis", {}).get("simply_success_trust_min", 0.25)
                if trust_ratio < min_trust:
                    self.logger.warning("Rejecting race with low trust_ratio for DB logging", venue=r.venue, race=r.race_number, trust_ratio=round(trust_ratio, 2), required=min_trust)
                    continue

            st = r.start_time
            if isinstance(st, str):
                try: st = from_storage_format(st.replace('Z', '+00:00'))
                except Exception: continue
            if st.tzinfo is None: st = st.replace(tzinfo=EASTERN)

            # Re-added 24h safety guard for DB logging (Fix for test_housekeeping)
            if st > future_limit:
                continue

            # Timing gate removed (Phase B7). Caller (run_score_now) has already filtered to 0 < MTP <= 15.

            # BUG-12: Secondary soft-key dedup guard (BUG-CR-04: Include discipline)
            disc_char = (r.discipline or 'T')[:1].upper()
            soft_key = f"{get_canonical_venue(r.venue)}|{r.race_number}|{st.strftime('%y%m%d')}|{disc_char}"
            if soft_key in already_handled_soft_keys:
                self.logger.debug("Skipping duplicate play (soft key match)", soft_key=soft_key)
                continue
            already_handled_soft_keys.add(soft_key)

            is_goldmine = r.metadata.get('is_goldmine', False)
            gap_abs = r.metadata.get('gap_abs', 0.0)

            tip_data = {
                "report_date": report_date,
                "race_id": r.id,
                "venue": r.venue,
                "race_number": r.race_number,
                "start_time": to_storage_format(r.start_time) if isinstance(r.start_time, datetime) else str(r.start_time),
                "is_goldmine": is_goldmine,
                "source": r.source or "Unknown",
                "gap_abs": gap_abs,
                "discipline": r.discipline,
                "top_five": r.top_five_numbers,
                "selection_number": r.metadata.get('selection_number'),
                "selection_name": r.metadata.get('selection_name'),
                "predicted_fav_odds": r.metadata.get('predicted_fav_odds'),
                "predicted_2nd_fav_odds": r.metadata.get('predicted_2nd_fav_odds'),
                "field_size": total_active,
                "market_depth": r.metadata.get('market_depth'),
                "place_prob": r.metadata.get('place_prob'),
                "predicted_ev": r.metadata.get('predicted_ev'),
                "race_type": getattr(r, 'race_type', None),
                "is_handicap": getattr(r, 'is_handicap', None),
                "condition_modifier": r.metadata.get('condition_modifier'),
                "qualification_grade": r.metadata.get('qualification_grade'),
                "composite_score": r.metadata.get('composite_score'),
                "is_best_bet": is_best_bet,
                "tip_tier": tip_tier,
                "is_superfecta_key":     is_superfecta_key,
                "superfecta_key_number": r.metadata.get('superfecta_key_number'),
                "superfecta_key_name":   r.metadata.get('superfecta_key_name'),
                "daypart":               daypart_tag,
                "goldmine_tier":          r.metadata.get('success_tier'),
                "goldmine_sources":       len(r.metadata.get('goldmine_sources', [])),
                "goldmine_gap_confirmed": 1 if r.metadata.get('goldmine_confidence') == 'high' else 0
            }
            new_tips.append(tip_data)

        try:
            # FIX-13: Process all tips in batches instead of truncating
            BATCH_SIZE = 100
            if len(new_tips) > BATCH_SIZE:
                self.logger.info("Processing tips in batches", total=len(new_tips), batch_size=BATCH_SIZE)

            for i in range(0, len(new_tips), BATCH_SIZE):
                batch = new_tips[i:i + BATCH_SIZE]
                await self.db.log_tips(batch)
                if i + BATCH_SIZE < len(new_tips):
                    self.logger.debug("Batch persisted", batch_num=i // BATCH_SIZE + 1, size=len(batch))

            self.logger.info("Hot tips processed", count=len(new_tips))
        except Exception as e:
            self.logger.error("Failed to log hot tips", error=str(e))


# ----------------------------------------
# MONITOR LOGIC
# ----------------------------------------
#!/usr/bin/env python3
"""
Fortuna Favorite-to-Place Betting Monitor
=========================================

This script monitors racing data from multiple adapters and identifies
betting opportunities based on:
1. Second favorite odds >= 4.0 decimal
2. Races under 120 minutes to post (MTP)
3. Superfecta availability preferred

Usage:
    python favorite_to_place_monitor.py [--date YYMMDD] [--refresh-interval 30]
"""

@dataclass
class RaceSummary:
    """Summary of a single race for display."""
    discipline: str  # T/H/G
    track: str
    race_number: int
    field_size: int
    superfecta_offered: bool
    adapter: str
    start_time: datetime
    mtp: Optional[int] = None  # Minutes to post
    second_fav_odds: Optional[float] = None
    second_fav_name: Optional[str] = None
    selection_number: Optional[int] = None
    favorite_odds: Optional[float] = None
    favorite_name: Optional[str] = None
    top_five_numbers: Optional[str] = None
    gap_abs: float = 0.0
    is_goldmine: bool = False
    is_best_bet: bool = False
    is_superfecta_key: bool = False
    superfecta_key_number: Optional[int] = None
    superfecta_key_name: Optional[str] = None
    superfecta_box_numbers: List[int] = Field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "discipline": self.discipline,
            "track": self.track,
            "race_number": self.race_number,
            "field_size": self.field_size,
            "superfecta_offered": self.superfecta_offered,
            "adapter": self.adapter,
            "start_time": to_storage_format(self.start_time),
            "mtp": self.mtp,
            "second_fav_odds": self.second_fav_odds,
            "second_fav_name": self.second_fav_name,
            "selection_number": self.selection_number,
            "favorite_odds": self.favorite_odds,
            "favorite_name": self.favorite_name,
            "top_five_numbers": self.top_five_numbers,
            "gap_abs": self.gap_abs,
            "is_goldmine": self.is_goldmine,
            "is_best_bet": self.is_best_bet,
            "is_superfecta_key": self.is_superfecta_key,
            "superfecta_key_number": self.superfecta_key_number,
            "superfecta_key_name":   self.superfecta_key_name,
            "superfecta_box_numbers": self.superfecta_box_numbers
        }


@lru_cache(maxsize=1)
def get_discovery_adapter_classes() -> List[Type[BaseAdapterV3]]:
    """Recursively discovers all discovery adapter classes (cached for performance - Capability Improvement)."""
    def get_all_subclasses(cls):
        return set(cls.__subclasses__()).union(
            [s for c in cls.__subclasses__() for s in get_all_subclasses(c)]
        )

    return [
        c for c in get_all_subclasses(BaseAdapterV3)
        if not getattr(c, "__abstractmethods__", None)
        and getattr(c, "ADAPTER_TYPE", "discovery") == "discovery"
        and not getattr(c, "DECOMMISSIONED", False)
    ]


class OddscheckerAdapter(BrowserHeadersMixin, DebugMixin, BaseAdapterV3):
    """Adapter for scraping horse racing odds from Oddschecker, migrated to BaseAdapterV3."""

    SOURCE_NAME = "Oddschecker"
    PROVIDES_ODDS: ClassVar[bool] = True
    BASE_URL = "https://www.oddschecker.com"

    def __init__(self, config=None):
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        # Oddschecker is heavily protected by Cloudflare; Playwright with high timeout and network idle
        return FetchStrategy(
            primary_engine=BrowserEngine.PLAYWRIGHT,
            enable_js=True,
            stealth_mode="camouflage",
            timeout=120,
            network_idle=True
        )

    async def make_request(self, method: str, url: str, **kwargs: Any) -> Any:
        # Playwright doesn't use impersonate but SmartFetcher handles it now
        return await super().make_request(method, url, **kwargs)

    def _get_headers(self) -> dict:
        return self._get_browser_headers(host="www.oddschecker.com")

    async def _fetch_data(self, date: str) -> Optional[dict]:
        """
        Fetches the raw HTML for all race pages for a given date. This involves a multi-level fetch.
        """
        # Success Strategy: Bootstrap session if using browser
        try:
            # FIX: Use longer timeout and mandatory delay for Cloudflare
            await self.make_request("GET", "https://www.oddschecker.com/horse-racing", timeout=45, raise_for_status=False)
            await asyncio.sleep(5)
        except Exception: pass

        sem = asyncio.Semaphore(3)
        dt = parse_date_string(date)
        date_iso = dt.strftime("%Y-%m-%d")
        index_url = f"/horse-racing/{date_iso}"
        index_response = await self.make_request("GET", index_url, headers=self._get_headers())
        if not index_response or not index_response.text:
            self.logger.warning("Failed to fetch Oddschecker index page", url=index_url)
            return None

        self._save_debug_html(index_response.text, f"oddschecker_index_{date}")

        parser = HTMLParser(index_response.text)
        # Find all links to individual race pages
        metadata = []

        try:
            target_date = parse_date_string(date).date()
        except Exception:
            target_date = datetime.now(EASTERN).date()

        site_tz = ZoneInfo("Europe/London")
        now_site = datetime.now(site_tz)

        # Group by track to pick "next" race
        track_map = defaultdict(list)

        # Broaden selectors for race links
        for selector in ["a.race-time-link[href]", "a[href*='/horse-racing/'][href*='/20']", ".rf__link"]:
            for a in parser.css(selector):
                href = a.attributes.get("href")
                if href and not href.endswith("/horse-racing"):
                    # Ensure absolute URL
                    full_url = href if href.startswith("http") else f"{self.BASE_URL}{href}"

                    # Extract track from URL if possible, or use parent
                    # URL usually /horse-racing/venue/date/time
                    parts = full_url.split("/")
                    if len(parts) >= 6:
                        track = parts[4]
                        txt = node_text(a) # Time is often in text
                        track_map[track].append({"url": full_url, "time_txt": txt})

        for track, races in track_map.items():
            for r in races:
                if re.match(r"\d{1,2}:\d{2}", r["time_txt"]):
                    try:
                        rt = datetime.strptime(r["time_txt"], "%H:%M").replace(
                            year=target_date.year, month=target_date.month, day=target_date.day, tzinfo=site_tz
                        )
                        # Broaden window to capture multiple races
                        diff = (rt - now_site).total_seconds() / 60
                        if not (-45 < diff <= 1080):
                            continue

                        metadata.append(r["url"])
                    except Exception: pass

        if not metadata:
            self.logger.warning("No metadata found", context="Oddschecker Index Parsing", url=index_url)
            self.metrics.record_parse_warning()
            return None

        async def fetch_single_html(url_path: str):
            async with sem:
                # Small delay to avoid ban
                await asyncio.sleep(0.5 + random.random() * 0.5)
                response = await self.make_request("GET", url_path, headers=self._get_headers())
                return response.text if response else ""

        tasks = [fetch_single_html(link) for link in metadata]
        html_pages = await asyncio.gather(*tasks)
        return {"pages": html_pages, "date": date}

    def _parse_races(self, raw_data: Any) -> List[Race]:
        """Parses a list of raw HTML strings from different races into Race objects."""
        if not raw_data or not raw_data.get("pages"):
            return []

        try:
            race_date = parse_date_string(raw_data["date"]).date()
        except ValueError:
            self.logger.error(
                "Invalid date format provided to OddscheckerAdapter",
                date=raw_data.get("date"),
            )
            return []

        all_races = []
        for html in raw_data["pages"]:
            if not html:
                continue
            try:
                parser = HTMLParser(html)
                # Phase D2: Detect discipline (Greyhound support)
                discipline = detect_discipline(html)
                race = self._parse_race_page(parser, race_date, discipline=discipline)
                if race:
                    all_races.append(race)
            except (AttributeError, IndexError, ValueError):
                self.logger.warning(
                    "Error parsing a race from Oddschecker, skipping race.",
                    exc_info=True,
                )
                continue
        return all_races

    def _parse_race_page(self, parser: HTMLParser, race_date, discipline: str = "Thoroughbred") -> Optional[Race]:
        track_name_node = parser.css_first("h1.meeting-name")
        if not track_name_node:
            return None
        track_name = node_text(track_name_node)

        race_time_node = parser.css_first("span.race-time")
        if not race_time_node:
            return None
        race_time_str = node_text(race_time_node)

        # Heuristic to find race number from navigation
        active_link = parser.css_first("a.race-time-link.active")
        race_number = 0
        if active_link:
            all_links = parser.css("a.race-time-link")
            try:
                for i, link in enumerate(all_links):
                    if link.html == active_link.html:
                        race_number = i + 1
                        break
            except Exception:
                pass

        start_time = datetime.combine(race_date, datetime.strptime(race_time_str, "%H:%M").time())
        runners = [runner for row in parser.css("tr.race-card-row") if (runner := self._parse_runner_row(row))]

        if not runners:
            return None

        # BUG-6 Fix: Use canonical venue for race ID
        venue_key = get_canonical_venue(track_name).lower().replace(' ', '')
        # BUG-11: Pass discipline to generate correct race IDs (Phase D2)
        return Race(
            id=generate_race_id("oc", track_name, start_time, race_number, discipline=discipline),
            venue=track_name,
            race_number=race_number,
            start_time=start_time,
            runners=runners,
            source=self.source_name,
            discipline=discipline
        )

    def _parse_runner_row(self, row: Node) -> Optional[Runner]:
        try:
            name_node = row.css_first("span.selection-name")
            if not name_node:
                return None
            name = node_text(name_node)

            odds_node = row.css_first("span.bet-button-odds-desktop, span.best-price")
            if not odds_node:
                return None
            odds_str = node_text(odds_node)

            number_node = row.css_first("td.runner-number")
            number = 0
            if number_node:
                num_txt = "".join(filter(str.isdigit, node_text(number_node)))
                if num_txt:
                    number = int(num_txt)

            if not name or not odds_str:
                return None

            win_odds = parse_odds_to_decimal(odds_str)

            # Advanced heuristic fallback
            if win_odds is None:
                win_odds = SmartOddsExtractor.extract_from_node(row)

            odds_dict = {}
            if odds_data := create_odds_data(self.source_name, win_odds):
                odds_dict[self.source_name] = odds_data

            return Runner(number=number, name=name, odds=odds_dict)
        except (AttributeError, ValueError):
            self.logger.warning("Failed to parse a runner on Oddschecker, skipping runner.")
            return None

# python_service/adapters/timeform_adapter.py





class TimeformAdapter(JSONParsingMixin, BrowserHeadersMixin, DebugMixin, BaseAdapterV3):
    """
    Adapter for timeform.com, migrated to BaseAdapterV3 and standardized on selectolax.
    """

    SOURCE_NAME = "Timeform"
    PROVIDES_ODDS: ClassVar[bool] = True
    BASE_URL = "https://www.timeform.com"

    def __init__(self, config=None):
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        # Timeform often blocks basic requests; Playwright is robust
        return FetchStrategy(
            primary_engine=BrowserEngine.PLAYWRIGHT,
            enable_js=True,
            stealth_mode="camouflage",
            timeout=90,
            network_idle=True
        )

    def _get_headers(self) -> dict:
        headers = self._get_browser_headers(host="www.timeform.com")
        headers.update({
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        return headers

    async def _fetch_data(self, date: str) -> Optional[dict]:
        """
        Fetches the raw HTML for all race pages for a given date.
        """
        sem = asyncio.Semaphore(5)
        dt = parse_date_string(date)
        date_iso = dt.strftime("%Y-%m-%d")
        index_url = f"/horse-racing/racecards/{date_iso}"
        # Hardening Fix: Establish session on the homepage to avoid 500 errors on dated racecards
        await self.make_request("GET", "https://www.timeform.com/horse-racing", headers=self._get_headers(), raise_for_status=False)

        index_response = await self.make_request("GET", index_url, headers=self._get_headers())
        if not index_response or not index_response.text:
            self.logger.warning("Failed to fetch Timeform index page", url=index_url)
            # Fallback to generic racecards if dated index fails
            index_response = await self.make_request("GET", "/horse-racing/racecards", headers=self._get_headers())
            if not index_response or not index_response.text:
                return None

        self._save_debug_snapshot(index_response.text, f"timeform_index_{date}")

        parser = HTMLParser(index_response.text)
        # Updated selector for race links
        try:
            target_date = parse_date_string(date).date()
        except Exception:
            target_date = datetime.now(EASTERN).date()

        site_tz = ZoneInfo("Europe/London")
        now_site = datetime.now(site_tz)

        track_map = defaultdict(list)
        # Broaden selectors for Timeform race links
        for selector in ["a[href*='/racecards/']", ".rf__link", "a.rf-meeting-race__time", ".rp-meetingItem__race__time"]:
            for a in parser.css(selector):
                href = a.attributes.get("href")
                if href and "/racecards/" in href and not href.endswith("/racecards"):
                    # URL usually: /horse-racing/racecards/venue/date/time/...
                    # or: /racecards/venue/date/time
                    parts = href.split("/")
                    # Handle both relative and absolute-ish paths
                    track = "unknown"
                    for i, p in enumerate(parts):
                        if p == "racecards" and i + 1 < len(parts):
                            track = parts[i+1]
                            break

                    txt = node_text(a)
                    track_map[track].append({"url": href, "time_txt": txt})

        links = []
        for track, races in track_map.items():
            for r in races:
                # Timeform often uses HH:MM in text
                time_match = re.search(r"(\d{1,2}:\d{2})", r["time_txt"])
                if time_match:
                    try:
                        rt = datetime.strptime(time_match.group(1), "%H:%M").replace(
                            year=target_date.year, month=target_date.month, day=target_date.day, tzinfo=site_tz
                        )
                        # Broaden window to capture multiple races
                        diff = (rt - now_site).total_seconds() / 60
                        if not (-45 < diff <= 1080):
                            continue

                        full_url = r["url"] if r["url"].startswith("http") else f"{self.BASE_URL}{r['url']}"
                        links.append(full_url)
                    except Exception: pass

        if not links:
            self.logger.warning("No metadata found", context="Timeform Index Parsing", url=index_url)
            self.metrics.record_parse_warning()
            return None

        async def fetch_single_html(url_path: str):
            async with sem:
                await asyncio.sleep(0.5)
                response = await self.make_request("GET", url_path, headers=self._get_headers())
                return (url_path, response.text) if response else (url_path, "")

        self.logger.info(f"Found {len(links)} race links on Timeform")
        tasks = [fetch_single_html(link) for link in links]
        results = await asyncio.gather(*tasks)
        return {"pages": [r for r in results if r[1]], "date": date}

    def _parse_races(self, raw_data: Any) -> List[Race]:
        """Parses a list of raw HTML strings into Race objects."""
        if not raw_data or not raw_data.get("pages"):
            return []

        try:
            race_date = parse_date_string(raw_data["date"]).date()
        except ValueError:
            self.logger.error("Invalid date format", date=raw_data.get("date"))
            return []

        all_races = []
        for url_path, html_content in raw_data["pages"]:
            if not html_content:
                continue
            try:
                parser = HTMLParser(html_content)

                # Extract via JSON-LD if possible
                venue = ""
                start_time = None
                is_handicap = None
                scripts = self._parse_all_jsons_from_scripts(parser, 'script[type="application/ld+json"]', context="Betfair Index")
                for data in scripts:
                    if data.get("@type") == "Event":
                        venue = normalize_venue_name(data.get("location", {}).get("name", ""))
                        if sd := data.get("startDate"):
                            # 2026-01-28T14:32:00
                            start_time = from_storage_format(sd.split('+')[0])
                        break

                title_node = parser.css_first("title")
                if title_node:
                    title_text = node_text(title_node)
                    if "HANDICAP" in title_text.upper():
                        is_handicap = True

                # BUG-17: Prefer URL-based time extraction to avoid shared card-start-time issues
                url_time = None
                time_match = re.search(r'/(\d{4})/?$', url_path.split('?')[0])
                if time_match:
                    try:
                        url_time = datetime.combine(
                            race_date,
                            datetime.strptime(time_match.group(1), "%H%M").time()
                        )
                    except Exception: pass

                if url_time:
                    start_time = url_time

                if not venue:
                    # Fallback to title
                    if title_node:
                        # 14:32 DUNDALK | Races 28 January 2026 ...
                        match = re.search(r'(\d{1,2}:\d{2})\s+([^|]+)', title_text)
                        if match:
                            time_str = match.group(1)
                            venue = normalize_venue_name(match.group(2).strip())
                            if not start_time:
                                start_time = datetime.combine(race_date, datetime.strptime(time_str, "%H:%M").time())

                if not venue or not start_time:
                    continue

                # Betting Forecast Parsing
                forecast_map = {}
                verdict_section = parser.css_first("section.rp-verdict")
                if verdict_section:
                    forecast_text = clean_text(node_text(verdict_section))
                    if "Betting Forecast :" in forecast_text:
                        # "Betting Forecast : 15/8 2.87 Spring Is Here, 3/1 4 This Guy, ..."
                        after_forecast = forecast_text.split("Betting Forecast :")[1]
                        # Split by comma
                        parts = after_forecast.split(',')
                        for part in parts:
                            # Match odds and then name
                            # Odds can be fractional space decimal
                            m = re.search(r'(\d+/\d+|EVENS)\s+([\d\.]+)?\s*(.+)', part.strip())
                            if m:
                                odds_str = m.group(1)
                                name = clean_text(m.group(3))
                                forecast_map[name.lower()] = odds_str

                # Runners
                runners = []
                # Use tbody as the main container for each runner
                for row in parser.css('tbody.rp-horse-row'):
                    if runner := self._parse_runner(row, forecast_map):
                        runners.append(runner)

                if not runners:
                    continue

                # Race number from URL or sequence
                race_number = 0
                num_match = re.search(r'/(\d+)/([^/]+)$', url_path)
                # .../1432/207/1/view... -> the '1' is the race number
                url_parts = url_path.split('/')
                if len(url_parts) >= 10:
                    try: race_number = int(url_parts[9])
                    except Exception: pass

                race = Race(
                    id=f"tf_{venue.lower().replace(' ', '')}_{start_time:%y%m%d}_R{race_number}",
                    venue=venue,
                    race_number=race_number,
                    start_time=start_time,
                    runners=runners,
                    is_handicap=is_handicap,
                    source=self.source_name,
                )
                all_races.append(race)
            except Exception as e:
                self.logger.warning(f"Error parsing Timeform race: {e}")
                continue
        return all_races

    def _parse_runner(self, row: Node, forecast_map: dict = None) -> Optional[Runner]:
        """Parses a single runner from a table row node."""
        try:
            name_node = row.css_first("a.rp-horse") or row.css_first("a.rp-horseTable_horse-name")
            if not name_node:
                return None
            name = clean_text(node_text(name_node))

            number = 0
            num_attr = row.attributes.get("data-entrynumber")
            if num_attr:
                try:
                    val = int(num_attr)
                    if val <= 40: number = val
                except Exception:
                    pass

            if not number:
                num_node = row.css_first(".rp-entry-number") or row.css_first("span.rp-horseTable_horse-number")
                if num_node:
                    num_text = clean_text(node_text(num_node)).strip("()")
                    num_match = re.search(r"\d+", num_text)
                    if num_match:
                        val = int(num_match.group())
                        if val <= 40: number = val

            win_odds = None
            odds_source = None
            if forecast_map:
                win_odds = parse_odds_to_decimal(forecast_map.get(name.lower()))
                if win_odds is not None:
                    odds_source = "morning_line"

            # Try to find live odds button if available (old selector)
            if not win_odds:
                odds_tag = row.css_first("button.rp-bet-placer-btn__odds")
                if odds_tag:
                    win_odds = parse_odds_to_decimal(clean_text(node_text(odds_tag)))
                    if win_odds is not None:
                        odds_source = "extracted"

            # Advanced heuristic fallback
            if win_odds is None:
                win_odds = SmartOddsExtractor.extract_from_node(row)
                if win_odds is not None:
                    odds_source = "smart_extractor"

            odds_data = {}
            if odds_val := create_odds_data(self.source_name, win_odds):
                odds_data[self.source_name] = odds_val

            return Runner(number=number, name=name, win_odds=win_odds, odds=odds_data, odds_source=odds_source)
        except (AttributeError, ValueError, TypeError):
            return None

# python_service/adapters/racingpost_adapter.py




class RacingPostAdapter(BrowserHeadersMixin, DebugMixin, BaseAdapterV3):
    """
    Adapter for scraping Racing Post racecards, migrated to BaseAdapterV3.
    """

    SOURCE_NAME = "RacingPost"
    PROVIDES_ODDS: ClassVar[bool] = True
    BASE_URL = "https://www.racingpost.com"

    def __init__(self, config=None):
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)
        self.headers.update({
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
        })

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return scraping_fetch_strategy(network_idle=True)

    def _get_headers(self) -> dict:
        headers = self._get_browser_headers(host="www.racingpost.com")
        headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
        })
        return headers

    async def _fetch_data(self, date: str) -> Any:
        """
        Fetches the raw HTML content for all races on a given date, including international.
        """
        # Hardening Fix: Establish session on the homepage before fetching racecards
        try:
            await self.make_request("GET", "https://www.racingpost.com/", headers=self._get_headers(), raise_for_status=False)
            await asyncio.sleep(1)
        except Exception: pass

        dt = parse_date_string(date)
        date_iso = dt.strftime("%Y-%m-%d")
        index_url = f"/racecards/{date_iso}"
        # RacingPost international URL sometimes varies
        intl_urls = [
            f"/racecards/international/{date_iso}",
            f"/racecards/{date_iso}/international",
            "/racecards/international"
        ]

        index_response = await self.make_request("GET", index_url, headers=self._get_headers())

        intl_response = None
        for url in intl_urls:
            try:
                resp = await self.make_request("GET", url, headers=self._get_headers(), raise_for_status=False)
                if resp and get_resp_status(resp) == 200:
                    intl_response = resp
                    break
            except Exception:
                continue

        race_card_urls = []
        try:
            target_date = parse_date_string(date).date()
        except Exception:
            target_date = datetime.now(EASTERN).date()

        site_tz = ZoneInfo("Europe/London")
        now_site = datetime.now(site_tz)

        if index_response and index_response.text:
            self._save_debug_html(index_response.text, f"racingpost_index_{date}")
            index_parser = HTMLParser(index_response.text)

            # Broaden window to capture multiple races
            meetings = index_parser.css('.rp-raceCourse__panel') or index_parser.css('.RC-meetingItem') or index_parser.css('.rp-meetingItem') or index_parser.css('.RC-courseCards')
            for meeting in meetings:
                # Broaden a tag selectors to catch new Racing Post structures
                for link in meeting.css('a[data-test-selector^="RC-meetingItem__link_race"], a.rp-raceCourse__panel__race__time, a.rp-meetingItem__race__time, a.RC-meetingItem__race__time, a.RC-meetingItem__link, a[href*="/racecards/"]'):
                    href = link.attributes.get("href", "")
                    if not href or "/results/" in href:
                        continue

                    txt = clean_text(node_text(link))
                    time_match = re.search(r"(\d{1,2}:\d{2})", txt)
                    if time_match:
                        try:
                            time_str = time_match.group(1)
                            tm = datetime.strptime(time_str, "%H:%M")
                            if tm.hour < 9:
                                tm = tm.replace(hour=tm.hour + 12)

                            rt = tm.replace(
                                year=target_date.year, month=target_date.month, day=target_date.day, tzinfo=site_tz
                            )
                            diff = (rt - now_site).total_seconds() / 60
                            if not (-45 < diff <= 1080):
                                continue
                        except Exception: pass

                    race_card_urls.append(href)

        elif index_response:
            self.logger.warning("Unexpected status", status=index_response.status, url=index_url)

        if intl_response and intl_response.text:
            self._save_debug_html(intl_response.text, f"racingpost_intl_index_{date}")
            intl_parser = HTMLParser(intl_response.text)

            meetings = intl_parser.css('.rp-raceCourse__panel') or intl_parser.css('.RC-meetingItem') or intl_parser.css('.rp-meetingItem') or intl_parser.css('.RC-courseCards')
            for meeting in meetings:
                for link in meeting.css('a[data-test-selector^="RC-meetingItem__link_race"], a.rp-raceCourse__panel__race__time, a.rp-meetingItem__race__time, a.RC-meetingItem__race__time, a.RC-meetingItem__link, a[href*="/racecards/"]'):
                    href = link.attributes.get("href", "")
                    if not href or "/results/" in href:
                        continue

                    txt = clean_text(node_text(link))
                    time_match = re.search(r"(\d{1,2}:\d{2})", txt)
                    if time_match:
                        try:
                            time_str = time_match.group(1)
                            tm = datetime.strptime(time_str, "%H:%M")
                            if tm.hour < 9:
                                tm = tm.replace(hour=tm.hour + 12)

                            rt = tm.replace(
                                year=target_date.year, month=target_date.month, day=target_date.day, tzinfo=site_tz
                            )
                            diff = (rt - now_site).total_seconds() / 60
                            if not (-45 < diff <= 1080):
                                continue
                        except Exception: pass

                    race_card_urls.append(href)
        elif intl_response:
            self.logger.warning("Unexpected status", status=intl_response.status, url=intl_url)

        if not race_card_urls:
            self.logger.warning("Standard RacingPost link discovery failed, trying aggressive fallback", date=date)
            for resp in [index_response, intl_response]:
                if resp and resp.text:
                    p = HTMLParser(resp.text)
                    # Even more aggressive: any link containing /racecards/ and a date-like pattern
                    for a in p.css('a[href*="/racecards/"]'):
                        href = a.attributes.get("href", "")
                        if re.search(r"/\d{4}-\d{2}-\d{2}/", href) or re.search(r"/\d+/.*/\d+/?$", href):
                            race_card_urls.append(href)

        if not race_card_urls:
            self.logger.warning("Failed to fetch RacingPost racecard links", date=date)
            self.metrics.record_parse_warning()
            return None

        # Deduplicate URLs to avoid redundant fetching
        race_card_urls = list(dict.fromkeys(race_card_urls))
        self.logger.info("Deduplicated RacingPost links", original=len(race_card_urls), unique=len(race_card_urls))

        async def fetch_single_html(url: str):
            response = await self.make_request("GET", url, headers=self._get_headers())
            return response.text if response else ""

        tasks = [fetch_single_html(url) for url in race_card_urls]
        html_contents = await asyncio.gather(*tasks)
        return {"date": date, "html_contents": html_contents}

    def _parse_races(self, raw_data: Any) -> List[Race]:
        """Parses a list of raw HTML strings into Race objects."""
        if not raw_data or not raw_data.get("html_contents"):
            return []

        date = raw_data["date"]
        html_contents = raw_data["html_contents"]
        all_races: List[Race] = []

        for html in html_contents:
            if not html:
                continue
            try:
                parser = HTMLParser(html)

                venue_node = (
                    parser.css_first('*[data-test-selector="RC-courseHeader__name"]')
                    or parser.css_first('a[data-test-selector="RC-courseHeader__name"]')
                    or parser.css_first('a[data-test-selector="RC-course__name"]')
                )
                if not venue_node:
                    continue
                venue_raw = node_text(venue_node)
                venue = normalize_venue_name(venue_raw)

                race_time_node = (
                    parser.css_first('*[data-test-selector="RC-courseHeader__time"]')
                    or parser.css_first('span[data-test-selector="RC-courseHeader__time"]')
                    or parser.css_first('span[data-test-selector="RC-course__time"]')
                )
                if not race_time_node:
                    continue
                race_time_str = node_text(race_time_node)

                # S5 — extract race type (independent review item)
                # Hardening Fix: Broaden race type detection to improve scoring population
                race_type = None
                header_text = node_text(
                    parser.css_first('.rp-raceCourse__panel__race__info')
                    or parser.css_first('.RC-course__info')
                    or parser.css_first('.RC-courseHeader')
                )
                rt_match = re.search(r'(Maiden\s+\w+|Claiming|Allowance|Graded\s+Stakes|Stakes|Handicap|Novice|Group\s+\d|Grade\s+\d|Listed|Condition|Hurdle|Chase|Bumper|National\s+Hunt|Steeplechase)', header_text, re.I)
                if rt_match: race_type = rt_match.group(1)

                is_handicap = None
                if "HANDICAP" in header_text.upper():
                    is_handicap = True

                race_datetime_str = f"{date} {race_time_str}"
                try:
                    start_time = datetime.strptime(race_datetime_str, f"{DATE_FORMAT} %H:%M")
                except ValueError:
                    # Handle cases where time might have extra text or different format
                    time_match = re.search(r"(\d{1,2}:\d{2})", race_time_str)
                    if time_match:
                        start_time = datetime.strptime(f"{date} {time_match.group(1)}", f"{DATE_FORMAT} %H:%M")
                    else:
                        continue

                runners = self._parse_runners(parser)

                if venue and runners:
                    race_number = self._get_race_number(parser, start_time)
                    race = Race(
                        id=f"rp_{venue.lower().replace(' ', '')}_{date}_{race_number}",
                        venue=venue,
                        race_number=race_number,
                        start_time=start_time,
                        runners=runners,
                        race_type=race_type,
                        is_handicap=is_handicap,
                        source=self.source_name,
                    )
                    all_races.append(race)
            except (AttributeError, ValueError):
                self.logger.error("Failed to parse RacingPost race from HTML content.", exc_info=True)
                continue
        return all_races

    def _get_race_number(self, parser: HTMLParser, start_time: datetime) -> int:
        """Derives the race number by finding the active time in the nav bar."""
        time_str_to_find = start_time.strftime("%H:%M")
        time_links = parser.css('a[data-test-selector="RC-raceTime"]')
        for i, link in enumerate(time_links):
            if node_text(link) == time_str_to_find:
                return i + 1
        return 1

    def _parse_runners(self, parser: HTMLParser) -> list[Runner]:
        """Parses all runners from a single race card page."""
        runners = []
        runner_nodes = (
            parser.css('div[data-test-selector="RC-runnerCard"]')
            or parser.css('.RC-runnerRow')
        )

        # Betting Forecast Fallback
        forecast_map = {}
        for group in parser.css('*[data-test-selector="RC-bettingForecast_group"]'):
            group_text = node_text(group)
            # Format: "2/1 Horse Name" or similar
            link = group.css_first('*[data-test-selector="RC-bettingForecast_link"]')
            if link:
                horse_name = clean_text(node_text(link))
                # Remove horse name from group_text to get odds
                odds_part = group_text.replace(horse_name, "").strip().rstrip(",")
                if val := parse_odds_to_decimal(odds_part):
                    forecast_map[horse_name.lower()] = val

        for node in runner_nodes:
            if runner := self._parse_runner(node, forecast_map):
                runners.append(runner)
        return runners

    def _parse_runner(self, node: Node, forecast_map: Optional[Dict[str, float]] = None) -> Optional[Runner]:
        try:
            number_node = (
                node.css_first('span[data-test-selector="RC-cardPage-runnerNumber-no"]')
                or node.css_first('span[data-test-selector="RC-runnerNumber"]')
                or node.css_first('.RC-runnerNumber__no')
            )
            name_node = (
                node.css_first('a[data-test-selector="RC-cardPage-runnerName"]')
                or node.css_first('a[data-test-selector="RC-runnerName"]')
                or node.css_first('.RC-runnerName')
            )
            odds_node = (
                node.css_first('span[data-test-selector="RC-cardPage-runnerPrice"]')
                or node.css_first('a[data-test-selector="RC-cardPage-runnerPrice"]')
                or node.css_first('span[data-test-selector="RC-runnerPrice"]')
                or node.css_first('.RC-runnerPrice')
            )

            if not name_node:
                return None

            name = clean_text(node_text(name_node))

            number = 0
            if number_node:
                number_str = clean_text(node_text(number_node))
                if number_str:
                    num_txt = "".join(filter(str.isdigit, number_str))
                    if num_txt:
                        val = int(num_txt)
                        if val <= 100: number = val

            odds_str = clean_text(node_text(odds_node)) if odds_node else ""
            scratched = "NR" in odds_str.upper() or "NON-RUNNER" in node_text(node).upper()

            odds = {}
            win_odds = None
            odds_source = None
            if not scratched:
                win_odds = parse_odds_to_decimal(odds_str)
                if win_odds is not None:
                    odds_source = "extracted"

                # Betting Forecast Fallback
                if win_odds is None and forecast_map and name.lower() in forecast_map:
                    win_odds = forecast_map[name.lower()]
                    odds_source = "betting_forecast"

                # Advanced heuristic fallback
                if win_odds is None:
                    win_odds = SmartOddsExtractor.extract_from_node(node)
                    if win_odds is not None:
                        odds_source = "smart_extractor"

                if odds_data := create_odds_data(self.source_name, win_odds):
                    odds[self.source_name] = odds_data

            return Runner(number=number, name=name, odds=odds, win_odds=win_odds, odds_source=odds_source, scratched=scratched)
        except Exception:
            self.logger.warning("Could not parse RacingPost runner, skipping.", exc_info=True)
            return None


class RacingPostToteAdapter(BrowserHeadersMixin, DebugMixin, BaseAdapterV3):
    """
    Adapter for fetching Tote dividends and results from Racing Post.
    """
    ADAPTER_TYPE = "results"
    SOURCE_NAME = "RacingPostTote"
    BASE_URL = "https://www.racingpost.com"
    DECOMMISSIONED: ClassVar[bool] = False

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return FetchStrategy(
            primary_engine=BrowserEngine.CURL_CFFI,
            enable_js=True,
            stealth_mode=StealthMode.CAMOUFLAGE,
            timeout=45
        )

    def _get_headers(self) -> dict:
        return self._get_browser_headers(host="www.racingpost.com")

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        dt = parse_date_string(date)
        date_iso = dt.strftime("%Y-%m-%d")
        url = f"/results/{date_iso}"
        resp = await self.make_request("GET", url, headers=self._get_headers())
        if not resp or not resp.text:
            return None

        self._save_debug_snapshot(resp.text, f"rp_tote_results_{date}")
        parser = HTMLParser(resp.text)

        # Extract links to individual race results
        links = set()
        selectors = [
            'a[data-test-selector="RC-meetingItem__link_race"]',
            'a[href*="/results/"]',
            '.ui-link.rp-raceCourse__panel__race__time',
            'a.rp-raceCourse__panel__race__time'
        ]
        target_venues = getattr(self, "target_venues", None)
        for s in selectors:
            for a in parser.css(s):
                href = a.attributes.get("href")
                if href:
                    # Filter by venue
                    if target_venues:
                        match_found = False
                        for v in target_venues:
                            if v in href.lower().replace("-", ""):
                                match_found = True
                                break
                        if not match_found:
                            v_text = get_canonical_venue(node_text(a))
                            if v_text in target_venues:
                                match_found = True
                        if not match_found:
                            continue

                    # Broaden regex to match various RP result link patterns
                    if re.search(r"/results/.*?\d{5,}", href) or \
                       re.search(r"/results/\d+/", href) or \
                       re.search(r"/\d{4}-\d{2}-\d{2}/", href) or \
                       len(href.split("/")) >= 4:
                        links.add(href if href.startswith("http") else f"{self.BASE_URL}{href}")

        async def fetch_result_page(link):
            r = await self.make_request("GET", link, headers=self._get_headers())
            return (link, r.text if r else "")

        tasks = [fetch_result_page(link) for link in links]
        pages = await asyncio.gather(*tasks)
        return {"pages": pages, "date": date}

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data or not raw_data.get("pages"):
            return []

        races = []
        date_str = raw_data["date"]

        for link, html_content in raw_data["pages"]:
            if not html_content:
                continue
            try:
                parser = HTMLParser(html_content)
                race = self._parse_result_page(parser, date_str, link)
                if race:
                    races.append(race)
            except Exception as e:
                self.logger.warning("Failed to parse RP result page", link=link, error=str(e))

        return races

    def _parse_result_page(self, parser: HTMLParser, date_str: str, url: str) -> Optional[Race]:
        venue_node = (
            parser.css_first('*[data-test-selector="RC-courseHeader__name"]')
            or parser.css_first('a[data-test-selector="RC-courseHeader__name"]')
            or parser.css_first('a[data-test-selector="RC-course__name"]')
        )
        if not venue_node: return None
        venue = normalize_venue_name(node_text(venue_node))

        time_node = (
            parser.css_first('*[data-test-selector="RC-courseHeader__time"]')
            or parser.css_first('span[data-test-selector="RC-courseHeader__time"]')
            or parser.css_first('span[data-test-selector="RC-course__time"]')
        )
        if not time_node: return None
        time_str = node_text(time_node)

        try:
            start_time = datetime.strptime(f"{date_str} {time_str}", f"{DATE_FORMAT} %H:%M").replace(tzinfo=EASTERN)
        except Exception:
            return None

        # Extract dividends
        dividends = {}
        tote_container = parser.css_first('div[data-test-selector="RC-toteReturns"]')
        if not tote_container:
             # Try alternate selector
             tote_container = parser.css_first('.rp-toteReturns')

        if tote_container:
            for row in (tote_container.css('div.rp-toteReturns__row') or tote_container.css('.rp-toteReturns__row')):
                try:
                    label_node = row.css_first('div.rp-toteReturns__label') or row.css_first('.rp-toteReturns__label')
                    val_node = row.css_first('div.rp-toteReturns__value') or row.css_first('.rp-toteReturns__value')
                    if label_node and val_node:
                        label = clean_text(node_text(label_node))
                        value = clean_text(node_text(val_node))
                        if label and value:
                            dividends[label] = value
                except Exception as e:
                    self.logger.debug("Failed parsing RP tote row", error=str(e))



        # Extract runners (finishers)
        runners = []
        # Try different row selectors for results
        runner_rows = (
            parser.css('div[data-test-selector="RC-resultRunner"]')
            or parser.css('.RC-runnerRow')
            or parser.css('.rp-horseTable__mainRow')
        )

        for row in runner_rows:
            name_node = (
                row.css_first('a[data-test-selector="RC-resultRunnerName"]')
                or row.css_first('*[data-test-selector="RC-cardPage-runnerName"]')
                or row.css_first('.RC-runnerName')
                or row.css_first('.rp-horseTable__horse__name')
            )
            if not name_node: continue
            name = clean_text(node_text(name_node))

            pos_node = (
                row.css_first('*[data-test-selector="RC-cardPage-runnerPosition"]')
                or row.css_first('span.rp-resultRunner__position')
                or row.css_first('.rp-horseTable__pos__number')
            )
            pos = clean_text(node_text(pos_node)) if pos_node else "?"

            # Try to find saddle number
            number = 0
            num_node = (
                row.css_first('*[data-test-selector="RC-cardPage-runnerNumber-no"]')
                or row.css_first('.RC-runnerNumber__no')
                or row.css_first(".rp-resultRunner__saddleClothNo")
                or row.css_first(".rp-horseTable__saddleClothNo")
            )
            if num_node:
                try: number = _safe_int(node_text(num_node))
                except Exception: pass

            # Extract SP (Starting Price) odds for audit comparison
            win_odds = None
            odds_source = None
            sp_node = (
                row.css_first('*[data-test-selector="RC-cardPage-runnerPrice"]')
                or row.css_first('.RC-runnerPrice')
                or row.css_first('span[data-test-selector="RC-resultRunnerSP"]')
                or row.css_first('.rp-resultRunner__sp')
                or row.css_first(".rp-horseTable__horse__sp")
            )
            if sp_node:
                win_odds = parse_odds_to_decimal(clean_text(node_text(sp_node)))
                if win_odds is not None:
                    odds_source = "starting_price"

            odds_data = {}
            if ov := create_odds_data(self.source_name, win_odds):
                odds_data[self.source_name] = ov

            runners.append(Runner(
                name=name,
                number=number,
                win_odds=win_odds,
                odds=odds_data,
                odds_source=odds_source,
                metadata={"position": pos}
            ))

        # Derive race number from header or navigation
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
            race_num_match = re.search(r'Race\s+(\d+)', node_text(parser))
            if race_num_match:
                race_num = int(race_num_match.group(1))

        race = Race(
            id=f"rp_tote_{get_canonical_venue(venue)}_{date_str.replace('-', '')}_R{race_num}",
            venue=venue,
            race_number=race_num,
            start_time=start_time,
            runners=runners,
            source=self.source_name,
            metadata={"dividends": dividends, "url": url}
        )
        return race

# ----------------------------------------
# MASTER ORCHESTRATOR
# ----------------------------------------

async def run_quarter_fetch(
    config: Dict,
    daypart_tag: str,
    adapter_filter: Optional[List[str]] = None,
    quality: Optional[str] = None,
    save_path: Optional[str] = None,
    force_fetch: bool = False,
    include_health_checks: bool = False,
) -> List[Race]:
    """
    Performs the structural discovery sweep and saves a snapshot.
    Does NOT score or persist tips.
    """
    logger = structlog.get_logger("run_quarter_fetch")
    db = FortunaDB()
    await db.initialize()

    if await db.is_quarter_fetched(daypart_tag) and not force_fetch:
        logger.info("quarter_already_fetched_skipping", daypart_tag=daypart_tag)
        return []

    # Get active daypart from tag (e.g. Q3_260225 -> Q3)
    try:
        dp_str = daypart_tag.split("_")[0]
        daypart = DayPart(dp_str)
    except (ValueError, IndexError):
        daypart = resolve_daypart()

    # Determine region from daypart (Phase A2 fix: mapping to GLOBAL for mixed Q3/Q4)
    if daypart in (DayPart.Q3, DayPart.Q4):
        region = "GLOBAL"
    elif daypart == DayPart.Q1:
        region = "INT"
    elif daypart == DayPart.Q2:
        region = "INT"
    else:
        region = "GLOBAL"

    # Get adapter names for this daypart
    if not adapter_filter:
        adapter_filter = list(get_daypart_discovery_adapters(daypart))

    # Filter out health-check-only adapters unless explicitly requested
    if not include_health_checks:
        all_adapter_classes = get_discovery_adapter_classes()
        adapter_classes = [
            cls for cls in all_adapter_classes
            if not getattr(cls, 'IS_HEALTH_CHECK_ONLY', False)
        ]
        # Sync filter with the filtered classes
        allowed_names = {getattr(c, "SOURCE_NAME", c.__name__) for c in adapter_classes}
        adapter_filter = [n for n in adapter_filter if n in allowed_names]

    # Apply quality filter on top
    if quality == "solid":
        adapter_filter = [n for n in adapter_filter if n in SOLID_DISCOVERY_ADAPTERS]
    elif quality == "lousy":
        adapter_filter = [n for n in adapter_filter if n not in SOLID_DISCOVERY_ADAPTERS]

    # Auto-discover discovery adapter classes
    all_adapter_classes = get_discovery_adapter_classes()
    adapter_classes = [c for c in all_adapter_classes if getattr(c, "SOURCE_NAME", c.__name__) in adapter_filter]

    # Partition adapters into tiers based on their class variables
    tier1_odds = []      # Adapters with PROVIDES_ODDS = True — highest value
    tier2_discovery = []  # Data adapters without odds
    tier3_health = []    # IS_HEALTH_CHECK_ONLY adapters (if included)

    for cls in adapter_classes:
        if getattr(cls, 'IS_HEALTH_CHECK_ONLY', False):
            tier3_health.append(cls)
        elif getattr(cls, 'PROVIDES_ODDS', False):
            tier1_odds.append(cls)
        else:
            tier2_discovery.append(cls)

    # Sort each tier by historical performance score (best first)
    try:
        adapter_scores = await db.get_adapter_scores(days=7)
    except Exception:
        adapter_scores = {}

    for tier in (tier1_odds, tier2_discovery):
        tier.sort(key=lambda c: adapter_scores.get(
            getattr(c, 'SOURCE_NAME', ''), 0), reverse=True)

    logger.info('adapter_tiers',
                tier1_count=len(tier1_odds),
                tier2_count=len(tier2_discovery),
                tier3_count=len(tier3_health))

    adapter_configs = config.get("adapters", {}) if config else {}
    harvest_summary = {}

    date_str = daypart_tag.split("_")[1] # 260225

    async def fetch_one(cls):
        name = getattr(cls, "SOURCE_NAME", cls.__name__)
        specific_config = adapter_configs.get(name, {}).copy()
        specific_config.update({"region": region})

        adapter = None
        try:
            adapter = cls(config=specific_config)
            # Determine if this adapter uses Playwright
            strategy = getattr(adapter, 'strategy', None)
            use_playwright_sem = False
            if strategy and strategy.primary_engine in (BrowserEngine.PLAYWRIGHT, BrowserEngine.PLAYWRIGHT_LEGACY, BrowserEngine.CAMOUFOX):
                use_playwright_sem = True

            # GEMINI_3: Increase per-adapter timeout in run_quarter_fetch to 180s
            # Increased to 300s to match hardened adapter timeouts (Hardening Fix)
            fetch_timeout = 300.0
            # FIX: Deadlock guard — do not acquire playwright_semaphore here!
            # SmartFetcher already handles the semaphore internally during session creation.
            races = await asyncio.wait_for(adapter.get_races(date_str), timeout=fetch_timeout)

            # Record last status for Phase 1 logging (Hardening Fix)
            status = getattr(adapter, 'last_response_status', None)

            # Update harvest summary
            if name not in harvest_summary:
                harvest_summary[name] = {"count": 0, "max_odds": 0.0, "trust_ratio": 0.0}

            harvest_summary[name]["count"] = len(races)
            harvest_summary[name]["status"] = status
            harvest_summary[name]["trust_ratio"] = getattr(adapter, "trust_ratio", 0.0)

            m_odds = 0.0
            for r in races:
                for run in r.runners:
                    if run.win_odds and run.win_odds > m_odds:
                        m_odds = float(run.win_odds)
            harvest_summary[name]["max_odds"] = m_odds

            return races
        except asyncio.TimeoutError:
            logger.warning("adapter_timeout", adapter=name, timeout=fetch_timeout)
            if name not in harvest_summary:
                harvest_summary[name] = {"count": 0, "max_odds": 0.0, "trust_ratio": 0.0}
            harvest_summary[name]["status"] = "timeout"
            return []
        except Exception as e:
            logger.error("Error fetching from adapter", adapter=name, date=date_str, error=str(e))
            if name not in harvest_summary:
                harvest_summary[name] = {"count": 0, "max_odds": 0.0, "trust_ratio": 0.0}

            error_str = str(e).lower()
            bot_keywords = ['captcha','cloudflare','bot','403','challenge','blocked','perimeterx','datadome']
            harvest_summary[name]["status"] = "blocked" if any(kw in error_str for kw in bot_keywords) else "error"
            harvest_summary[name]["error"] = str(e)
            return []
        finally:
            if adapter:
                try: await adapter.close()
                except Exception: pass

    all_races_raw = []

    # Phase 1: Odds-providing + discovery adapters (main time budget)
    phase1_adapters = tier1_odds + tier2_discovery
    # Capability Improvement: Increased to 10 minutes to allow multiple browser adapters to finish
    phase1_timeout = 600

    if phase1_adapters:
        pending = [asyncio.create_task(fetch_one(cls)) for cls in phase1_adapters]
        try:
            done, pending = await asyncio.wait(pending, timeout=phase1_timeout)
            for task in done:
                try:
                    r = task.result()
                    if isinstance(r, list):
                        all_races_raw.extend(r)
                except Exception as e:
                    logger.error("task_error", error=str(e))

            if pending:
                logger.warning('Some discovery adapters are taking longer than expected, but we have collected what we could! ✨',
                            completed_races=len(all_races_raw),
                            adapters_pending=len(pending))
                for task in pending:
                    task.cancel()
        except Exception as e:
            logger.error('phase1_execution_error', error=str(e))

    # Phase 2: Health checks only if time permits and explicitly included
    if tier3_health:
        try:
            await asyncio.wait_for(
                asyncio.gather(*[
                    fetch_one(cls) for cls in tier3_health
                ], return_exceptions=True),
                timeout=60  # 1 minute max for health checks
            )
        except asyncio.TimeoutError:
            pass  # Health checks are expendable

    # Results processing (already handled inside fetch_one for the summary)
    for adapter_name, stats in harvest_summary.items():
        count = stats.get("count", 0)
        status = stats.get("status")
        if count > 0:
            logger.info("Great news! Adapter fetch successful! ✨", adapter=adapter_name, count=count, status=status)
        else:
            logger.info("adapter_fetch_complete", adapter=adapter_name, count=count, status=status)

    # Deduplicate
    race_map = {}
    for race in all_races_raw:
        canonical_venue = get_canonical_venue(race.venue)
        st = race.start_time
        if isinstance(st, str):
            try: st = from_storage_format(st.replace('Z', '+00:00'))
            except Exception: pass

        d_str = st.strftime('%y%m%d') if hasattr(st, 'strftime') else "Unknown"
        # IMP-CR-04: Drop time from dedup key to handle slight start time variations
        key = f"{canonical_venue}|{race.race_number}|{d_str}|{race.discipline}"

        if key not in race_map:
            race_map[key] = race
        else:
            existing = race_map[key]
            # Merge runners with shadow-runner / double-header detection
            for nr in race.runners:
                # Find matching runner in existing race by number or name
                er = next((r for r in existing.runners if (r.number != 0 and r.number == nr.number) or (r.name.lower() == nr.name.lower())), None)
                if er:
                    # Update odds and metadata
                    er.odds.update(nr.odds)
                    if not er.win_odds and nr.win_odds: er.win_odds = nr.win_odds
                    if not er.number and nr.number: er.number = nr.number
                    er.metadata.update(nr.metadata)
                else:
                    # New runner discovered for this race (e.g. from a different aggregator)
                    existing.runners.append(nr)
            sources = set((existing.source or "").split(", "))
            sources.add(race.source or "Unknown")
            existing.source = ", ".join(sorted(list(filter(None, sources))))

    unique_races = list(race_map.values())
    logger.info("Unique races identified", count=len(unique_races))

    # Save snapshot
    snapshot_path = save_quarter_snapshot(daypart_tag, unique_races)

    # Log to DB
    await db.log_quarter_fetch(daypart_tag, region, str(daypart), len(unique_races), snapshot_path)

    # Save harvest_summary (existing pattern)
    try:
        harvest_file = get_writable_path("discovery_harvest.json")
        with open(harvest_file, "w") as f:
            json.dump(harvest_summary, f)
        await db.log_harvest(harvest_summary, region=region)

        # Generate and save health dashboard (P2-ENH-1 / P2-ENH-6)
        health_report = generate_adapter_health_report(harvest_summary)
        with open(get_writable_path("adapter_health_report.txt"), "w", encoding="utf-8") as f:
            f.write(health_report)
        logger.info("adapter_health_report_generated")
        # Print to console for GHA log visibility
        print(health_report)
    except Exception as e:
        logger.warning("failed_saving_harvest_summary", error=str(e))

    # Generate summary_grid.txt and field_matrix.txt (minimal versions)
    # This part is omitted here for brevity but could be added if needed for GHA visibility.

    return unique_races


async def run_score_now(
    config: Dict,
    daypart_tag: str,
    loaded_races: Optional[List[Race]] = None,
) -> List[Race]:
    """
    Loads a snapshot, refreshes odds for approaching races, scores, and persists tips.
    """
    logger = structlog.get_logger("run_score_now")
    db = FortunaDB()
    await db.initialize()

    # 2. Load races
    if loaded_races:
        logger.info("Using loaded races for scoring", count=len(loaded_races))
        cached_races = loaded_races
    else:
        cached_races = load_quarter_snapshot(daypart_tag)

    if cached_races is None:
        logger.warning("No snapshot file found for scoring", daypart_tag=daypart_tag)
        return []

    if not cached_races:
        logger.info("Snapshot exists but contains 0 races", daypart_tag=daypart_tag)
        return []

    # 4. Get already scored
    already_scored = await db.get_scored_race_ids(daypart_tag)

    # 5. Filter scorable
    scorable = get_scorable_races(cached_races, already_scored)

    if not scorable:
        logger.info("No races within scoring window", mtp_max=SCORING_MTP_MAX)
        return []

    logger.info("Scoring races approaching post", count=len(scorable))

    # 8. Refresh odds
    await refresh_odds_for_races(scorable, config)

    # 9. Score
    analyzer = SimplySuccessAnalyzer(config)
    analysis_result = analyzer.qualify_races(scorable)
    qualified = analysis_result.get("races", [])

    # 11. Persist
    if qualified:
        tracker = HotTipsTracker(db, config)
        await tracker.log_tips(qualified, daypart_tag=daypart_tag)
        logger.info("Success! We have persisted some incredible qualified tips! 🚀", count=len(qualified))

        # [CHANGE 6] Enhanced analytical logging for qualified races
        for race in qualified:
            active_runners = [r for r in race.runners if not r.scratched]
            total_active = len(active_runners)
            profile_tag = race.metadata.get('race_profile', 'standard')
            furlongs    = race.metadata.get('race_furlongs')
            dist_label  = f"{furlongs:.1f}f" if furlongs else (race.distance or '?')

            logger.info(
                "Qualified race",
                venue=race.venue,
                race=race.race_number,
                grade=race.metadata.get('qualification_grade'),
                score=race.qualification_score,
                profile=profile_tag,          # e.g. 'r9_6f_sweet_spot', 'goldmine_high_fav2'
                distance=dist_label,
                sec_fav=race.metadata.get('predicted_2nd_fav_odds'),
                runners=total_active,
                superfecta=race.metadata.get('is_superfecta_key', False),
            )

    # 12. Log scoring run
    await db.log_scoring_run(daypart_tag, len(qualified))

    # 13. Reports (goldmine_report.txt, summary_grid.txt for this batch)
    try:
        grid = generate_summary_grid(qualified, all_races=scorable)
        field_matrix = generate_field_matrix(scorable)
        gm_report = generate_goldmine_report(qualified, all_races=scorable)
        with open(get_writable_path('summary_grid.txt'), 'w', encoding='utf-8') as f:
            f.write(grid)
        with open(get_writable_path('field_matrix.txt'), 'w', encoding='utf-8') as f:
            f.write(field_matrix)
        with open(get_writable_path('goldmine_report.txt'), 'w', encoding='utf-8') as f:
            f.write(gm_report)
    except Exception as e:
        logger.error('failed_saving_text_reports', error=str(e))

    return qualified


async def run_discovery(
    target_dates: List[str],
    window_hours: Optional[int] = 8,
    loaded_races: Optional[List[Race]] = None,
    adapter_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    fetch_only: bool = False,
    live_dashboard: bool = False,
    track_odds: bool = False,
    region: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    now: Optional[datetime] = None,
    daypart_tag: Optional[str] = None
):
    """Legacy discovery wrapper that calls the new two-tier architecture."""
    logger = structlog.get_logger("run_discovery_legacy")

    # 1. Fetch
    if loaded_races is not None:
        races = loaded_races
    else:
        if not daypart_tag:
            dt = now or now_eastern()
            daypart = resolve_daypart_from_dt(dt)
            daypart_tag = f"{daypart}_{dt.strftime(DATE_FORMAT)}"

        if adapter_names:
            adapter_filter = adapter_names
        else:
            adapter_filter = list(get_daypart_discovery_adapters(daypart))

        races = await run_quarter_fetch(
            config=config or {},
            daypart_tag=daypart_tag,
            adapter_filter=adapter_filter,
            save_path=save_path,
            force_fetch=True
        )

    if fetch_only:
        return races

    db = FortunaDB()
    await db.initialize()

    analyzer = SimplySuccessAnalyzer(config or {})
    result = analyzer.qualify_races(races)
    qualified = result.get("races", [])

    logger.info("Legacy analysis complete", total=len(races), qualified=len(qualified))

    tracker = HotTipsTracker(db, config or {})
    # Use the tag computed during fetch or passed by caller
    if not daypart_tag:
        dt = now or now_eastern()
        daypart = resolve_daypart_from_dt(dt)
        daypart_tag = f"{daypart}_{dt.strftime(DATE_FORMAT)}"
    await tracker.log_tips(qualified, daypart_tag=daypart_tag)

    # 4. Post-run reporting (legacy fallback)
    try:
        from scripts.generate_gha_summary import EASTERN as SUMMARY_EASTERN
        timestamp = datetime.now(SUMMARY_EASTERN).strftime("%y%m%dT%H:%M:%S")
    except ImportError:
        timestamp = to_storage_format(now_eastern())

    report_data = {
        "qualified_races": [r.model_dump(mode='json') for r in qualified],
        "races": [r.model_dump(mode='json') for r in qualified],
        "analysis_metadata": result.get("criteria", {}),
        "timestamp": timestamp,
    }

    try:
        with open(get_writable_path("qualified_races.json"), "w") as f:
            json.dump(report_data, f, indent=4)
    except Exception: pass

    return qualified

async def start_desktop_app():
    """Starts a FastAPI server and opens a webview window for the Fortuna Dashboard."""
    try:
        import uvicorn
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse
        import webview
        import threading
        import time
    except ImportError as e:
        print(f"GUI dependencies missing: {e}. Install with 'pip install fastapi uvicorn pywebview'")
        return

    app = FastAPI(title="Fortuna Desktop Intelligence")

    @app.get("/", response_class=HTMLResponse)
    async def get_dashboard():
        # Retrieve latest Goldmines from the database
        db = FortunaDB()
        try:
            async with db.get_connection() as conn:
                try:
                    async with conn.execute(
                        "SELECT venue, race_number, selection_number, predicted_2nd_fav_odds, start_time "
                        "FROM tips ORDER BY id DESC LIMIT 50"
                    ) as cursor:
                        tips = await cursor.fetchall()
                except Exception as e:
                    print(f"DB query failed: {e}")
                    tips = []
        except Exception as e:
            print(f"Failed to connect to database: {e}")
            tips = []

        tips_html = "".join([
            f"<tr><td>{t[4]}</td><td>{t[0]}</td><td>R{t[1]}</td><td>#{t[2]}</td><td>{t[3]}</td></tr>"
            for t in tips
        ])

        return f"""
        <html>
            <head>
                <title>Fortuna Intelligence Desktop</title>
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; padding: 30px; }}
                    .container {{ max-width: 1200px; margin: auto; }}
                    h1 {{ color: #fbbf24; border-bottom: 2px solid #fbbf24; padding-bottom: 10px; text-transform: uppercase; letter-spacing: 2px; }}
                    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: #1e293b; border-radius: 8px; overflow: hidden; }}
                    th, td {{ padding: 15px; text-align: left; border-bottom: 1px solid #334155; }}
                    th {{ background: #334155; color: #fbbf24; }}
                    tr:hover {{ background: #475569; }}
                    .footer {{ margin-top: 30px; font-size: 0.8em; color: #94a3b8; text-align: center; }}
                    .btn {{ display: inline-block; background: #fbbf24; color: #0f172a; padding: 10px 20px; border-radius: 5px; text-decoration: none; font-weight: bold; margin-bottom: 20px; }}
                </style>
                <script>
                    setTimeout(() => {{ location.reload(); }}, 30000);
                </script>
            </head>
            <body>
                <div class="container">
                    <h1>Fortuna Intelligence Dashboard</h1>
                    <p>Monitoring global racing markets for Goldmine opportunities...</p>
                    <a href="/" class="btn">REFRESH NOW</a>
                    <table>
                        <thead>
                            <tr><th>Time Discovered</th><th>Venue</th><th>Race</th><th>Selection</th><th>Odds</th></tr>
                        </thead>
                        <tbody>
                            {tips_html or "<tr><td colspan='5'>No opportunities found yet. Run discovery to populate the database.</td></tr>"}
                        </tbody>
                    </table>
                    <div class="footer">Fortuna Intelligence Monolith - Sci-Fi Future Edition - Auto-refreshing every 30s</div>
                </div>
            </body>
        </html>
        """

    def run_server():
        uvicorn.run(app, host="127.0.0.1", port=8013, log_level="error")

    # Start FastAPI in a background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Wait a moment for server to initialize
    time.sleep(2.0)

    # Create and start the webview window if server is up
    if server_thread.is_alive():
        print("Launching Fortuna Desktop Window...")
        webview.create_window('Fortuna Intelligence Desktop', 'http://127.0.0.1:8013', width=1300, height=900)
        webview.start()
    else:
        print("⚠️ Error: GUI Server failed to start.")

async def ensure_browsers(force_install: bool = False):
    """Ensure browser dependencies are available for scraping."""

    # Skip Playwright in frozen apps if binary doesn't exist - use HTTP-only adapters
    if is_frozen():
        playwright_path = os.path.expanduser("~\\AppData\\Local\\ms-playwright")
        if not os.path.exists(playwright_path) and platform.system() == 'Windows':
            structlog.get_logger().info("Running as frozen app - Playwright disabled (binary not found)")
            return True

    try:
        # Check if playwright is installed and has a chromium binary
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            try:
                # We try to launch a headless browser to verify installation
                browser = await p.chromium.launch(headless=True)
                await browser.close()
                return True
            except Exception as e:
                structlog.get_logger().debug("Playwright launch failed during verification", error=str(e))
                if is_frozen():
                    structlog.get_logger().info("Frozen app: Playwright launch failed, using HTTP-only fallbacks")
                    return True
    except ImportError:
        structlog.get_logger().debug("Playwright not imported")
        if is_frozen(): return True

    if is_frozen():
        return True

    # Capability Improvement: Instead of auto-installing, warn the user unless opt-in
    # For now, we will assume it's NOT opt-in and ask for manual installation
    # because auto-pip-installing can be surprising.
    structlog.get_logger().warning("Browser dependencies (Playwright Chromium) missing.")
    print("\nBrowser dependencies missing!")
    print("To use browser-based adapters, please run:")
    print(f"  {sys.executable} -m pip install playwright==1.49.1")
    print(f"  {sys.executable} -m playwright install chromium")
    print("Alternatively, run Fortuna with: --install-browsers\n")

    # Check if we should auto-install via flag or environment variable
    if force_install or os.getenv("FORTUNA_AUTO_INSTALL_BROWSERS") == "1":
        structlog.get_logger().info("Auto-installing browser dependencies as requested...")
        try:
            # Remove version pin to avoid conflicts
            subprocess.run([sys.executable, "-m", "pip", "install", "playwright"], check=True, capture_output=True, text=True)
            subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True, capture_output=True, text=True)
            structlog.get_logger().info("Browser dependencies installed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            structlog.get_logger().error("Failed to auto-install browsers", error=str(e))
            return False

    return True # Continue with HTTP-only adapters

async def handle_early_exit_args(args: argparse.Namespace, config: Dict[str, Any]) -> bool:
    """Handles CLI arguments that should trigger an immediate exit (Capability Improvement)."""
    if args.quick_help:
        print_quick_help()
        return True
    if args.status:
        print_status_card(config)
        return True
    if args.show_log:
        await print_recent_logs()
        return True
    if args.open_dashboard:
        open_report_in_browser()
        return True
    return False

async def main_all_in_one():
    # Configure logging at the start of main
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO)
    )
    # Ensure DB path env is set if passed via argument or already in environment
    # Actually, we should probably add a --db-path arg here too for parity with analytics
    config = load_config()
    logger = structlog.get_logger("main")
    parser = argparse.ArgumentParser(description="Fortuna All-In-One - Professional Racing Intelligence")
    parser.add_argument("--date", type=str, help="Target date (YYMMDD)")
    parser.add_argument("--hours", type=int, default=8, help="Discovery time window in hours (default: 8)")
    parser.add_argument("--daypart", type=str, choices=["auto", "Q1", "Q2", "Q3", "Q4"], default="auto", help="Specify daypart for fetching/scoring")
    parser.add_argument("--quarter-fetch", action="store_true", help="Structural discovery sweep for current daypart. Saves snapshot for later scoring. Does NOT score races.")
    parser.add_argument("--score-now", action="store_true", help="Load quarter snapshot, refresh odds for races within 15min of post, score and persist tips.")
    parser.add_argument("--force-fetch", action="store_true", help="Force structural re-fetch even if quarter already cached. Use with --quarter-fetch.")
    parser.add_argument("--region", type=str, choices=["USA", "INT", "GLOBAL"], help="Filter by region (USA, INT or GLOBAL)")
    parser.add_argument("--quality", choices=["solid", "lousy"], help="Filter by adapter quality (Solid Top 3 vs others)")
    parser.add_argument("--include", type=str, help="Comma-separated adapter names to include")
    parser.add_argument("--save", type=str, help="Save races to JSON file")
    parser.add_argument("--load", type=str, help="Load races from JSON file(s), comma-separated")
    parser.add_argument("--fetch-only", action="store_true", help="Only fetch and save data, skip analysis and reporting")
    parser.add_argument("--db-path", type=str, help="Path to tip history database")
    parser.add_argument("--clear-db", action="store_true", help="Clear all tips from the database and exit")
    parser.add_argument("--gui", action="store_true", help="Start the Fortuna Desktop GUI")
    parser.add_argument("--live-dashboard", action="store_true", help="Show live updating terminal dashboard")
    parser.add_argument("--track-odds", action="store_true", help="Monitor live odds and send notifications")
    parser.add_argument("--status", action="store_true", help="Show application status card and latest metrics")
    parser.add_argument("--show-log", action="store_true", help="Print recent fetch/audit highlights")
    parser.add_argument("--quick-help", action="store_true", help="Show friendly onboarding guide")
    parser.add_argument("--open-dashboard", action="store_true", help="Open the HTML intelligence report in browser")
    parser.add_argument("--install-browsers", action="store_true", help="Install required browser dependencies (Playwright Chromium)")
    parser.add_argument("--include-health-checks", action="store_true", help="Include Official_* track health-check adapters in discovery. These confirm track websites are reachable but produce no race data. Off by default. Also affects --test-all-adapters.")
    parser.add_argument("--test-adapter", type=str, help="Test a single discovery adapter by name")
    parser.add_argument("--test-all-adapters", action="store_true", help="Run a health check on all available discovery adapters")
    args = parser.parse_args()

    # Handle early-exit arguments via helper (Hardening Fix/Improvement)
    if await handle_early_exit_args(args, config):
        return

    # Phase 5: Implement adapter testing tool
    if args.test_adapter or args.test_all_adapters:
        await ensure_browsers()
        test_date = args.date or now_eastern().strftime(DATE_FORMAT)
        all_classes = get_discovery_adapter_classes()

        if args.test_adapter:
            classes_to_test = [c for c in all_classes if getattr(c, "SOURCE_NAME", c.__name__) == args.test_adapter]
            if not classes_to_test:
                print(f"Error: Adapter '{args.test_adapter}' not found.")
                return
        else:
            if args.include_health_checks:
                classes_to_test = all_classes
            else:
                classes_to_test = [c for c in all_classes if not getattr(c, 'IS_HEALTH_CHECK_ONLY', False)]

        print(f"\n{'='*60}")
        print(f" ADAPTER TESTING MODE - DATE: {test_date}")
        print(f"{'='*60}\n")

        for cls in classes_to_test:
            name = getattr(cls, "SOURCE_NAME", cls.__name__)
            print(f"Testing {name}...")
            try:
                adapter = cls()
                start_time = time.time()
                # Use a generous timeout for testing
                races = await asyncio.wait_for(adapter.get_races(test_date), timeout=120.0)
                duration = time.time() - start_time

                status_emoji = "✅" if races else "⚠️"
                print(f"  STATUS:  {status_emoji} {len(races)} races returned")
                print(f"  TIME:    {duration:.1f}s")

                if races:
                    venues = Counter([r.venue for r in races])
                    venue_summary = ", ".join([f"{v} ({c})" for v, c in venues.most_common(3)])
                    print(f"  VENUES:  {venue_summary}...")

                    # Sample detail
                    sample = races[0]
                    runner_count = len(sample.runners)
                    trust_count = sum(1 for run in sample.runners if run.metadata.get("odds_source_trustworthy"))
                    trust_pct = (trust_count / runner_count * 100) if runner_count else 0

                    print(f"  SAMPLE:  {sample.venue} R{sample.race_number} | {sample.start_time.strftime('%H:%M')} | {runner_count} runners | discipline={sample.discipline}")
                    print(f"  ODDS:    {trust_count}/{runner_count} runners have trustworthy odds ({trust_pct:.0f}%)")

                await adapter.close()
            except Exception as e:
                print(f"  STATUS:  ❌ FAILED")
                print(f"  ERROR:   {str(e)}")
            print(f"\n{'-'*60}\n")
        return

    if args.db_path:
        os.environ["FORTUNA_DB_PATH"] = args.db_path

    # Print status card for all normal runs
    print_status_card(config)
    try:
        _gm_db = FortunaDB()
        await _gm_db.initialize()
        await print_goldmine_intelligence(_gm_db)
    except Exception:
        pass

    if args.install_browsers:
        await ensure_browsers(force_install=True)
        print("Installation complete.")
        return

    if args.gui:
        # Start GUI. It runs its own event loop for the webview.
        await ensure_browsers()
        await start_desktop_app()
        return

    if args.clear_db:
        db = FortunaDB()
        await db.clear_all_tips()
        await db.close()
        print("Database cleared successfully.")
        return

    adapter_filter = [n.strip() for n in args.include.split(",")] if args.include else None

    # Use default region if not specified
    if not args.region:
        args.region = config.get("region", {}).get("default", DEFAULT_REGION)
        structlog.get_logger().info("Using default region", region=args.region)

    # Region-based adapter filtering
    if args.region:
        if args.region == "USA":
            target_set = USA_DISCOVERY_ADAPTERS
        elif args.region == "INT":
            target_set = INT_DISCOVERY_ADAPTERS
        else:
            target_set = GLOBAL_DISCOVERY_ADAPTERS

        if adapter_filter:
            adapter_filter = [n for n in adapter_filter if n in target_set]
        else:
            adapter_filter = list(target_set)

    # Quality-based adapter filtering (Council of Superbrains Strategy)
    if args.quality:
        if args.quality == "solid":
            if adapter_filter:
                adapter_filter = [n for n in adapter_filter if n in SOLID_DISCOVERY_ADAPTERS]
            else:
                adapter_filter = list(SOLID_DISCOVERY_ADAPTERS)
        else:
            if adapter_filter:
                adapter_filter = [n for n in adapter_filter if n not in SOLID_DISCOVERY_ADAPTERS]
            else:
                # All adapters except solid
                all_names = [getattr(c, "SOURCE_NAME", c.__name__) for c in get_discovery_adapter_classes()]
                adapter_filter = [n for n in all_names if n not in SOLID_DISCOVERY_ADAPTERS]

        # Special case: TwinSpires needs to know its region internally if it's not filtered out
        # We can pass the region via config if we were creating adapters manually,
        # but here we use names.
        # Actually, I updated TwinSpiresAdapter to check self.config.get("region").
        # I need to ensure the adapter gets this config.

    loaded_races = None
    if args.load:
        loaded_races = []
        for path in args.load.split(","):
            path = path.strip()
            if not os.path.exists(path):
                print(f"Warning: File not found: {path}")
                logger.warning("Race data file not found", path=path)
                continue
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    loaded_races.extend([Race.model_validate(r) for r in data])
            except Exception as e:
                print(f"Error loading {path}: {e}")
                logger.error("Failed to load race data", path=path, error=str(e), exc_info=True)

    # Phase B8: CLI Execution Logic
    if args.quarter_fetch:
        if args.score_now:
            parser.error("--quarter-fetch and --score-now are mutually exclusive")

        await ensure_browsers()
        daypart_tag = get_daypart_tag(args)

        await run_quarter_fetch(
            config,
            daypart_tag,
            adapter_filter=adapter_filter,
            quality=args.quality,
            save_path=args.save,
            force_fetch=args.force_fetch,
            include_health_checks=args.include_health_checks
        )
        return

    if args.score_now:
        daypart_tag = get_daypart_tag(args)

        await run_score_now(
            config,
            daypart_tag,
            loaded_races=loaded_races
        )
        return

    if args.date:
        target_dates = [args.date]
    else:
        now = datetime.now(EASTERN)
        future = now + timedelta(hours=args.hours)

        target_dates = [now.strftime(DATE_FORMAT)]
        if future.date() > now.date():
            target_dates.append(future.strftime(DATE_FORMAT))

    await ensure_browsers()
    await run_discovery(
        target_dates,
        window_hours=args.hours,
        loaded_races=loaded_races,
        adapter_names=adapter_filter,
        save_path=args.save,
        fetch_only=args.fetch_only,
        live_dashboard=args.live_dashboard,
        track_odds=args.track_odds,
        region=args.region, # Pass region to run_discovery
        config=config
    )
    # Post-run UI enhancements (Council of Superbrains Directive)
    if config.get("ui", {}).get("auto_open_report", True) and not os.getenv("GITHUB_ACTIONS"):
        open_report_in_browser()

if __name__ == "__main__":
    if os.getenv("DEBUG_SNAPSHOTS"):
        os.makedirs("debug_snapshots", exist_ok=True)
    
    # Windows Event Loop Policy Fix (Project Hardening)
    if sys.platform == 'win32' and not getattr(sys, 'frozen', False):
        try:
            # For non-frozen mode, we prefer Proactor for full feature support
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        except AttributeError:
            try:
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            except AttributeError:
                pass

    try:
        asyncio.run(main_all_in_one())
    except KeyboardInterrupt:
        pass
