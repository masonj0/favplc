from __future__ import annotations
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
    Tuple,
    Type,
    TypeVar,
    Union,
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
EASTERN = ZoneInfo("America/New_York")

# Region-based adapter lists
USA_DISCOVERY_ADAPTERS: Final[set] = {"Equibase", "TwinSpires", "RacingPostB2B", "StandardbredCanada"}
INT_DISCOVERY_ADAPTERS: Final[set] = {
    "SkyRacingWorld", "AtTheRaces", "AtTheRacesGreyhound", "RacingPost",
    "TAB", "BetfairDataScientist", "Oddschecker", "Timeform", "BoyleSports",
    "SportingLife", "SkySports"
}

USA_RESULTS_ADAPTERS: Final[set] = {"EquibaseResults"}
INT_RESULTS_ADAPTERS: Final[set] = {
    "RacingPostResults", "RacingPostTote", "AtTheRacesResults",
    "SportingLifeResults", "SkySportsResults"
}

MAX_VALID_ODDS: Final[float] = 1000.0
MIN_VALID_ODDS: Final[float] = 1.01
DEFAULT_ODDS_FALLBACK: Final[float] = 2.75
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
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

CHROME_SEC_CH_UA: Final[str] = (
    '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"'
)

# Bet type keywords mapping (lowercase key -> display name)
BET_TYPE_KEYWORDS: Final[Dict[str, str]] = {
    "superfecta": "Superfecta",
    "spr": "Superfecta",
    "trifecta": "Trifecta",
    "tri": "Trifecta",
    "exacta": "Exacta",
    "ex": "Exacta",
    "quinella": "Quinella",
    "qn": "Quinella",
    "daily double": "Daily Double",
    "dbl": "Daily Double",
    "pick 3": "Pick 3",
    "pick 4": "Pick 4",
    "pick 5": "Pick 5",
    "pick 6": "Pick 6",
    "first 4": "Superfecta",
    "forecast": "Exacta",
    "tricast": "Trifecta",
}

# Discipline detection keywords
DISCIPLINE_KEYWORDS: Final[Dict[str, List[str]]] = {
    "Harness": ["harness", "trotter", "pacer", "standardbred", "trot", "pace"],
    "Greyhound": ["greyhound", "dog", "dogs"],
    "Quarter Horse": ["quarter horse", "quarterhorse"],
}


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


class Runner(FortunaBaseModel):
    id: Optional[str] = None
    name: str
    number: Optional[int] = Field(None, alias="saddleClothNumber")
    scratched: bool = False
    odds: Dict[str, OddsData] = Field(default_factory=dict)
    win_odds: Optional[float] = Field(None, alias="winOdds")
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
        return name.strip() or "Unknown"


class Race(FortunaBaseModel):
    id: str
    venue: str
    race_number: int = Field(..., alias="raceNumber", ge=1, le=100)
    start_time: datetime = Field(..., alias="startTime")
    runners: List[Runner] = Field(default_factory=list)

    @field_validator("start_time", mode="after")
    @classmethod
    def validate_eastern(cls, v: datetime) -> datetime:
        """Ensures all race start times are in US Eastern Time."""
        return ensure_eastern(v)
    source: str
    discipline: Optional[str] = None
    distance: Optional[str] = None
    field_size: Optional[int] = None
    available_bets: List[str] = Field(default_factory=list, alias="availableBets")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    qualification_score: Optional[float] = None
    is_error_placeholder: bool = False
    top_five_numbers: Optional[str] = None
    error_message: Optional[str] = None

# --- UTILITIES ---
def get_field(obj: Any, field_name: str, default: Any = None) -> Any:
    """Helper to get a field from either an object or a dictionary."""
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    return getattr(obj, field_name, default)


def clean_text(text: Any) -> str:
    """Strips leading/trailing whitespace and collapses internal whitespace."""
    if not text:
        return ""
    return " ".join(str(text).strip().split())


def node_text(n: Any) -> str:
    """Consistently extracts text from Scrapling Selectors and Selectolax Nodes."""
    if n is None:
        return ""
    # Selectolax nodes have a .text() method, Scrapling Selectors have a .text property
    txt = getattr(n, "text", None)
    if txt is None:
        return ""
    return txt().strip() if callable(txt) else str(txt).strip()


def get_canonical_venue(name: Optional[str]) -> str:
    """Returns a sanitized canonical form for deduplication keys."""
    if not name:
        return "unknown"
    # Call normalization first to strip race titles and ads
    norm = normalize_venue_name(name)
    # Remove everything in parentheses (extra safety)
    norm = re.sub(r"[\(\[（].*?[\)\]）]", "", norm)
    # Remove special characters, lowercase, strip
    res = re.sub(r"[^a-z0-9]", "", norm.lower())
    return res or "unknown"


def now_eastern() -> datetime:
    """Returns the current time in US Eastern Time."""
    return datetime.now(EASTERN)


def get_optimal_region_at_time(dt: datetime) -> str:
    """Determine which region has the most active racing at given time."""
    et_hour = dt.astimezone(EASTERN).hour
    # US Racing Window: 9am - 11pm ET
    if 9 <= et_hour < 23:
        return "USA"
    # International Window: 11pm - 9am ET (covers AUS/UK morning)
    else:
        return "INT"


def to_eastern(dt: datetime) -> datetime:
    """Converts a datetime object to US Eastern Time."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=EASTERN)
    return dt.astimezone(EASTERN)


def ensure_eastern(dt: datetime) -> datetime:
    """Ensures datetime is timezone-aware and in Eastern time. More strict than to_eastern."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=EASTERN)
    if dt.tzinfo is not EASTERN:
        return dt.astimezone(EASTERN)
    return dt


RACING_KEYWORDS = [
"PRIX", "CHASE", "HURDLE", "HANDICAP", "STAKES", "CUP", "LISTED", "GBB",
"RACE", "MEETING", "NOVICE", "TRIAL", "PLATE", "TROPHY", "CHAMPIONSHIP",
"JOCKEY", "TRAINER", "BEST ODDS", "GUARANTEED", "PRO/AM", "AUCTION",
"HUNT", "MARES", "FILLIES", "COLTS", "GELDINGS", "JUVENILE", "SELLING",
"CLAIMING", "OPTIONAL", "ALLOWANCE", "MAIDEN", "OPEN", "INVITATIONAL",
"CLASS ", "GRADE ", "GROUP ", "DERBY", "OAKS", "GUINEAS", "ELIE DE",
"FREDERIK", "CONNOLLY'S", "QUINNBET", "RED MILLS", "IRISH EBF", "SKY BET",
"CORAL", "BETFRED", "WILLIAM HILL", "UNIBET", "PADDY POWER", "BETFAIR",
"GET THE BEST", "CHELTENHAM TRIALS", "PORSCHE", "IMPORTED", "IMPORTE", "THE JOC",
"PREMIO", "GRANDE", "CLASSIC", "SPRINT", "DASH", "MILE", "STAYERS",
"BOWL", "MEMORIAL", "PURSE", "CONDITION", "NIGHT", "EVENING", "DAY",
"4RACING", "WILGERBOSDRIFT", "YOUCANBETONUS", "FOR HOSPITALITY", "SA ", "TAB ",
"DE ", "DU ", "DES ", "LA ", "LE ", "AU ", "WELCOME", "BET ", "WITH ", "AND ",
"NEXT", "WWW", "GAMBLE", "BETMGM", "TV", "ONLINE", "LUCKY", "RACEWAY",
"SPEEDWAY", "DOWNS", "PARK", "HARNESS", " STANDARDBRED", "FORM GUIDE", "FULL FIELDS"
]


VENUE_MAP = {
"ABU DHABI": "Abu Dhabi",
"AQUEDUCT": "Aqueduct",
"ARGENTAN": "Argentan",
"ASCOT": "Ascot",
"AYR": "Ayr",
"BAHRAIN": "Bahrain",
"BANGOR ON DEE": "Bangor-on-Dee",
"CATTERICK": "Catterick",
"CATTERICK BRIDGE": "Catterick",
"CENTRAL PARK": "Central Park",
"CHELMSFORD": "Chelmsford",
"CHELMSFORD CITY": "Chelmsford",
"CURRAGH": "Curragh",
"DEAUVILLE": "Deauville",
"DELTA DOWNS": "Delta Downs",
"DONCASTER": "Doncaster",
"DOVER DOWNS": "Dover Downs",
"DOWN ROYAL": "Down Royal",
"DUNDALK": "Dundalk",
"DUNSTALL PARK": "Wolverhampton",
"EPSOM": "Epsom",
"EPSOM DOWNS": "Epsom",
"FAIR GROUNDS": "Fair Grounds",
"FONTWELL": "Fontwell Park",
"FONTWELL PARK": "Fontwell Park",
"GREAT YARMOUTH": "Great Yarmouth",
"GULFSTREAM": "Gulfstream Park",
"GULFSTREAM PARK": "Gulfstream Park",
"HAYDOCK": "Haydock Park",
"HAYDOCK PARK": "Haydock Park",
"HOOSIER PARK": "Hoosier Park",
"HOVE": "Hove",
"KEMPTON": "Kempton Park",
"KEMPTON PARK": "Kempton Park",
"LAUREL PARK": "Laurel Park",
"LINGFIELD": "Lingfield Park",
"LINGFIELD PARK": "Lingfield Park",
"LOS ALAMITOS": "Los Alamitos",
"MARONAS": "Maronas",
"MEADOWLANDS": "Meadowlands",
"MEYDAN": "Meydan",
"MIAMI VALLEY": "Miami Valley",
"MIAMI VALLEY RACEWAY": "Miami Valley",
"MOHAWK": "Mohawk",
"MOHAWK PARK": "Mohawk",
"MUSSELBURGH": "Musselburgh",
"NAAS": "Naas",
"NEWCASTLE": "Newcastle",
"NEWMARKET": "Newmarket",
"NORTHFIELD PARK": "Northfield Park",
"OXFORD": "Oxford",
"PAU": "Pau",
"POCONO DOWNS": "Pocono Downs",
"SAM HOUSTON": "Sam Houston",
"SAM HOUSTON RACE PARK": "Sam Houston",
"SANDOWN": "Sandown Park",
"SANDOWN PARK": "Sandown Park",
"SANTA ANITA": "Santa Anita",
"SARATOGA": "Saratoga",
"SARATOGA HARNESS": "Saratoga Harness",
"SCIOTO DOWNS": "Scioto Downs",
"SHEFFIELD": "Sheffield",
"STRATFORD": "Stratford-on-Avon",
"SUNLAND PARK": "Sunland Park",
"TAMPA BAY DOWNS": "Tampa Bay Downs",
"THURLES": "Thurles",
"TURF PARADISE": "Turf Paradise",
"TURFFONTEIN": "Turffontein",
"UTTOXETER": "Uttoxeter",
"VINCENNES": "Vincennes",
"WARWICK": "Warwick",
"WETHERBY": "Wetherby",
"WOLVERHAMPTON": "Wolverhampton",
"WOODBINE": "Woodbine",
"WOODBINE MOHAWK": "Mohawk",
"WOODBINE MOHAWK PARK": "Mohawk",
"YARMOUTH": "Great Yarmouth",
"YONKERS": "Yonkers",
"YONKERS RACEWAY": "Yonkers",
}


def normalize_venue_name(name: Optional[str]) -> str:
    """
    Normalizes a racecourse name to a standard format.
    Aggressively strips race names, sponsorships, and country noise.
    """
    if not name:
        return "Unknown"

    # 1. Initial Cleaning: Replace dashes and strip all parenthetical info
    # Handle full-width parentheses and brackets often found in international data
    name = str(name).replace("-", " ")
    name = re.sub(r"[\(\[（].*?[\)\]）]", " ", name)

    cleaned = clean_text(name)
    if not cleaned:
        return "Unknown"

    # 2. Aggressive Race/Meeting Name Stripping
    # If these keywords are found, assume everything after is the race name.

    upper_name = cleaned.upper()
    earliest_idx = len(cleaned)
    for kw in RACING_KEYWORDS:
        # Check for keyword with leading space
        idx = upper_name.find(" " + kw)
        if idx != -1:
            earliest_idx = min(earliest_idx, idx)

    track_part = cleaned[:earliest_idx].strip()
    if not track_part:
        track_part = cleaned

    # Handle repetition check (e.g., "Bahrain Bahrain" -> "Bahrain")
    words = track_part.split()
    if len(words) > 1 and words[0].lower() == words[1].lower():
        track_part = words[0]

    upper_track = track_part.upper()

    # 3. High-Confidence Mapping
    # Map raw/cleaned names to canonical display names.

    # Direct match
    if upper_track in VENUE_MAP:
        return VENUE_MAP[upper_track]

    # Prefix match (sort by length desc to avoid partial matches on shorter names)
    for known_track in sorted(VENUE_MAP.keys(), key=len, reverse=True):
        if upper_name.startswith(known_track):
            return VENUE_MAP[known_track]

    return track_part.title()


def parse_odds_to_decimal(odds_str: Any) -> Optional[float]:
    """
    Parses various odds formats (fractional, decimal) into a float decimal.
    Uses advanced heuristics to extract odds from noisy strings.
    """
    if odds_str is None: return None
    s = str(odds_str).strip().upper()

    # Remove common non-odds noise and currency symbols
    s = re.sub(r"[$\s\xa0]", "", s)
    s = re.sub(r"(ML|MTP|AM|PM|LINE|ODDS|PRICE)[:=]*", "", s)

    if s in ("EVN", "EVEN", "EVS", "EVENS"): return 2.0
    if any(kw in s for kw in ("SCR", "SCRATCHED", "N/A", "NR", "VOID")): return None

    try:
        # 1. Fractional Format: "7/4", "7-4", "7 TO 4"
        groups = re.search(r"(\d+)\s*(?:[/\-]|TO)\s*(\d+)", s)
        if groups:
            num, den = int(groups.group(1)), int(groups.group(2))
            if den > 0: return round((num / den) + 1.0, 2)

        # 2. Decimal Format: "5.00", "10.5"
        decimal_match = re.search(r"(\d+\.\d+)", s)
        if decimal_match:
            value = float(decimal_match.group(1))
            if MIN_VALID_ODDS <= value < MAX_VALID_ODDS: return round(value, 2)

        # 3. Simple Integer as fractional odds (e.g., "5" often means "5/1")
        # Only apply if it's a likely odds value (not saddle cloth 1-20)
        int_match = re.match(r"^(\d+)$", s)
        if int_match:
            val = int(int_match.group(1))
            # Heuristic: only treat as fractional odds if it's in a likely range (1-50)
            # to avoid misinterpreting horse numbers or race numbers.
            if 1 <= val <= 50:
                return float(val + 1)

    except Exception: pass
    return None


def is_valid_odds(odds: Any) -> bool:
    if odds is None: return False
    try:
        odds_float = float(odds)
        return MIN_VALID_ODDS <= odds_float < MAX_VALID_ODDS
    except Exception: return False


def create_odds_data(source_name: str, win_odds: Any, place_odds: Any = None) -> Optional[OddsData]:
    if not is_valid_odds(win_odds): return None
    return OddsData(win=float(win_odds), place=float(place_odds) if is_valid_odds(place_odds) else None, source=source_name)


def scrape_available_bets(html_content: str) -> List[str]:
    if not html_content: return []
    available_bets: List[str] = []
    html_lower = html_content.lower()
    for kw, bet_name in BET_TYPE_KEYWORDS.items():
        if re.search(rf"\b{re.escape(kw)}\b", html_lower) and bet_name not in available_bets:
            available_bets.append(bet_name)
    return available_bets


def detect_discipline(html_content: str) -> str:
    if not html_content: return "Thoroughbred"
    html_lower = html_content.lower()
    for disc, keywords in DISCIPLINE_KEYWORDS.items():
        if any(kw in html_lower for kw in keywords): return disc
    return "Thoroughbred"


class SmartOddsExtractor:
    """
    Advanced heuristics for extracting odds from noisy HTML or text.
    Scans for various patterns and returns the first plausible odds found.
    """
    @staticmethod
    def extract_from_text(text: str) -> Optional[float]:
        if not text: return None
        # Try to find common odds patterns in the text
        # 1. Decimal odds (e.g. 5.00, 10.5)
        decimals = re.findall(r"(\d+\.\d+)", text)
        for d in decimals:
            val = float(d)
            if MIN_VALID_ODDS <= val < MAX_VALID_ODDS: return round(val, 2)

        # 2. Fractional odds (e.g. 7/4, 10-1)
        fractions = re.findall(r"(\d+)\s*[/\-]\s*(\d+)", text)
        for num, den in fractions:
            n, d = int(num), int(den)
            if d > 0 and (n/d) > 0.1: return round((n / d) + 1.0, 2)

        return None

    @staticmethod
    def extract_from_node(node: Any) -> Optional[float]:
        """Scans a selectolax node for odds using multiple strategies."""
        # Strategy 1: Look at text content of the entire node
        if hasattr(node, 'text'):
            if val := SmartOddsExtractor.extract_from_text(node.text()):
                return val

        # Strategy 2: Look at attributes
        if hasattr(node, 'attributes'):
            for attr in ["data-odds", "data-price", "data-bestprice", "title"]:
                if val_str := node.attributes.get(attr):
                    if val := parse_odds_to_decimal(val_str):
                        return val

        return None


def generate_race_id(prefix: str, venue: str, start_time: datetime, race_number: int, discipline: Optional[str] = None) -> str:
    venue_slug = re.sub(r"[^a-z0-9]", "", venue.lower())
    date_str = start_time.strftime("%Y%m%d")
    time_str = start_time.strftime("%H%M")

    # Always include a discipline suffix for consistency and better matching
    dl = (discipline or "Thoroughbred").lower()
    if "harness" in dl: disc_suffix = "_h"
    elif "greyhound" in dl: disc_suffix = "_g"
    elif "quarter" in dl: disc_suffix = "_q"
    else: disc_suffix = "_t"

    return f"{prefix}_{venue_slug}_{date_str}_{time_str}_R{race_number}{disc_suffix}"


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
class GlobalResourceManager:
    """Manages shared resources like HTTP clients and semaphores."""
    _httpx_client: Optional[httpx.AsyncClient] = None
    _locks: ClassVar[dict[asyncio.AbstractEventLoop, asyncio.Lock]] = {}
    _lock_initialized: ClassVar[threading.Lock] = threading.Lock()
    _global_semaphore: Optional[asyncio.Semaphore] = None

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
        Returns a shared httpx client.
        If timeout is provided and differs from current client, the client is recreated.
        """
        lock = await cls._get_lock()
        async with lock:
            if cls._httpx_client is not None:
                if timeout is not None and abs(cls._httpx_client.timeout.read - timeout) > 0.001:
                    try:
                        await cls._httpx_client.aclose()
                    except Exception:
                        pass
                    cls._httpx_client = None

            if cls._httpx_client is None:
                use_timeout = timeout or DEFAULT_REQUEST_TIMEOUT
                cls._httpx_client = httpx.AsyncClient(
                    follow_redirects=True,
                    timeout=httpx.Timeout(use_timeout),
                    headers={**DEFAULT_BROWSER_HEADERS, "User-Agent": CHROME_USER_AGENT},
                    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
                )
        return cls._httpx_client

    @classmethod
    def get_global_semaphore(cls) -> asyncio.Semaphore:
        if cls._global_semaphore is None:
            try:
                # Attempt to get running loop to ensure we are in async context
                asyncio.get_running_loop()
                cls._global_semaphore = asyncio.Semaphore(DEFAULT_CONCURRENT_REQUESTS * 2)
            except RuntimeError:
                # Fallback if called outside a loop
                cls._global_semaphore = asyncio.Semaphore(DEFAULT_CONCURRENT_REQUESTS * 2)
                return cls._global_semaphore
        return cls._global_semaphore

    @classmethod
    async def cleanup(cls):
        if cls._httpx_client:
            await cls._httpx_client.aclose()
            cls._httpx_client = None


class BrowserEngine(Enum):
    CAMOUFOX = "camoufox"
    PLAYWRIGHT = "playwright"
    CURL_CFFI = "curl_cffi"
    PLAYWRIGHT_LEGACY = "playwright_legacy"
    HTTPX = "httpx"


class FetchStrategy(FortunaBaseModel):
    primary_engine: BrowserEngine = BrowserEngine.PLAYWRIGHT
    enable_js: bool = True
    stealth_mode: str = "fast"
    block_resources: bool = True
    max_retries: int = Field(3, ge=0, le=10)
    timeout: int = Field(DEFAULT_REQUEST_TIMEOUT, ge=1, le=300)
    page_load_strategy: str = "domcontentloaded"
    wait_for_selector: Optional[str] = None


class SmartFetcher:
    BOT_DETECTION_KEYWORDS: ClassVar[List[str]] = ["datadome", "perimeterx", "access denied", "captcha", "cloudflare", "please verify"]
    def __init__(self, strategy: Optional[FetchStrategy] = None):
        self.strategy = strategy or FetchStrategy()
        self.logger = structlog.get_logger(self.__class__.__name__)
        self._engine_health = {
            BrowserEngine.CAMOUFOX: 0.9,
            BrowserEngine.CURL_CFFI: 0.8,
            BrowserEngine.PLAYWRIGHT: 0.7,
            BrowserEngine.HTTPX: 0.5
        }
        self.last_engine: str = "unknown"
        if BROWSERFORGE_AVAILABLE:
            self.header_gen = HeaderGenerator()
            self.fingerprint_gen = FingerprintGenerator()
        else:
            self.header_gen = None
            self.fingerprint_gen = None

    async def fetch(self, url: str, **kwargs: Any) -> Any:
        method = kwargs.pop("method", "GET").upper()
        kwargs.pop("url", None)
        # Check if engines are available before sorting
        available_engines = [e for e in self._engine_health.keys()]
        if not curl_requests and BrowserEngine.CURL_CFFI in available_engines:
            available_engines.remove(BrowserEngine.CURL_CFFI)
        if not ASYNC_SESSIONS_AVAILABLE:
            for e in [BrowserEngine.CAMOUFOX, BrowserEngine.PLAYWRIGHT]:
                if e in available_engines: available_engines.remove(e)

        if not available_engines:
            self.logger.error("no_fetch_engines_available", url=url)
            raise FetchError("No fetch engines available (install curl_cffi or scrapling)")

        engines = sorted(available_engines, key=lambda e: self._engine_health[e], reverse=True)
        if self.strategy.primary_engine in engines:
            engines.remove(self.strategy.primary_engine)
            engines.insert(0, self.strategy.primary_engine)
        last_error: Optional[Exception] = None
        for engine in engines:
            try:
                response = await self._fetch_with_engine(engine, url, method=method, **kwargs)
                self._engine_health[engine] = min(1.0, self._engine_health[engine] + 0.1)
                self.last_engine = engine.value
                return response
            except Exception as e:
                self.logger.debug(f"Engine {engine.value} failed", error=str(e))
                self._engine_health[engine] = max(0.0, self._engine_health[engine] - 0.2)
                last_error = e
                continue
        err_msg = repr(last_error) if last_error else "All fetch engines failed"
        self.logger.error("all_engines_failed", url=url, error=err_msg)
        raise last_error or FetchError("All fetch engines failed")

    
    async def _fetch_with_engine(self, engine: BrowserEngine, url: str, method: str, **kwargs: Any) -> Any:
        # Generate browserforge headers if available and not explicitly provided
        if BROWSERFORGE_AVAILABLE and "headers" not in kwargs:
            try:
                # Generate headers and a corresponding user agent
                fingerprint = self.fingerprint_gen.generate()
                headers = self.header_gen.generate()
                # Ensure User-Agent is consistent between fingerprint and headers
                headers['User-Agent'] = getattr(fingerprint.navigator, 'userAgent', getattr(fingerprint.navigator, 'user_agent', CHROME_USER_AGENT))
                kwargs["headers"] = headers
                self.logger.debug("Generated browserforge headers", engine=engine.value)
            except Exception as e:
                self.logger.warning("Failed to generate browserforge headers", error=str(e))

        # Define browser-specific arguments to strip for non-browser engines
        BROWSER_SPECIFIC_KWARGS = [
            "network_idle", "wait_selector", "wait_until", "impersonate",
            "stealth", "block_resources", "wait_for_selector", "stealth_mode"
        ]

        if engine == BrowserEngine.HTTPX:
            # Pass strategy timeout if present in kwargs or use default
            timeout = kwargs.get("timeout", self.strategy.timeout)
            client = await GlobalResourceManager.get_httpx_client(timeout=timeout)

            # Remove timeout and browser-specific keys from kwargs
            req_kwargs = {
                k: v for k, v in kwargs.items()
                if k != "timeout" and k not in BROWSER_SPECIFIC_KWARGS
            }
            resp = await client.request(method, url, timeout=timeout, **req_kwargs)
            resp.status = resp.status_code
            return resp
        
        if engine == BrowserEngine.CURL_CFFI:
            if not curl_requests:
                raise ImportError("curl_cffi is not available")
            
            self.logger.debug(f"Using curl_cffi for {url}")
            timeout = kwargs.get("timeout", self.strategy.timeout)

            # Default headers if still not present after browserforge attempt
            headers = kwargs.get("headers", {**DEFAULT_BROWSER_HEADERS, "User-Agent": CHROME_USER_AGENT})
            # Respect impersonate if provided, otherwise default
            impersonate = kwargs.get("impersonate", "chrome110")
            
            # Remove keys that curl_requests.AsyncSession.request doesn't like
            clean_kwargs = {
                k: v for k, v in kwargs.items()
                if k not in ["timeout", "headers", "impersonate"] + BROWSER_SPECIFIC_KWARGS
            }
            
            async with curl_requests.AsyncSession() as s:
                resp = await s.request(
                    method, 
                    url, 
                    timeout=timeout, 
                    headers=headers, 
                    impersonate=impersonate,
                    **clean_kwargs
                )
                resp.status = resp.status_code
                return resp

        if not ASYNC_SESSIONS_AVAILABLE:
            raise ImportError("scrapling not available")

        # Scrapling specific kwargs
        SCRAPLING_KWARGS = ["network_idle", "wait_selector", "wait_until", "stealth_mode", "block_resources", "timeout"]
        scrapling_kwargs = {k: v for k, v in kwargs.items() if k in SCRAPLING_KWARGS}
        if "timeout" not in scrapling_kwargs:
             scrapling_kwargs["timeout"] = kwargs.get("timeout", self.strategy.timeout)
            
        # For other engines, we use AsyncFetcher from scrapling
        if engine == BrowserEngine.CAMOUFOX:
            async with AsyncStealthySession(headless=True) as s:
                resp = await s.fetch(url, method=method, **scrapling_kwargs)
            return resp
        elif engine == BrowserEngine.PLAYWRIGHT:
            async with AsyncDynamicSession(headless=True) as s:
                resp = await s.fetch(url, method=method, **scrapling_kwargs)
            return resp
        else:
            # Fallback to simple fetcher
            async with AsyncFetcher() as fetcher:
                if method.upper() == "GET":
                    resp = await fetcher.get(url, **kwargs)
                else:
                    resp = await fetcher.post(url, **kwargs)
            return resp


    async def close(self) -> None:
        """
        Shared resources are managed by GlobalResourceManager.
        This remains for API compatibility.
        """
        pass


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


@dataclass
class RateLimiter:
    requests_per_second: float = 10.0
    _tokens: float = field(default=10.0, init=False)
    _last_update: float = field(default_factory=time.time, init=False)
    _lock: Optional[asyncio.Lock] = field(default=None, init=False)
    _lock_sentinel: ClassVar[threading.Lock] = threading.Lock()

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            with self._lock_sentinel:
                if self._lock is None:
                    try:
                        loop = asyncio.get_running_loop()
                        self._lock = asyncio.Lock()
                    except RuntimeError:
                        pass
        return self._lock

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


class AdapterMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency_ms = 0.0
        self.consecutive_failures = 0
        self.last_failure_reason: Optional[str] = None
    @property
    def success_rate(self) -> float:
        return self.successful_requests / self.total_requests if self.total_requests > 0 else 1.0
    async def record_success(self, latency_ms: float) -> None:
        with self._lock:
            self.total_requests += 1
        self.successful_requests += 1
        self.total_latency_ms += latency_ms
        self.consecutive_failures = 0
        self.last_failure_reason: Optional[str] = None
    async def record_failure(self, error: str) -> None:
        with self._lock:
            self.total_requests += 1
        self.failed_requests += 1
        self.consecutive_failures += 1
        self.last_failure_reason = error
    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "success_rate": self.success_rate,
            "failed_requests": self.failed_requests,
            "consecutive_failures": self.consecutive_failures,
            "last_failure_reason": getattr(self, "last_failure_reason", None)
        }


# --- MIXINS ---
class JSONParsingMixin:
    """Mixin for safe JSON extraction from HTML and scripts."""
    def _parse_json_from_script(self, parser: HTMLParser, selector: str, context: str = "script") -> Optional[Any]:
        script = parser.css_first(selector)
        if not script:
            return None
        try:
            return json.loads(script.text())
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
                results.append(json.loads(script.text()))
            except json.JSONDecodeError as e:
                if hasattr(self, 'logger'):
                    self.logger.error("failed_parsing_json_in_list", context=context, selector=selector, error=str(e))
        return results


class BrowserHeadersMixin:
    def _get_browser_headers(self, host: Optional[str] = None, referer: Optional[str] = None, **extra: str) -> Dict[str, str]:
        h = {**DEFAULT_BROWSER_HEADERS, "User-Agent": CHROME_USER_AGENT, "sec-ch-ua": CHROME_SEC_CH_UA, "sec-ch-ua-mobile": "?0", "sec-ch-ua-platform": '"Windows"'}
        if host: h["Host"] = host
        if referer: h["Referer"] = referer
        h.update(extra)
        return h


class DebugMixin:
    def _save_debug_snapshot(self, content: str, context: str, url: Optional[str] = None) -> None:
        if not content: return
        try:
            d = Path("debug_snapshots")
            d.mkdir(parents=True, exist_ok=True)
            f = d / f"{context}_{datetime.now(EASTERN).strftime('%Y%m%d_%H%M%S')}.html"
            with open(f, "w", encoding="utf-8") as out:
                if url: out.write(f"<!-- URL: {url} -->\n")
                out.write(content)
        except Exception: pass
    def _save_debug_html(self, content: str, filename: str, **kwargs) -> None:
        self._save_debug_snapshot(content, filename)


class RacePageFetcherMixin:
    async def _fetch_race_pages_concurrent(self, metadata: List[Dict[str, Any]], headers: Dict[str, str], semaphore_limit: int = 5, delay_range: tuple[float, float] = (0.5, 1.5)) -> List[Dict[str, Any]]:
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
                        resp = await self.make_request("GET", url, headers=headers)
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

    def __init__(self, source_name: str, base_url: str, rate_limit: float = 10.0, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        self.source_name = source_name
        self.base_url = base_url.rstrip("/")
        self.config = config or {}
        # Merge kwargs into config
        self.config.update(kwargs)

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
            if is_frozen() and strategy.primary_engine in [BrowserEngine.PLAYWRIGHT, BrowserEngine.CAMOUFOX]:
                self.logger.info("Skipping browser-dependent adapter in monolith mode")
                return []

            if not await self.circuit_breaker.allow_request(): return []
            await self.rate_limiter.acquire()
            raw = await self._fetch_data(date)
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
        for r in races:
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
                self.logger.warning("suspicious_runner_numbers", venue=r.venue, field_size=field_size)
                for i, run in enumerate(r.runners):
                    run.number = i + 1

            for runner in r.runners:
                if not runner.scratched:
                    # Explicitly enrich win_odds using all available sources (including fallbacks)
                    best = _get_best_win_odds(runner)
                    if best:
                        runner.win_odds = float(best)
        valid, warnings = DataValidationPipeline.validate_parsed_races(races, adapter_name=self.source_name)
        return valid

    async def make_request(self, method: str, url: str, **kwargs: Any) -> Any:
        full_url = url if url.startswith("http") else f"{self.base_url}/{url.lstrip('/')}"
        self.logger.debug("Requesting", method=method, url=full_url)
        # Apply global concurrency limit (Memory Directive Fix)
        async with GlobalResourceManager.get_global_semaphore():
            try:
                resp = await self.smart_fetcher.fetch(full_url, method=method, **kwargs)
                status = get_resp_status(resp)
                self.logger.debug("Response received", method=method, url=full_url, status=status)
                return resp
            except Exception as e:
                self.logger.error("Request failed", method=method, url=full_url, error=str(e))
                return None

    async def close(self) -> None: await self.smart_fetcher.close()
    async def shutdown(self) -> None: await self.close()

# ============================================================================
# ADAPTER IMPLEMENTATIONS
# ============================================================================

# ----------------------------------------
# EquibaseAdapter
# ----------------------------------------
class RacingAndSportsAdapter(BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
    """
    Adapter for Racing & Sports (RAS).
    Note: Highly protected by Cloudflare; requires advanced impersonation.
    """
    SOURCE_NAME: ClassVar[str] = "RacingAndSports"
    BASE_URL: ClassVar[str] = "https://www.racingandsports.com.au"

    async def fetch_races(self, date_str: str) -> List[Race]:
        url = f"{self.BASE_URL}/racing-index?date={date_str}"
        headers = self.get_browser_headers()

        try:
            resp = await self.client.get(url, headers=headers)
            if resp.status_code != 200:
                self.logger.warning(f"RAS blocked with status {resp.status_code}")
                return []

            tree = HTMLParser(resp.text)
            races = []
            # RAS uses tables for different regions (Australia, UK, etc.)
            tables = tree.css("table.table-index")
            for table in tables:
                for row in table.css("tbody tr"):
                    venue_cell = row.css_first("td.venue-name")
                    if not venue_cell: continue
                    venue_name = venue_cell.text(strip=True)

                    for link in row.css("td a.race-link"):
                        race_url = self.BASE_URL + link.attributes.get("href", "")
                        r_num_match = re.search(r"R(\d+)", link.text(strip=True))
                        if not r_num_match: continue
                        r_num = int(r_num_match.group(1))

                        # We'll queue these for detail fetching
                        # (In a real implementation, we'd use a pool)
                        detail = await self._fetch_race_detail(race_url, venue_name, r_num, date_str)
                        if detail: races.append(detail)
            return races
        except Exception as e:
            self.logger.error(f"RAS Fetch Error: {e}")
            return []

    async def _fetch_race_detail(self, url: str, venue: str, num: int, date_str: str) -> Optional[Race]:
        try:
            resp = await self.client.get(url, headers=self.get_browser_headers())
            if resp.status_code != 200: return None
            tree = HTMLParser(resp.text)

            runners = []
            for row in tree.css("tr.runner-row"):
                name = row.css_first(".runner-name").text(strip=True) if row.css_first(".runner-name") else "Unknown"
                saddle = int(row.css_first(".runner-number").text(strip=True)) if row.css_first(".runner-number") else 0
                odds_text = row.css_first(".odds-win").text(strip=True) if row.css_first(".odds-win") else "0"
                odds = float(odds_text) if odds_text.replace(".","").isdigit() else 0.0
                runners.append(Runner(name=name, selection_number=saddle, win_odds=odds))

            if not runners: return None
            return Race(venue=venue, race_number=num, start_time=datetime.now(EASTERN), runners=runners, source=self.SOURCE_NAME, url=url)
        except Exception: return None

class SkyRacingWorldAdapter(BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
    SOURCE_NAME: ClassVar[str] = "SkyRacingWorld"
    BASE_URL: ClassVar[str] = "https://www.skyracingworld.com"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return FetchStrategy(
            primary_engine=BrowserEngine.CURL_CFFI,
            enable_js=True,
            stealth_mode="camouflage",
            timeout=60
        )

    def _get_headers(self) -> Dict[str, str]:
        return self._get_browser_headers(host="www.skyracingworld.com")

    async def make_request(self, method: str, url: str, **kwargs: Any) -> Any:
        kwargs.setdefault("impersonate", "chrome120")
        return await super().make_request(method, url, **kwargs)

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        # Index for the day
        index_url = f"/form-guide/thoroughbred/{date}"
        resp = await self.make_request("GET", index_url, headers=self._get_headers())
        if not resp or not resp.text:
            if resp: self.logger.warning("Unexpected status", status=resp.status, url=index_url)
            return None
        self._save_debug_snapshot(resp.text, f"skyracing_index_{date}")

        parser = HTMLParser(resp.text)
        track_links = defaultdict(list)
        now = now_eastern()
        today_str = now.strftime("%Y-%m-%d")

        # Optimization: If it's late in ET, skip countries that are finished
        # Europe/Turkey/SA usually finished by 18:00 ET
        skip_finished_countries = (now.hour >= 18 or now.hour < 6) and (date == today_str)
        finished_keywords = ["turkey", "south-africa", "united-kingdom", "france", "germany", "dubai", "bahrain"]

        for link in parser.css("a.fg-race-link"):
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
            return None
        # Limit to first 50 to avoid hammering
        pages = await self._fetch_race_pages_concurrent(metadata[:50], self._get_headers(), semaphore_limit=5)
        return {"pages": pages, "date": date}

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data or not raw_data.get("pages"): return []
        try: race_date = datetime.strptime(raw_data["date"], "%Y-%m-%d").date()
        except Exception: return []
        races: List[Race] = []
        for item in raw_data["pages"]:
            html_content = item.get("html")
            if not html_content: continue
            try:
                race = self._parse_single_race(html_content, item.get("url", ""), race_date)
                if race: races.append(race)
            except Exception: pass
        return races

    def _parse_single_race(self, html_content: str, url: str, race_date: date) -> Optional[Race]:
        parser = HTMLParser(html_content)

        # Extract venue and time from header
        # Format usually: "14:30 LINGFIELD" or similar
        header = parser.css_first(".sdc-site-racing-header__name") or parser.css_first("h1") or parser.css_first("h2")
        if not header: return None

        header_text = clean_text(header.text())
        match = re.search(r"(\d{1,2}:\d{2})\s+(.+)", header_text)
        if match:
            time_str = match.group(1)
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
        for row in parser.css(".runner_row") or parser.css(".mobile-runner"):
            try:
                name_node = row.css_first(".horseName") or row.css_first("a[href*='/horse/']")
                if not name_node: continue
                name = clean_text(name_node.text())

                num_node = row.css_first(".tdContent b") or row.css_first("[data-tab-no]")
                number = 0
                if num_node:
                    if num_node.attributes.get("data-tab-no"):
                        number = int(num_node.attributes.get("data-tab-no"))
                    else:
                        digits = "".join(filter(str.isdigit, num_node.text()))
                        if digits: number = int(digits)

                scratched = "strikeout" in (row.attributes.get("class") or "").lower() or row.attributes.get("data-scratched") == "True"

                win_odds = None
                odds_node = row.css_first(".pa_odds") or row.css_first(".odds")
                if odds_node:
                    win_odds = parse_odds_to_decimal(clean_text(odds_node.text()))

                od = {}
                if ov := create_odds_data(self.SOURCE_NAME, win_odds):
                    od[self.SOURCE_NAME] = ov

                runners.append(Runner(name=name, number=number, scratched=scratched, odds=od, win_odds=win_odds))
            except Exception: continue

        if not runners: return None

        disc = detect_discipline(html_content)
        return Race(
            id=generate_race_id("srw", venue, start_time, race_num, disc),
            venue=venue,
            race_number=race_num,
            start_time=start_time,
            runners=runners,
            discipline=disc,
            source=self.SOURCE_NAME,
            available_bets=scrape_available_bets(html_content)
        )

# ----------------------------------------
# AtTheRacesAdapter
# ----------------------------------------
class AtTheRacesAdapter(BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
    SOURCE_NAME: ClassVar[str] = "AtTheRaces"
    BASE_URL: ClassVar[str] = "https://www.attheraces.com"

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return FetchStrategy(primary_engine=BrowserEngine.CURL_CFFI, enable_js=True, stealth_mode="camouflage")

    async def make_request(self, method: str, url: str, **kwargs: Any) -> Any:
        kwargs.setdefault("impersonate", "chrome120")
        return await super().make_request(method, url, **kwargs)

    SELECTORS: ClassVar[Dict[str, List[str]]] = {
        "race_links": ['a.race-navigation-link', 'a.sidebar-racecardsigation-link', 'a[href^="/racecard/"]', 'a[href*="/racecard/"]'],
        "details_container": [".race-header__details--primary", "atr-racecard-race-header .container", ".racecard-header .container"],
        "track_name": ["h2", "h1 a", "h1"],
        "race_time": ["h2 b", "h1 span", ".race-time"],
        "distance": [".race-header__details--secondary .p--large", ".race-header__details--secondary div"],
        "runners": [".card-cell--horse", ".odds-grid-horse", "atr-horse-in-racecard", ".horse-in-racecard"],
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return FetchStrategy(primary_engine=BrowserEngine.HTTPX, enable_js=False)

    def _get_headers(self) -> Dict[str, str]:
        return self._get_browser_headers(host="www.attheraces.com", referer="https://www.attheraces.com/racecards")

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        index_url = f"/racecards/{date}"
        intl_url = f"/racecards/international/{date}"

        resp = await self.make_request("GET", index_url, headers=self._get_headers())
        intl_resp = await self.make_request("GET", intl_url, headers=self._get_headers())

        metadata = []
        if resp and resp.text:
            self._save_debug_snapshot(resp.text, f"atr_index_{date}")
            parser = HTMLParser(resp.text)
            metadata.extend(self._extract_race_metadata(parser, date))

        elif resp:
            self.logger.warning("Unexpected status", status=resp.status, url=index_url)

        if intl_resp and intl_resp.text:
            self._save_debug_snapshot(intl_resp.text, f"atr_intl_index_{date}")
            intl_parser = HTMLParser(intl_resp.text)
            metadata.extend(self._extract_race_metadata(intl_parser, date))
        elif intl_resp:
            self.logger.warning("Unexpected status", status=intl_resp.status, url=intl_url)

        if not metadata:
            self.logger.warning("No metadata found", context="ATR Index Parsing", date=date)
            return None
        pages = await self._fetch_race_pages_concurrent(metadata, self._get_headers(), semaphore_limit=5)
        return {"pages": pages, "date": date}

    def _extract_race_metadata(self, parser: HTMLParser, date_str: str) -> List[Dict[str, Any]]:
        meta: List[Dict[str, Any]] = []
        track_map = defaultdict(list)

        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            target_date = datetime.now(EASTERN).date()

        for link in parser.css('a[href*="/racecard/"]'):
            url = link.attributes.get("href")
            if not url: continue
            # Look for time at end of URL: /racecard/venue/date/1330
            time_match = re.search(r"/(\d{4})$", url)
            if not time_match:
                # Might be just a race number: /racecard/venue/date/1
                if not re.search(r"/\d{1,2}$", url): continue

            parts = url.split("/")
            if len(parts) >= 3:
                track_name = parts[2]
                time_str = time_match.group(1) if time_match else None
                track_map[track_name].append({"url": url, "time_str": time_str})

        # Site usually shows UK time
        site_tz = ZoneInfo("Europe/London")
        now_site = datetime.now(site_tz)

        for track, race_infos in track_map.items():
            # Sort by time if possible
            sorted_races = sorted(race_infos, key=lambda x: x["time_str"] or "9999")

            next_race = None
            for r in sorted_races:
                if r["time_str"]:
                    try:
                        rt = datetime.strptime(r["time_str"], "%H%M").replace(
                            year=target_date.year, month=target_date.month, day=target_date.day, tzinfo=site_tz
                        )
                        # Optimization: Skip if race is more than 5 mins in the past (Only for today's races)
                        if target_date == now_site.date() and rt < now_site - timedelta(minutes=5):
                            continue
                        next_race = r
                        break
                    except Exception: pass

            # Only append if we found an upcoming race (Optimization: Skip finished meetings)
            if next_race:
                meta.append({"url": next_race["url"], "race_number": 1, "venue_raw": track})

        if not meta:
            for meeting in (parser.css(".meeting-summary") or parser.css(".p-meetings__item")):
                for link in meeting.css('a[href*="/racecard/"]'):
                    if url := link.attributes.get("href"):
                        meta.append({"url": url, "race_number": 1})
        return meta

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data or not raw_data.get("pages"): return []
        try: race_date = datetime.strptime(raw_data["date"], "%Y-%m-%d").date()
        except Exception: return []
        races: List[Race] = []
        for item in raw_data["pages"]:
            html_content = item.get("html")
            if not html_content: continue
            try:
                race = self._parse_single_race(html_content, item.get("url", ""), race_date, item.get("race_number"))
                if race: races.append(race)
            except Exception: pass
        return races

    def _parse_single_race(self, html_content: str, url_path: str, race_date: date, race_number_fallback: Optional[int]) -> Optional[Race]:
        parser = HTMLParser(html_content)
        track_name, time_str, header_text = None, None, ""
        header = parser.css_first(".race-header__details") or parser.css_first(".racecard-header")
        if header:
            header_text = clean_text(header.text()) or ""
            time_match = re.search(r"(\d{1,2}:\d{2})", header_text)
            if time_match:
                time_str = time_match.group(1)
                track_raw = re.sub(r"\d{1,2}\s+[A-Za-z]{3}\s+\d{4}", "", header_text.replace(time_str, "")).strip()
                track_raw = re.split(r"\s+Race\s+\d+", track_raw, flags=re.I)[0]
                track_raw = re.sub(r"^\d+\s+", "", track_raw).split(" - ")[0].split("|")[0].strip()
                track_name = normalize_venue_name(track_raw)
        if not track_name:
            details = parser.css_first(".race-header__details--primary")
            if details:
                track_node = details.css_first("h2") or details.css_first("h1 a") or details.css_first("h1")
                if track_node: track_name = normalize_venue_name(clean_text(track_node.text()))
                if not time_str:
                    time_node = details.css_first("h2 b") or details.css_first(".race-time")
                    if time_node: time_str = clean_text(time_node.text()).replace(" ATR", "")
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
        race_number = race_number_fallback or 1
        distance = None
        dist_match = re.search(r"\|\s*(\d+[mfy].*)", header_text, re.I)
        if dist_match: distance = dist_match.group(1).strip()
        runners = self._parse_runners(parser)
        if not runners: return None
        return Race(discipline="Thoroughbred", id=generate_race_id("atr", track_name, start_time, race_number), venue=track_name, race_number=race_number, start_time=start_time, runners=runners, distance=distance, source=self.source_name, available_bets=scrape_available_bets(html_content))

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
            name = clean_text(name_node.text())
            if not name: return None
            num_node = row.css_first(".horse-in-racecard__saddle-cloth-number") or row.css_first(".odds-grid-horse__no")
            number = 0
            if num_node:
                ns = clean_text(num_node.text())
                if ns:
                    digits = "".join(filter(str.isdigit, ns))
                    if digits: number = int(digits)

            if number == 0 or number > 40:
                number = fallback_number
            win_odds = None
            if horse_link := row.css_first('a[href*="/form/horse/"]'):
                if m := re.search(r"/(\d+)(\?|$)", horse_link.attributes.get("href", "")):
                    win_odds = odds_map.get(m.group(1))
            if win_odds is None:
                if odds_node := row.css_first(".horse-in-racecard__odds"):
                    win_odds = parse_odds_to_decimal(clean_text(odds_node.text()))

            # Advanced heuristic fallback
            if win_odds is None:
                win_odds = SmartOddsExtractor.extract_from_node(row)

            odds: Dict[str, OddsData] = {}
            if od := create_odds_data(self.source_name, win_odds): odds[self.source_name] = od
            return Runner(number=number, name=name, odds=odds, win_odds=win_odds)
        except Exception: return None

# ----------------------------------------
# AtTheRacesGreyhoundAdapter
# ----------------------------------------
class AtTheRacesGreyhoundAdapter(JSONParsingMixin, BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
    SOURCE_NAME: ClassVar[str] = "AtTheRacesGreyhound"
    BASE_URL: ClassVar[str] = "https://greyhounds.attheraces.com"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return FetchStrategy(primary_engine=BrowserEngine.CURL_CFFI, enable_js=True, stealth_mode="camouflage", timeout=45)

    def _get_headers(self) -> Dict[str, str]:
        return self._get_browser_headers(host="greyhounds.attheraces.com", referer="https://greyhounds.attheraces.com/racecards")

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        index_url = f"/racecards/{date}" if date else "/racecards"
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
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            target_date = datetime.now(EASTERN).date()

        # Usually UK time
        site_tz = ZoneInfo("Europe/London")
        now_site = datetime.now(site_tz)

        try:
            modules = json.loads(html.unescape(items_raw))
            for module in modules:
                for meeting in module.get("data", {}).get("items", []):
                    # Only take the "next" race for each meeting (Memory Directive Fix)
                    races = [r for r in meeting.get("items", []) if r.get("type") == "racecard"]

                    next_race = None
                    for race in races:
                        r_time_str = race.get("time") # Usually HH:MM
                        if r_time_str:
                            try:
                                rt = datetime.strptime(r_time_str, "%H:%M").replace(
                                    year=target_date.year, month=target_date.month, day=target_date.day, tzinfo=site_tz
                                )
                                # Skip if in past (Today only)
                                if target_date == now_site.date() and rt < now_site - timedelta(minutes=5):
                                    continue
                                next_race = race
                                break
                            except Exception: pass

                    # No fallback to first race - skip if meeting is likely finished
                    if next_race:
                        r_num = next_race.get("raceNumber") or next_race.get("number") or 1
                        if u := next_race.get("cta", {}).get("href"):
                            if "/racecard/" in u:
                                meta.append({"url": u, "race_number": r_num})
        except Exception: pass
        return meta

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data or not raw_data.get("pages"): return []
        try: race_date = datetime.strptime(raw_data.get("date", ""), "%Y-%m-%d").date()
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
            title_text = title_node.text().strip()
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
            if m_type == "OddsGrid":
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

                    # Advanced heuristic fallback
                    if win_odds is None:
                        win_odds = SmartOddsExtractor.extract_from_text(str(t))

                    odds_data = {}
                    if ov := create_odds_data(self.source_name, win_odds): odds_data[self.source_name] = ov
                    runners.append(Runner(number=trap_num or 0, name=name, odds=odds_data, win_odds=win_odds))

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
# BoyleSportsAdapter
# ----------------------------------------
class BoyleSportsAdapter(BrowserHeadersMixin, DebugMixin, BaseAdapterV3):
    SOURCE_NAME: ClassVar[str] = "BoyleSports"
    BASE_URL: ClassVar[str] = "https://www.boylesports.com"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        # Use CURL_CFFI with chrome120 for better reliability against bot detection
        return FetchStrategy(primary_engine=BrowserEngine.CURL_CFFI, enable_js=True, stealth_mode="camouflage", timeout=45)

    async def make_request(self, method: str, url: str, **kwargs: Any) -> Any:
        kwargs.setdefault("impersonate", "chrome120")
        return await super().make_request(method, url, **kwargs)

    def _get_headers(self) -> Dict[str, str]:
        return self._get_browser_headers(host="www.boylesports.com", referer="https://www.google.com/")

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        url = "/sports/horse-racing"
        resp = await self.make_request("GET", url, headers=self._get_headers())
        if not resp or not resp.text:
            if resp: self.logger.warning("Unexpected status", status=resp.status, url=url)
            return None
        self._save_debug_snapshot(resp.text, f"boylesports_index_{date}")
        return {"pages": [{"url": url, "html": resp.text}], "date": date}

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data or not raw_data.get("pages"): return []
        try: race_date = datetime.strptime(raw_data["date"], "%Y-%m-%d").date()
        except Exception: race_date = datetime.now(EASTERN).date()
        item = raw_data["pages"][0]
        parser = HTMLParser(item.get("html", ""))
        races: List[Race] = []
        meeting_groups = parser.css('.meeting-group') or parser.css('.race-meeting') or parser.css('div[class*="meeting"]')
        for meeting in meeting_groups:
            tnn = meeting.css_first('.meeting-name') or meeting.css_first('h2') or meeting.css_first('.title')
            if not tnn: continue
            trw = clean_text(tnn.text())
            track_name = normalize_venue_name(trw)
            if not track_name: continue
            m_harness = any(kw in trw.lower() for kw in ['harness', 'trot', 'pace', 'standardbred'])
            is_grey = any(kw in trw.lower() for kw in ['greyhound', 'dog'])
            race_nodes = meeting.css('.race-time-row') or meeting.css('.race-details') or meeting.css('a[href*="/race/"]')
            for i, rn in enumerate(race_nodes):
                txt = clean_text(rn.text())
                r_harness = m_harness or any(kw in txt.lower() for kw in ['trot', 'pace', 'attele', 'mounted'])
                tm = re.search(r'(\d{1,2}:\d{2})', txt)
                if not tm: continue
                fm = re.search(r'\((\d+)\s+runners\)', txt, re.I)
                fs = int(fm.group(1)) if fm else 0
                dm = re.search(r'(\d+(?:\.\d+)?\s*[kmf]|1\s*mile)', txt, re.I)
                dist = dm.group(1) if dm else None
                try: st = datetime.combine(race_date, datetime.strptime(tm.group(1), "%H:%M").time())
                except Exception: continue
                runners = [Runner(number=j+1, name=f"Runner {j+1}", scratched=False, odds={}) for j in range(fs)]
                disc = "Harness" if r_harness else "Greyhound" if is_grey else "Thoroughbred"
                ab = []
                if 'superfecta' in txt.lower(): ab.append('Superfecta')
                elif r_harness or ' (us)' in trw.lower():
                    if fs >= 6: ab.append('Superfecta')
                races.append(Race(id=f"boyle_{track_name.lower().replace(' ', '')}_{st:%Y%m%d_%H%M}", venue=track_name, race_number=i + 1, start_time=st, runners=runners, distance=dist, source=self.source_name, discipline=disc, available_bets=ab))
        return races


# ----------------------------------------
# SportingLifeAdapter
# ----------------------------------------
class SportingLifeAdapter(JSONParsingMixin, BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
    SOURCE_NAME: ClassVar[str] = "SportingLife"
    BASE_URL: ClassVar[str] = "https://www.sportinglife.com"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return FetchStrategy(primary_engine=BrowserEngine.HTTPX, enable_js=False, stealth_mode="camouflage", timeout=30)

    def _get_headers(self) -> Dict[str, str]:
        return self._get_browser_headers(host="www.sportinglife.com", referer="https://www.sportinglife.com/racing/racecards")

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        index_url = f"/racing/racecards/{date}/" if date else "/racing/racecards/"
        resp = await self.make_request("GET", index_url, headers=self._get_headers(), follow_redirects=True)
        if not resp or not resp.text:
            if resp: self.logger.warning("Unexpected status", status=resp.status, url=index_url)
            raise AdapterHttpError(self.source_name, getattr(resp, 'status', 500), index_url)
        self._save_debug_snapshot(resp.text, f"sportinglife_index_{date}")
        parser = HTMLParser(resp.text)
        metadata = self._extract_race_metadata(parser, date)
        if not metadata:
            self.logger.warning("No metadata found", context="SportingLife Index Parsing", url=index_url)
            return None
        pages = await self._fetch_race_pages_concurrent(metadata, self._get_headers(), semaphore_limit=8)
        return {"pages": pages, "date": date}

    def _extract_race_metadata(self, parser: HTMLParser, date_str: str) -> List[Dict[str, Any]]:
        meta: List[Dict[str, Any]] = []
        data = self._parse_json_from_script(parser, "script#__NEXT_DATA__", context="SportingLife Index")

        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            target_date = datetime.now(EASTERN).date()

        site_tz = ZoneInfo("Europe/London")
        now_site = datetime.now(site_tz)

        if data:
            for meeting in data.get("props", {}).get("pageProps", {}).get("meetings", []):
                # Only take the "next" race for each meeting (Memory Directive Fix)
                races = meeting.get("races", [])
                next_race_info = None
                for i, race in enumerate(races):
                    r_time_str = race.get("time") # Usually HH:MM
                    if r_time_str:
                        try:
                            rt = datetime.strptime(r_time_str, "%H:%M").replace(
                                year=target_date.year, month=target_date.month, day=target_date.day, tzinfo=site_tz
                            )
                            # Skip if in past (Today only)
                            if target_date == now_site.date() and rt < now_site - timedelta(minutes=5):
                                continue
                            next_race_info = (race, i + 1)
                            break
                        except Exception: pass

                if next_race_info:
                    race, r_num = next_race_info
                    if url := race.get("racecard_url"):
                        meta.append({"url": url, "race_number": r_num})
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
        try: race_date = datetime.strptime(raw_data["date"], "%Y-%m-%d").date()
        except Exception: return []
        races: List[Race] = []
        for item in raw_data["pages"]:
            html_content = item.get("html")
            if not html_content: continue
            try:
                parser = HTMLParser(html_content)
                race = self._parse_from_next_data(parser, race_date, item.get("race_number"), html_content)
                if not race: race = self._parse_from_html(parser, race_date, item.get("race_number"), html_content)
                if race: races.append(race)
            except Exception: pass
        return races

    def _parse_from_next_data(self, parser: HTMLParser, race_date: date, race_number_fallback: Optional[int], html_content: str) -> Optional[Race]:
        data = self._parse_json_from_script(parser, "script#__NEXT_DATA__", context="SportingLife Race")
        if not data: return None
        race_info = data.get("props", {}).get("pageProps", {}).get("race")
        if not race_info: return None
        summary = race_info.get("race_summary") or {}
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
            odds_data = {}
            if ov := create_odds_data(self.source_name, wo): odds_data[self.source_name] = ov
            runners.append(Runner(number=num, name=name, scratched=rd.get("is_non_runner") or rd.get("ride_status") == "NON_RUNNER", odds=odds_data, win_odds=wo))
        if not runners: return None
        return Race(id=generate_race_id("sl", track_name or "Unknown", start_time, race_info.get("race_number") or race_number_fallback or 1), venue=track_name or "Unknown", race_number=race_info.get("race_number") or race_number_fallback or 1, start_time=start_time, runners=runners, distance=summary.get("distance") or race_info.get("distance"), source=self.source_name, discipline="Thoroughbred", available_bets=scrape_available_bets(html_content))

    def _parse_from_html(self, parser: HTMLParser, race_date: date, race_number_fallback: Optional[int], html_content: str) -> Optional[Race]:
        h1 = parser.css_first('h1[class*="RacingRacecardHeader__Title"]')
        if not h1: return None
        ht = clean_text(h1.text())
        if not ht: return None
        parts = ht.split()
        if not parts: return None
        try: start_time = datetime.combine(race_date, datetime.strptime(parts[0], "%H:%M").time())
        except Exception: return None
        track_name = normalize_venue_name(" ".join(parts[1:]))
        runners = []
        for row in parser.css('div[class*="RunnerCard"]'):
            try:
                nn = row.css_first('a[href*="/racing/profiles/horse/"]')
                if not nn: continue
                name = clean_text(nn.text()).splitlines()[0].strip()
                num_node = row.css_first('span[class*="SaddleCloth__Number"]')
                number = int("".join(filter(str.isdigit, clean_text(num_node.text())))) if num_node else 0
                on = row.css_first('span[class*="Odds__Price"]')
                wo = parse_odds_to_decimal(clean_text(on.text()) if on else "")

                # Advanced heuristic fallback
                if wo is None:
                    wo = SmartOddsExtractor.extract_from_node(row)

                od = {}
                if ov := create_odds_data(self.source_name, wo): od[self.source_name] = ov
                runners.append(Runner(number=number, name=name, odds=od, win_odds=wo))
            except Exception: continue
        if not runners: return None
        dn = parser.css_first('span[class*="RacecardHeader__Distance"]') or parser.css_first(".race-distance")
        return Race(id=generate_race_id("sl", track_name or "Unknown", start_time, race_number_fallback or 1), venue=track_name or "Unknown", race_number=race_number_fallback or 1, start_time=start_time, runners=runners, distance=clean_text(dn.text()) if dn else None, source=self.source_name, available_bets=scrape_available_bets(html_content))

# ----------------------------------------
# SkySportsAdapter
# ----------------------------------------
class SkySportsAdapter(JSONParsingMixin, BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
    SOURCE_NAME: ClassVar[str] = "SkySports"
    BASE_URL: ClassVar[str] = "https://www.skysports.com"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return FetchStrategy(primary_engine=BrowserEngine.HTTPX, enable_js=False, stealth_mode="fast", timeout=30)

    def _get_headers(self) -> Dict[str, str]:
        return self._get_browser_headers(host="www.skysports.com", referer="https://www.skysports.com/racing")

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        dt = datetime.strptime(date, "%Y-%m-%d")
        index_url = f"/racing/racecards/{dt.strftime('%d-%m-%Y')}"
        resp = await self.make_request("GET", index_url, headers=self._get_headers())
        if not resp or not resp.text:
            if resp: self.logger.warning("Unexpected status", status=resp.status, url=index_url)
            raise AdapterHttpError(self.source_name, getattr(resp, 'status', 500), index_url)
        self._save_debug_snapshot(resp.text, f"skysports_index_{date}")
        parser = HTMLParser(resp.text)
        metadata = []

        try:
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
        except Exception:
            target_date = datetime.now(EASTERN).date()

        site_tz = ZoneInfo("Europe/London")
        now_site = datetime.now(site_tz)

        meetings = parser.css(".sdc-site-concertina-block") or parser.css(".page-details__section") or parser.css(".racing-meetings__meeting")
        for meeting in meetings:
            hn = meeting.css_first(".sdc-site-concertina-block__title") or meeting.css_first(".racing-meetings__meeting-title")
            if not hn: continue
            vr = clean_text(hn.text()) or ""
            if "ABD:" in vr: continue

            for link in meeting.css('a[href*="/racecards/"]'):
                if h := link.attributes.get("href"):
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

                    metadata.append({"url": h, "venue_raw": vr, "race_number": 1})
                    break

        if not metadata:
            self.logger.warning("No metadata found", context="SkySports Index Parsing", url=index_url)
            return None
        pages = await self._fetch_race_pages_concurrent(metadata, self._get_headers(), semaphore_limit=10)
        return {"pages": pages, "date": date}

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data or not raw_data.get("pages"): return []
        try: race_date = datetime.strptime(raw_data.get("date", ""), "%Y-%m-%d").date()
        except Exception: race_date = datetime.now(EASTERN).date()
        races: List[Race] = []
        for item in raw_data["pages"]:
            html_content = item.get("html")
            if not html_content: continue
            parser = HTMLParser(html_content)
            h = parser.css_first(".sdc-site-racing-header__name")
            if not h: continue
            ht = clean_text(h.text()) or ""
            m = re.match(r"(\d{1,2}:\d{2})\s+(.+)", ht)
            if not m:
                tn, cn = parser.css_first(".sdc-site-racing-header__time"), parser.css_first(".sdc-site-racing-header__course")
                if tn and cn: rts, tnr = clean_text(tn.text()) or "", clean_text(cn.text()) or ""
                else: continue
            else: rts, tnr = m.group(1), m.group(2)
            track_name = normalize_venue_name(tnr)
            if not track_name: continue
            try: start_time = datetime.combine(race_date, datetime.strptime(rts, "%H:%M").time())
            except Exception: continue
            dist = None
            for d in parser.css(".sdc-site-racing-header__detail-item"):
                dt = clean_text(d.text()) or ""
                if "Distance:" in dt: dist = dt.replace("Distance:", "").strip(); break
            runners = []
            for i, node in enumerate(parser.css(".sdc-site-racing-card__item")):
                nn = node.css_first(".sdc-site-racing-card__name a")
                if not nn: continue
                name = clean_text(nn.text())
                if not name: continue
                nnode = node.css_first(".sdc-site-racing-card__number strong")
                number = i + 1
                if nnode:
                    nt = clean_text(nnode.text())
                    if nt:
                        try: number = int(nt)
                        except Exception: pass
                onode = node.css_first(".sdc-site-racing-card__betting-odds")
                wo = parse_odds_to_decimal(clean_text(onode.text()) if onode else "")

                # Advanced heuristic fallback
                if wo is None:
                    wo = SmartOddsExtractor.extract_from_node(node)

                ntxt = clean_text(node.text()) or ""
                scratched = "NR" in ntxt or "Non-runner" in ntxt
                od = {}
                if ov := create_odds_data(self.source_name, wo): od[self.source_name] = ov
                runners.append(Runner(number=number, name=name, scratched=scratched, odds=od, win_odds=wo))
            if not runners: continue
            disc = detect_discipline(html_content)
            ab = scrape_available_bets(html_content)
            if not ab and (disc == "Harness" or "(us)" in tnr.lower()) and len([r for r in runners if not r.scratched]) >= 6: ab.append("Superfecta")
            races.append(Race(id=generate_race_id("sky", track_name, start_time, item.get("race_number", 0), disc), venue=track_name, race_number=item.get("race_number", 0), start_time=start_time, runners=runners, distance=dist, discipline=disc, source=self.source_name, available_bets=ab))
        return races

# ----------------------------------------
# RacingPostB2BAdapter
# ----------------------------------------
class RacingPostB2BAdapter(BaseAdapterV3):
    SOURCE_NAME: ClassVar[str] = "RacingPostB2B"
    BASE_URL: ClassVar[str] = "https://backend-us-racecards.widget.rpb2b.com"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config, enable_cache=True, cache_ttl=300.0, rate_limit=5.0)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return FetchStrategy(primary_engine=BrowserEngine.HTTPX, enable_js=False, max_retries=3, timeout=20)

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        endpoint = f"/v2/racecards/daily/{date}"
        resp = await self.make_request("GET", endpoint)
        if not resp: return None
        try: data = resp.json()
        except Exception: return None
        if not isinstance(data, list): return None
        return {"venues": data, "date": date, "fetched_at": datetime.now(EASTERN).isoformat()}

    def _parse_races(self, raw_data: Optional[Dict[str, Any]]) -> List[Race]:
        if not raw_data or not raw_data.get("venues"): return []
        races: List[Race] = []
        for vd in raw_data["venues"]:
            if vd.get("isAbandoned"): continue
            vn, cc, rd = vd.get("name", "Unknown"), vd.get("countryCode", "USA"), vd.get("races", [])
            for r in rd:
                if r.get("raceStatusCode") == "ABD": continue
                parsed = self._parse_single_race(r, vn, cc)
                if parsed: races.append(parsed)
        return races

    def _parse_single_race(self, rd: Dict[str, Any], vn: str, cc: str) -> Optional[Race]:
        rid, rnum, dts, nr = rd.get("id"), rd.get("raceNumber"), rd.get("datetimeUtc"), rd.get("numberOfRunners", 0)
        if not all([rid, rnum, dts]): return None
        try: st = datetime.fromisoformat(dts.replace("Z", "+00:00"))
        except Exception: return None
        runners = [Runner(number=i + 1, name=f"Runner {i + 1}", scratched=False, odds={}) for i in range(nr)]
        return Race(discipline="Thoroughbred", id=f"rpb2b_{rid.replace('-', '')[:16]}", venue=normalize_venue_name(vn), race_number=rnum, start_time=st, runners=runners, source=self.source_name, metadata={"original_race_id": rid, "country_code": cc, "num_runners": nr})


# ----------------------------------------
# StandardbredCanadaAdapter
# ----------------------------------------
class StandardbredCanadaAdapter(BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
    SOURCE_NAME: ClassVar[str] = "StandardbredCanada"
    BASE_URL: ClassVar[str] = "https://standardbredcanada.ca"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)
        self._semaphore = asyncio.Semaphore(3)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        engine = BrowserEngine.PLAYWRIGHT if ASYNC_SESSIONS_AVAILABLE else BrowserEngine.HTTPX
        return FetchStrategy(primary_engine=engine, enable_js=(engine != BrowserEngine.HTTPX), stealth_mode="fast", timeout=45)

    def _get_headers(self) -> Dict[str, str]:
        return self._get_browser_headers(host="standardbredcanada.ca", referer="https://standardbredcanada.ca/racing")

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        dt = datetime.strptime(date, "%Y-%m-%d")
        date_label = dt.strftime(f"%A %b {dt.day}, %Y")
        date_short = dt.strftime("%m%d") # e.g. 0208

        index_html = None

        # 1. Try browser-based fetch if available
        try:
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
            tn = clean_text(tnn.text()) or ""
            isf = "*" in tn or "*" in (clean_text(container.text()) or "")
            for link in container.css('a[href*="/entries/"]'):
                if u := link.attributes.get("href"):
                    metadata.append({"url": u, "venue": tn.replace("*", "").strip(), "finalized": isf})
        if not metadata:
            self.logger.warning("No metadata found", context="StandardbredCanada Index Parsing")
            return None
        pages = await self._fetch_race_pages_concurrent(metadata, self._get_headers(), semaphore_limit=3)
        return {"pages": pages, "date": date}

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data or not raw_data.get("pages"): return []
        try: race_date = datetime.strptime(raw_data.get("date", ""), "%Y-%m-%d").date()
        except Exception: race_date = datetime.now(EASTERN).date()
        races: List[Race] = []
        for item in raw_data["pages"]:
            html_content = item.get("html")
            if not html_content or ("Final Changes Made" not in html_content and not item.get("finalized")): continue
            track_name = normalize_venue_name(item["venue"])
            for pre in HTMLParser(html_content).css("pre"):
                text = pre.text()
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
            m = re.search(r"^\s*(\d+)\s+([^(]+)", line)
            if m:
                num, name = int(m.group(1)), m.group(2).strip()
                name = re.sub(r"\(L\)$|\(L\)\s+", "", name).strip()
                sc = "SCR" in line or "Scratched" in line
                # Try smarter odds extraction from the line
                wo = SmartOddsExtractor.extract_from_text(line)
                if wo is None:
                    om = re.search(r"(\d+-\d+|[0-9.]+)\s*$", line)
                    if om: wo = parse_odds_to_decimal(om.group(1))

                odds_data = {}
                if ov := create_odds_data(self.source_name, wo): odds_data[self.source_name] = ov
                runners.append(Runner(number=num, name=name, scratched=sc, odds=odds_data, win_odds=wo))
        if not runners: return None
        return Race(discipline="Harness", id=generate_race_id("sc", track_name, st, race_num, "Harness"), venue=track_name, race_number=race_num, start_time=st, runners=runners, distance=dist, source=self.source_name, available_bets=ab)

# ----------------------------------------
# TabAdapter
# ----------------------------------------
class TabAdapter(BaseAdapterV3):
    SOURCE_NAME: ClassVar[str] = "TAB"
    # Note: api.tab.com.au often has DNS resolution issues in some environments.
    # api.beta.tab.com.au is more reliable.
    BASE_URL: ClassVar[str] = "https://api.beta.tab.com.au/v1/tab-info-service/racing"
    BASE_URL_STABLE: ClassVar[str] = "https://api.tab.com.au/v1/tab-info-service/racing"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config, rate_limit=2.0)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        engine = BrowserEngine.PLAYWRIGHT if ASYNC_SESSIONS_AVAILABLE else BrowserEngine.HTTPX
        return FetchStrategy(primary_engine=engine, enable_js=(engine != BrowserEngine.HTTPX), stealth_mode="fast", timeout=45)

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/dates/{date}/meetings"
        resp = await self.make_request("GET", url, headers={"Accept": "application/json", "User-Agent": CHROME_USER_AGENT})

        if not resp or resp.status != 200:
            self.logger.info("Falling back to STABLE TAB API")
            url = f"{self.BASE_URL_STABLE}/dates/{date}/meetings"
            resp = await self.make_request("GET", url, headers={"Accept": "application/json", "User-Agent": CHROME_USER_AGENT})

        if not resp: return None
        try: data = resp.json() if hasattr(resp, "json") else json.loads(resp.text)
        except Exception: return None
        if not data or "meetings" not in data: return None

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

                try: st = datetime.fromisoformat(rst.replace("Z", "+00:00"))
                except Exception: continue

                runners = []
                # If detail data was fetched, extract runners
                for runner_data in rd.get("runners", []):
                    name = runner_data.get("runnerName", "Unknown")
                    num = runner_data.get("runnerNumber")

                    # Try to get win odds
                    win_odds = None
                    fixed_odds = runner_data.get("fixedOdds", {})
                    if fixed_odds:
                        win_odds = fixed_odds.get("returnWin") or fixed_odds.get("win")

                    odds_dict = {}
                    if win_odds:
                        if ov := create_odds_data(self.source_name, win_odds):
                            odds_dict[self.source_name] = ov

                    runners.append(Runner(
                        name=name,
                        number=num,
                        win_odds=win_odds,
                        odds=odds_dict,
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

    def __init__(self, model_name: str = "Ratings", url: str = "https://www.betfair.com.au/hub/ratings/model/horse-racing/", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=f"{self.ADAPTER_NAME}_{model_name}", base_url=url, config=config)
        self.model_name = model_name

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return FetchStrategy(primary_engine=BrowserEngine.HTTPX)

    async def _fetch_data(self, date: str) -> Optional[StringIO]:
        endpoint = f"?date={date}&presenter=RatingsPresenter&csv=true"
        resp = await self.make_request("GET", endpoint)
        return StringIO(resp.text) if resp and resp.text else None

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
                                start_time = to_eastern(datetime.fromisoformat(st_val.replace("Z", "+00:00")))
                            break
                        except Exception: pass

                races.append(Race(id=str(mid), venue=vn, race_number=int(ri.get("race_number", 0)), start_time=start_time, runners=runners, source=self.source_name, discipline="Thoroughbred"))
            return races
        except Exception: return []

# ----------------------------------------
# EquibaseAdapter
# ----------------------------------------
class EquibaseAdapter(BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
    SOURCE_NAME: ClassVar[str] = "Equibase"
    BASE_URL: ClassVar[str] = "https://www.equibase.com"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        # Equibase uses Instart Logic / Imperva; try CURL_CFFI first as it's often more successful if Playwright is detected as headless
        return FetchStrategy(
            primary_engine=BrowserEngine.PLAYWRIGHT,
            enable_js=True,
            stealth_mode="camouflage",
            timeout=60
        )

    async def make_request(self, method: str, url: str, **kwargs: Any) -> Any:
        # For Equibase, sometimes a simple curl_cffi with chrome120 works better if Playwright is detected
        # but let's try to be consistent with headers
        kwargs.setdefault("headers", self._get_browser_headers(host="www.equibase.com", referer="https://www.equibase.com/"))
        return await super().make_request(method, url, **kwargs)

    def _get_headers(self) -> Dict[str, str]:
        return self._get_browser_headers(host="www.equibase.com")

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        dt = datetime.strptime(date, "%Y-%m-%d")
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
            try:
                # dt = datetime.strptime(date, "%Y-%m-%d")
                resp = await self.make_request("GET", url, headers=self._get_headers(), impersonate="chrome120")
                if resp and resp.status == 200 and resp.text and len(resp.text) > 1000 and "Pardon Our Interruption" not in resp.text:
                    self.logger.info("Found Equibase index", url=url)
                    break
                else:
                    self.logger.debug("Equibase index candidate failed", url=url, status=getattr(resp, 'status', 'N/A'))
            except Exception as e:
                self.logger.debug("Equibase request exception", url=url, error=str(e))
                continue

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
            if "/static/entry/" in url_norm and date_str in url_norm:
                links.append(url_norm)

        for a in parser.css("a"):
            h = a.attributes.get("href") or ""
            c = a.attributes.get("class") or ""
            # Normalize backslashes (Project fix for Equibase path separators)
            h_norm = h.replace("\\", "/")
            if "/static/entry/" in h_norm and date_str in h_norm:
                links.append(h_norm)
            elif "entry-race-level" in c:
                links.append(h_norm)

        if not links:
            self.logger.warning("No links found", context="Equibase Index Parsing", date=date)
            return None

        # Fetch initial set of pages
        pages = await self._fetch_race_pages_concurrent([{"url": l} for l in set(links)], self._get_headers(), semaphore_limit=5)

        all_htmls = []
        extra_links = []
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
        except Exception:
            target_date = datetime.now(EASTERN).date()

        now = now_eastern()
        for p in pages:
            html_content = p.get("html")
            if not html_content: continue

            # If it's an index page for a track, we need to extract individual race links
            if "RaceCardIndex" in p.get("url", ""):
                sub_parser = HTMLParser(html_content)
                # Only take the "next" race link for this track (Memory Directive Fix)
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
            extra_pages = await self._fetch_race_pages_concurrent([{"url": l} for l in set(extra_links)], self._get_headers(), semaphore_limit=5)
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
                venue = clean_text(vn.text())
                rnum_txt = rn.text().replace("Race", "").strip()
                if not venue or not rnum_txt.isdigit(): continue
                st = self._parse_post_time(ds, pt.text().strip())
                ab = scrape_available_bets(html_content)
                runners = [r for node in p.css("table.entries-table tbody tr") if (r := self._parse_runner(node))]
                if not runners: continue
                races.append(Race(id=f"eqb_{venue.lower().replace(' ', '')}_{ds}_{rnum_txt}", venue=venue, race_number=int(rnum_txt), start_time=st, runners=runners, source=self.source_name, discipline="Thoroughbred", available_bets=ab))
            except Exception: continue
        return races

    def _parse_runner(self, node: Node) -> Optional[Runner]:
        try:
            cols = node.css("td")
            if len(cols) < 3: return None

            # P1: Try to find number in first col
            number = 0
            num_text = clean_text(cols[0].text())
            if num_text.isdigit():
                number = int(num_text)

            # P2: Horse name usually in 3rd col, but can vary
            name = None
            for idx in [2, 1, 3]:
                if len(cols) > idx:
                    n_text = clean_text(cols[idx].text())
                    if n_text and not n_text.isdigit() and len(n_text) > 2:
                        name = n_text
                        break

            if not name: return None

            sc = "scratched" in node.attributes.get("class", "").lower() or "SCR" in (clean_text(node.text()) or "")

            odds, wo = {}, None
            if not sc:
                # Odds column can be 9 or 10 (blind indexing fallback)
                for idx in [9, 8, 10]:
                    if len(cols) > idx:
                        o_text = clean_text(cols[idx].text())
                        if o_text:
                            wo = parse_odds_to_decimal(o_text)
                            if wo: break

                if wo is None: wo = SmartOddsExtractor.extract_from_node(node)
                if od := create_odds_data(self.source_name, wo): odds[self.source_name] = od

            return Runner(number=number, name=name, odds=odds, win_odds=wo, scratched=sc)
        except Exception as e:
            self.logger.debug("equibase_runner_parse_failed", error=str(e))
            return None

    def _parse_post_time(self, ds: str, ts: str) -> datetime:
        try:
            parts = ts.replace("Post Time:", "").strip().split()
            if len(parts) >= 2:
                dt = datetime.strptime(f"{ds} {parts[0]} {parts[1]}", "%Y-%m-%d %I:%M %p")
                return dt.replace(tzinfo=EASTERN)
        except Exception: pass
        # Fallback to noon UTC for the given date if time parsing fails
        try:
            dt = datetime.strptime(ds, "%Y-%m-%d")
            return dt.replace(hour=12, minute=0, tzinfo=EASTERN)
        except Exception:
            return datetime.now(EASTERN)

# ----------------------------------------
# TwinSpiresAdapter
# ----------------------------------------
class TwinSpiresAdapter(JSONParsingMixin, DebugMixin, BaseAdapterV3):
    SOURCE_NAME: ClassVar[str] = "TwinSpires"
    BASE_URL: ClassVar[str] = "https://www.twinspires.com"

    RACE_CONTAINER_SELECTORS: ClassVar[List[str]] = ['div[class*="RaceCard"]', 'div[class*="race-card"]', 'div[data-testid*="race"]', 'div[data-race-id]', 'section[class*="race"]', 'article[class*="race"]', ".race-container", "[data-race]", 'div[class*="card"][class*="race" i]', 'div[class*="event"]']
    TRACK_NAME_SELECTORS: ClassVar[List[str]] = ['[class*="track-name"]', '[class*="trackName"]', '[data-track-name]', 'h2[class*="track"]', 'h3[class*="track"]', ".track-title", '[class*="venue"]']
    RACE_NUMBER_SELECTORS: ClassVar[List[str]] = ['[class*="race-number"]', '[class*="raceNumber"]', '[class*="race-num"]', '[data-race-number]', 'span[class*="number"]']
    POST_TIME_SELECTORS: ClassVar[List[str]] = ["time[datetime]", '[class*="post-time"]', '[class*="postTime"]', '[class*="mtp"]', "[data-post-time]", '[class*="race-time"]']
    RUNNER_ROW_SELECTORS: ClassVar[List[str]] = ['tr[class*="runner"]', 'div[class*="runner"]', 'li[class*="runner"]', "[data-runner-id]", 'div[class*="horse-row"]', 'tr[class*="horse"]', 'div[class*="entry"]', ".runner-row", ".horse-entry"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config, enable_cache=True, cache_ttl=180.0, rate_limit=1.5)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        # Try CURL_CFFI for TwinSpires as well, sometimes it's less aggressive than Playwright detection
        return FetchStrategy(
            primary_engine=BrowserEngine.PLAYWRIGHT,
            enable_js=True,
            stealth_mode="camouflage",
            timeout=60
        )

    async def make_request(self, method: str, url: str, **kwargs: Any) -> Any:
        # Force chrome120 for TwinSpires to bypass basic bot checks
        kwargs.setdefault("impersonate", "chrome120")
        kwargs.setdefault("headers", {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"})
        return await super().make_request(method, url, **kwargs)

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        ard = []
        last_err = None

        # Respect region from config if provided
        target_region = self.config.get("region") # "USA", "INT", or None for both

        async def fetch_disc(disc, region="USA"):
            suffix = "" if region == "USA" else "?region=INT"
            # Try date-specific URL first, fallback to todays-races
            # TwinSpires uses YYYY-MM-DD for races URL
            if date == datetime.now(EASTERN).strftime("%Y-%m-%d"):
                url = f"{self.BASE_URL}/bet/todays-races/{disc}{suffix}"
            else:
                url = f"{self.BASE_URL}/bet/races/{date}/{disc}{suffix}"
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
        tasks = []
        for d in ["thoroughbred", "harness", "greyhound"]:
            if target_region in [None, "USA"]:
                tasks.append(fetch_disc(d, "USA"))
            if target_region in [None, "INT"]:
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
        rl, ds, parsed = raw_data["races"], raw_data.get("date", datetime.now(EASTERN).strftime("%Y-%m-%d")), []
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
        bd = datetime.strptime(ds, "%Y-%m-%d").date()
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
                            dt = datetime.fromisoformat(da.replace('Z', '+00:00'))
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
        if not sc:
            for s in ['[class*="odds"]', '[class*="ml"]', '[class*="morning-line"]', '[data-odds]']:
                try:
                    oe = e.css_first(s)
                    if oe:
                        ot = node_text(oe)
                        if ot and ot.upper() not in ['SCR', 'SCRATCHED', '--', 'N/A']:
                            wo = parse_odds_to_decimal(ot)
                            if od := create_odds_data(self.source_name, wo): odds[self.source_name] = od; break
                except Exception: continue

            # Advanced heuristic fallback
            if wo is None:
                wo = SmartOddsExtractor.extract_from_node(e)
                if od := create_odds_data(self.source_name, wo): odds[self.source_name] = od

        return Runner(number=num or dn, name=name, scratched=sc, odds=odds, win_odds=wo)

    async def cleanup(self):
        await self.close()
        self.logger.info("TwinSpires adapter cleaned up")


# ----------------------------------------
# ANALYZER LOGIC
# ----------------------------------------

log = structlog.get_logger(__name__)


def _get_best_win_odds(runner: Runner) -> Optional[Decimal]:
    """Gets the best win odds for a runner, filtering out invalid or placeholder values."""
    if not runner.odds:
        # Fallback to win_odds if available
        if runner.win_odds and 0 < runner.win_odds < 999:
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

        if win is not None and 0 < win < 999:
            valid_odds.append(Decimal(str(win)))

    if valid_odds:
        return min(valid_odds)

    # Final fallback to win_odds if present
    if runner.win_odds and 0 < runner.win_odds < 999:
        return Decimal(str(runner.win_odds))

    # Absolute fallback to prevent empty results if requested (heuristics)
    if DEFAULT_ODDS_FALLBACK > 0:
        return Decimal(str(DEFAULT_ODDS_FALLBACK))

    return None


class BaseAnalyzer(ABC):
    """The abstract interface for all future analyzer plugins."""

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def qualify_races(self, races: List[Race]) -> Dict[str, Any]:
        """The core method every analyzer must implement."""
        pass


class TrifectaAnalyzer(BaseAnalyzer):
    """Analyzes races and assigns a qualification score based on the 'Trifecta of Factors'."""

    @property
    def name(self) -> str:
        return "trifecta_analyzer"

    def __init__(
        self,
        max_field_size: int = 11,
        min_favorite_odds: float = 0.01,
        min_second_favorite_odds: float = 0.01,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_field_size = max_field_size
        self.min_favorite_odds = Decimal(str(min_favorite_odds))
        self.min_second_favorite_odds = Decimal(str(min_second_favorite_odds))
        self.notifier = RaceNotifier()

    def is_race_qualified(self, race: Race) -> bool:
        """A race is qualified for a trifecta if it has at least 3 non-scratched runners."""
        if not race or not race.runners:
            return False

        # Apply global timing cutoff (45m ago, 120m future)
        now = datetime.now(EASTERN)
        past_cutoff = now - timedelta(minutes=45)
        future_cutoff = now + timedelta(minutes=120)
        st = race.start_time
        if st.tzinfo is None:
            st = st.replace(tzinfo=EASTERN)
        if st < past_cutoff or st > future_cutoff:
            return False

        active_runners = sum(1 for r in race.runners if not r.scratched)
        return active_runners >= 3

    def qualify_races(self, races: List[Race]) -> Dict[str, Any]:
        """Scores all races and returns a dictionary with criteria and a sorted list."""
        qualified_races = []
        for race in races:
            if not self.is_race_qualified(race):
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
        race.qualification_score = score
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


class SimplySuccessAnalyzer(BaseAnalyzer):
    """An analyzer that qualifies every race to show maximum successes (HTTP 200)."""

    @property
    def name(self) -> str:
        return "simply_success"

    def qualify_races(self, races: List[Race]) -> Dict[str, Any]:
        """Returns races with a perfect score, applying global timing and chalk filters."""
        qualified = []
        now = datetime.now(EASTERN)

        for race in races:
            # 1. Timing Filter: Relaxed for "News" mode (GPT5: Caller handles strict timing)
            st = race.start_time
            if st.tzinfo is None:
                st = st.replace(tzinfo=EASTERN)

            # 2. Chalk Filter: Exclude races with an odds-on favorite (< 2.0)
            # if best_odds is not None and best_odds < 2.0:
            #     log.debug("Excluding chalk race", venue=race.venue, favorite_odds=best_odds)
            #     continue

            # Goldmine Detection: 2nd favorite >= 4.5 decimal
            # A race cannot be a goldmine if field size is over 11
            is_goldmine = False
            active_runners = [r for r in race.runners if not r.scratched]
            gap12 = 0.0
            all_odds = []
            if active_runners:
                for runner in active_runners:
                    odds = _get_best_win_odds(runner)
                    if odds is not None:
                        all_odds.append(odds)
                if len(all_odds) >= 2:
                    all_odds.sort()
                    fav, sec = all_odds[0], all_odds[1]
                    gap12 = round(float(sec - fav), 2)
                    if gap12 == 0:
                        log.debug("Zero gap detected (1Gap2=0), ineligible for Best Bet treatment", venue=race.venue, race=race.race_number)
                    if len(active_runners) <= 11 and sec >= Decimal("4.5") and gap12 > 0:
                        is_goldmine = True

                # Calculate Top 5 for all races
                # Collect valid odds once to avoid repetitive calculation/conversion
                valid_r_with_odds = []
                for r in active_runners:
                    wo = _get_best_win_odds(r)
                    if wo is not None:
                        # Propagate fresh odds to runner object for reporting
                        r.win_odds = float(wo)
                        valid_r_with_odds.append((r, wo))

                r_with_odds = sorted(valid_r_with_odds, key=lambda x: x[1])
                race.top_five_numbers = ", ".join([str(r[0].number or '?') for r in r_with_odds[:5]])

                if len(r_with_odds) >= 2:
                    sec_fav = r_with_odds[1][0]
                    race.metadata['selection_number'] = sec_fav.number
                    race.metadata['selection_name'] = sec_fav.name

            # Stability Check: Ensure we have at least 2 active runners to compare
            if len(active_runners) < 2:
                log.debug("Excluding race with < 2 runners", venue=race.venue)
                continue

            if len(all_odds) < 2:
                # If we still don't have enough odds despite enrichment, use fallback
                while len(all_odds) < 2:
                    all_odds.append(Decimal(str(DEFAULT_ODDS_FALLBACK)))

            # Best Bet Detection:
            # Goldmine = 2nd Fav >= 4.5, Field <= 11, Gap > 0
            # You Might Like = 2nd Fav >= 3.5, Field <= 11, Gap > 0
            is_best_bet = (len(active_runners) <= 11 and all_odds[1] >= Decimal("3.5") and gap12 > 0)

            race.metadata['is_goldmine'] = is_goldmine
            race.metadata['is_best_bet'] = is_best_bet
            race.metadata['1Gap2'] = gap12
            race.metadata['predicted_2nd_fav_odds'] = float(all_odds[1])
            race.qualification_score = 100.0
            qualified.append(race)

        if not qualified:
            log.warning("🔭 SimplySuccess analyzer pass returned 0 qualified races", input_count=len(races))

        return {
            "criteria": {
                "mode": "simply_success",
                "timing_filter": "45m_past_to_120m_future",
                "chalk_filter": "disabled",
                "goldmine_threshold": 4.5
            },
            "races": qualified
        }


class AnalyzerEngine:
    """Discovers and manages all available analyzer plugins."""

    def __init__(self):
        self.analyzers: Dict[str, Type[BaseAnalyzer]] = {}
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
        return analyzer_class(**kwargs)


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
        message = f"{race.venue} - Race {race.race_number}\nScore: {race.qualification_score:.0f}%\nPost Time: {race.start_time.strftime('%I:%M %p')}"

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

        # Only interested in field sizes 3-11 for this report
        if 3 <= field_size <= 11:
            is_gold = get_field(r, 'metadata', {}).get('is_goldmine', False)
            race_num = get_field(r, 'race_number')
            matrix[track][field_size].append((race_num, is_gold))

    if not matrix:
        return "No qualifying races for field matrix (3-11 runners)."

    # Header: Display sizes 3 to 11
    display_sizes = range(3, 12)

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

    # Pre-calculate track categories
    track_categories = {}
    source_races_for_cat = all_races if all_races is not None else races
    races_by_track = defaultdict(list)
    for r in source_races_for_cat:
        v = get_field(r, 'venue')
        track = normalize_venue_name(v)
        races_by_track[track].append(r)
    for track, tr_races in races_by_track.items():
        track_categories[track] = get_track_category(tr_races)

    def is_superfecta_effective(r):
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

    goldmines = [r for r in races if get_field(r, 'metadata', {}).get('is_goldmine') and is_superfecta_effective(r)]

    if not goldmines:
        lines.append("No qualifying races.")
        return "\n".join(lines)

    track_to_nums = defaultdict(list)
    for r in goldmines:
        v = get_field(r, 'venue')
        if v:
            track = normalize_venue_name(v)
            track_to_nums[track].append(get_field(r, 'race_number'))

    # Sort tracks descending by category (T > H > G)
    cat_map = {'T': 3, 'H': 2, 'G': 1}

    formatted_tracks = []
    for track in track_to_nums.keys():
        cat = track_categories.get(track, 'T')
        display_name = f"{cat}~{track}"
        formatted_tracks.append((cat, track, display_name))

    # Sort: Category Descending, then Track Name Ascending
    formatted_tracks.sort(key=lambda x: (-cat_map.get(x[0], 0), x[1]))

    for cat, track, display_name in formatted_tracks:
        nums = sorted(list(set(track_to_nums[track])))
        lines.append(f"{display_name}: {', '.join(map(str, nums))}")
    return "\n".join(lines)


def generate_goldmine_report(races: List[Any], all_races: Optional[List[Any]] = None) -> str:
    """Generate a detailed report for Goldmine races."""
    # 1. Reuse category logic
    track_categories = {}
    source_races_for_cat = all_races if all_races is not None else races
    races_by_track = defaultdict(list)
    for r in source_races_for_cat:
        v = get_field(r, 'venue')
        track = normalize_venue_name(v)
        races_by_track[track].append(r)
    for track, tr_races in races_by_track.items():
        track_categories[track] = get_track_category(tr_races)

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

    # Include all goldmines (2nd fav >= 4.5)
    # Deduplicate to prevent double-reporting (e.g. from multiple sources)
    goldmines = []
    seen_gold = set()
    for r in races:
        if get_field(r, 'metadata', {}).get('is_goldmine'):
            track = get_canonical_venue(get_field(r, 'venue'))
            num = get_field(r, 'race_number')
            st = get_field(r, 'start_time')
            st_str = st.strftime('%Y%m%d') if isinstance(st, datetime) else str(st)
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
    immediate_gold_superfecta = []
    immediate_gold = []
    remaining_gold = []

    for r in goldmines:
        start_time = get_field(r, 'start_time')
        if isinstance(start_time, str):
            try:
                start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            except ValueError:
                remaining_gold.append(r)
                continue

        if start_time:
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=EASTERN)

            diff = (start_time - now).total_seconds() / 60
            if 0 <= diff <= 20:
                if is_superfecta_available(r):
                    immediate_gold_superfecta.append(r)
                else:
                    immediate_gold.append(r)
            else:
                remaining_gold.append(r)
        else:
            remaining_gold.append(r)

    report_lines = ["LIST OF BEST BETS - GOLDMINE REPORT", "===================================", ""]

    def render_races(races_to_render, label):
        if not races_to_render:
            return
        report_lines.append(f"--- {label.upper()} ---")
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

            gap12 = get_field(r, 'metadata', {}).get('1Gap2', 0.0)
            report_lines.append(f"{cat}~{track} - Race {race_num} ({time_str})")
            report_lines.append(f"PREDICTED TOP 5: [{top_5_nums}] | 1Gap2: {gap12:.2f}")
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

    if immediate_gold_superfecta:
        render_races(immediate_gold_superfecta, "Immediate Gold (superfecta)")

    if immediate_gold:
        render_races(immediate_gold, "Immediate Gold")

    if remaining_gold:
        render_races(remaining_gold, "All Remaining Goldmine Races")

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
            st = datetime.fromisoformat(start_time_raw.replace('Z', '+00:00'))
            time_str = to_eastern(st).strftime("%Y-%m-%d %H:%M ET")
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
                r_time = datetime.fromisoformat(r_time.replace('Z', '+00:00'))
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


async def generate_friendly_html_report(races: List[Any], stats: Dict[str, Any]) -> str:
    """Generates a high-impact, friendly HTML report for the Fortuna Faucet."""
    now_str = datetime.now(EASTERN).strftime('%Y-%m-%d %H:%M:%S')

    # 1. Best Bet Opportunities
    rows = []
    for r in sorted(races, key=lambda x: getattr(x, 'start_time', '')):
        # Get selection (2nd favorite)
        runners = getattr(r, 'runners', [])
        active = [run for run in runners if not getattr(run, 'scratched', False)]
        if len(active) < 2: continue

        active.sort(key=lambda x: getattr(x, 'win_odds', 999.0) or 999.0)
        sel = active[1]

        st = getattr(r, 'start_time', '')
        if isinstance(st, datetime):
            # Ensure it's in Eastern for display (GPT5 Improvement)
            st_str = to_eastern(st).strftime('%H:%M')
        elif isinstance(st, str):
            try:
                dt = datetime.fromisoformat(st.replace('Z', '+00:00'))
                st_str = to_eastern(dt).strftime('%H:%M')
            except Exception:
                st_str = str(st)[11:16]
        else:
            st_str = str(st)[11:16]

        is_gold = getattr(r, 'metadata', {}).get('is_goldmine', False)
        gold_badge = '<span class="badge gold">GOLD</span>' if is_gold else ''

        rows.append(f"""
            <tr>
                <td>{st_str}</td>
                <td>{getattr(r, 'venue', 'Unknown')}</td>
                <td>R{getattr(r, 'race_number', '?')}</td>
                <td>#{getattr(sel, 'number', '?')} {getattr(sel, 'name', 'Unknown')}</td>
                <td>{ (getattr(sel, 'win_odds') or 0.0):.2f}</td>
                <td>{gold_badge}</td>
            </tr>
        """)

    tips_count = stats.get('tips', 0)
    cashed_count = stats.get('cashed', 0)
    profit = stats.get('profit', 0.0)

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fortuna Faucet Intelligence Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; background-color: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }}
            .container {{ max-width: 1000px; margin: 0 auto; background-color: #1e293b; padding: 30px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
            h1 {{ color: #fbbf24; text-align: center; text-transform: uppercase; letter-spacing: 3px; border-bottom: 2px solid #fbbf24; padding-bottom: 15px; }}
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
            .footer {{ margin-top: 40px; text-align: center; font-size: 12px; color: #64748b; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Fortuna Faucet Intelligence</h1>
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

            {await _generate_audit_history_html()}

            <div class="footer">
                Fortuna Faucet Portable App - Sci-Fi Intelligence Edition<br>
                Powered by the Council of Superbrains
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
    Sorted by MTP, ceiling of 4 hours from now.
    """
    now = datetime.now(EASTERN)
    cutoff = now + timedelta(hours=18)

    # 1. Pre-calculate track categories
    track_categories = {}
    source_races = all_races if all_races is not None else races
    races_by_track = defaultdict(list)
    for r in source_races:
        venue = get_field(r, 'venue')
        track = normalize_venue_name(venue)
        races_by_track[track].append(r)

    for track, tr_races in races_by_track.items():
        track_categories[track] = get_track_category(tr_races)

    table_races = []
    seen = set()
    for race in (all_races or races):
        st = get_field(race, 'start_time')
        if isinstance(st, str):
            try: st = datetime.fromisoformat(st.replace('Z', '+00:00'))
            except Exception: continue
        if st and st.tzinfo is None: st = st.replace(tzinfo=EASTERN)

        # Ceiling of 18 hours, ignore races more than 10 mins past
        if not st or st < now - timedelta(minutes=10) or st > cutoff:
            continue

        track = normalize_venue_name(get_field(race, 'venue'))
        canonical_track = get_canonical_venue(get_field(race, 'venue'))
        num = get_field(race, 'race_number')
        # Deduplication key: Use canonical track/num/date
        key = (canonical_track, num, st.strftime('%Y%m%d'))
        if key in seen: continue
        seen.add(key)

        mtp = int((st - now).total_seconds() / 60)
        runners = get_field(race, 'runners', [])
        field_size = len([run for run in runners if not get_field(run, 'scratched', False)])
        top5 = getattr(race, 'top_five_numbers', 'N/A')
        gap12 = get_field(race, 'metadata', {}).get('1Gap2', 0.0)
        is_gold = get_field(race, 'metadata', {}).get('is_goldmine', False)

        table_races.append({
            'mtp': mtp,
            'cat': track_categories.get(track, 'T'),
            'track': track,
            'num': num,
            'field': field_size,
            'top5': top5,
            'gap': gap12,
            'gold': '[G]' if is_gold else ''
        })

    # Sort by MTP
    table_races.sort(key=lambda x: x['mtp'])

    if not table_races:
        return "No upcoming races in the next 4 hours."

    lines = [
        "| MTP | CAT | TRACK | R# | FLD | TOP 5 | GAP | |",
        "|:---:|:---:|:---|:---:|:---:|:---|:---:|:---:|"
    ]
    for tr in table_races:
        # Better alignment: leading zero for single digits (Memory Directive Fix)
        mtp_val = tr['mtp']
        mtp_str = f"{mtp_val:02d}" if 0 <= mtp_val < 10 else str(mtp_val)
        lines.append(f"| {mtp_str}m | {tr['cat']} | {tr['track'][:20]} | {tr['num']} | {tr['field']} | `{tr['top5']}` | {tr['gap']:.2f} | {tr['gold']} |")

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


def format_prediction_row(race: Race) -> str:
    """Formats a single race prediction for the GHA Job Summary table."""
    metadata = getattr(race, 'metadata', {})
    gold = '✅' if metadata.get('is_goldmine') else '—'
    selection = metadata.get('selection_name') or f"#{metadata.get('selection_number', '?')}"
    odds = metadata.get('predicted_2nd_fav_odds')
    odds_str = f"{odds:.2f}" if odds else 'N/A'
    top5 = getattr(race, 'top_five_numbers', 'TBD')
    gap = metadata.get('1Gap2', 0.0)
    gap_str = f"{gap:.2f}"

    payouts = []
    # Check both metadata and attributes for payouts
    for label in ('top1_place_payout', 'trifecta_payout', 'superfecta_payout'):
        val = metadata.get(label) or getattr(race, label, None)
        if val:
            display_label = label.replace('_', ' ').title().replace('Top1 ', '')
            payouts.append(f"{display_label}: ${float(val):.2f}")

    payout_text = ' | '.join(payouts) or 'Awaiting Results'
    return f"| {race.venue} | {race.race_number} | {selection} | {odds_str} | {gap_str} | {gold} | {top5} | {payout_text} |"


def format_predictions_section(qualified_races: List[Race]) -> str:
    """Generates the Predictions & Proof section for the GHA Job Summary."""
    lines = ["### 🔮 Fortuna Predictions & Proof", ""]
    if not qualified_races:
        lines.append("No Goldmine predictions available for this run.")
        return "\n".join(lines)

    now = datetime.now(EASTERN)

    def get_mtp(r):
        st = r.start_time
        if isinstance(st, str):
            try:
                st = datetime.fromisoformat(st.replace('Z', '+00:00'))
            except Exception:
                return 9999
        if st and st.tzinfo is None:
            st = st.replace(tzinfo=EASTERN)
        return (st - now).total_seconds() / 60 if st else 9999

    # Sort by MTP ascending
    sorted_races = sorted(qualified_races, key=get_mtp)
    # Take top 10 opportunities
    top_10 = sorted_races[:10]

    lines.extend([
        "| Venue | Race# | Selection | Odds | Gap | Goldmine? | Pred Top 5 | Payout Proof |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    ])
    for r in top_10:
        lines.append(format_prediction_row(r))
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

        lines.extend([
            "| Verdict | Profit | Venue | R# | Actual Top 5 | Actual 2nd Fav Odds | Payout Details |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
        ])
        for tip in tips:
            payouts = []
            if tip.get('superfecta_payout'):
                payouts.append(f"Superfecta ${tip['superfecta_payout']:.2f}")
            if tip.get('trifecta_payout'):
                payouts.append(f"Trifecta ${tip['trifecta_payout']:.2f}")
            if tip.get('top1_place_payout'):
                payouts.append(f"Place ${tip['top1_place_payout']:.2f}")

            payout_text = ' / '.join(payouts) if payouts else 'No payout data'

            verdict = tip.get("verdict", "?")
            emoji = "✅" if verdict == "CASHED" else "❌" if verdict == "BURNED" else "⚪"
            profit = tip.get('net_profit', 0.0)
            actual_odds = tip.get('actual_2nd_fav_odds')
            actual_odds_str = f"{actual_odds:.2f}" if actual_odds else "N/A"

            lines.append(
                f"| {emoji} {verdict} | ${profit:+.2f} | {tip['venue']} | {tip['race_number']} | {tip.get('actual_top_5', 'N/A')} | {actual_odds_str} | {payout_text} |"
            )
    except Exception as e:
        lines.append(f"Error generating audited proof: {e}")

    return "\n".join(lines)


def build_harvest_table(summary: Dict[str, Any], title: str) -> str:
    """Generates a harvest performance table for the GHA Job Summary."""
    lines = [f"### {title}", ""]
    if not summary:
        lines.extend([
            "| Adapter | Races | Max Odds | Status |",
            "| --- | --- | --- | --- |",
            "| N/A | 0 | 0.0 | ⚠️ No harvest data |"
        ])
        return "\n".join(lines)

    lines.extend([
        "| Adapter | Races | Max Odds | Status |",
        "| --- | --- | --- | --- |"
    ])

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
        lines.append(f"| {adapter} | {count} | {max_odds:.1f} | {status} |")
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


def write_job_summary(predictions_md: str, harvest_md: str, proof_md: str, artifacts_md: str) -> None:
    """Writes the consolidated sections to $GITHUB_STEP_SUMMARY."""
    path = os.environ.get('GITHUB_STEP_SUMMARY')
    if not path:
        return

    # Narrate the entire workflow
    summary = '\n'.join([
        '## 🔔 Fortuna Intelligence Job Summary',
        '',
        predictions_md,
        '',
        harvest_md,
        '',
        proof_md,
        '',
        artifacts_md,
    ])

    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(summary + '\n')
    except Exception:
        # Silently fail if writing summary fails
        pass


def get_db_path() -> str:
    """Returns the path to the SQLite database, using AppData in frozen mode."""
    if is_frozen() and sys.platform == "win32":
        appdata = os.getenv('APPDATA')
        if appdata:
            db_dir = Path(appdata) / "Fortuna"
            db_dir.mkdir(parents=True, exist_ok=True)
            return str(db_dir / "fortuna.db")

    return os.environ.get("FORTUNA_DB_PATH", "fortuna.db")


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
        with self._conn_lock:
            if not self._conn:
                self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._conn.execute("PRAGMA journal_mode=WAL")
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

    async def initialize(self):
        """Creates the database schema if it doesn't exist."""
        if self._initialized: return

        def _init():
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
                        gap12 TEXT,
                        top_five TEXT,
                        selection_number INTEGER,
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
                        audit_timestamp TEXT
                    )
                """)
                # Composite index for deduplication - changed to race_id only for better deduplication
                conn.execute("DROP INDEX IF EXISTS idx_race_report")

                # Cleanup potential duplicates before creating unique index (Memory Directive Fix)
                try:
                    self.logger.info("Cleaning up duplicate race_ids before indexing")
                    conn.execute("""
                        DELETE FROM tips
                        WHERE id NOT IN (
                            SELECT MAX(id)
                            FROM tips
                            GROUP BY race_id
                        )
                    """)
                    self.logger.info("Duplicates removed, creating unique index")
                    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_race_id ON tips (race_id)")
                except Exception as e:
                    self.logger.error("Failed to cleanup or create unique index", error=str(e))
                    # If index exists but table has duplicates, we might get IntegrityError
                    # Just log it and continue - better than crashing the whole app
                # Composite index for audit performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_time ON tips (audit_completed, start_time)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_venue ON tips (venue)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_discipline ON tips (discipline)")

                # Add missing columns for existing databases
                cursor = conn.execute("PRAGMA table_info(tips)")
                columns = [column[1] for column in cursor.fetchall()]
                if "superfecta_payout" not in columns:
                    conn.execute("ALTER TABLE tips ADD COLUMN superfecta_payout REAL")
                if "superfecta_combination" not in columns:
                    conn.execute("ALTER TABLE tips ADD COLUMN superfecta_combination TEXT")
                if "top1_place_payout" not in columns:
                    conn.execute("ALTER TABLE tips ADD COLUMN top1_place_payout REAL")
                if "top2_place_payout" not in columns:
                    conn.execute("ALTER TABLE tips ADD COLUMN top2_place_payout REAL")
                if "discipline" not in columns:
                    conn.execute("ALTER TABLE tips ADD COLUMN discipline TEXT")
                if "predicted_2nd_fav_odds" not in columns:
                    conn.execute("ALTER TABLE tips ADD COLUMN predicted_2nd_fav_odds REAL")
                if "actual_2nd_fav_odds" not in columns:
                    conn.execute("ALTER TABLE tips ADD COLUMN actual_2nd_fav_odds REAL")

        await self._run_in_executor(_init)

        # Track and execute migrations based on schema version
        def _get_version():
            cursor = self._get_conn().execute("SELECT MAX(version) FROM schema_version")
            row = cursor.fetchone()
            return row[0] if row and row[0] is not None else 0

        current_version = await self._run_in_executor(_get_version)

        if current_version < 2:
            await self.migrate_utc_to_eastern()
            def _update_version():
                with self._get_conn() as conn:
                    conn.execute("INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (2, ?)", (datetime.now(EASTERN).isoformat(),))
            await self._run_in_executor(_update_version)
            self.logger.info("Schema migrated to version 2")

        if current_version < 3:
            def _declutter():
                # Delete old records to keep database lean (30-day retention cleanup)
                cutoff = (datetime.now(EASTERN) - timedelta(days=30)).isoformat()
                with self._get_conn() as conn:
                    cursor = conn.execute("DELETE FROM tips WHERE report_date < ?", (cutoff,))
                    self.logger.info("Database decluttered (30-day retention cleanup)", deleted_count=cursor.rowcount)
                    conn.execute("INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (3, ?)", (datetime.now(EASTERN).isoformat(),))
            await self._run_in_executor(_declutter)
            self.logger.info("Schema migrated to version 3")

        if current_version < 4:
            # Migration to version 4: Housekeeping & Long-term retention.
            # 1. Clear the tips table for a fresh start as requested by JB.
            # 2. Historical retention is now enabled (auto-cleanup removed from future migrations).
            def _housekeeping():
                self.logger.warning("Applying destructive migration: Clearing all historical tips for version 4 fresh start.")
                with self._get_conn() as conn:
                    conn.execute("DELETE FROM tips")
                    conn.execute("INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (4, ?)", (datetime.now(EASTERN).isoformat(),))
            await self._run_in_executor(_housekeeping)
            self.logger.info("Schema migrated to version 4 (Housekeeping complete, long-term retention enabled)")

        self._initialized = True
        self.logger.info("Database initialized", path=self.db_path, schema_version=max(current_version, 4))

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

            self.logger.info("Migrating legacy UTC timestamps to Eastern", count=len(rows))
            converted = 0
            errors = 0
            with conn:
                for row in rows:
                    updates = {}
                    for col in ["start_time", "report_date", "audit_timestamp"]:
                        if col not in row.keys(): continue
                        val = row[col]
                        if val:
                            try:
                                dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
                                dt_eastern = ensure_eastern(dt)
                                updates[col] = dt_eastern.isoformat()
                            except Exception: pass
                    if updates:
                        try:
                            set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
                            conn.execute(f"UPDATE tips SET {set_clause} WHERE id = ?", (*updates.values(), row["id"]))
                            converted += 1
                        except Exception as e:
                            errors += 1
                            self.logger.warning("Failed to migrate row", row_id=row["id"], error=str(e))
            self.logger.info("Migration complete", total=len(rows), converted=converted, errors=errors)
        await self._run_in_executor(_migrate)

    async def log_harvest(self, harvest_summary: Dict[str, Any], region: Optional[str] = None):
        """Logs harvest performance metrics to the database."""
        if not self._initialized: await self.initialize()

        def _log():
            conn = self._get_conn()
            now = datetime.now(EASTERN).isoformat()
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
            cutoff = (datetime.now(EASTERN) - timedelta(days=days)).isoformat()
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
        """Logs new tips to the database with batch deduplication."""
        if not self._initialized: await self.initialize()

        def _log():
            conn = self._get_conn()
            now = datetime.now(EASTERN)

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

            to_insert = []
            for tip in tips:
                rid = tip.get("race_id")
                if rid and rid not in already_logged:
                    report_date = tip.get("report_date") or now.isoformat()
                    to_insert.append((
                        rid, tip.get("venue"), tip.get("race_number"),
                        tip.get("discipline"), tip.get("start_time"), report_date,
                        1 if tip.get("is_goldmine") else 0,
                        str(tip.get("1Gap2", 0.0)),
                        tip.get("top_five"), tip.get("selection_number"),
                        float(tip.get("predicted_2nd_fav_odds")) if tip.get("predicted_2nd_fav_odds") is not None else None
                    ))
                    already_logged.add(rid) # Avoid duplicates within the same batch

            if to_insert:
                with conn:
                    conn.executemany("""
                        INSERT OR IGNORE INTO tips (
                            race_id, venue, race_number, discipline, start_time, report_date,
                            is_goldmine, gap12, top_five, selection_number, predicted_2nd_fav_odds
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, to_insert)
                self.logger.info("Hot tips batch logged", count=len(to_insert))

        await self._run_in_executor(_log)

    async def get_unverified_tips(self, lookback_hours: int = 48) -> List[Dict[str, Any]]:
        """Returns tips that haven't been audited yet but have likely finished."""
        if not self._initialized: await self.initialize()

        def _get():
            conn = self._get_conn()
            now = datetime.now(EASTERN)
            cutoff = (now - timedelta(hours=lookback_hours)).isoformat()

            cursor = conn.execute(
                """SELECT * FROM tips
                   WHERE audit_completed = 0
                   AND report_date > ?
                   AND start_time < ?""",
                (cutoff, now.isoformat())
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
                        audit_timestamp = ?
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
                    datetime.now(EASTERN).isoformat(),
                    race_id
                ))
        await self._run_in_executor(_update)

    async def get_all_audited_tips(self) -> List[Dict[str, Any]]:
        """Returns all audited tips for reporting."""
        if not self._initialized: await self.initialize()
        def _get():
            cursor = self._get_conn().execute(
                "SELECT * FROM tips WHERE audit_completed = 1 ORDER BY start_time DESC"
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

    async def clear_all_tips(self):
        """Wipes all records from the tips table."""
        if not self._initialized: await self.initialize()
        def _clear():
            conn = self._get_conn()
            with conn:
                conn.execute("DELETE FROM tips")
            conn.execute("VACUUM")
            self.logger.info("Database cleared (all tips deleted)")
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
                                    is_goldmine, gap12, top_five, selection_number,
                                    audit_completed, verdict, net_profit, selection_position,
                                    actual_top_5, actual_2nd_fav_odds, trifecta_payout,
                                    trifecta_combination, superfecta_payout,
                                    superfecta_combination, top1_place_payout,
                                    top2_place_payout, audit_timestamp
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                entry.get("race_id"), entry.get("venue"), entry.get("race_number"),
                                entry.get("start_time"), entry.get("report_date"),
                                1 if entry.get("is_goldmine") else 0, str(entry.get("1Gap2", 0.0)),
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
    def __init__(self, db_path: Optional[str] = None):
        self.db = FortunaDB(db_path) if db_path else FortunaDB()
        self.logger = structlog.get_logger(self.__class__.__name__)

    async def log_tips(self, races: List[Race]):
        if not races:
            return

        await self.db.initialize()
        now = datetime.now(EASTERN)
        report_date = now.isoformat()
        new_tips = []

        # Strict future cutoff to prevent leakage (Never log more than 20 mins ahead)
        future_limit = now + timedelta(minutes=45)

        for r in races:
            # Only store "Best Bets" (Goldmine, BET NOW, or You Might Like)
            # These are marked in metadata by the analyzer.
            if not r.metadata.get('is_best_bet') and not r.metadata.get('is_goldmine'):
                continue

            st = r.start_time
            if isinstance(st, str):
                try: st = datetime.fromisoformat(st.replace('Z', '+00:00'))
                except Exception: continue
            if st.tzinfo is None: st = st.replace(tzinfo=EASTERN)

            # Reject races too far in the future
            if st > future_limit or st < now - timedelta(minutes=10):
                self.logger.debug("Rejecting far-future race", venue=r.venue, start_time=st)
                continue

            is_goldmine = r.metadata.get('is_goldmine', False)
            gap12 = r.metadata.get('1Gap2', 0.0)

            tip_data = {
                "report_date": report_date,
                "race_id": r.id,
                "venue": r.venue,
                "race_number": r.race_number,
                "start_time": r.start_time.isoformat() if isinstance(r.start_time, datetime) else str(r.start_time),
                "is_goldmine": is_goldmine,
                "1Gap2": gap12,
                "discipline": r.discipline,
                "top_five": r.top_five_numbers,
                "selection_number": r.metadata.get('selection_number'),
                "selection_name": r.metadata.get('selection_name'),
                "predicted_2nd_fav_odds": r.metadata.get('predicted_2nd_fav_odds')
            }
            new_tips.append(tip_data)

        try:
            await self.db.log_tips(new_tips)
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
    python favorite_to_place_monitor.py [--date YYYY-MM-DD] [--refresh-interval 30]
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
    favorite_odds: Optional[float] = None
    favorite_name: Optional[str] = None
    top_five_numbers: Optional[str] = None
    gap12: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "discipline": self.discipline,
            "track": self.track,
            "race_number": self.race_number,
            "field_size": self.field_size,
            "superfecta_offered": self.superfecta_offered,
            "adapter": self.adapter,
            "start_time": self.start_time.isoformat(),
            "mtp": self.mtp,
            "second_fav_odds": self.second_fav_odds,
            "second_fav_name": self.second_fav_name,
            "favorite_odds": self.favorite_odds,
            "favorite_name": self.favorite_name,
            "top_five_numbers": self.top_five_numbers,
            "gap12": self.gap12,
        }


def get_discovery_adapter_classes() -> List[Type[BaseAdapterV3]]:
    """Returns all non-abstract discovery adapter classes."""
    def get_all_subclasses(cls):
        return set(cls.__subclasses__()).union(
            [s for c in cls.__subclasses__() for s in get_all_subclasses(c)]
        )

    return [
        c for c in get_all_subclasses(BaseAdapterV3)
        if not getattr(c, "__abstractmethods__", None)
        and getattr(c, "ADAPTER_TYPE", "discovery") == "discovery"
    ]


class FavoriteToPlaceMonitor:
    """Monitor for favorite-to-place betting opportunities."""

    def __init__(self, target_dates: Optional[List[str]] = None, refresh_interval: int = 30, config: Optional[Dict] = None):
        """
        Initialize monitor.

        Args:
            target_dates: Dates to fetch races for (YYYY-MM-DD), defaults to today + tomorrow
            refresh_interval: Seconds between refreshes for BET NOW list
        """
        if target_dates:
            self.target_dates = target_dates
        else:
            today = datetime.now(EASTERN)
            tomorrow = today + timedelta(days=1)
            self.target_dates = [today.strftime("%Y-%m-%d"), tomorrow.strftime("%Y-%m-%d")]

        self.refresh_interval = refresh_interval
        self.config = config or {}
        self.all_races: List[RaceSummary] = []
        self.adapters: List = []
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.tracker = HotTipsTracker()

    async def initialize_adapters(self, adapter_names: Optional[List[str]] = None):
        """Initialize all adapters, optionally filtered by name."""
        all_discovery_classes = get_discovery_adapter_classes()

        classes_to_init = all_discovery_classes
        if adapter_names:
            classes_to_init = [c for c in all_discovery_classes if c.__name__ in adapter_names or getattr(c, "SOURCE_NAME", "") in adapter_names]

        self.logger.info("Initializing adapters", count=len(classes_to_init))

        for adapter_class in classes_to_init:
            try:
                adapter = adapter_class(config={"region": self.config.get("region")})
                self.adapters.append(adapter)
                self.logger.debug("Adapter initialized", adapter=adapter_class.__name__)
            except Exception as e:
                self.logger.error("Adapter initialization failed", adapter=adapter_class.__name__, error=str(e))

        self.logger.info("Adapters initialization complete", initialized=len(self.adapters))

    async def fetch_all_races(self) -> List[Tuple[Race, str]]:
        """Fetch races from all adapters."""
        self.logger.info("Fetching races", dates=self.target_dates)

        all_races_with_adapters = []

        # Run fetches in parallel for speed
        async def fetch_one(adapter, date_str):
            name = adapter.__class__.__name__
            try:
                races = await adapter.get_races(date_str)
                self.logger.info("Fetch complete", adapter=name, date=date_str, count=len(races))
                return [(r, name) for r in races]
            except Exception as e:
                self.logger.error("Fetch failed", adapter=name, date=date_str, error=str(e))
                return []

        fetch_tasks = []
        for d in self.target_dates:
            for a in self.adapters:
                fetch_tasks.append(fetch_one(a, d))

        results = await asyncio.gather(*fetch_tasks)
        for r_list in results:
            all_races_with_adapters.extend(r_list)

        self.logger.info("Total races fetched", total=len(all_races_with_adapters))
        return all_races_with_adapters

    def _get_discipline_code(self, race: Race) -> str:
        """Get discipline code (T/H/G)."""
        if not race.discipline:
            return "T"

        d = race.discipline.lower()
        if "harness" in d or "standardbred" in d: return "H"
        if "greyhound" in d or "dog" in d: return "G"
        return "T"

    def _calculate_field_size(self, race: Race) -> int:
        """Calculate active field size."""
        return len([r for r in race.runners if not r.scratched])

    def _has_superfecta(self, race: Race) -> bool:
        """Check if race offers Superfecta."""
        ab = race.available_bets or []
        # Support metadata fallback if field not populated
        if not ab and hasattr(race, 'metadata'):
            ab = race.metadata.get('available_bets', [])
        return "Superfecta" in ab

    def _get_top_runners(self, race: Race, limit: int = 5) -> List[Runner]:
        """Get top runners by odds, sorted lowest first."""
        # Get active runners with valid odds
        r_with_odds = []
        for r in race.runners:
            if r.scratched:
                continue
            # Refresh odds to avoid stale metadata in continuous monitor mode
            wo = _get_best_win_odds(r)
            if wo is not None and wo > 1.0:
                # Update runner object with fresh odds for downstream summaries
                r.win_odds = float(wo)
                # Store the Decimal odds directly for sorting to avoid conversion
                r_with_odds.append((r, wo))

        if not r_with_odds:
            return []

        # Sort by odds (lowest first)
        sorted_r = sorted(r_with_odds, key=lambda x: x[1])
        return [x[0] for x in sorted_r[:limit]]

    def _calculate_mtp(self, start_time: Optional[datetime]) -> int:
        """Calculate minutes to post. Returns -9999 if start_time is None."""
        if not start_time: return -9999
        now = now_eastern()
        # Use ensure_eastern to handle naive or other timezones correctly
        st = ensure_eastern(start_time)
        delta = st - now
        return int(delta.total_seconds() / 60)

    def _get_top_n_runners(self, race: Race, n: int = 5) -> str:
        """Get top N runners by win odds."""
        top_runners = self._get_top_runners(race, limit=n)
        return ", ".join([str(r.number) if r.number is not None else "?" for r in top_runners])

    def _create_race_summary(self, race: Race, adapter_name: str) -> RaceSummary:
        """Create a RaceSummary from a Race object."""
        top_runners = self._get_top_runners(race, limit=5)
        favorite = top_runners[0] if len(top_runners) >= 1 else None
        second_fav = top_runners[1] if len(top_runners) >= 2 else None

        gap12 = 0.0
        if favorite and second_fav and favorite.win_odds and second_fav.win_odds:
            gap12 = round(second_fav.win_odds - favorite.win_odds, 2)

        return RaceSummary(
            discipline=self._get_discipline_code(race),
            track=normalize_venue_name(race.venue),
            race_number=race.race_number,
            field_size=self._calculate_field_size(race),
            superfecta_offered=self._has_superfecta(race),
            adapter=adapter_name,
            start_time=race.start_time,
            mtp=self._calculate_mtp(race.start_time),
            second_fav_odds=second_fav.win_odds if second_fav else None,
            second_fav_name=second_fav.name if second_fav else None,
            favorite_odds=favorite.win_odds if favorite else None,
            favorite_name=favorite.name if favorite else None,
            top_five_numbers=self._get_top_n_runners(race, 5),
            gap12=gap12
        )

    async def build_race_summaries(self, races_with_adapters: List[Tuple[Race, str]], window_hours: Optional[int] = 12):
        """Build and deduplicate summary list, with optional time window filtering."""
        race_map = {}
        now = datetime.now(EASTERN)
        cutoff = now + timedelta(hours=window_hours) if window_hours else None

        for race, adapter_name in races_with_adapters:
            try:
                # Time window filtering
                st = race.start_time
                if st.tzinfo is None: st = st.replace(tzinfo=EASTERN)

                # Time window filtering removed to ensure all unique races are counted

                summary = self._create_race_summary(race, adapter_name)
                # Stable key: Canonical Venue + Race Number + Date
                canonical_venue = get_canonical_venue(summary.track)
                date_str = summary.start_time.strftime('%Y%m%d') if summary.start_time else "Unknown"
                key = f"{canonical_venue}|{summary.race_number}|{date_str}"

                if key not in race_map:
                    race_map[key] = summary
                else:
                    existing = race_map[key]
                    # Prefer the one with valid second favorite odds
                    if summary.second_fav_odds and not existing.second_fav_odds:
                        race_map[key] = summary
                    # Or prefer more detailed available bets
                    elif summary.superfecta_offered and not existing.superfecta_offered:
                        race_map[key] = summary
            except Exception: pass

        unique_summaries = list(race_map.values())
        self.all_races = sorted(unique_summaries, key=lambda x: x.start_time)

        # GPT5 Improvement: Keep all races within window for analysis, not just one per track.
        # Window broadened to 18 hours (News Mode)
        timing_window_summaries = []
        now = datetime.now(EASTERN)
        for summary in unique_summaries:
            st = summary.start_time
            if st.tzinfo is None: st = st.replace(tzinfo=EASTERN)

            # Calculate Minutes to Post
            diff = st - now
            mtp = diff.total_seconds() / 60

            # Broaden window to 18 hours to ensure yield for "News"
            if -45 < mtp <= 1080: # 18 hours
                timing_window_summaries.append(summary)

        self.golden_zone_races = timing_window_summaries
        if not self.golden_zone_races:
            self.logger.warning("🔭 Monitor found 0 races in the Broadened Window (-45m to 18h)", total_unique=len(unique_summaries))

    def print_full_list(self):
        """Log all fetched races."""
        lines = [
            "=" * 120,
            "FULL RACE LIST".center(120),
            "=" * 120,
            f"{'DISC':<5} {'TRACK':<25} {'R#':<4} {'FIELD':<6} {'SUPER':<6} {'ADAPTER':<25} {'START TIME':<20}",
            "-" * 120
        ]
        for r in sorted(self.all_races, key=lambda x: (x.discipline, x.track, x.race_number)):
            superfecta = "Yes" if r.superfecta_offered else "No"
            # Display time in Eastern with ET suffix
            st = r.start_time.strftime("%Y-%m-%d %H:%M ET") if r.start_time else "Unknown"
            lines.append(f"{r.discipline:<5} {r.track[:24]:<25} {r.race_number:<4} {r.field_size:<6} {superfecta:<6} {r.adapter[:24]:<25} {st:<20}")
        lines.append("-" * 120)
        lines.append(f"Total races: {len(self.all_races)}")
        self.logger.info("\n".join(lines))

    def get_bet_now_races(self) -> List[RaceSummary]:
        """Get races meeting BET NOW criteria."""
        # 1. MTP <= 120 (Broadened for yield)
        # 2. 2nd Fav Odds >= 4.0
        # 3. Field size <= 11 (User Directive)
        # 4. Gap > 0 (User Directive)
        bet_now = [
            r for r in self.golden_zone_races
            if r.mtp is not None and -10 < r.mtp <= 120
            and r.second_fav_odds is not None and r.second_fav_odds >= 4.0
            and r.field_size <= 11
            and r.gap12 > 0
        ]
        # Sort by Superfecta desc, then MTP asc
        bet_now.sort(key=lambda r: (not r.superfecta_offered, r.mtp))
        return bet_now

    def get_you_might_like_races(self) -> List[RaceSummary]:
        """Get 'You Might Like' races with relaxed criteria."""
        # Criteria: Not in BET NOW, but -10 < MTP <= 240 (4h) and 2nd Fav Odds >= 3.0
        # and field size <= 11 and Gap > 0
        bet_now_keys = {(r.track, r.race_number) for r in self.get_bet_now_races()}
        yml = [
            r for r in self.golden_zone_races
            if r.mtp is not None and -10 < r.mtp <= 240
            and r.second_fav_odds is not None and r.second_fav_odds >= 3.0
            and r.field_size <= 11
            and r.gap12 > 0
            and (r.track, r.race_number) not in bet_now_keys
        ]
        # Sort by MTP asc
        yml.sort(key=lambda r: r.mtp)
        return yml[:5]  # Limit to top 5 recommendations

    async def print_bet_now_list(self):
        """Log filtered BET NOW list and recent audited goldmine results."""
        bet_now = self.get_bet_now_races()
        lines = [
            "=" * 140,
            "🎯 BET NOW - FAVORITE TO PLACE OPPORTUNITIES".center(140),
            "=" * 140,
            f"Updated: {datetime.now(EASTERN).strftime('%Y-%m-%d %H:%M:%S')} ET",
            "Criteria: -10 < MTP <= 120 minutes AND 2nd Favorite Odds >= 4.0",
            "-" * 140
        ]
        if not bet_now:
            lines.append("⏳ No races currently meet BET NOW criteria.")
            yml = self.get_you_might_like_races()
            if yml:
                lines.extend([
                    "=" * 160,
                    "🌟 YOU MIGHT LIKE - NEAR-MISS OPPORTUNITIES".center(160),
                    "=" * 160,
                    f"{'SUPER':<6} {'MTP':<5} {'DISC':<5} {'TRACK':<20} {'R#':<4} {'FIELD':<6} {'ODDS':<20} {'TOP 5':<20}",
                    "-" * 160
                ])
                for r in yml:
                    sup = "✅" if r.superfecta_offered else "❌"
                    fo = f"{r.favorite_odds:.2f}" if r.favorite_odds else "N/A"
                    so = f"{r.second_fav_odds:.2f}" if r.second_fav_odds else "N/A"
                    top5 = r.top_five_numbers or "N/A"
                    # Leading zero alignment (Memory Directive Fix)
                    m_str = f"{r.mtp:02d}" if 0 <= r.mtp < 10 else str(r.mtp)
                    lines.append(f"{sup:<6} {m_str:<5} {r.discipline:<5} {r.track[:19]:<20} {r.race_number:<4} {r.field_size:<6}  ~ {fo}, {so:<15} [{top5}]")
                lines.append("-" * 160)
            self.logger.info("\n".join(lines))
            return

        lines.extend([
            f"{'SUPER':<6} {'MTP':<5} {'DISC':<5} {'TRACK':<20} {'R#':<4} {'FIELD':<6} {'ODDS':<20} {'TOP 5':<20}",
            "-" * 160
        ])
        for r in bet_now:
            sup = "✅" if r.superfecta_offered else "❌"
            fo = f"{r.favorite_odds:.2f}" if r.favorite_odds else "N/A"
            so = f"{r.second_fav_odds:.2f}" if r.second_fav_odds else "N/A"
            top5 = r.top_five_numbers or "N/A"
            m_str = f"{r.mtp:02d}" if 0 <= r.mtp < 10 else str(r.mtp)
            lines.append(f"{sup:<6} {m_str:<5} {r.discipline:<5} {r.track[:19]:<20} {r.race_number:<4} {r.field_size:<6}  ~ {fo}, {so:<15} [{top5}]")
        lines.extend(["-" * 160, f"Total opportunities: {len(bet_now)}"])
        self.logger.info("\n".join(lines))

        # Include recent audited results to provide proof of system performance
        history = await self.tracker.db.get_recent_audited_goldmines(limit=10)
        if history:
            historical_report = generate_historical_goldmine_report(history)
            self.logger.info(historical_report)

    def save_to_json(self, filename: str = "race_data.json"):
        """Export to JSON."""
        bn = self.get_bet_now_races()
        yml = self.get_you_might_like_races()

        if not bn:
            self.logger.warning("🔭 Monitor found 0 BET NOW opportunities", total_checked=len(self.golden_zone_races))
            # Structured telemetry for monitoring
            structlog.get_logger("FortunaTelemetry").warning("empty_bet_now_list", golden_zone_count=len(self.golden_zone_races))
            # Create an indicator file for downstream monitoring (GPT5 Improvement)
            try:
                Path("monitor_empty.alert").write_text(datetime.now(EASTERN).isoformat())
            except Exception: pass
        else:
            # Clear alert if it exists
            try:
                alert_file = Path("monitor_empty.alert")
                if alert_file.exists(): alert_file.unlink()
            except Exception: pass

        data = {
            "generated_at": datetime.now(EASTERN).isoformat(),
            "target_dates": self.target_dates,
            "total_races": len(self.all_races),
            "bet_now_count": len(bn),
            "you_might_like_count": len(yml),
            "all_races": [r.to_dict() for r in self.all_races],
            "bet_now_races": [r.to_dict() for r in bn],
            "you_might_like_races": [r.to_dict() for r in yml],
        }
        try:
            # Ensure parent directory exists (GPT5 Improvement)
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error("failed_saving_race_data", path=filename, error=str(e))

        # Persistent history log
        self._append_to_history(bn + yml)

    def _append_to_history(self, races: List[RaceSummary]):
        """Append races to persistent history for future result matching."""
        if not races: return
        history_file = "prediction_history.jsonl"
        timestamp = datetime.now(EASTERN).isoformat()
        try:
            with open(history_file, 'a') as f:
                for r in races:
                    record = r.to_dict()
                    record["logged_at"] = timestamp
                    f.write(json.dumps(record) + "\n")
        except Exception as e:
            self.logger.error("History logging failed", error=str(e))

    async def run_once(self, loaded_races: Optional[List[Race]] = None, adapter_names: Optional[List[str]] = None):
        try:
            if loaded_races is not None:
                self.logger.info("Using loaded races", count=len(loaded_races))
                # Map to (Race, AdapterName) tuple expected by build_race_summaries
                raw = [(r, r.source) for r in loaded_races]
            else:
                await self.initialize_adapters(adapter_names=adapter_names)
                raw = await self.fetch_all_races()

            await self.build_race_summaries(raw, window_hours=12) # Use 12h window for monitor
            self.print_full_list()
            await self.print_bet_now_list()
            self.save_to_json()
        finally:
            for a in self.adapters: await a.shutdown()
            await GlobalResourceManager.cleanup()

    async def run_continuous(self):
        await self.initialize_adapters()
        raw = await self.fetch_all_races()
        await self.build_race_summaries(raw, window_hours=12)
        self.print_full_list()
        try:
            for _ in range(1000): # Iteration limit to prevent potential hangs
                for r in self.all_races: r.mtp = self._calculate_mtp(r.start_time)
                await self.print_bet_now_list()
                self.save_to_json()
                await asyncio.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            self.logger.info("Stopped by user")
        finally:
            for a in self.adapters: await a.shutdown()
            await GlobalResourceManager.cleanup()





# ----------------------------------------
# EXPANDED ADAPTERS
# ----------------------------------------
# python_service/adapters/oddschecker_adapter.py





class OddscheckerAdapter(BrowserHeadersMixin, DebugMixin, BaseAdapterV3):
    """Adapter for scraping horse racing odds from Oddschecker, migrated to BaseAdapterV3."""

    SOURCE_NAME = "Oddschecker"
    BASE_URL = "https://www.oddschecker.com"

    def __init__(self, config=None):
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        # Oddschecker is heavily protected by Cloudflare; try Playwright with high wait time
        return FetchStrategy(
            primary_engine=BrowserEngine.PLAYWRIGHT,
            enable_js=True,
            stealth_mode="camouflage",
            timeout=90,
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
        index_url = f"/horse-racing/{date}"
        index_response = await self.make_request("GET", index_url, headers=self._get_headers())
        if not index_response or not index_response.text:
            self.logger.warning("Failed to fetch Oddschecker index page", url=index_url)
            return None

        self._save_debug_html(index_response.text, f"oddschecker_index_{date}")

        parser = HTMLParser(index_response.text)
        # Find all links to individual race pages
        metadata = []

        try:
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
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
            next_race = None
            for r in races:
                if re.match(r"\d{1,2}:\d{2}", r["time_txt"]):
                    try:
                        rt = datetime.strptime(r["time_txt"], "%H:%M").replace(
                            year=target_date.year, month=target_date.month, day=target_date.day, tzinfo=site_tz
                        )
                        # Skip if in past (Today only)
                        if target_date == now_site.date() and rt < now_site - timedelta(minutes=5):
                            continue
                        next_race = r
                        break
                    except Exception: pass

            if next_race:
                metadata.append(next_race["url"])

        if not metadata:
            self.logger.warning("No metadata found", context="Oddschecker Index Parsing", url=index_url)
            return None

        async def fetch_single_html(url_path: str):
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
            race_date = datetime.strptime(raw_data["date"], "%Y-%m-%d").date()
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
                race = self._parse_race_page(parser, race_date)
                if race:
                    all_races.append(race)
            except (AttributeError, IndexError, ValueError):
                self.logger.warning(
                    "Error parsing a race from Oddschecker, skipping race.",
                    exc_info=True,
                )
                continue
        return all_races

    def _parse_race_page(self, parser: HTMLParser, race_date) -> Optional[Race]:
        track_name_node = parser.css_first("h1.meeting-name")
        if not track_name_node:
            return None
        track_name = track_name_node.text(strip=True)

        race_time_node = parser.css_first("span.race-time")
        if not race_time_node:
            return None
        race_time_str = race_time_node.text(strip=True)

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

        return Race(
            id=f"oc_{track_name.lower().replace(' ', '')}_{start_time.strftime('%Y%m%d')}_r{race_number}",
            venue=track_name,
            race_number=race_number,
            start_time=start_time,
            runners=runners,
            source=self.source_name,
        )

    def _parse_runner_row(self, row: Node) -> Optional[Runner]:
        try:
            name_node = row.css_first("span.selection-name")
            if not name_node:
                return None
            name = name_node.text(strip=True)

            odds_node = row.css_first("span.bet-button-odds-desktop, span.best-price")
            if not odds_node:
                return None
            odds_str = odds_node.text(strip=True)

            number_node = row.css_first("td.runner-number")
            number = 0
            if number_node:
                num_txt = "".join(filter(str.isdigit, number_node.text(strip=True)))
                if num_txt:
                    number = int(num_txt)

            if not name or not odds_str:
                return None

            win_odds = parse_odds_to_decimal(odds_str)
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
    BASE_URL = "https://www.timeform.com"

    def __init__(self, config=None):
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)
        self._semaphore = asyncio.Semaphore(5)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        # Timeform often blocks basic requests; try Playwright
        return FetchStrategy(
            primary_engine=BrowserEngine.PLAYWRIGHT,
            enable_js=True,
            stealth_mode="camouflage",
            timeout=60
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
        index_url = f"/horse-racing/racecards/{date}"
        index_response = await self.make_request("GET", index_url, headers=self._get_headers())
        if not index_response or not index_response.text:
            self.logger.warning("Failed to fetch Timeform index page", url=index_url)
            return None

        self._save_debug_snapshot(index_response.text, f"timeform_index_{date}")

        parser = HTMLParser(index_response.text)
        # Updated selector for race links
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
        except Exception:
            target_date = datetime.now(EASTERN).date()

        site_tz = ZoneInfo("Europe/London")
        now_site = datetime.now(site_tz)

        track_map = defaultdict(list)
        for selector in ["a[href*='/racecards/'][href*='/20']", ".rf__link", "a.rf-meeting-race__time"]:
            for a in parser.css(selector):
                href = a.attributes.get("href")
                if href and not href.endswith("/racecards"):
                    # URL: /horse-racing/racecards/venue/date/time/...
                    parts = href.split("/")
                    if len(parts) >= 6:
                        track = parts[3]
                        txt = node_text(a)
                        track_map[track].append({"url": href, "time_txt": txt})

        links = []
        for track, races in track_map.items():
            next_race = None
            for r in races:
                if re.match(r"\d{1,2}:\d{2}", r["time_txt"]):
                    try:
                        rt = datetime.strptime(r["time_txt"], "%H:%M").replace(
                            year=target_date.year, month=target_date.month, day=target_date.day, tzinfo=site_tz
                        )
                        # Skip if in past (Today only)
                        if target_date == now_site.date() and rt < now_site - timedelta(minutes=5):
                            continue
                        next_race = r
                        break
                    except Exception: pass

            if next_race:
                links.append(next_race["url"])

        if not links:
            self.logger.warning("No metadata found", context="Timeform Index Parsing", url=index_url)
            return None

        async def fetch_single_html(url_path: str):
            async with self._semaphore:
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
            race_date = datetime.strptime(raw_data["date"], "%Y-%m-%d").date()
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
                scripts = self._parse_all_jsons_from_scripts(parser, 'script[type="application/ld+json"]', context="Betfair Index")
                for data in scripts:
                    if data.get("@type") == "Event":
                        venue = normalize_venue_name(data.get("location", {}).get("name", ""))
                        if sd := data.get("startDate"):
                            # 2026-01-28T14:32:00
                            start_time = datetime.fromisoformat(sd.split('+')[0])
                        break

                if not venue:
                    # Fallback to title
                    title = parser.css_first("title")
                    if title:
                        # 14:32 DUNDALK | Races 28 January 2026 ...
                        match = re.search(r'(\d{1,2}:\d{2})\s+([^|]+)', title.text())
                        if match:
                            time_str = match.group(1)
                            venue = normalize_venue_name(match.group(2).strip())
                            start_time = datetime.combine(race_date, datetime.strptime(time_str, "%H:%M").time())

                if not venue or not start_time:
                    continue

                # Betting Forecast Parsing
                forecast_map = {}
                verdict_section = parser.css_first("section.rp-verdict")
                if verdict_section:
                    forecast_text = clean_text(verdict_section.text())
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
                    id=f"tf_{venue.lower().replace(' ', '')}_{start_time:%Y%m%d}_R{race_number}",
                    venue=venue,
                    race_number=race_number,
                    start_time=start_time,
                    runners=runners,
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
            name = clean_text(name_node.text())

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
                    num_text = clean_text(num_node.text()).strip("()")
                    num_match = re.search(r"\d+", num_text)
                    if num_match:
                        val = int(num_match.group())
                        if val <= 40: number = val

            win_odds = None
            if forecast_map:
                win_odds = parse_odds_to_decimal(forecast_map.get(name.lower()))

            # Try to find live odds button if available (old selector)
            if not win_odds:
                odds_tag = row.css_first("button.rp-bet-placer-btn__odds")
                if odds_tag:
                    win_odds = parse_odds_to_decimal(clean_text(odds_tag.text()))

            odds_data = {}
            if odds_val := create_odds_data(self.source_name, win_odds):
                odds_data[self.source_name] = odds_val

            return Runner(number=number, name=name, odds=odds_data)
        except (AttributeError, ValueError, TypeError):
            return None

# python_service/adapters/racingpost_adapter.py




class RacingPostAdapter(BrowserHeadersMixin, DebugMixin, BaseAdapterV3):
    """
    Adapter for scraping Racing Post racecards, migrated to BaseAdapterV3.
    """

    SOURCE_NAME = "RacingPost"
    BASE_URL = "https://www.racingpost.com"

    def __init__(self, config=None):
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        # RacingPost has strong anti-bot measures. Playwright is most reliable.
        return FetchStrategy(
            primary_engine=BrowserEngine.PLAYWRIGHT,
            enable_js=True,
            stealth_mode="camouflage",
            timeout=60,
            block_resources=False,
        )

    def _get_headers(self) -> dict:
        return self._get_browser_headers(host="www.racingpost.com")

    async def _fetch_data(self, date: str) -> Any:
        """
        Fetches the raw HTML content for all races on a given date, including international.
        """
        index_url = f"/racecards/{date}"
        intl_url = f"/racecards/international/{date}"

        index_response = await self.make_request("GET", index_url, headers=self._get_headers())
        intl_response = await self.make_request("GET", intl_url, headers=self._get_headers())

        race_card_urls = []
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
        except Exception:
            target_date = datetime.now(EASTERN).date()

        site_tz = ZoneInfo("Europe/London")
        now_site = datetime.now(site_tz)

        if index_response and index_response.text:
            self._save_debug_html(index_response.text, f"racingpost_index_{date}")
            index_parser = HTMLParser(index_response.text)

            # Group by meeting to pick "next" race (Memory Directive Fix)
            meetings = index_parser.css('.rp-raceCourse__panel') or index_parser.css('.RC-meetingItem') or index_parser.css('.rp-meetingItem')
            for meeting in meetings:
                for link in meeting.css('a[data-test-selector^="RC-meetingItem__link_race"], a.rp-raceCourse__panel__race__time, a.rp-meetingItem__race__time, a.RC-meetingItem__race__time'):
                    txt = node_text(link)
                    if re.match(r"\d{1,2}:\d{2}", txt):
                        try:
                            rt = datetime.strptime(txt, "%H:%M").replace(
                                year=target_date.year, month=target_date.month, day=target_date.day, tzinfo=site_tz
                            )
                            if rt < now_site - timedelta(minutes=5):
                                continue
                        except Exception: pass

                    race_card_urls.append(link.attributes["href"])
                    break

        elif index_response:
            self.logger.warning("Unexpected status", status=index_response.status, url=index_url)

        if intl_response and intl_response.text:
            self._save_debug_html(intl_response.text, f"racingpost_intl_index_{date}")
            intl_parser = HTMLParser(intl_response.text)

            meetings = intl_parser.css('.rp-raceCourse__panel') or intl_parser.css('.RC-meetingItem') or intl_parser.css('.rp-meetingItem')
            for meeting in meetings:
                for link in meeting.css('a[data-test-selector^="RC-meetingItem__link_race"], a.rp-raceCourse__panel__race__time, a.rp-meetingItem__race__time, a.RC-meetingItem__race__time'):
                    txt = node_text(link)
                    if re.match(r"\d{1,2}:\d{2}", txt):
                        try:
                            rt = datetime.strptime(txt, "%H:%M").replace(
                                year=target_date.year, month=target_date.month, day=target_date.day, tzinfo=site_tz
                            )
                            if rt < now_site - timedelta(minutes=5):
                                continue
                        except Exception: pass

                    race_card_urls.append(link.attributes["href"])
                    break
        elif intl_response:
            self.logger.warning("Unexpected status", status=intl_response.status, url=intl_url)

        if not race_card_urls:
            self.logger.warning("Failed to fetch RacingPost racecard links", date=date)
            return None

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

                venue_node = parser.css_first('a[data-test-selector="RC-course__name"]')
                if not venue_node:
                    continue
                venue_raw = venue_node.text(strip=True)
                venue = normalize_venue_name(venue_raw)

                race_time_node = parser.css_first('span[data-test-selector="RC-course__time"]')
                if not race_time_node:
                    continue
                race_time_str = race_time_node.text(strip=True)

                race_datetime_str = f"{date} {race_time_str}"
                start_time = datetime.strptime(race_datetime_str, "%Y-%m-%d %H:%M")

                runners = self._parse_runners(parser)

                if venue and runners:
                    race_number = self._get_race_number(parser, start_time)
                    race = Race(
                        id=f"rp_{venue.lower().replace(' ', '')}_{date}_{race_number}",
                        venue=venue,
                        race_number=race_number,
                        start_time=start_time,
                        runners=runners,
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
            if link.text(strip=True) == time_str_to_find:
                return i + 1
        return 1

    def _parse_runners(self, parser: HTMLParser) -> list[Runner]:
        """Parses all runners from a single race card page."""
        runners = []
        runner_nodes = parser.css('div[data-test-selector="RC-runnerCard"]')
        for node in runner_nodes:
            if runner := self._parse_runner(node):
                runners.append(runner)
        return runners

    def _parse_runner(self, node: Node) -> Optional[Runner]:
        try:
            number_node = node.css_first('span[data-test-selector="RC-runnerNumber"]')
            name_node = node.css_first('a[data-test-selector="RC-runnerName"]')
            odds_node = node.css_first('span[data-test-selector="RC-runnerPrice"]')

            if not all([number_node, name_node, odds_node]):
                return None

            number_str = clean_text(number_node.text())
            number = 0
            if number_str:
                num_txt = "".join(filter(str.isdigit, number_str))
                if num_txt:
                    val = int(num_txt)
                    if val <= 40: number = val
            name = clean_text(name_node.text())
            odds_str = clean_text(odds_node.text())
            scratched = "NR" in odds_str.upper() or not odds_str

            odds = {}
            if not scratched:
                win_odds = parse_odds_to_decimal(odds_str)
                if odds_data := create_odds_data(self.source_name, win_odds):
                    odds[self.source_name] = odds_data

            return Runner(number=number, name=name, odds=odds, scratched=scratched)
        except (ValueError, AttributeError):
            self.logger.warning("Could not parse RacingPost runner, skipping.", exc_info=True)
            return None


class RacingPostToteAdapter(BrowserHeadersMixin, DebugMixin, BaseAdapterV3):
    """
    Adapter for fetching Tote dividends and results from Racing Post.
    """
    ADAPTER_TYPE = "results"
    SOURCE_NAME = "RacingPostTote"
    BASE_URL = "https://www.racingpost.com"

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
        url = f"/results/{date}"
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

                    # Broaden regex to match various RP result link patterns (Memory Directive Fix)
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
        venue_node = parser.css_first('a[data-test-selector="RC-course__name"]')
        if not venue_node: return None
        venue = normalize_venue_name(venue_node.text(strip=True))

        time_node = parser.css_first('span[data-test-selector="RC-course__time"]')
        if not time_node: return None
        time_str = time_node.text(strip=True)

        try:
            start_time = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=EASTERN)
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
                        label = clean_text(label_node.text())
                        value = clean_text(val_node.text())
                        if label and value:
                            dividends[label] = value
                except Exception as e:
                    self.logger.debug("Failed parsing RP tote row", error=str(e))



        # Extract runners (finishers)
        runners = []
        for row in parser.css('div[data-test-selector="RC-resultRunner"]'):
            name_node = row.css_first('a[data-test-selector="RC-resultRunnerName"]')
            if not name_node: continue
            name = clean_text(name_node.text())
            pos_node = row.css_first('span.rp-resultRunner__position')
            pos = clean_text(pos_node.text()) if pos_node else "?"

            # Try to find saddle number
            number = 0
            num_node = row.css_first(".rp-resultRunner__saddleClothNo")
            if num_node:
                try: number = int(clean_text(num_node.text()))
                except Exception: pass

            runners.append(Runner(
                name=name,
                number=number,
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
            race_num_match = re.search(r'Race\s+(\d+)', parser.text())
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

async def run_discovery(
    target_dates: List[str],
    window_hours: Optional[int] = 8,
    loaded_races: Optional[List[Race]] = None,
    adapter_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    fetch_only: bool = False,
    live_dashboard: bool = False,
    track_odds: bool = False,
    region: Optional[str] = None
):
    logger = structlog.get_logger("run_discovery")
    logger.info("Running Discovery", dates=target_dates, window_hours=window_hours)

    try:
        now = datetime.now(EASTERN)
        cutoff = now + timedelta(hours=window_hours) if window_hours else None

        all_races_raw = []
        harvest_summary = {}

        if loaded_races is not None:
            logger.info("Using loaded races", count=len(loaded_races))
            all_races_raw = loaded_races
            adapters = []
            # Ensure harvest files exist even for loaded runs (Memory Directive Fix)
            try:
                if not os.path.exists("discovery_harvest.json"):
                    with open("discovery_harvest.json", "w") as f:
                        json.dump({}, f)
            except Exception: pass
        else:
            # Auto-discover discovery adapter classes
            adapter_classes = get_discovery_adapter_classes()

            if adapter_names:
                adapter_classes = [c for c in adapter_classes if c.__name__ in adapter_names or getattr(c, "SOURCE_NAME", "") in adapter_names]

            # Load historical performance scores to prioritize adapters
            db = FortunaDB()
            adapter_scores = await db.get_adapter_scores(days=30)

            # Prioritize adapters by score (descending)
            adapter_classes = sorted(
                adapter_classes,
                key=lambda c: adapter_scores.get(getattr(c, "SOURCE_NAME", c.__name__), 0),
                reverse=True
            )

            adapters = []
            for cls in adapter_classes:
                try:
                    adapters.append(cls(config={"region": region}))
                except Exception as e:
                    logger.error("Failed to initialize adapter", adapter=cls.__name__, error=str(e))

            try:
                async def fetch_one(a, date_str):
                    try:
                        races = await a.get_races(date_str)
                        return a.source_name, races
                    except Exception as e:
                        logger.error("Error fetching from adapter", adapter=a.source_name, date=date_str, error=str(e))
                        return a.source_name, []

                fetch_tasks = []
                for d in target_dates:
                    for a in adapters:
                        fetch_tasks.append(fetch_one(a, d))

                results = await asyncio.gather(*fetch_tasks)
                for adapter_name, r_list in results:
                    all_races_raw.extend(r_list)

                    # Track count and MaxOdds (Proxy for successful odds fetching)
                    m_odds = 0.0
                    for r in r_list:
                        for run in r.runners:
                            if run.win_odds and run.win_odds > m_odds:
                                m_odds = float(run.win_odds)

                    if adapter_name not in harvest_summary:
                        harvest_summary[adapter_name] = {"count": 0, "max_odds": 0.0}

                    harvest_summary[adapter_name]["count"] += len(r_list)
                    if m_odds > harvest_summary[adapter_name]["max_odds"]:
                        harvest_summary[adapter_name]["max_odds"] = m_odds

                logger.info("Fetched total races", count=len(all_races_raw))
            finally:
                # Save discovery harvest summary for GHA reporting and DB persistence
                try:
                    # Only create if it doesn't exist or we have data
                    if harvest_summary or not os.path.exists("discovery_harvest.json"):
                        with open("discovery_harvest.json", "w") as f:
                            json.dump(harvest_summary, f)

                    if harvest_summary:
                        await db.log_harvest(harvest_summary, region=region)
                except Exception: pass

                # Shutdown adapters
                for a in adapters:
                    try: await a.close()
                    except Exception: pass

        # Apply time window filter if requested to avoid overloading
        # Initial time window filtering removed to ensure all unique races are tracked for reporting

        if not all_races_raw:
            logger.error("No races fetched from any adapter. Discovery aborted.")
            if save_path:
                try:
                    with open(save_path, "w") as f:
                        json.dump([], f)
                    logger.info("Saved empty race list to file", path=save_path)
                except Exception as e:
                    logger.error("Failed to save empty race list", error=str(e))
            return
        
        # Deduplicate
        race_map = {}
        for race in all_races_raw:
            canonical_venue = get_canonical_venue(race.venue)
            # Use Canonical Venue + Race Number + Date + Discipline as stable key
            st = race.start_time
            if isinstance(st, str):
                try:
                    st = datetime.fromisoformat(st.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    pass

            date_str = st.strftime('%Y%m%d') if hasattr(st, 'strftime') else "Unknown"
            # Removing discipline from key to allow better merging across adapters
            key = f"{canonical_venue}|{race.race_number}|{date_str}"
            
            if key not in race_map:
                race_map[key] = race
            else:
                existing = race_map[key]
                # Merge runners/odds
                for nr in race.runners:
                    # Match by number OR name (if numbers are missing)
                    er = next((r for r in existing.runners if (r.number != 0 and r.number == nr.number) or (r.name.lower() == nr.name.lower())), None)
                    if er:
                        er.odds.update(nr.odds)
                        if not er.win_odds and nr.win_odds:
                            er.win_odds = nr.win_odds
                        if not er.number and nr.number:
                            er.number = nr.number
                    else:
                        existing.runners.append(nr)

                # Update source
                sources = set((existing.source or "").split(", "))
                sources.add(race.source or "Unknown")
                existing.source = ", ".join(sorted(list(filter(None, sources))))

        unique_races = list(race_map.values())
        logger.info("Unique races identified", count=len(unique_races))

        # GPT5 Improvement: Keep all races within window for analysis, not just one per track.
        # Window broadened to 18 hours to match grid cutoff (News Mode)
        timing_window_races = []
        now = datetime.now(EASTERN)
        for race in unique_races:
            st = race.start_time
            if isinstance(st, str):
                try:
                    st = datetime.fromisoformat(st.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    continue
            if st.tzinfo is None:
                st = st.replace(tzinfo=EASTERN)

            # Calculate Minutes to Post
            diff = st - now
            mtp = diff.total_seconds() / 60

            # Broaden window to 18 hours to ensure yield for "News"
            if -45 < mtp <= 1080: # 18 hours = 1080 mins
                timing_window_races.append(race)
                if mtp <= 45:
                    logger.info(f"  💰 Found Gold Candidate: {race.venue} R{race.race_number} ({mtp:.1f} MTP)")
                else:
                    logger.debug(f"  🔭 Found Upcoming Candidate: {race.venue} R{race.race_number} ({mtp:.1f} MTP)")

        golden_zone_races = timing_window_races
        if not golden_zone_races:
            logger.warning("🔭 No races found in the broadened window (-45m to 18h).")

        logger.info("Total unique races available for analysis", count=len(unique_races))

        # Save raw fetched/merged races if requested (Save EVERYTHING unique)
        if save_path:
            try:
                with open(save_path, "w") as f:
                    json.dump([r.model_dump(mode='json') for r in unique_races], f, indent=4)
                logger.info("Saved all unique races to file", path=save_path)
            except Exception as e:
                logger.error("Failed to save races", error=str(e))

        if fetch_only:
            logger.info("Fetch-only mode active. Skipping analysis and reporting.")
            return

        # Analyze ALL unique races to ensure Grid is populated with Top 5 info (News Mode)
        analyzer = SimplySuccessAnalyzer()
        result = analyzer.qualify_races(unique_races)
        qualified = result.get("races", [])

        # Generate Grid & Goldmine (Grid uses unique_races for the broader context)
        grid = generate_summary_grid(qualified, all_races=unique_races)
        logger.info("Summary Grid Generated")

        # Generate Field Matrix for all unique races
        field_matrix = generate_field_matrix(unique_races)
        logger.info("Field Matrix Generated")

        # Log Hot Tips & Fetch recent historical results for the report
        tracker = HotTipsTracker()
        await tracker.log_tips(qualified)

        historical_goldmines = await tracker.db.get_recent_audited_goldmines(limit=15)
        historical_report = generate_historical_goldmine_report(historical_goldmines)

        gm_report = generate_goldmine_report(qualified, all_races=unique_races)
        if historical_report:
            gm_report += "\n" + historical_report

        # NEW: Dashboard and Live Tracking
        goldmines = [r for r in qualified if get_field(r, 'metadata', {}).get('is_goldmine')]

        # Calculate today's stats for dashboard
        recent_tips = await tracker.db.get_recent_tips(limit=100)
        today_str = datetime.now(EASTERN).strftime("%Y-%m-%d")
        today_tips = [t for t in recent_tips if t.get("report_date", "").startswith(today_str)]

        cashed = sum(1 for t in today_tips if t.get("verdict") == "CASHED")
        total_tips = len(today_tips)
        profit = sum((t.get("net_profit") or 0.0) for t in today_tips)

        stats = {
            "tips": total_tips,
            "cashed": cashed,
            "profit": profit
        }

        # Generate friendly HTML report
        try:
            html_content = await generate_friendly_html_report(qualified, stats)
            html_path = Path("fortuna_report.html")
            html_path.write_text(html_content, encoding="utf-8")
            logger.info("Friendly HTML report generated", path=str(html_path))

            # Launch the report if running as a portable app (not in GHA)
            if not os.getenv("GITHUB_ACTIONS"):
                try:
                    # Use absolute path for reliable opening
                    abs_path = html_path.absolute()
                    if sys.platform == "win32":
                        os.startfile(abs_path)
                    else:
                        webbrowser.open(f"file://{abs_path}")
                except Exception as e:
                    logger.warning("Failed to automatically launch report", error=str(e))
        except Exception as e:
            logger.error("Failed to generate HTML report", error=str(e))

        if live_dashboard:
            try:
                from rich.live import Live
                from rich.console import Console
                # Check if our custom dashboard exists
                try:
                    from dashboard import FortunaDashboard
                    dash = FortunaDashboard()
                    dash.update(goldmines, stats)

                    # Start odds tracker if requested
                    if track_odds:
                        try:
                            from odds_tracker import LiveOddsTracker
                            adapter_classes = get_discovery_adapter_classes()
                            odds_tracker = LiveOddsTracker(goldmines, adapter_classes)
                            asyncio.create_task(odds_tracker.start_tracking())
                        except ImportError:
                            logger.warning("LiveOddsTracker not available")

                    await dash.run_live()
                except (ImportError, Exception) as e:
                    logger.warning(f"Rich dashboard component missing or failed: {e}")
                    # Fallback to simple rich display if possible
                    console = Console()
                    console.print("\n" + grid + "\n")
            except ImportError:
                logger.warning("Rich library not available, falling back to static display")
                print("\n" + grid + "\n")
        else:
            # Fallback to static print
            try:
                from dashboard import print_dashboard
                print_dashboard(goldmines, stats)
            except Exception as e:
                # Silently fallback to standard print if dashboard fails
                pass

            print("\n" + grid + "\n")
            if historical_report:
                print("\n" + historical_report + "\n")

        # Always save reports to files (GPT5 Improvement: Defensive guards)
        try:
            with open("summary_grid.txt", "w", encoding='utf-8') as f: f.write(grid)
            with open("field_matrix.txt", "w", encoding='utf-8') as f: f.write(field_matrix)
            with open("goldmine_report.txt", "w", encoding='utf-8') as f: f.write(gm_report)
        except Exception as e:
            logger.error("failed_saving_text_reports", error=str(e))

        # Save qualified races to JSON
        report_data = {
            "races": [r.model_dump(mode='json') for r in qualified],
            "analysis_metadata": result.get("criteria", {}),
            "timestamp": datetime.now(EASTERN).isoformat(),
        }
        try:
            with open("qualified_races.json", "w", encoding='utf-8') as f:
                json.dump(report_data, f, indent=4)
        except Exception as e:
            logger.error("failed_saving_qualified_races", error=str(e))

        # NEW: Write GHA Job Summary
        if 'GITHUB_STEP_SUMMARY' in os.environ:
            try:
                predictions_md = format_predictions_section(qualified)
                # We need a db instance for format_proof_section
                proof_md = await format_proof_section(tracker.db)
                harvest_md = build_harvest_table(harvest_summary, "🛰️ Discovery Harvest Performance")
                artifacts_md = format_artifact_links()
                write_job_summary(predictions_md, harvest_md, proof_md, artifacts_md)
                logger.info("GHA Job Summary written")
            except Exception as e:
                logger.error("Failed to write GHA summary", error=str(e))

    finally:
        await GlobalResourceManager.cleanup()
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
    time.sleep(1.5)

    # Create and start the webview window
    print("Launching Fortuna Desktop Window...")
    webview.create_window('Fortuna Intelligence Desktop', 'http://127.0.0.1:8013', width=1300, height=900)
    webview.start()

async def ensure_browsers():
    """Ensure browser dependencies are available for scraping."""
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
    except ImportError:
        structlog.get_logger().debug("Playwright not imported")

    if is_frozen():
        print("━" * 60)
        print("⚠️  PLAYWRIGHT NOT DETECTED IN MONOLITH")
        print("━" * 60)
        print("Playwright is required for some adapters but cannot be auto-installed in EXE mode.")
        print("\nStandard HTTP-based adapters will still function.")
        print("━" * 60)
        return False

    structlog.get_logger().info("Installing browser dependencies (Playwright Chromium)...")
    try:
        # Run installation in a separate process to avoid blocking the loop too much
        # We explicitly don't use 'pip install playwright' here if possible because it might conflict
        # but for local non-frozen runs it's a helpful fallback.
        subprocess.run([sys.executable, "-m", "pip", "install", "playwright==1.49.1"], check=True, capture_output=True, text=True)
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True, capture_output=True, text=True)
        structlog.get_logger().info("Browser dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        structlog.get_logger().error(
            "Failed to install browsers",
            error=str(e),
            returncode=e.returncode,
            stdout=e.stdout,
            stderr=e.stderr
        )
        return False
    except Exception as e:
        structlog.get_logger().error("Unexpected error installing browsers", error=str(e))
        return False

async def main_all_in_one():
    # Configure logging at the start of main
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO)
    )
    # Ensure DB path env is set if passed via argument or already in environment
    # Actually, we should probably add a --db-path arg here too for parity with analytics
    logger = structlog.get_logger("main")
    parser = argparse.ArgumentParser(description="Fortuna All-In-One")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--hours", type=int, default=8, help="Discovery time window in hours (default: 8)")
    parser.add_argument("--monitor", action="store_true", help="Run in monitor mode")
    parser.add_argument("--once", action="store_true", help="Run monitor once")
    parser.add_argument("--region", type=str, choices=["USA", "INT"], help="Filter by region (USA or INT)")
    parser.add_argument("--include", type=str, help="Comma-separated adapter names to include")
    parser.add_argument("--save", type=str, help="Save races to JSON file")
    parser.add_argument("--load", type=str, help="Load races from JSON file(s), comma-separated")
    parser.add_argument("--fetch-only", action="store_true", help="Only fetch and save data, skip analysis and reporting")
    parser.add_argument("--db-path", type=str, help="Path to tip history database")
    parser.add_argument("--clear-db", action="store_true", help="Clear all tips from the database and exit")
    parser.add_argument("--gui", action="store_true", help="Start the Fortuna Desktop GUI")
    parser.add_argument("--live-dashboard", action="store_true", help="Show live updating terminal dashboard")
    parser.add_argument("--track-odds", action="store_true", help="Monitor live odds and send notifications")
    args = parser.parse_args()

    if args.db_path:
        os.environ["FORTUNA_DB_PATH"] = args.db_path

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

    # Auto-select region if not specified
    if not args.region:
        args.region = get_optimal_region_at_time(datetime.now(EASTERN))
        structlog.get_logger().info("Auto-selected region", region=args.region)

    # Region-based adapter filtering
    if args.region:
        target_set = USA_DISCOVERY_ADAPTERS if args.region == "USA" else INT_DISCOVERY_ADAPTERS

        if adapter_filter:
            adapter_filter = [n for n in adapter_filter if n in target_set]
        else:
            adapter_filter = list(target_set)

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

    if args.date:
        target_dates = [args.date]
    else:
        now = datetime.now(EASTERN)
        future = now + timedelta(hours=args.hours)

        target_dates = [now.strftime("%Y-%m-%d")]
        if future.date() > now.date():
            target_dates.append(future.strftime("%Y-%m-%d"))

    if args.monitor:
        await ensure_browsers()
        monitor = FavoriteToPlaceMonitor(target_dates=target_dates)
        # Pass region config to monitor
        monitor.config["region"] = args.region
        if args.once: await monitor.run_once(loaded_races=loaded_races, adapter_names=adapter_filter)
        else: await monitor.run_continuous() # Continuous mode doesn't support load/filter yet for simplicity
    else:
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
            region=args.region # Pass region to run_discovery
        )

if __name__ == "__main__":
    if os.getenv("DEBUG_SNAPSHOTS"):
        os.makedirs("debug_snapshots", exist_ok=True)
    
    try:
        asyncio.run(main_all_in_one())
    except KeyboardInterrupt:
        pass
