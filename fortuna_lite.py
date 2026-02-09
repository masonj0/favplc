from __future__ import annotations
# adapter_anthology.py
# Aggregated monolithic discovery adapters for Fortuna
# This anthology serves as a high-reliability fallback for the Fortuna discovery system.

"""
Fortuna Adapter Anthology - Production-grade racing data aggregation.

This module provides a unified collection of adapters for fetching racecard data
from various racing websites. It serves as a high-reliability fallback system.
"""
import argparse
import asyncio
import functools
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
except ImportError:
    curl_requests = None

try:
    from scrapling import AsyncFetcher, Fetcher
    from scrapling.parser import Selector
    ASYNC_SESSIONS_AVAILABLE = True
except ImportError:
    ASYNC_SESSIONS_AVAILABLE = False
    Selector = None  # type: ignore

try:
    from scrapling.fetchers import AsyncDynamicSession, AsyncStealthySession
except ImportError:
    ASYNC_SESSIONS_AVAILABLE = False

try:
    from scrapling.core.custom_types import StealthMode
except ImportError:
    class StealthMode:  # type: ignore
        FAST = "fast"
        CAMOUFLAGE = "camouflage"

try:
    import winsound
except (ImportError, RuntimeError):
    winsound = None


def is_frozen() -> bool:
    """Check if running as a frozen executable (PyInstaller, cx_Freeze, etc.)"""
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')

try:
    from notifications import DesktopNotifier
    HAS_NOTIFICATIONS = True
except ImportError:
    HAS_NOTIFICATIONS = False

try:
    from browserforge.headers import HeaderGenerator
    from browserforge.fingerprints import FingerprintGenerator
    BROWSERFORGE_AVAILABLE = True
except ImportError:
    BROWSERFORGE_AVAILABLE = False


# --- TYPE VARIABLES ---
T = TypeVar("T")
RaceT = TypeVar("RaceT", bound="Race")

# --- CONSTANTS ---
EASTERN = ZoneInfo("America/New_York")

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
    return float(value)


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


def clean_text(text: Optional[str]) -> Optional[str]:
    """Strips leading/trailing whitespace and collapses internal whitespace."""
    if not text:
        return ""
    return " ".join(str(text).strip().split())


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
    if dt.tzinfo != EASTERN:
        return dt.astimezone(EASTERN)
    return dt


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
        "SPEEDWAY", "DOWNS", "PARK", "HARNESS", " STANDARDBRED"
    ]

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
            if val >= 2: # "2" -> 2/1 -> 3.0
                return float(val + 1)

    except: pass
    return None


def is_valid_odds(odds: Any) -> bool:
    if odds is None: return False
    try:
        odds_float = float(odds)
        return MIN_VALID_ODDS <= odds_float < MAX_VALID_ODDS
    except: return False


def create_odds_data(source_name: str, win_odds: Any, place_odds: Any = None) -> Optional[OddsData]:
    if not is_valid_odds(win_odds): return None
    return OddsData(win=float(win_odds), place=float(place_odds) if is_valid_odds(place_odds) else None, source=source_name)


def scrape_available_bets(html_content: str) -> List[str]:
    if not html_content: return []
    available_bets: List[str] = []
    html_lower = html_content.lower()
    for kw, bet_name in BET_TYPE_KEYWORDS.items():
        if kw in html_lower and bet_name not in available_bets:
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
        return valid_races, warnings


# --- CORE INFRASTRUCTURE ---
class GlobalResourceManager:
    """Manages shared resources like HTTP clients and semaphores."""
    _httpx_client: Optional[httpx.AsyncClient] = None
    _lock: asyncio.Lock = asyncio.Lock()
    _global_semaphore: Optional[asyncio.Semaphore] = None

    @classmethod
    async def get_httpx_client(cls, timeout: Optional[int] = None) -> httpx.AsyncClient:
        """
        Returns a shared httpx client.
        If timeout is provided and differs from current client, the client is recreated.
        """
        async with cls._lock:
            if cls._httpx_client is not None:
                if timeout is not None and cls._httpx_client.timeout.read != timeout:
                    await cls._httpx_client.aclose()
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
            cls._global_semaphore = asyncio.Semaphore(DEFAULT_CONCURRENT_REQUESTS * 2)
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
                raise ImportError("curl_cffi not available")

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

        # For other engines, we use AsyncFetcher from scrapling
        if engine == BrowserEngine.CAMOUFOX:
            async with AsyncStealthySession(headless=True) as s:
                resp = await s.fetch(url, method=method, **kwargs)
            return resp
        elif engine == BrowserEngine.PLAYWRIGHT:
            async with AsyncDynamicSession(headless=True) as s:
                resp = await s.fetch(url, method=method, **kwargs)
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
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    async def acquire(self) -> None:
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            self._tokens = min(self.requests_per_second, self._tokens + elapsed * self.requests_per_second)
            self._last_update = now
            if self._tokens < 1:
                wait_time = (1 - self._tokens) / self.requests_per_second
                await asyncio.sleep(wait_time)
                self._tokens = 0
            else: self._tokens -= 1


class AdapterMetrics:
    def __init__(self) -> None:
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency_ms = 0.0
        self.consecutive_failures = 0
    @property
    def success_rate(self) -> float:
        return self.successful_requests / self.total_requests if self.total_requests > 0 else 1.0
    async def record_success(self, latency_ms: float) -> None:
        self.total_requests += 1
        self.successful_requests += 1
        self.total_latency_ms += latency_ms
        self.consecutive_failures = 0
    async def record_failure(self, error: str) -> None:
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
        h = {**DEFAULT_BROWSER_HEADERS, "User-Agent": CHROME_USER_AGENT, "sec-ch-ua": CHROME_SEC_CH_UA, "sec-ch-ua-mobile": "0", "sec-ch-ua-platform": '"Windows"'}
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
        except: pass
    def _save_debug_html(self, content: str, filename: str, **kwargs) -> None:
        self._save_debug_snapshot(content, filename)


class RacePageFetcherMixin:
    async def _fetch_race_pages_concurrent(self, metadata: List[Dict[str, Any]], headers: Dict[str, str], semaphore_limit: int = 5, delay_range: tuple[float, float] = (0.5, 1.5)) -> List[Dict[str, Any]]:
        global_sem = GlobalResourceManager.get_global_semaphore()
        local_sem = asyncio.Semaphore(semaphore_limit)
        async def fetch_single(item):
            url = item.get("url")
            if not url: return None
            async with global_sem:
                async with local_sem:
                    await asyncio.sleep(delay_range[0] + random.random() * (delay_range[1] - delay_range[0]))
                    try:
                        if hasattr(self, 'logger'):
                            self.logger.debug("fetching_race_page", url=url)
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
                if not runner.scratched and (runner.win_odds is None or runner.win_odds <= 0):
                    runner.win_odds = DEFAULT_ODDS_FALLBACK
        valid, warnings = DataValidationPipeline.validate_parsed_races(races, adapter_name=self.source_name)
        return valid

    async def make_request(self, method: str, url: str, **kwargs: Any) -> Any:
        full_url = url if url.startswith("http") else f"{self.base_url}/{url.lstrip('/')}"
        self.logger.debug("Requesting", method=method, url=full_url)
        # Apply global concurrency limit (Memory Directive Fix)
        async with GlobalResourceManager.get_global_semaphore():
            try:
                resp = await self.smart_fetcher.fetch(full_url, method=method, **kwargs)
                status = getattr(resp, 'status', 'unknown')
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
# SkyRacingWorldAdapter
# ----------------------------------------
class EquibaseAdapter(BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
    SOURCE_NAME: ClassVar[str] = "Equibase"
    BASE_URL: ClassVar[str] = "https://www.equibase.com"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return FetchStrategy(
            primary_engine=BrowserEngine.CURL_CFFI,
            enable_js=True,
            stealth_mode="camouflage",
            block_resources=True,
            timeout=60
        )

    def _get_headers(self) -> Dict[str, str]:
        return self._get_browser_headers(host="www.equibase.com")

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        dt = datetime.strptime(date, "%Y-%m-%d")
        date_str = dt.strftime("%m%d%y")

        # Try different possible index URLs
        index_urls = [
            f"/static/entry/index.html",
            f"/entries/{date}",
        ]

        resp = None
        for url in index_urls:
            try:
                resp = await self.make_request("GET", url, headers=self._get_headers(), impersonate="chrome120")
                if resp and resp.text and len(resp.text) > 1000 and "Pardon Our Interruption" not in resp.text:
                    break
            except: continue

        if not resp or not resp.text:
            if resp: self.logger.warning("Unexpected status", status=resp.status, url=resp.url)
            return None

        self._save_debug_snapshot(resp.text, f"equibase_index_{date}")
        parser, links = HTMLParser(resp.text), []
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
        for p in pages:
            html_content = p.get("html")
            if not html_content: continue

            # If it's an index page for a track, we need to extract individual race links
            if "RaceCardIndex" in p.get("url", ""):
                sub_parser = HTMLParser(html_content)
                # Only take the first race link for this track (Memory Directive Fix)
                for a in sub_parser.css("a"):
                    sh = (a.attributes.get("href") or "").replace("\\", "/")
                    if "/static/entry/" in sh and date_str in sh and "RaceCardIndex" not in sh:
                        extra_links.append(sh)
                        break
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
            except: continue
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
        except: pass
        # Fallback to noon UTC for the given date if time parsing fails
        try:
            dt = datetime.strptime(ds, "%Y-%m-%d")
            return dt.replace(hour=12, minute=0, tzinfo=EASTERN)
        except:
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
        # Fallback to CURL_CFFI if browser engines are missing in frozen mode
        return FetchStrategy(
            primary_engine=BrowserEngine.CURL_CFFI,
            enable_js=True,
            stealth_mode="camouflage",
            block_resources=True,
            max_retries=3,
            timeout=60
        )

    async def make_request(self, method: str, url: str, **kwargs: Any) -> Any:
        # Force chrome120 for TwinSpires to bypass basic bot checks
        kwargs.setdefault("impersonate", "chrome120")
        return await super().make_request(method, url, **kwargs)

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        ard = []
        last_err = None

        # Respect region from config if provided
        target_region = self.config.get("region") # "USA", "INT", or None for both

        async def fetch_disc(disc, region="USA"):
            suffix = "" if region == "USA" else "?region=INT"
            url = f"{self.BASE_URL}/bet/todays-races/{disc}{suffix}"
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
            except: continue

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
            except: continue
        return rd

    def _find_with_selectors(self, el, selectors: List[str]) -> Optional[str]:
        for s in selectors:
            try:
                f = el.css_first(s)
                if f:
                    t = f.text.strip() if hasattr(f, 'text') else str(f).strip()
                    if t: return t
            except: continue
        return None

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data or "races" not in raw_data: return []
        rl, ds, parsed = raw_data["races"], raw_data.get("date", datetime.now(EASTERN).strftime("%Y-%m-%d")), []
        for rd in rl:
            try:
                r = self._parse_single_race(rd, ds)
                if r and r.runners: parsed.append(r)
            except: continue
        return parsed

    def _parse_single_race(self, rd: dict, ds: str) -> Optional[Race]:
        page = rd.get("selector")
        hc = rd.get("html", "")
        if not page:
            if not hc or Selector is None: return None
            page = Selector(hc)
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
                    da = e.attrib.get('datetime')
                    if da:
                        try:
                            dt = datetime.fromisoformat(da.replace('Z', '+00:00'))
                            # Only trust the date from HTML if it's within 1 day of what we expected
                            if abs((dt.date() - bd).days) <= 1:
                                return dt
                            else:
                                self.logger.debug("Suspicious date in HTML datetime attribute", html_dt=da, expected_date=bd)
                        except: pass
                    p = self._parse_time_string(e.text.strip() if hasattr(e, 'text') else str(e).strip(), bd)
                    if p: return p
            except: continue
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
            except: continue
        return None

    def _parse_runners(self, page) -> List[Runner]:
        runners = []
        relems = []
        for s in self.RUNNER_ROW_SELECTORS:
            try:
                el = page.css(s)
                if el: relems = el; break
            except: continue
        for i, e in enumerate(relems):
            try:
                r = self._parse_single_runner(e, i + 1)
                if r: runners.append(r)
            except: continue
        return runners

    def _parse_single_runner(self, e, dn: int) -> Optional[Runner]:
        es = str(e.html) if hasattr(e, 'html') else str(e)
        sc = any(s in es.lower() for s in ['scratched', 'scr', 'scratch'])
        num = None
        for s in ['[class*="program"]', '[class*="saddle"]', '[class*="post"]', '[class*="number"]', '[data-program-number]', 'td:first-child']:
            try:
                ne = e.css_first(s)
                if ne:
                    nt = ne.text.strip() if hasattr(ne, 'text') else str(ne)
                    dig = "".join(filter(str.isdigit, nt))
                    if dig:
                        val = int(dig)
                        if val <= 40:
                            num = val
                            break
            except: continue
        name = None
        for s in ['[class*="horse-name"]', '[class*="horseName"]', '[class*="runner-name"]', 'a[class*="name"]', '[data-horse-name]', 'td:nth-child(2)']:
            try:
                ne = e.css_first(s)
                if ne:
                    nt = ne.text.strip() if hasattr(ne, 'text') else None
                    if nt and len(nt) > 1: name = re.sub(r"\(.*\)", "", nt).strip(); break
            except: continue
        if not name: return None
        odds, wo = {}, None
        if not sc:
            for s in ['[class*="odds"]', '[class*="ml"]', '[class*="morning-line"]', '[data-odds]']:
                try:
                    oe = e.css_first(s)
                    if oe:
                        ot = oe.text.strip() if hasattr(oe, 'text') else None
                        if ot and ot.upper() not in ['SCR', 'SCRATCHED', '--', 'N/A']:
                            wo = parse_odds_to_decimal(ot)
                            if od := create_odds_data(self.source_name, wo): odds[self.source_name] = od; break
                except: continue

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


def _get_best_win_odds(runner: Runner, refresh: bool = False) -> Optional[Decimal]:
    """Gets the best win odds for a runner, filtering out invalid or placeholder values."""
    # Check if we have already calculated and cached a valid best odds in metadata
    if not refresh and "best_win_odds_decimal" in runner.metadata:
        return runner.metadata["best_win_odds_decimal"]

    if not runner.odds:
        # Fallback to win_odds if available
        if runner.win_odds and 0 < runner.win_odds < 999:
            val = Decimal(str(runner.win_odds))
            runner.metadata["best_win_odds_decimal"] = val
            return val
        return None

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

    res = min(valid_odds) if valid_odds else None
    if res is not None:
        runner.metadata["best_win_odds_decimal"] = res
    return res


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
        max_field_size: int = 14,
        min_favorite_odds: float = 0.01,
        min_second_favorite_odds: float = 0.01,
    ):
        self.max_field_size = max_field_size
        self.min_favorite_odds = Decimal(str(min_favorite_odds))
        self.min_second_favorite_odds = Decimal(str(min_second_favorite_odds))
        self.notifier = RaceNotifier()

    def is_race_qualified(self, race: Race) -> bool:
        """A race is qualified for a trifecta if it has at least 3 non-scratched runners."""
        if not race or not race.runners:
            return False

        # Apply global timing cutoff (30m ago)
        now = datetime.now(EASTERN)
        cutoff = now - timedelta(minutes=30)
        st = race.start_time
        if st.tzinfo is None:
            st = st.replace(tzinfo=EASTERN)
        if st < cutoff:
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
            return 0.0

        runners_with_odds.sort(key=lambda x: x[1])
        favorite_odds = runners_with_odds[0][1]
        second_favorite_odds = runners_with_odds[1][1]

        # --- Calculate Qualification Score (as inspired by the TypeScript Genesis) ---
        field_score = (self.max_field_size - len(active_runners)) / self.max_field_size

        # Normalize odds scores - cap influence of extremely high odds
        fav_odds_score = min(float(favorite_odds) / FAV_ODDS_NORMALIZATION, 1.0)
        sec_fav_odds_score = min(float(second_favorite_odds) / SEC_FAV_ODDS_NORMALIZATION, 1.0)

        # Weighted average
        odds_score = (fav_odds_score * FAV_ODDS_WEIGHT) + (sec_fav_odds_score * SEC_FAV_ODDS_WEIGHT)
        final_score = (field_score * FIELD_SIZE_SCORE_WEIGHT) + (odds_score * ODDS_SCORE_WEIGHT)

        # --- Apply hard filters before scoring ---
        # User requested to exclude every race with an odds-on favorite (< 2.0 decimal)
        if (
            len(active_runners) > self.max_field_size
            or favorite_odds < 2.0
            or favorite_odds < self.min_favorite_odds
            or second_favorite_odds < self.min_second_favorite_odds
        ):
            return 0.0

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
        cutoff = now - timedelta(minutes=30)

        for race in races:
            # 1. Timing Filter: Ignore races more than 30 minutes in the past
            st = race.start_time
            if st.tzinfo is None:
                st = st.replace(tzinfo=EASTERN)

            if st < cutoff:
                log.debug("Excluding past race", venue=race.venue, start_time=st)
                continue

            # 2. Chalk Filter: Exclude races with an odds-on favorite (< 2.0)
            # if best_odds is not None and best_odds < 2.0:
            #     log.debug("Excluding chalk race", venue=race.venue, favorite_odds=best_odds)
            #     continue

            # Goldmine Detection: 2nd favorite >= 4:1 (5.0 decimal)
            # A race cannot be a goldmine if field size is over 8
            is_goldmine = False
            active_runners = [r for r in race.runners if not r.scratched]
            gap12 = 0.0
            if active_runners:
                all_odds = []
                for runner in active_runners:
                    odds = _get_best_win_odds(runner)
                    if odds is not None:
                        all_odds.append(odds)
                if len(all_odds) >= 2:
                    all_odds.sort()
                    fav, sec = all_odds[0], all_odds[1]
                    gap12 = round(float(sec - fav), 2)
                    if len(active_runners) <= 8 and sec >= 5.0:
                        is_goldmine = True

                # Calculate Top 5 for all races
                # Collect valid odds once to avoid repetitive calculation/conversion
                valid_r_with_odds = []
                for r in active_runners:
                    wo = _get_best_win_odds(r)
                    if wo is not None:
                        valid_r_with_odds.append((r, wo))

                r_with_odds = sorted(valid_r_with_odds, key=lambda x: x[1])
                race.top_five_numbers = ", ".join([str(r[0].number or '?') for r in r_with_odds[:5]])

            # Best Bet Detection:
            # Goldmine = 2nd Fav >= 5.0, Field <= 8
            # You Might Like = 2nd Fav >= 4.0, Field <= 8
            is_best_bet = (len(active_runners) <= 8 and active_runners and len(all_odds) >= 2 and all_odds[1] >= 4.0)

            race.metadata['is_goldmine'] = is_goldmine
            race.metadata['is_best_bet'] = is_best_bet
            race.metadata['1Gap2'] = gap12
            if active_runners and len(all_odds) >= 2:
                race.metadata['predicted_2nd_fav_odds'] = all_odds[1]
            race.qualification_score = 100.0
            qualified.append(race)

        return {
            "criteria": {
                "mode": "simply_success",
                "timing_filter": "30m_past_cutoff",
                "chalk_filter": "disabled",
                "goldmine_threshold": 5.0
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
            "high_value": Path(__file__).parent.parent.parent / "assets" / "sounds" / "alert_premium.wav",
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
            self.notifier.send({
                "title": title,
                "message": message,
                "urgency": "high" if race.qualification_score >= 80 else "normal"
            })
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
    for race in races:
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

    # Include all goldmines (2nd fav >= 5.0)
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
        except:
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
            st_str = st.strftime('%H:%M')
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
                <td>{getattr(sel, 'win_odds', 0.0):.2f}</td>
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
    cutoff = now + timedelta(hours=4)

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
    for race in races:
        st = get_field(race, 'start_time')
        if isinstance(st, str):
            try: st = datetime.fromisoformat(st.replace('Z', '+00:00'))
            except: continue
        if st and st.tzinfo is None: st = st.replace(tzinfo=EASTERN)

        # Ceiling of 4 hours, ignore races more than 10 mins past
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


def is_frozen() -> bool:
    """Check if the script is running as a bundled PyInstaller executable."""
    return getattr(sys, 'frozen', False)


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
        self._initialized = False
        self.logger = structlog.get_logger(self.__class__.__name__)

    def _get_conn(self):
        if not self._conn:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    @asynccontextmanager
    async def get_connection(self):
        """Returns an async context manager for a database connection."""
        import aiosqlite
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
                        val = row[col]
                        if val and ("+00:00" in val or val.endswith("Z")):
                            try:
                                dt_utc = datetime.fromisoformat(val.replace("Z", "+00:00"))
                                dt_eastern = dt_utc.astimezone(EASTERN)
                                updates[col] = dt_eastern.isoformat()
                            except: pass
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

        now = datetime.now(EASTERN)
        report_date = now.isoformat()
        new_tips = []

        # Strict future cutoff to prevent leakage (Never log more than 20 mins ahead)
        future_limit = now + timedelta(minutes=20)

        for r in races:
            # Only store "Best Bets" (Goldmine, BET NOW, or You Might Like)
            # These are marked in metadata by the analyzer.
            if not r.metadata.get('is_best_bet') and not r.metadata.get('is_goldmine'):
                continue

            st = r.start_time
            if isinstance(st, str):
                try: st = datetime.fromisoformat(st.replace('Z', '+00:00'))
                except: continue
            if st.tzinfo is None: st = st.replace(tzinfo=EASTERN)

            # Reject races too far in the future
            if st > future_limit:
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
1. Second favorite odds >= 5.0 decimal
2. Races under 20 minutes to post (MTP)
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
        }


@functools.lru_cache(None)
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
            wo = _get_best_win_odds(r, refresh=True)
            if wo is not None and wo > 1.0:
                # Store the Decimal odds directly for sorting to avoid conversion
                r_with_odds.append((r, wo))

        if not r_with_odds:
            return []

        # Sort by odds (lowest first)
        sorted_r = sorted(r_with_odds, key=lambda x: x[1])
        return [x[0] for x in sorted_r[:limit]]

    def _calculate_mtp(self, start_time: datetime) -> Optional[int]:
        """Calculate minutes to post."""
        if not start_time: return None
        now = datetime.now(EASTERN)
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=EASTERN)
        delta = start_time - now
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
        top_five_str = "|".join([str(r.number) for r in top_runners if r.number is not None])

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

                if cutoff and (st < now - timedelta(minutes=30) or st > cutoff):
                    continue

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
            except: pass

        unique_summaries = list(race_map.values())

        # Filter: Only keep THE NEXT RACE per track within the GOLDEN ZONE (Memory Directive)
        # We keep the earliest upcoming race (or very recently started) for each venue,
        # but only if it falls within the -5 to 20 minute window.
        next_races_map = {}
        now = datetime.now(EASTERN)
        for summary in unique_summaries:
            st = summary.start_time
            if st.tzinfo is None: st = st.replace(tzinfo=EASTERN)

            # Calculate Minutes to Post
            diff = st - now
            mtp = diff.total_seconds() / 60

            v = get_canonical_venue(summary.track)
            # THE GOLDEN ZONE: -5 to 20 mins
            if -5 < mtp <= 20:
                if v not in next_races_map or st < next_races_map[v].start_time:
                    next_races_map[v] = summary

        self.all_races = list(next_races_map.values())

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
        # 1. MTP <= 20 (Inclusive to match Grid)
        # 2. 2nd Fav Odds >= 5.0
        # 3. Field size <= 8 (User Directive)
        bet_now = [
            r for r in self.all_races
            if r.mtp is not None and 0 < r.mtp <= 20
            and r.second_fav_odds is not None and r.second_fav_odds >= 5.0
            and r.field_size <= 8
        ]
        # Sort by Superfecta desc, then MTP asc
        bet_now.sort(key=lambda r: (not r.superfecta_offered, r.mtp))
        return bet_now

    def get_you_might_like_races(self) -> List[RaceSummary]:
        """Get 'You Might Like' races with relaxed criteria."""
        # Criteria: Not in BET NOW, but 0 < MTP <= 20 and 2nd Fav Odds >= 4.0
        # and field size <= 8
        bet_now_keys = {(r.track, r.race_number) for r in self.get_bet_now_races()}
        yml = [
            r for r in self.all_races
            if r.mtp is not None and 0 < r.mtp <= 20
            and r.second_fav_odds is not None and r.second_fav_odds >= 4.0
            and r.field_size <= 8
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
            "Criteria: MTP <= 20 minutes AND 2nd Favorite Odds >= 5.0",
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
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

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
            while True:
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

        if loaded_races is not None:
            logger.info("Using loaded races", count=len(loaded_races))
            all_races_raw = loaded_races
            adapters = []
            # Ensure harvest files exist even for loaded runs (Memory Directive Fix)
            try:
                if not os.path.exists("discovery_harvest.json"):
                    with open("discovery_harvest.json", "w") as f:
                        json.dump({}, f)
            except: pass
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
                harvest_summary = {}

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

                # Save discovery harvest summary for GHA reporting and DB persistence
                try:
                    with open("discovery_harvest.json", "w") as f:
                        json.dump(harvest_summary, f)

                    db = FortunaDB()
                    await db.log_harvest(harvest_summary, region=region)
                except: pass

            finally:
                # Shutdown adapters
                for a in adapters:
                    try: await a.close()
                    except: pass

        # Apply time window filter if requested to avoid overloading
        if cutoff:
            original_count = len(all_races_raw)
            filtered_races = []
            for r in all_races_raw:
                st = r.start_time
                if isinstance(st, str):
                    try:
                        st = datetime.fromisoformat(st.replace('Z', '+00:00'))
                    except (ValueError, TypeError):
                        continue

                if st.tzinfo is None:
                    st = st.replace(tzinfo=EASTERN)

                if now <= st <= cutoff:
                    filtered_races.append(r)

            all_races_raw = filtered_races
            logger.info(
                "Filtered races by time window",
                window_hours=window_hours,
                before=original_count,
                after=len(all_races_raw)
            )

        if not all_races_raw:
            logger.error("No races fetched from any adapter. Discovery aborted.")
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

        # Filter: Only keep THE NEXT RACE per track within the GOLDEN ZONE (Memory Directive)
        # We keep the earliest upcoming race (or very recently started) for each venue,
        # but only if it falls within the -5 to 20 minute window.
        next_races_map = {}
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

            v = get_canonical_venue(race.venue)
            # THE GOLDEN ZONE: -5 to 20 mins
            if -5 < mtp <= 20:
                if v not in next_races_map or st < next_races_map[v].start_time:
                    next_races_map[v] = race
                    logger.info(f"  💰 Found Gold Candidate: {race.venue} R{race.race_number} ({mtp:.1f} MTP)")

        unique_races = list(next_races_map.values())
        if not unique_races:
            logger.warning("🔭 No 'Immediate Gold' races found (0-20 mins).")
            # We continue instead of returning to allow the rest of the discovery process (saving reports, etc)
            # but no tips will be logged or processed.

        logger.info("Filtered to Next Race per track in Golden Zone", count=len(unique_races))

        # Save raw fetched/merged races if requested
        if save_path:
            try:
                with open(save_path, "w") as f:
                    json.dump([r.model_dump(mode='json') for r in unique_races], f, indent=4)
                logger.info("Saved races to file", path=save_path)
            except Exception as e:
                logger.error("Failed to save races", error=str(e))

        if fetch_only:
            logger.info("Fetch-only mode active. Skipping analysis and reporting.")
            return

        # Analyze
        analyzer = SimplySuccessAnalyzer()
        result = analyzer.qualify_races(unique_races)
        qualified = result.get("races", [])

        # Generate Grid & Goldmine
        grid = generate_summary_grid(qualified, all_races=unique_races)
        logger.info("Summary Grid Generated")

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

        # Always save reports to files
        with open("summary_grid.txt", "w") as f: f.write(grid)
        with open("goldmine_report.txt", "w") as f: f.write(gm_report)

        # Save qualified races to JSON
        report_data = {
            "races": [r.model_dump(mode='json') for r in qualified],
            "analysis_metadata": result.get("criteria", {}),
            "timestamp": datetime.now(EASTERN).isoformat(),
        }
        with open("qualified_races.json", "w") as f:
            json.dump(report_data, f, indent=4)

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
        async with db.get_connection() as conn:  # ✅ Fixed Council's syntax (removed await)
            try:
                async with conn.execute(
                    "SELECT venue, race_number, selection_name, win_odds, discovered_at "
                    "FROM tips ORDER BY id DESC LIMIT 50"
                ) as cursor:
                    tips = await cursor.fetchall()
            except Exception as e:
                print(f"DB query failed: {e}")
                tips = []

        tips_html = "".join([
            f"<tr><td>{t[4]}</td><td>{t[0]}</td><td>R{t[1]}</td><td>{t[2]}</td><td>{t[3]}</td></tr>"
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
            except Exception:
                pass
    except ImportError:
        pass

    if is_frozen():
        print("━" * 60)
        print("⚠️  PLAYWRIGHT NOT DETECTED IN MONOLITH")
        print("━" * 60)
        print("Playwright is required for some adapters but cannot be auto-installed in EXE mode.")
        print("\nStandard HTTP-based adapters will still function.")
        print("━" * 60)
        return False

    print("Installing browser dependencies (Playwright Chromium)...")
    try:
        # Run installation in a separate process to avoid blocking the loop too much
        subprocess.run([sys.executable, "-m", "pip", "install", "playwright"], check=True)
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
        print("Browser dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install browsers: {e}")
        return False

async def main_all_in_one():
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
    parser.add_argument("--clear-db", action="store_true", help="Clear all tips from the database and exit")
    parser.add_argument("--gui", action="store_true", help="Start the Fortuna Desktop GUI")
    parser.add_argument("--live-dashboard", action="store_true", help="Show live updating terminal dashboard")
    parser.add_argument("--track-odds", action="store_true", help="Monitor live odds and send notifications")
    args = parser.parse_args()

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
        usa_adapters = {"Equibase", "TwinSpires"}
        int_adapters = {}

        target_set = usa_adapters if args.region == "USA" else int_adapters

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
            try:
                with open(path.strip(), "r") as f:
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
