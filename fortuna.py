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
import hashlib
import html
import json
import logging
import random
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
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
    Selector = None  # type: ignore

try:
    from scrapling.core.custom_types import StealthMode
except ImportError:
    class StealthMode:  # type: ignore
        FAST = "fast"
        CAMOUFLAGE = "camouflage"


# --- TYPE VARIABLES ---
T = TypeVar("T")
RaceT = TypeVar("RaceT", bound="Race")

# --- CONSTANTS ---
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
    last_updated: datetime = Field(default_factory=datetime.now)


class Runner(FortunaBaseModel):
    id: Optional[str] = None
    name: str
    number: Optional[int] = Field(None, alias="saddleClothNumber")
    scratched: bool = False
    odds: Dict[str, OddsData] = Field(default_factory=dict)
    win_odds: Optional[float] = Field(None, alias="winOdds")

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
    source: str
    discipline: Optional[str] = None
    distance: Optional[str] = None
    field_size: Optional[int] = None
    available_bets: List[str] = Field(default_factory=list, alias="availableBets")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    qualification_score: Optional[float] = None
    is_error_placeholder: bool = False
    error_message: Optional[str] = None

# --- UTILITIES ---
def clean_text(text: Optional[str]) -> Optional[str]:
    if not text: return None
    return " ".join(text.strip().split())


def normalize_venue_name(name: Optional[str]) -> Optional[str]:
    if not name: return name
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name)
    name = re.sub(r"\s+(IRE|USA|UK|FR|AUS|NZ|GB)$", "", name, flags=re.I)
    cleaned = clean_text(name)
    return cleaned.title() if cleaned else None


def parse_odds_to_decimal(odds_str: Any) -> Optional[float]:
    """
    Parses various odds formats (fractional, decimal) into a float decimal.
    Uses 'deep dark sorcery' to extract odds from noisy strings.
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
    Deep dark sorcery for extracting odds from noisy HTML or text.
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
    disc_suffix = ""
    if discipline:
        dl = discipline.lower()
        if "harness" in dl: disc_suffix = "_h"
        elif "greyhound" in dl: disc_suffix = "_g"
        elif "quarter" in dl: disc_suffix = "_q"
    return f"{prefix}_{venue_slug}_{date_str}_R{race_number}{disc_suffix}"


# --- VALIDATORS ---
class RaceValidator(BaseModel):
    venue: str = Field(..., min_length=1)
    race_number: int = Field(..., ge=1, le=100)
    start_time: datetime
    runners: List[Any] = Field(..., min_length=2)


class DataValidationPipeline:
    @staticmethod
    def validate_raw_response(adapter_name: str, raw_data: Any) -> tuple[bool, str]:
        if raw_data is None: return False, "Null response"
        return True, "OK"
    @staticmethod
    def validate_parsed_races(races: List[Race]) -> tuple[List[Race], List[str]]:
        valid_races: List[Race] = []
        warnings: List[str] = []
        for i, race in enumerate(races):
            try:
                data = race.model_dump() if hasattr(race, "model_dump") else race.dict()
                RaceValidator(**data)
                valid_races.append(race)
            except Exception as e:
                warnings.append(f"Race {i} validation failed: {str(e)}")
        return valid_races, warnings


# --- CORE INFRASTRUCTURE ---
class BrowserEngine(Enum):
    CAMOUFOX = "camoufox"
    PLAYWRIGHT = "playwright"
    CURL_CFFI = "curl_cffi"
    PLAYWRIGHT_LEGACY = "playwright_legacy"
    HTTPX = "httpx"


@dataclass
class FetchStrategy:
    primary_engine: BrowserEngine = BrowserEngine.PLAYWRIGHT
    enable_js: bool = True
    stealth_mode: str = "fast"
    block_resources: bool = True
    max_retries: int = 3
    timeout: int = DEFAULT_REQUEST_TIMEOUT
    page_load_strategy: str = "domcontentloaded"
    wait_for_selector: Optional[str] = None


class SmartFetcher:
    BOT_DETECTION_KEYWORDS: ClassVar[List[str]] = ["datadome", "perimeterx", "access denied", "captcha", "cloudflare", "please verify"]
    def __init__(self, strategy: Optional[FetchStrategy] = None):
        self.strategy = strategy or FetchStrategy()
        self.logger = structlog.get_logger(self.__class__.__name__)
        self._httpx_client: Optional[httpx.AsyncClient] = None
        self._curl_session: Optional[Any] = None
        self._lock = asyncio.Lock()
        self._engine_health = {
            BrowserEngine.CAMOUFOX: 0.9,
            BrowserEngine.CURL_CFFI: 0.8,
            BrowserEngine.PLAYWRIGHT: 0.7,
            BrowserEngine.HTTPX: 0.5
        }
        self.last_engine: str = "unknown"

    async def _get_httpx_client(self) -> httpx.AsyncClient:
        if self._httpx_client is None:
            async with self._lock:
                if self._httpx_client is None:
                    self._httpx_client = httpx.AsyncClient(
                        follow_redirects=True,
                        timeout=httpx.Timeout(self.strategy.timeout),
                        headers={**DEFAULT_BROWSER_HEADERS, "User-Agent": CHROME_USER_AGENT},
                    )
        return self._httpx_client

    async def fetch(self, url: str, **kwargs: Any) -> Any:
        method = kwargs.pop("method", "GET").upper()
        kwargs.pop("url", None)
        engines = sorted(self._engine_health.keys(), key=lambda e: self._engine_health[e], reverse=True)
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
        raise last_error or FetchError("All fetch engines failed")

    
    async def _fetch_with_engine(self, engine: BrowserEngine, url: str, method: str, **kwargs: Any) -> Any:
        if engine == BrowserEngine.HTTPX:
            client = await self._get_httpx_client()
            resp = await client.request(method, url, **kwargs)
            resp.status = resp.status_code
            return resp
        
        if engine == BrowserEngine.CURL_CFFI:
            if not curl_requests:
                raise ImportError("curl_cffi not available")
            
            self.logger.debug(f"Using curl_cffi for {url}")
            timeout = kwargs.get("timeout", self.strategy.timeout)
            headers = kwargs.get("headers", {**DEFAULT_BROWSER_HEADERS, "User-Agent": CHROME_USER_AGENT})
            impersonate = kwargs.get("impersonate", "chrome110")
            
            # Remove keys that curl_requests.AsyncSession.request doesn't like
            clean_kwargs = {k: v for k, v in kwargs.items() if k not in ["timeout", "headers", "impersonate", "network_idle", "wait_selector", "wait_until"]}
            
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
        # But we need to use it correctly.
        fetcher = AsyncFetcher()
        # In scrapling 0.3.x, AsyncFetcher has get(), post() etc.
        # If we want JS/Stealth, we should use AsyncStealthySession or AsyncDynamicSession
        
        if engine == BrowserEngine.CAMOUFOX:
            async with AsyncStealthySession(headless=True) as s:
                return await s.fetch(url, method=method, **kwargs)
        elif engine == BrowserEngine.PLAYWRIGHT:
            async with AsyncDynamicSession(headless=True) as s:
                return await s.fetch(url, method=method, **kwargs)
        else:
            # Fallback to simple fetcher
            if method.upper() == "GET":
                return await fetcher.get(url, **kwargs)
            else:
                return await fetcher.post(url, **kwargs)


    async def close(self) -> None:
        if self._httpx_client:
            await self._httpx_client.aclose()
            self._httpx_client = None


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
    def snapshot(self) -> Dict[str, Any]:
        return {"total_requests": self.total_requests, "success_rate": self.success_rate}


# --- MIXINS ---
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
            f = d / f"{context}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(f, "w", encoding="utf-8") as out:
                if url: out.write(f"<!-- URL: {url} -->\n")
                out.write(content)
        except: pass
    def _save_debug_html(self, content: str, filename: str, **kwargs) -> None:
        self._save_debug_snapshot(content, filename)


class RacePageFetcherMixin:
    async def _fetch_race_pages_concurrent(self, metadata: List[Dict[str, Any]], headers: Dict[str, str], semaphore_limit: int = 5, delay_range: tuple[float, float] = (0.5, 1.5)) -> List[Dict[str, Any]]:
        sem = asyncio.Semaphore(semaphore_limit)
        async def fetch_single(item):
            url = item.get("url")
            if not url: return None
            async with sem:
                await asyncio.sleep(delay_range[0] + random.random() * (delay_range[1] - delay_range[0]))
                try:
                    resp = await self.make_request("GET", url, headers=headers)
                    if resp and hasattr(resp, "text") and resp.text: return {**item, "html": resp.text}
                except: pass
                return None
        tasks = [fetch_single(m) for m in metadata]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception) and r is not None]


# --- BASE ADAPTER ---
class BaseAdapterV3(ABC):
    def __init__(self, source_name: str, base_url: str, rate_limit: float = 10.0, **kwargs: Any) -> None:
        self.source_name = source_name
        self.base_url = base_url.rstrip("/")
        self.logger = structlog.get_logger(adapter_name=self.source_name)
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimiter(requests_per_second=rate_limit)
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
            for runner in r.runners:
                if not runner.scratched and (runner.win_odds is None or runner.win_odds <= 0):
                    runner.win_odds = DEFAULT_ODDS_FALLBACK
        valid, warnings = DataValidationPipeline.validate_parsed_races(races)
        return valid

    async def make_request(self, method: str, url: str, **kwargs: Any) -> Any:
        full_url = url if url.startswith("http") else f"{self.base_url}/{url.lstrip('/')}"
        return await self.smart_fetcher.fetch(full_url, method=method, **kwargs)

    async def close(self) -> None: await self.smart_fetcher.close()
    async def shutdown(self) -> None: await self.close()

# ============================================================================
# ADAPTER IMPLEMENTATIONS
# ============================================================================

# ----------------------------------------
# AtTheRacesAdapter
# ----------------------------------------
class AtTheRacesAdapter(BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
    SOURCE_NAME: ClassVar[str] = "AtTheRaces"
    BASE_URL: ClassVar[str] = "https://www.attheraces.com"

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
        resp = await self.make_request("GET", index_url, headers=self._get_headers())
        if not resp or not resp.text: raise AdapterHttpError(self.source_name, 500, index_url)
        self._save_debug_snapshot(resp.text, f"atr_index_{date}")
        parser = HTMLParser(resp.text)
        metadata = self._extract_race_metadata(parser)
        if not metadata: return None
        pages = await self._fetch_race_pages_concurrent(metadata, self._get_headers(), semaphore_limit=5)
        return {"pages": pages, "date": date}

    def _extract_race_metadata(self, parser: HTMLParser) -> List[Dict[str, Any]]:
        meta: List[Dict[str, Any]] = []
        track_map = defaultdict(list)
        for link in parser.css('a[href*="/racecard/"]'):
            url = link.attributes.get("href")
            if not url or not (re.search(r"/\d{4}$", url) or re.search(r"/\d{1,2}$", url)): continue
            parts = url.split("/")
            if len(parts) >= 3: track_map[parts[2]].append(url)
        for track, urls in track_map.items():
            for i, url in enumerate(sorted(set(urls))):
                meta.append({"url": url, "race_number": i + 1, "venue_raw": track})
        if not meta:
            for meeting in (parser.css(".meeting-summary") or parser.css(".p-meetings__item")):
                for i, link in enumerate(meeting.css('a[href*="/racecard/"]')):
                    if url := link.attributes.get("href"): meta.append({"url": url, "race_number": i + 1})
        return meta

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data or not raw_data.get("pages"): return []
        try: race_date = datetime.strptime(raw_data["date"], "%Y-%m-%d").date()
        except: return []
        races: List[Race] = []
        for item in raw_data["pages"]:
            html_content = item.get("html")
            if not html_content: continue
            try:
                race = self._parse_single_race(html_content, item.get("url", ""), race_date, item.get("race_number"))
                if race: races.append(race)
            except: pass
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
        except: return None
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
                    except: pass
        runners: List[Runner] = []
        for selector in self.SELECTORS["runners"]:
            nodes = parser.css(selector)
            if nodes:
                for node in nodes:
                    runner = self._parse_runner(node, odds_map)
                    if runner: runners.append(runner)
                break
        return runners

    def _parse_runner(self, row: Node, odds_map: Dict[str, float]) -> Optional[Runner]:
        try:
            name_node = row.css_first("h3") or row.css_first("a.horse__link") or row.css_first('a[href*="/form/horse/"]')
            if not name_node: return None
            name = clean_text(name_node.text())
            if not name: return None
            num_node = row.css_first(".horse-in-racecard__saddle-cloth-number") or row.css_first(".odds-grid-horse__no") or row.css_first("span")
            number = 0
            if num_node:
                ns = clean_text(num_node.text())
                if ns:
                    digits = "".join(filter(str.isdigit, ns))
                    if digits: number = int(digits)
            win_odds = None
            if horse_link := row.css_first('a[href*="/form/horse/"]'):
                if m := re.search(r"/(\d+)(\?|$)", horse_link.attributes.get("href", "")):
                    win_odds = odds_map.get(m.group(1))
            if win_odds is None:
                if odds_node := row.css_first(".horse-in-racecard__odds"):
                    win_odds = parse_odds_to_decimal(clean_text(odds_node.text()))

            # Deep dark sorcery fallback
            if win_odds is None:
                win_odds = SmartOddsExtractor.extract_from_node(row)

            odds: Dict[str, OddsData] = {}
            if od := create_odds_data(self.source_name, win_odds): odds[self.source_name] = od
            return Runner(number=number, name=name, odds=odds, win_odds=win_odds)
        except: return None

# ----------------------------------------
# AtTheRacesGreyhoundAdapter
# ----------------------------------------
class AtTheRacesGreyhoundAdapter(BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
    SOURCE_NAME: ClassVar[str] = "AtTheRacesGreyhound"
    BASE_URL: ClassVar[str] = "https://greyhounds.attheraces.com"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return FetchStrategy(primary_engine=BrowserEngine.PLAYWRIGHT, enable_js=True, stealth_mode="fast", timeout=45)

    def _get_headers(self) -> Dict[str, str]:
        return self._get_browser_headers(host="greyhounds.attheraces.com", referer="https://greyhounds.attheraces.com/racecards")

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        index_url = f"/racecards/{date}" if date else "/racecards"
        resp = await self.make_request("GET", index_url, headers=self._get_headers())
        if not resp: return None
        self._save_debug_snapshot(resp.text, f"atr_grey_index_{date}")
        parser = HTMLParser(resp.text)
        metadata = self._extract_race_metadata(parser)
        if not metadata:
            links = []
            for script in parser.css('script[type="application/ld+json"]'):
                try:
                    d = json.loads(script.text())
                    items = d.get("@graph", [d]) if isinstance(d, dict) else []
                    for item in items:
                        if item.get("@type") == "SportsEvent":
                            loc = item.get("location")
                            if isinstance(loc, list):
                                for l in loc:
                                    if u := l.get("url"): links.append(u)
                            elif isinstance(loc, dict):
                                if u := loc.get("url"): links.append(u)
                except: continue
            metadata = [{"url": l, "race_number": 0} for l in set(links)]
        if not metadata: return None
        pages = await self._fetch_race_pages_concurrent(metadata, self._get_headers(), semaphore_limit=5)
        return {"pages": pages, "date": date}

    def _extract_race_metadata(self, parser: HTMLParser) -> List[Dict[str, Any]]:
        meta: List[Dict[str, Any]] = []
        pc = parser.css_first("page-content")
        if not pc: return []
        items_raw = pc.attributes.get(":items") or pc.attributes.get(":modules")
        if not items_raw: return []
        try:
            modules = json.loads(html.unescape(items_raw))
            for module in modules:
                for meeting in module.get("data", {}).get("items", []):
                    for i, race in enumerate(meeting.get("items", [])):
                        if race.get("type") == "racecard":
                            r_num = race.get("raceNumber") or race.get("number") or (i + 1)
                            if u := race.get("cta", {}).get("href"):
                                meta.append({"url": u, "race_number": r_num})
        except: pass
        return meta

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data or not raw_data.get("pages"): return []
        try: race_date = datetime.strptime(raw_data.get("date", ""), "%Y-%m-%d").date()
        except: race_date = datetime.now().date()
        races: List[Race] = []
        for item in raw_data["pages"]:
            if not item or not item.get("html"): continue
            try:
                race = self._parse_single_race(item["html"], item.get("url", ""), race_date, item.get("race_number"))
                if race: races.append(race)
            except: pass
        return races

    def _parse_single_race(self, html_content: str, url_path: str, race_date: date, race_number: Optional[int]) -> Optional[Race]:
        parser = HTMLParser(html_content)
        pc = parser.css_first("page-content")
        if not pc: return None
        items_raw = pc.attributes.get(":items") or pc.attributes.get(":modules")
        if not items_raw: return None
        try: modules = json.loads(html.unescape(items_raw))
        except: return None
        venue, race_time_str, distance, runners, odds_map = "", "", "", [], {}
        for module in modules:
            m_type, m_data = module.get("type"), module.get("data", {})
            if m_type == "RacecardHero":
                venue = normalize_venue_name(m_data.get("track", "")) or ""
                race_time_str = m_data.get("time", "")
                distance = m_data.get("distance", "")
                if not race_number: race_number = m_data.get("raceNumber") or m_data.get("number")
            if m_type == "OddsGrid":
                odds_grid = m_data.get("oddsGrid", {})
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
                            p_val = float(price)
                            if is_valid_odds(p_val): odds_map[str(g_id)] = p_val
                for t in odds_grid.get("traps", []):
                    trap_num = t.get("trap", 0)
                    name = clean_text(t.get("name", "")) or ""
                    g_id_match = re.search(r"/greyhound/(\d+)", t.get("href", ""))
                    g_id = g_id_match.group(1) if g_id_match else None
                    win_odds = odds_map.get(str(g_id)) if g_id else None

                    # Deep dark sorcery fallback
                    if win_odds is None:
                        win_odds = SmartOddsExtractor.extract_from_text(str(t))

                    odds_data = {}
                    if ov := create_odds_data(self.source_name, win_odds): odds_data[self.source_name] = ov
                    runners.append(Runner(number=trap_num or 0, name=name, odds=odds_data, win_odds=win_odds))
        if not venue or not runners:
            url_parts = url_path.split("/")
            if len(url_parts) >= 5:
                venue = normalize_venue_name(url_parts[3]) or ""
                race_time_str = url_parts[-1]
        if not venue or not runners: return None
        try:
            if ":" not in race_time_str and len(race_time_str) == 4: race_time_str = f"{race_time_str[:2]}:{race_time_str[2:]}"
            start_time = datetime.combine(race_date, datetime.strptime(race_time_str, "%H:%M").time())
        except: return None
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
        return FetchStrategy(primary_engine=BrowserEngine.HTTPX, enable_js=False, timeout=30)

    def _get_headers(self) -> Dict[str, str]:
        return self._get_browser_headers(host="www.boylesports.com", referer="https://www.boylesports.com/sports/horse-racing")

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        url = "/sports/horse-racing/race-card"
        resp = await self.make_request("GET", url, headers=self._get_headers())
        if not resp or not resp.text: return None
        self._save_debug_snapshot(resp.text, f"boylesports_index_{date}")
        return {"pages": [{"url": url, "html": resp.text}], "date": date}

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data or not raw_data.get("pages"): return []
        try: race_date = datetime.strptime(raw_data["date"], "%Y-%m-%d").date()
        except: race_date = datetime.now().date()
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
                except: continue
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
class SportingLifeAdapter(BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
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
        if not resp or not resp.text: raise AdapterHttpError(self.source_name, 500, index_url)
        self._save_debug_snapshot(resp.text, f"sportinglife_index_{date}")
        parser = HTMLParser(resp.text)
        metadata = self._extract_race_metadata(parser)
        if not metadata: return None
        pages = await self._fetch_race_pages_concurrent(metadata, self._get_headers(), semaphore_limit=8)
        return {"pages": pages, "date": date}

    def _extract_race_metadata(self, parser: HTMLParser) -> List[Dict[str, Any]]:
        meta: List[Dict[str, Any]] = []
        script = parser.css_first("script#__NEXT_DATA__")
        if script:
            try:
                data = json.loads(script.text())
                for meeting in data.get("props", {}).get("pageProps", {}).get("meetings", []):
                    for i, race in enumerate(meeting.get("races", [])):
                        if url := race.get("racecard_url"): meta.append({"url": url, "race_number": i + 1})
            except: pass
        if not meta:
            meetings = parser.css('section[class^="MeetingSummary"]') or parser.css(".meeting-summary")
            for meeting in meetings:
                for i, link in enumerate(meeting.css('a[href*="/racecard/"]')):
                    if url := link.attributes.get("href"): meta.append({"url": url, "race_number": i + 1})
        return meta

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data or not raw_data.get("pages"): return []
        try: race_date = datetime.strptime(raw_data["date"], "%Y-%m-%d").date()
        except: return []
        races: List[Race] = []
        for item in raw_data["pages"]:
            html_content = item.get("html")
            if not html_content: continue
            try:
                parser = HTMLParser(html_content)
                race = self._parse_from_next_data(parser, race_date, item.get("race_number"), html_content)
                if not race: race = self._parse_from_html(parser, race_date, item.get("race_number"), html_content)
                if race: races.append(race)
            except: pass
        return races

    def _parse_from_next_data(self, parser: HTMLParser, race_date: date, race_number_fallback: Optional[int], html_content: str) -> Optional[Race]:
        script = parser.css_first("script#__NEXT_DATA__")
        if not script: return None
        try:
            data = json.loads(script.text())
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
            except: return None
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
        except: return None

    def _parse_from_html(self, parser: HTMLParser, race_date: date, race_number_fallback: Optional[int], html_content: str) -> Optional[Race]:
        h1 = parser.css_first('h1[class*="RacingRacecardHeader__Title"]')
        if not h1: return None
        ht = clean_text(h1.text())
        if not ht: return None
        parts = ht.split()
        if not parts: return None
        try: start_time = datetime.combine(race_date, datetime.strptime(parts[0], "%H:%M").time())
        except: return None
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

                # Deep dark sorcery fallback
                if wo is None:
                    wo = SmartOddsExtractor.extract_from_node(row)

                od = {}
                if ov := create_odds_data(self.source_name, wo): od[self.source_name] = ov
                runners.append(Runner(number=number, name=name, odds=od, win_odds=wo))
            except: continue
        if not runners: return None
        dn = parser.css_first('span[class*="RacecardHeader__Distance"]') or parser.css_first(".race-distance")
        return Race(id=generate_race_id("sl", track_name or "Unknown", start_time, race_number_fallback or 1), venue=track_name or "Unknown", race_number=race_number_fallback or 1, start_time=start_time, runners=runners, distance=clean_text(dn.text()) if dn else None, source=self.source_name, available_bets=scrape_available_bets(html_content))

# ----------------------------------------
# SkySportsAdapter
# ----------------------------------------
class SkySportsAdapter(BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
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
        if not resp or not resp.text: raise AdapterHttpError(self.source_name, 500, index_url)
        self._save_debug_snapshot(resp.text, f"skysports_index_{date}")
        parser = HTMLParser(resp.text)
        metadata = []
        meetings = parser.css(".sdc-site-concertina-block") or parser.css(".page-details__section") or parser.css(".racing-meetings__meeting")
        for meeting in meetings:
            hn = meeting.css_first(".sdc-site-concertina-block__title") or meeting.css_first(".racing-meetings__meeting-title")
            if not hn: continue
            vr = clean_text(hn.text()) or ""
            if "ABD:" in vr: continue
            for i, link in enumerate(meeting.css('a[href*="/racecards/"]')):
                if h := link.attributes.get("href"): metadata.append({"url": h, "venue_raw": vr, "race_number": i + 1})
        if not metadata: return None
        pages = await self._fetch_race_pages_concurrent(metadata, self._get_headers(), semaphore_limit=10)
        return {"pages": pages, "date": date}

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data or not raw_data.get("pages"): return []
        try: race_date = datetime.strptime(raw_data.get("date", ""), "%Y-%m-%d").date()
        except: race_date = datetime.now().date()
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
            except: continue
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
                        except: pass
                onode = node.css_first(".sdc-site-racing-card__betting-odds")
                wo = parse_odds_to_decimal(clean_text(onode.text()) if onode else "")

                # Deep dark sorcery fallback
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
        except: return None
        if not isinstance(data, list): return None
        return {"venues": data, "date": date, "fetched_at": datetime.now().isoformat()}

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
        except: return None
        runners = [Runner(number=i + 1, name=f"Runner {i + 1}", scratched=False, odds={}) for i in range(nr)]
        return Race(discipline="Thoroughbred", id=f"rpb2b_{rid.replace('-', '')[:16]}", venue=normalize_venue_name(vn) or vn, race_number=rnum, start_time=st, runners=runners, source=self.source_name, metadata={"original_race_id": rid, "country_code": cc, "num_runners": nr})


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
        return FetchStrategy(primary_engine=BrowserEngine.PLAYWRIGHT, enable_js=True, stealth_mode="fast", timeout=45)

    def _get_headers(self) -> Dict[str, str]:
        return self._get_browser_headers(host="standardbredcanada.ca", referer="https://standardbredcanada.ca/racing")

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        dt = datetime.strptime(date, "%Y-%m-%d")
        date_label = dt.strftime(f"%A %b {dt.day}, %Y")
        try: from playwright.async_api import async_playwright
        except: return None
        index_html = None
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                try:
                    await page.goto(f"{self.base_url}/entries", wait_until="networkidle")
                    await page.evaluate("() => { document.querySelectorAll('details').forEach(d => d.open = true); }")
                    try: await page.select_option("#edit-entries-track", label="View All Tracks")
                    except: pass
                    try: await page.select_option("#edit-entries-date", label=date_label)
                    except: pass
                    try: await page.click("#edit-custom-submit-entries", force=True, timeout=5000)
                    except: pass
                    try: await page.wait_for_selector("#entries-results-container a[href*='/entries/']", timeout=10000)
                    except: pass
                    index_html = await page.content()
                finally:
                    await page.close()
                    await browser.close()
        except Exception as e:
            self.logger.error("Playwright failed", error=str(e))
            return None
        if not index_html: return None
        self._save_debug_snapshot(index_html, f"sc_index_{date}")
        parser = HTMLParser(index_html)
        metadata = []
        for container in parser.css("#entries-results-container .racing-results-ex-wrap > div"):
            tnn = container.css_first("h4.track-name")
            if not tnn: continue
            tn = clean_text(tnn.text()) or ""
            isf = "*" in tn or "*" in (clean_text(container.text()) or "")
            for link in container.css('a[href*="/entries/"]'):
                if u := link.attributes.get("href"): metadata.append({"url": u, "venue": tn.replace("*", "").strip(), "finalized": isf})
        if not metadata: return None
        pages = await self._fetch_race_pages_concurrent(metadata, self._get_headers(), semaphore_limit=3)
        return {"pages": pages, "date": date}

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data or not raw_data.get("pages"): return []
        try: race_date = datetime.strptime(raw_data.get("date", ""), "%Y-%m-%d").date()
        except: race_date = datetime.now().date()
        races: List[Race] = []
        for item in raw_data["pages"]:
            html_content = item.get("html")
            if not html_content or ("Final Changes Made" not in html_content and not item.get("finalized")): continue
            track_name = normalize_venue_name(item["venue"]) or item["venue"]
            for pre in HTMLParser(html_content).css("pre"):
                text = pre.text()
                race_chunks = re.split(r"(\d+)\s+--\s+", text)
                for i in range(1, len(race_chunks), 2):
                    try:
                        r = self._parse_single_race(race_chunks[i+1], int(race_chunks[i]), race_date, track_name)
                        if r: races.append(r)
                    except: continue
        return races

    def _parse_single_race(self, content: str, race_num: int, race_date: date, track_name: str) -> Optional[Race]:
        tm = re.search(r"Post\s+Time:\s*(\d{1,2}:\d{2}\s*[APM]{2})", content, re.I)
        st = None
        if tm:
            try: st = datetime.combine(race_date, datetime.strptime(tm.group(1), "%I:%M %p").time())
            except: pass
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
    BASE_URL: ClassVar[str] = "https://api.beta.tab.com.au/v1/tab-info-service/racing"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config, rate_limit=2.0)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return FetchStrategy(primary_engine=BrowserEngine.PLAYWRIGHT, enable_js=True, stealth_mode="fast", timeout=45)

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/dates/{date}/meetings"
        resp = await self.make_request("GET", url, headers={"Accept": "application/json", "User-Agent": CHROME_USER_AGENT})
        if not resp: return None
        try: data = resp.json() if hasattr(resp, "json") else json.loads(resp.text)
        except: return None
        if not data or "meetings" not in data: return None
        return {"meetings": data["meetings"], "date": date}

    def _parse_races(self, raw_data: Any) -> List[Race]:
        if not raw_data or "meetings" not in raw_data: return []
        races: List[Race] = []
        for m in raw_data["meetings"]:
            vn = normalize_venue_name(m.get("meetingName")) or "Unknown"
            mt = m.get("meetingType", "R")
            disc = {"R": "Thoroughbred", "H": "Harness", "G": "Greyhound"}.get(mt, "Thoroughbred")
            for rd in m.get("races", []):
                rn, rst = rd.get("raceNumber"), rd.get("raceStartTime")
                if not rst: continue
                try: st = datetime.fromisoformat(rst.replace("Z", "+00:00"))
                except: continue
                races.append(Race(id=generate_race_id("tab", vn, st, rn, disc), venue=vn, race_number=rn, start_time=st, runners=[], discipline=disc, source=self.source_name, available_bets=[]))
        return races

# ----------------------------------------
# BetfairDataScientistAdapter
# ----------------------------------------
class BetfairDataScientistAdapter(BaseAdapterV3):
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
                vn = normalize_venue_name(str(ri.get("meeting_name", ""))) or "Unknown"
                races.append(Race(id=str(mid), venue=vn, race_number=int(ri.get("race_number", 0)), start_time=datetime.now(), runners=runners, source=self.source_name, discipline="Thoroughbred"))
            return races
        except: return []

# ----------------------------------------
# EquibaseAdapter
# ----------------------------------------
class EquibaseAdapter(BrowserHeadersMixin, DebugMixin, RacePageFetcherMixin, BaseAdapterV3):
    SOURCE_NAME: ClassVar[str] = "Equibase"
    BASE_URL: ClassVar[str] = "https://www.equibase.com"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        return FetchStrategy(primary_engine=BrowserEngine.PLAYWRIGHT, enable_js=True, block_resources=True)

    def _get_headers(self) -> Dict[str, str]:
        return self._get_browser_headers(host="www.equibase.com")

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        for url in [f"/entries/{date}", "/static/entry/index.html", f"/static/entry/{date}/index.html"]:
            try:
                resp = await self.make_request("GET", url, headers=self._get_headers())
                if resp and resp.text and len(resp.text) > 1000: break
            except: continue
        else: raise AdapterHttpError(self.source_name, 500, "Equibase index failed")
        self._save_debug_snapshot(resp.text, f"equibase_index_{date}")
        parser, links = HTMLParser(resp.text), []
        for a in parser.css("a"):
            h, c = a.attributes.get("href", ""), a.attributes.get("class", "")
            if "/static/entry/" in h or "entry-race-level" in c: links.append(h)
        pages = await self._fetch_race_pages_concurrent([{"url": l} for l in set(links)], self._get_headers(), semaphore_limit=5)
        return {"pages": [p.get("html") for p in pages if p and p.get("html")], "date": date}

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
            nn, nmn, on = node.css_first("td:nth-child(1)"), node.css_first("td:nth-child(3)"), node.css_first("td:nth-child(10)")
            if not nn or not nn.text(strip=True).isdigit() or not nmn: return None
            number, name = int(nn.text(strip=True)), clean_text(nmn.text())
            if not name: return None
            sc = "scratched" in node.attributes.get("class", "").lower() or "SCR" in (clean_text(node.text()) or "")
            odds, wo = {}, None
            if not sc:
                wo = parse_odds_to_decimal(clean_text(on.text()) if on else "")
                if wo is None: wo = SmartOddsExtractor.extract_from_node(node)
                if od := create_odds_data(self.source_name, wo): odds[self.source_name] = od
            return Runner(number=number, name=name, odds=odds, win_odds=wo, scratched=sc)
        except: return None

    def _parse_post_time(self, ds: str, ts: str) -> datetime:
        try:
            parts = ts.replace("Post Time:", "").strip().split()
            if len(parts) >= 2:
                dt = datetime.strptime(f"{ds} {parts[0]} {parts[1]}", "%Y-%m-%d %I:%M %p")
                return dt.replace(tzinfo=timezone.utc)
        except: pass
        # Fallback to noon UTC for the given date if time parsing fails
        try:
            dt = datetime.strptime(ds, "%Y-%m-%d")
            return dt.replace(hour=12, minute=0, tzinfo=timezone.utc)
        except:
            return datetime.now(timezone.utc)

# ----------------------------------------
# TwinSpiresAdapter
# ----------------------------------------
class TwinSpiresAdapter(DebugMixin, BaseAdapterV3):
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
        return FetchStrategy(primary_engine=BrowserEngine.CAMOUFOX, enable_js=True, stealth_mode="camouflage", block_resources=True, max_retries=3, timeout=60)

    async def _fetch_data(self, date: str) -> Optional[Dict[str, Any]]:
        ard = []
        last_err = None
        for disc in ["thoroughbred", "harness", "greyhound"]:
            url = f"{self.BASE_URL}/bet/todays-races/{disc}"
            try:
                resp = await self.make_request("GET", url, network_idle=True, wait_selector='div[class*="race"], [class*="RaceCard"], [class*="track"]')
                if resp and resp.status == 200:
                    self._save_debug_snapshot(resp.text, f"ts_{disc}_{date}")
                    dr = self._extract_races_from_page(resp, date)
                    for r in dr: r["assigned_discipline"] = disc.capitalize()
                    ard.extend(dr)
            except Exception as e: last_err = e
        if not ard:
            try:
                resp = await self.make_request("GET", f"{self.BASE_URL}/bet/todays-races/time", network_idle=True)
                if resp and resp.status == 200: ard = self._extract_races_from_page(resp, date)
            except Exception as e: last_err = last_err or e
        if not ard and last_err: raise last_err
        return {"races": ard, "date": date, "source": self.source_name} if ard else None

    def _extract_races_from_page(self, resp, date: str) -> List[Dict[str, Any]]:
        rd, page = [], Selector(resp.text)
        relems, used = [], None
        for s in self.RACE_CONTAINER_SELECTORS:
            try:
                el = page.css(s)
                if el:
                    relems, used = el, s
                    break
            except: continue
        if not relems: return [{"html": resp.text, "track": "Unknown", "race_number": 0, "date": date, "full_page": True}]
        for i, relem in enumerate(relems, 1):
            try:
                html_str = str(relem.html) if hasattr(relem, 'html') else str(relem)
                tn = self._find_with_selectors(relem, self.TRACK_NAME_SELECTORS) or f"Track {i}"
                rn_txt = self._find_with_selectors(relem, self.RACE_NUMBER_SELECTORS)
                rnum = i
                if rn_txt:
                    digits = "".join(filter(str.isdigit, rn_txt))
                    if digits: rnum = int(digits)
                rd.append({"html": html_str, "track": tn.strip(), "race_number": rnum, "post_time_text": self._find_with_selectors(relem, self.POST_TIME_SELECTORS), "distance": self._find_with_selectors(relem, ['[class*="distance"]', '[class*="Distance"]', '[data-distance]', ".race-distance"]), "date": date, "full_page": False, "available_bets": scrape_available_bets(html_str)})
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
        rl, ds, parsed = raw_data["races"], raw_data.get("date", datetime.now().strftime("%Y-%m-%d")), []
        for rd in rl:
            try:
                r = self._parse_single_race(rd, ds)
                if r and r.runners: parsed.append(r)
            except: continue
        return parsed

    def _parse_single_race(self, rd: dict, ds: str) -> Optional[Race]:
        hc = rd.get("html", "")
        if not hc: return None
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
                        try: return datetime.fromisoformat(da.replace('Z', '+00:00'))
                        except: pass
                    p = self._parse_time_string(e.text.strip() if hasattr(e, 'text') else str(e).strip(), bd)
                    if p: return p
            except: continue
        return datetime.combine(bd, datetime.now().time()) + timedelta(hours=1)

    def _parse_time_string(self, ts: str, bd) -> Optional[datetime]:
        if not ts: return None
        tc = re.sub(r"\s+(EST|EDT|CST|CDT|MST|MDT|PST|PDT|ET|PT|CT|MT)$", "", ts, flags=re.I).strip()
        m = re.search(r"(\d+)\s*(?:min|mtp)", tc, re.I)
        if m: return datetime.now() + timedelta(minutes=int(m.group(1)))
        for f in ['%I:%M %p', '%I:%M%p', '%H:%M', '%I:%M:%S %p']:
            try: return datetime.combine(bd, datetime.strptime(tc, f).time())
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
                    if dig: num = int(dig); break
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

            # Deep dark sorcery fallback
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
from abc import ABC
from abc import abstractmethod
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

import structlog




try:
    # winsound is a built-in Windows library
    import winsound
except ImportError:
    winsound = None
try:
    from win10toast_py3 import ToastNotifier
except (ImportError, RuntimeError):
    # Fails gracefully on non-Windows systems
    ToastNotifier = None

log = structlog.get_logger(__name__)


def _get_best_win_odds(runner: Runner) -> Optional[Decimal]:
    """Gets the best win odds for a runner, filtering out invalid or placeholder values."""
    if not runner.odds:
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
            valid_odds.append(win)

    return min(valid_odds) if valid_odds else None


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
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=30)
        st = race.start_time
        if st.tzinfo is None:
            st = st.replace(tzinfo=timezone.utc)
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
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=30)

        for race in races:
            # 1. Timing Filter: Ignore races more than 30 minutes in the past
            st = race.start_time
            if st.tzinfo is None:
                st = st.replace(tzinfo=timezone.utc)

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
            if active_runners and len(active_runners) <= 8:
                all_odds = []
                for runner in active_runners:
                    odds = _get_best_win_odds(runner)
                    if odds is not None:
                        all_odds.append(odds)
                if len(all_odds) >= 2:
                    all_odds.sort()
                    if all_odds[1] >= 5.0:
                        is_goldmine = True

            race.metadata['is_goldmine'] = is_goldmine
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
    """Handles sending native Windows notifications and audio alerts for high-value races."""

    def __init__(self):
        self.toaster = ToastNotifier("Fortuna") if ToastNotifier else None
        self.audio_system = AudioAlertSystem()
        self.notified_races = set()

    def notify_qualified_race(self, race):
        if not self.toaster or race.id in self.notified_races:
            return

        title = "🐎 High-Value Opportunity!"
        message = f"""{race.venue} - Race {race.race_number}
Score: {race.qualification_score:.0f}%
Post Time: {race.start_time.strftime("%I:%M %p")}"""

        try:
            # The `threaded=True` argument is crucial to prevent blocking the main application thread.
            self.toaster.show_toast(title, message, duration=10, threaded=True)
            self.notified_races.add(race.id)
            self.audio_system.play("high_value")
            log.info("Notification and audio alert sent for high-value race", race_id=race.id)
        except Exception as e:
            # Catch potential exceptions from the notification library itself
            log.error("Failed to send notification", error=str(e), exc_info=True)


# ----------------------------------------
# TEXT UTILITIES
# ----------------------------------------
# python_service/utils/text.py
# Centralized text and name normalization utilities
import re
import os
from typing import Optional, Any, List, Union
from collections import defaultdict
from datetime import datetime, timezone


def get_field(obj: Any, field_name: str, default: Any = None) -> Any:
    """Helper to get a field from either an object or a dictionary."""
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    return getattr(obj, field_name, default)


def clean_text(text: Optional[str]) -> Optional[str]:
    """Strips leading/trailing whitespace and collapses internal whitespace."""
    if not text:
        return None
    return " ".join(text.strip().split())


def normalize_venue_name(name: Optional[str]) -> Optional[str]:
    """
    Normalizes a racecourse name to a standard format.
    Handles common abbreviations, variations, and trims country suffixes.
    """
    if not name:
        return None

    # Trim parenthetical info like (USA), (IRE), (GB), etc. and extra whitespace
    name = re.sub(r'\s*\([^)]*\)\s*$', '', name)

    # Use a temporary variable for matching, but return the properly cased name
    cleaned_name_upper = clean_text(name).upper()

    VENUE_MAP = {
        "ASCOT": "Ascot",
        "AQUEDUCT": "Aqueduct",
        "AYR": "Ayr",
        "BANGOR-ON-DEE": "Bangor-on-Dee",
        "CATTERICK BRIDGE": "Catterick",
        "CHELMSFORD CITY": "Chelmsford",
        "EPSOM DOWNS": "Epsom",
        "FONTWELL": "Fontwell Park",
        "GULFSTREAM": "Gulfstream Park",
        "GULFSTREAM PARK": "Gulfstream Park",
        "HAYDOCK": "Haydock Park",
        "KEMPTON": "Kempton Park",
        "LINGFIELD": "Lingfield Park",
        "LINGFIELD PARK": "Lingfield Park",
        "NEWMARKET (ROWLEY)": "Newmarket",
        "NEWMARKET (JULY)": "Newmarket",
        "SAM HOUSTON": "Sam Houston",
        "SAM HOUSTON RACE PARK": "Sam Houston",
        "SANDOWN": "Sandown Park",
        "SANDOWN PARK": "Sandown Park",
        "SANTA ANITA": "Santa Anita",
        "STRATFORD": "Stratford-on-Avon",
        "YARMOUTH": "Great Yarmouth",
        "CURRAGH": "Curragh",
        "DOWN ROYAL": "Down Royal",
        "DELTA DOWNS": "Delta Downs",
        "FAIR GROUNDS": "Fair Grounds",
        "LAUREL PARK": "Laurel Park",
        "LOS ALAMITOS": "Los Alamitos",
        "MUSSELBURGH": "Musselburgh",
        "NEWCASTLE": "Newcastle",
        "SUNLAND PARK": "Sunland Park",
        "TAMPA BAY DOWNS": "Tampa Bay Downs",
        "TURF PARADISE": "Turf Paradise",
        "VINCENNES": "Vincennes",
        "WETHERBY": "Wetherby",
    }

    # Check primary map first
    if cleaned_name_upper in VENUE_MAP:
        return VENUE_MAP[cleaned_name_upper]

    # Handle cases where the key is the desired output but needs to be mapped from a variation
    # e.g. CHELMSFORD maps to Chelmsford
    # Title case the cleaned name for a sensible default
    title_cased_name = clean_text(name).title()
    if title_cased_name in VENUE_MAP.values():
        return title_cased_name

    # Return the title-cased cleaned name as a fallback
    return title_cased_name


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

    # Distance consistency check (4 or more times at that venue)
    dist_counts = defaultdict(int)
    for r in races_at_track:
        dist = get_field(r, 'distance')
        if dist:
            dist_counts[dist] += 1
    if dist_counts and max(dist_counts.values()) >= 4:
        return 'H'

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
        if not v: continue
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
        if v:
            races_by_track[normalize_venue_name(v)].append(r)
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
        if v:
            races_by_track[normalize_venue_name(v)].append(r)
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
    goldmines = [r for r in races if get_field(r, 'metadata', {}).get('is_goldmine')]

    if not goldmines:
        return "No Goldmine races found."

    # Sort goldmines: Cat descending, Track asc, Race num asc
    cat_map = {'T': 3, 'H': 2, 'G': 1}
    def goldmine_sort_key(r):
        track = normalize_venue_name(get_field(r, 'venue'))
        cat = track_categories.get(track, 'T')
        return (-cat_map.get(cat, 0), track, get_field(r, 'race_number', 0))

    goldmines.sort(key=goldmine_sort_key)

    now = datetime.now(timezone.utc)
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
                start_time = start_time.replace(tzinfo=timezone.utc)

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

    report_lines = ["GOLDMINE RACE INTELLIGENCE REPORT", "================================", ""]

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
                time_str = start_time.strftime("%H:%M UTC")
            else:
                time_str = str(start_time)

            report_lines.append(f"{cat}~{track} - Race {race_num} ({time_str})")
            report_lines.append("-" * 40)

            runners = get_field(r, 'runners', [])
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


def generate_next_to_jump(races: List[Any]) -> str:
    """Generate the NEXT TO JUMP section."""
    lines = ["", "", "NEXT TO JUMP", "------------"]
    now = datetime.now(timezone.utc)
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
                r_time = r_time.replace(tzinfo=timezone.utc)
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


def generate_summary_grid(races: List[Any], all_races: Optional[List[Any]] = None) -> str:
    """
    Generates a tiered summary grid.
    Primary section: Races with Superfectas (Explicit or T-tracks with field > 6).
    Secondary section: All remaining races.
    """
    now = datetime.now(timezone.utc)

    def is_superfecta_explicit(r):
        available_bets = get_field(r, 'available_bets', [])
        metadata_bets = get_field(r, 'metadata', {}).get('available_bets', [])
        return 'Superfecta' in available_bets or 'Superfecta' in metadata_bets

    track_categories = {}
    all_field_sizes = set()
    WRAP_WIDTH = 4

    # 1. Pre-calculate track categories
    races_by_track = defaultdict(list)
    source_races = all_races if all_races is not None else races
    for r in source_races:
        venue = get_field(r, 'venue')
        if venue:
            races_by_track[normalize_venue_name(venue)].append(r)

    for track, tr_races in races_by_track.items():
        track_categories[track] = get_track_category(tr_races)

    # 2. Partition races based on explicit Superfecta OR T-track field size rule
    primary_stats = defaultdict(lambda: defaultdict(list))
    secondary_stats = defaultdict(lambda: defaultdict(list))

    for race in races:
        track = normalize_venue_name(get_field(race, 'venue'))
        runners = get_field(race, 'runners', [])
        field_size = len([r for r in runners if not get_field(r, 'scratched', False)])
        race_num = get_field(race, 'race_number') or 0
        is_goldmine = get_field(race, 'metadata', {}).get('is_goldmine', False)

        all_field_sizes.add(field_size)
        cat = track_categories.get(track, 'T')

        is_primary = is_superfecta_explicit(race) or (cat == 'T' and field_size >= 6)

        if is_primary:
            primary_stats[track][field_size].append((race_num, is_goldmine))
        else:
            secondary_stats[track][field_size].append((race_num, is_goldmine))

    if not all_field_sizes:
        return "\nNo races found to display in grid."

    sorted_field_sizes = sorted(list(all_field_sizes))
    cat_map = {'T': 3, 'H': 2, 'G': 1}
    col_widths = {fs: max(len(str(fs)), WRAP_WIDTH) for fs in sorted_field_sizes}

    header_parts = [f"{'CATEG':<5}", f"{'Track':<25}"]
    for fs in sorted_field_sizes:
        header_parts.append(f"{str(fs):^{col_widths[fs]}}")

    header = " | ".join(header_parts)
    grid_lines = ["\n" + header, "-" * len(header)]

    def render_stats(stats_dict, label=None):
        if not stats_dict:
            return
        if label:
            label_row = f"--- {label.upper()} ---"
            grid_lines.append(f"{label_row:^{len(header)}}")
            grid_lines.append("-" * len(header))

        sorted_tracks = sorted(stats_dict.keys(), key=lambda t: (-cat_map.get(track_categories.get(t, 'T'), 0), t))
        for track in sorted_tracks:
            wrapped_stats = {}
            max_lines = 1
            for fs in sorted_field_sizes:
                wrapped = format_grid_code(stats_dict[track].get(fs, []), WRAP_WIDTH)
                wrapped_stats[fs] = wrapped
                max_lines = max(max_lines, len(wrapped))

            for line_idx in range(max_lines):
                if line_idx == 0:
                    row_prefix = f"{track_categories.get(track, 'T'):<5} | {track[:25]:<25} | "
                else:
                    row_prefix = f"{' ':<5} | {' ':<25} | "

                row_vals = []
                for fs in sorted_field_sizes:
                    wrapped = wrapped_stats[fs]
                    val = wrapped[line_idx] if line_idx < len(wrapped) else ""
                    row_vals.append(f"{val:^{col_widths[fs]}}")

                grid_lines.append(row_prefix + " | ".join(row_vals))
            grid_lines.append("-" * len(header))

    # 3. Identify Immediate Goldmine races for prime display
    immediate_gold_super_snippet = []
    immediate_gold_snippet = []

    # We need track categories for the goldmine partitioning check
    for race in races:
        if get_field(race, 'metadata', {}).get('is_goldmine'):
            track = normalize_venue_name(get_field(race, 'venue'))
            cat = track_categories.get(track, 'T')

            # Use same Superfecta filter as generate_goldmine_report
            available_bets = get_field(race, 'available_bets', [])
            metadata_bets = get_field(race, 'metadata', {}).get('available_bets', [])
            runners = get_field(race, 'runners', [])
            field_size = len([run for run in runners if not get_field(run, 'scratched', False)])

            is_super = 'Superfecta' in available_bets or 'Superfecta' in metadata_bets or (cat == 'T' and field_size >= 6)

            start_time = get_field(race, 'start_time')
            if isinstance(start_time, str):
                try:
                    start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                except ValueError:
                    start_time = None

            if start_time:
                if start_time.tzinfo is None:
                    start_time = start_time.replace(tzinfo=timezone.utc)

                diff = (start_time - now).total_seconds() / 60
                if 0 <= diff <= 20:
                    entry = f"{cat}~{track} R{get_field(race, 'race_number')} in {int(diff)}m"
                    if is_super:
                        immediate_gold_super_snippet.append(entry)
                    else:
                        immediate_gold_snippet.append(entry)

    final_grid_lines = []
    if immediate_gold_super_snippet:
        final_grid_lines.append("!!! IMMEDIATE GOLD (SUPERFECTA) !!!")
        final_grid_lines.extend(immediate_gold_super_snippet)
        final_grid_lines.append("")
    if immediate_gold_snippet:
        final_grid_lines.append("!!! IMMEDIATE GOLD !!!")
        final_grid_lines.extend(immediate_gold_snippet)
        final_grid_lines.append("")

    # Render sections BEFORE extending final_grid_lines with grid_lines
    if primary_stats:
        render_stats(primary_stats, label="Preferred Superfecta Races")

    if secondary_stats:
        # Use label if primary also existed
        label = "All Remaining Races" if primary_stats else None
        render_stats(secondary_stats, label=label)

    final_grid_lines.extend(grid_lines)

    appendix = generate_fortuna_fives(races, all_races=all_races)
    goldmines = generate_goldmines(races, all_races=all_races)
    next_to_jump = generate_next_to_jump(races)

    # Unified spacing management (Memory Directive Fix)
    # Ensure each appendix part is appended without doubling the newlines from the list join
    full_report = "\n".join(final_grid_lines) + appendix + goldmines + next_to_jump

    return full_report


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
        }


class FavoriteToPlaceMonitor:
    """Monitor for favorite-to-place betting opportunities."""

    # Adapter configuration
    ADAPTER_CLASSES = [
        AtTheRacesAdapter,
        AtTheRacesGreyhoundAdapter,
        BoyleSportsAdapter,
        SportingLifeAdapter,
        SkySportsAdapter,
        RacingPostB2BAdapter,
        StandardbredCanadaAdapter,
        TabAdapter,
        BetfairDataScientistAdapter,
        EquibaseAdapter,
        TwinSpiresAdapter,
    ]

    def __init__(self, target_date: Optional[str] = None, refresh_interval: int = 30):
        """
        Initialize monitor.

        Args:
            target_date: Date to fetch races for (YYYY-MM-DD), defaults to today
            refresh_interval: Seconds between refreshes for BET NOW list
        """
        self.target_date = target_date or datetime.now().strftime("%Y-%m-%d")
        self.refresh_interval = refresh_interval
        self.all_races: List[RaceSummary] = []
        self.adapters: List = []

    async def initialize_adapters(self):
        """Initialize all adapters."""
        print(f"🔧 Initializing {len(self.ADAPTER_CLASSES)} adapters...")

        for adapter_class in self.ADAPTER_CLASSES:
            try:
                adapter = adapter_class()
                self.adapters.append(adapter)
                print(f"  ✅ {adapter_class.__name__}")
            except Exception as e:
                print(f"  ❌ {adapter_class.__name__}: {e}")

        print(f"✅ Initialized {len(self.adapters)} adapters\n")

    async def fetch_all_races(self) -> List[Tuple[Race, str]]:
        """Fetch races from all adapters."""
        print(f"📡 Fetching races for {self.target_date}...\n")

        all_races_with_adapters = []

        # Run fetches in parallel for speed
        async def fetch_one(adapter):
            name = adapter.__class__.__name__
            try:
                races = await adapter.get_races(self.target_date)
                print(f"  ✅ {name}: {len(races)} races")
                return [(r, name) for r in races]
            except Exception as e:
                print(f"  ❌ {name}: {e}")
                return []

        results = await asyncio.gather(*[fetch_one(a) for a in self.adapters])
        for r_list in results:
            all_races_with_adapters.extend(r_list)

        print(f"\n📊 Total races fetched: {len(all_races_with_adapters)}\n")
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

    def _get_favorite_and_second(self, race: Race) -> Tuple[Optional[Runner], Optional[Runner]]:
        """Get favorite and second favorite by odds."""
        # Get active runners with valid odds
        r_with_odds = [r for r in race.runners if not r.scratched and r.win_odds and r.win_odds > 1.0]
        if len(r_with_odds) < 2: return None, None

        # Sort by odds (lowest first)
        sorted_r = sorted(r_with_odds, key=lambda r: r.win_odds)
        return sorted_r[0], sorted_r[1]

    def _calculate_mtp(self, start_time: datetime) -> Optional[int]:
        """Calculate minutes to post."""
        if not start_time: return None
        now = datetime.now(timezone.utc)
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        delta = start_time - now
        return int(delta.total_seconds() / 60)

    def _create_race_summary(self, race: Race, adapter_name: str) -> RaceSummary:
        """Create a RaceSummary from a Race object."""
        favorite, second_fav = self._get_favorite_and_second(race)
        return RaceSummary(
            discipline=self._get_discipline_code(race),
            track=race.venue,
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
        )

    async def build_race_summaries(self, races_with_adapters: List[Tuple[Race, str]]):
        """Build and deduplicate summary list."""
        race_map = {}
        for race, adapter_name in races_with_adapters:
            try:
                summary = self._create_race_summary(race, adapter_name)
                # Stable key: Venue + Race Number
                key = f"{summary.track.lower().strip()}|{summary.race_number}"

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

        self.all_races = list(race_map.values())

    def print_full_list(self):
        """Print all fetched races."""
        print("=" * 120)
        print("FULL RACE LIST".center(120))
        print("=" * 120)
        print(f"{'DISC':<5} {'TRACK':<25} {'R#':<4} {'FIELD':<6} {'SUPER':<6} {'ADAPTER':<25} {'START TIME':<20}")
        print("-" * 120)
        for r in sorted(self.all_races, key=lambda x: (x.discipline, x.track, x.race_number)):
            superfecta = "Yes" if r.superfecta_offered else "No"
            st = r.start_time.strftime("%Y-%m-%d %H:%M") if r.start_time else "Unknown"
            print(f"{r.discipline:<5} {r.track[:24]:<25} {r.race_number:<4} {r.field_size:<6} {superfecta:<6} {r.adapter[:24]:<25} {st:<20}")
        print("-" * 120)
        print(f"Total races: {len(self.all_races)}\n")

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
        # Criteria: Not in BET NOW, but 0 < MTP <= 30 and 2nd Fav Odds >= 4.0
        # and field size <= 8
        bet_now_keys = {(r.track, r.race_number) for r in self.get_bet_now_races()}
        yml = [
            r for r in self.all_races
            if r.mtp is not None and 0 < r.mtp <= 30
            and r.second_fav_odds is not None and r.second_fav_odds >= 4.0
            and r.field_size <= 8
            and (r.track, r.race_number) not in bet_now_keys
        ]
        # Sort by MTP asc
        yml.sort(key=lambda r: r.mtp)
        return yml[:5]  # Limit to top 5 recommendations

    def print_bet_now_list(self):
        """Print filtered BET NOW list."""
        bet_now = self.get_bet_now_races()
        print("=" * 140)
        print("🎯 BET NOW - FAVORITE TO PLACE OPPORTUNITIES".center(140))
        print("=" * 140)
        print(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Criteria: MTP <= 20 minutes AND 2nd Favorite Odds >= 5.0")
        print("-" * 140)
        if not bet_now:
            print("\n⏳ No races currently meet BET NOW criteria.\n")

            yml = self.get_you_might_like_races()
            if yml:
                print("=" * 140)
                print("🌟 YOU MIGHT LIKE - NEAR-MISS OPPORTUNITIES".center(140))
                print("=" * 140)
                print(f"{'SUPER':<6} {'MTP':<5} {'DISC':<5} {'TRACK':<20} {'R#':<4} {'FIELD':<6} {'ODDS':<20}")
                print("-" * 140)
                for r in yml:
                    sup = "✅" if r.superfecta_offered else "❌"
                    fo = f"{r.favorite_odds:.2f}" if r.favorite_odds else "N/A"
                    so = f"{r.second_fav_odds:.2f}" if r.second_fav_odds else "N/A"
                    print(f"{sup:<6} {r.mtp:<5} {r.discipline:<5} {r.track[:19]:<20} {r.race_number:<4} {r.field_size:<6}  ~ {fo}, {so}")
                print("-" * 140)
            return

        print(f"{'SUPER':<6} {'MTP':<5} {'DISC':<5} {'TRACK':<20} {'R#':<4} {'FIELD':<6} {'ODDS':<20} {'ADAPTER':<20}")
        print("-" * 140)
        for r in bet_now:
            sup = "✅" if r.superfecta_offered else "❌"
            fo = f"{r.favorite_odds:.2f}" if r.favorite_odds else "N/A"
            so = f"{r.second_fav_odds:.2f}" if r.second_fav_odds else "N/A"
            print(f"{sup:<6} {r.mtp:<5} {r.discipline:<5} {r.track[:19]:<20} {r.race_number:<4} {r.field_size:<6}  ~ {fo}, {so:<15} {r.adapter[:19]:<20}")
        print("-" * 140)
        print(f"Total opportunities: {len(bet_now)}\n")

    def save_to_json(self, filename: str = "race_data.json"):
        """Export to JSON."""
        bn = self.get_bet_now_races()
        yml = self.get_you_might_like_races()
        data = {
            "generated_at": datetime.now().isoformat(),
            "target_date": self.target_date,
            "total_races": len(self.all_races),
            "bet_now_count": len(bn),
            "you_might_like_count": len(yml),
            "all_races": [r.to_dict() for r in self.all_races],
            "bet_now_races": [r.to_dict() for r in bn],
            "you_might_like_races": [r.to_dict() for r in yml],
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    async def run_once(self):
        await self.initialize_adapters()
        raw = await self.fetch_all_races()
        await self.build_race_summaries(raw)
        self.print_full_list()
        self.print_bet_now_list()
        self.save_to_json()
        for a in self.adapters: await a.shutdown()

    async def run_continuous(self):
        await self.initialize_adapters()
        raw = await self.fetch_all_races()
        await self.build_race_summaries(raw)
        self.print_full_list()
        try:
            while True:
                for r in self.all_races: r.mtp = self._calculate_mtp(r.start_time)
                self.print_bet_now_list()
                self.save_to_json()
                await asyncio.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            for a in self.adapters: await a.shutdown()





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
        return FetchStrategy(
            primary_engine=BrowserEngine.CURL_CFFI,
            enable_js=True,
            stealth_mode=StealthMode.CAMOUFLAGE,
            timeout=45
        )

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
        race_links = {a.attributes["href"] for a in parser.css("a.race-time-link[href]") if a.attributes.get("href")}

        async def fetch_single_html(url_path: str):
            response = await self.make_request("GET", url_path, headers=self._get_headers())
            return response.text if response else ""

        tasks = [fetch_single_html(link) for link in race_links]
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
            if not number_node or not number_node.text(strip=True).isdigit():
                return None
            number = int(number_node.text(strip=True))

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





class TimeformAdapter(BrowserHeadersMixin, DebugMixin, BaseAdapterV3):
    """
    Adapter for timeform.com, migrated to BaseAdapterV3 and standardized on selectolax.
    """

    SOURCE_NAME = "Timeform"
    BASE_URL = "https://www.timeform.com"

    def __init__(self, config=None):
        super().__init__(source_name=self.SOURCE_NAME, base_url=self.BASE_URL, config=config)
        self._semaphore = asyncio.Semaphore(5)

    def _configure_fetch_strategy(self) -> FetchStrategy:
        """Timeform works with HTTPX and good headers."""
        return FetchStrategy(primary_engine=BrowserEngine.CURL_CFFI, enable_js=False)

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
        links = {a.attributes["href"] for a in parser.css("a[href*='/racecards/'][href*='/20']") if a.attributes.get("href") and not a.attributes.get("href").endswith("/racecards")}

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
                for script in parser.css('script[type="application/ld+json"]'):
                    try:
                        data = json.loads(script.text())
                        if data.get("@type") == "Event":
                            venue = normalize_venue_name(data.get("location", {}).get("name", ""))
                            if sd := data.get("startDate"):
                                # 2026-01-28T14:32:00
                                start_time = datetime.fromisoformat(sd.split('+')[0])
                            break
                    except: continue

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
                    except: pass

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
                    number = int(num_attr)
                except:
                    pass

            if not number:
                num_node = row.css_first(".rp-entry-number") or row.css_first("span.rp-horseTable_horse-number")
                if num_node:
                    num_text = clean_text(num_node.text()).strip("()")
                    num_match = re.search(r"\d+", num_text)
                    if num_match:
                        number = int(num_match.group())

            win_odds = None
            if forecast_map and name.lower() in forecast_map:
                win_odds = parse_odds_to_decimal(forecast_map[name.lower()])

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
        """
        RacingPost has strong anti-bot measures. We need to use a full
        browser with the highest stealth settings to avoid being blocked.
        """
        return FetchStrategy(
            primary_engine=BrowserEngine.CURL_CFFI,
            enable_js=True,
            stealth_mode=StealthMode.CAMOUFLAGE,  # Strongest stealth
            block_resources=False,  # Load all resources to appear more human
        )

    def _get_headers(self) -> dict:
        return self._get_browser_headers(host="www.racingpost.com")

    async def _fetch_data(self, date: str) -> Any:
        """
        Fetches the raw HTML content for all races on a given date.
        """
        index_url = f"/racecards/{date}"
        index_response = await self.make_request("GET", index_url, headers=self._get_headers())
        if not index_response or not index_response.text:
            self.logger.warning("Failed to fetch RacingPost index page", url=index_url)
            return None

        self._save_debug_html(index_response.text, f"racingpost_index_{date}")

        index_parser = HTMLParser(index_response.text)
        links = index_parser.css('a[data-test-selector^="RC-meetingItem__link_race"]')
        race_card_urls = [link.attributes["href"] for link in links]

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
            number = int(number_str) if number_str and number_str.isdigit() else 0
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

# ----------------------------------------
# MASTER ORCHESTRATOR
# ----------------------------------------

async def run_discovery(target_date: str):
    print(f"🚀 Running Discovery for {target_date}...")
    
    # Initialize all adapters
    adapter_classes = [
        AtTheRacesAdapter,
        AtTheRacesGreyhoundAdapter,
        BoyleSportsAdapter,
        SportingLifeAdapter,
        SkySportsAdapter,
        RacingPostB2BAdapter,
        StandardbredCanadaAdapter,
        TabAdapter,
        BetfairDataScientistAdapter,
        EquibaseAdapter,
        TwinSpiresAdapter,
            OddscheckerAdapter,
        TimeformAdapter,
        RacingPostAdapter,
]
    
    adapters = []
    for cls in adapter_classes:
        try:
            adapters.append(cls())
        except Exception as e:
            print(f"Failed to init {cls.__name__}: {e}")

    all_races_raw = []
    
    async def fetch_one(a):
        try:
            races = await a.get_races(target_date)
            return races
        except Exception as e:
            print(f"Error fetching from {a.source_name}: {e}")
            return []

    results = await asyncio.gather(*[fetch_one(a) for a in adapters])
    for r_list in results:
        all_races_raw.extend(r_list)

    print(f"Fetched {len(all_races_raw)} total races.")
    
    # Deduplicate
    race_map = {}
    for race in all_races_raw:
        venue = normalize_venue_name(race.venue)
        # Use Venue + Race Number + Date as stable key
        key = f"{venue.lower()}|{race.race_number}|{race.start_time.strftime('%Y%m%d')}"
        
        if key not in race_map:
            race_map[key] = race
        else:
            existing = race_map[key]
            # Merge runners/odds
            for nr in race.runners:
                er = next((r for r in existing.runners if r.number == nr.number), None)
                if er:
                    er.odds.update(nr.odds)
                    if not er.win_odds and nr.win_odds:
                        er.win_odds = nr.win_odds
                else:
                    existing.runners.append(nr)
            
            # Update source
            sources = set((existing.source or "").split(", "))
            sources.add(race.source or "Unknown")
            existing.source = ", ".join(sorted(list(filter(None, sources))))
    
    unique_races = list(race_map.values())
    print(f"Unique races: {len(unique_races)}")

    # Analyze
    analyzer = SimplySuccessAnalyzer()
    result = analyzer.qualify_races(unique_races)
    qualified = result.get("races", [])

    # Generate Grid & Goldmine
    grid = generate_summary_grid(qualified, all_races=unique_races)
    print(grid)
    with open("summary_grid.txt", "w") as f: f.write(grid)
    
    gm_report = generate_goldmine_report(qualified, all_races=unique_races)
    with open("goldmine_report.txt", "w") as f: f.write(gm_report)
    
    # Save qualified races to JSON
    report_data = {
        "races": [r.model_dump(mode='json') for r in qualified],
        "analysis_metadata": result.get("criteria", {}),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open("qualified_races.json", "w") as f:
        json.dump(report_data, f, indent=4)

    # Shutdown
    for a in adapters: await a.close()

async def main_all_in_one():
    parser = argparse.ArgumentParser(description="Fortuna All-In-One")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--monitor", action="store_true", help="Run in monitor mode")
    parser.add_argument("--once", action="store_true", help="Run monitor once")
    args = parser.parse_args()

    target_date = args.date or datetime.now().strftime("%Y-%m-%d")

    if args.monitor:
        monitor = FavoriteToPlaceMonitor(target_date=target_date)
        if args.once: await monitor.run_once()
        else: await monitor.run_continuous()
    else:
        await run_discovery(target_date)

if __name__ == "__main__":
    if os.getenv("DEBUG_SNAPSHOTS"):
        os.makedirs("debug_snapshots", exist_ok=True)
    
    try:
        asyncio.run(main_all_in_one())
    except KeyboardInterrupt:
        pass