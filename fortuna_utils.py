import re
import structlog
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union, Final
from zoneinfo import ZoneInfo
from functools import lru_cache

EASTERN = ZoneInfo("America/New_York")

MAX_VALID_ODDS: Final[float] = 1000.0
MIN_VALID_ODDS: Final[float] = 1.01
# 2.75 is chosen as a conservative implied place-odds estimate (approx 7/4)
# for runners where real odds are missing during discovery.
DEFAULT_ODDS_FALLBACK: Final[float] = 2.75
COMMON_PLACEHOLDERS: Final[set] = set()

VENUE_MAP = {
"ABU DHABI": "Abu Dhabi",
"LUDLOW": "Ludlow",
"SOUTHWELL": "Southwell",
"PUNCHESTOWN": "Punchestown",
"MARKET RASEN": "Market Rasen",
"NEWBURY": "Newbury",
"TAUNTON": "Taunton",
"HEREFORD": "Hereford",
"WOLVERHAMPTON": "Wolverhampton",
"AQU": "Aqueduct",
"AQUEDUCT": "Aqueduct",
"BEL": "Belmont Park",
"BELMONT": "Belmont Park",
"BELMONT PARK": "Belmont Park",
"ARGENTAN": "Argentan",
"ASCOT": "Ascot",
"AYR": "Ayr",
"BAHRAIN": "Bahrain",
"BANGOR ON DEE": "Bangor-on-Dee",
"CATTERICK": "Catterick",
"CATTERICK BRIDGE": "Catterick",
"CD": "Churchill Downs",
"CHURCHILL DOWNS": "Churchill Downs",
"CT": "Charles Town",
"CENTRAL PARK": "Central Park",
"CHELMSFORD": "Chelmsford",
"CHELMSFORD CITY": "Chelmsford",
"CURRAGH": "Curragh",
"DEAUVILLE": "Deauville",
"DED": "Delta Downs",
"DEL MAR": "Del Mar",
"DMR": "Del Mar",
"DELTA DOWNS": "Delta Downs",
"Dover Downs": "Dover Downs",
"DONCASTER": "Doncaster",
"DOVER DOWNS": "Dover Downs",
"DOWN ROYAL": "Down Royal",
"DUNDALK": "Dundalk",
"DUNSTALL PARK": "Wolverhampton",
"EPSOM": "Epsom",
"EPSOM DOWNS": "Epsom",
"FG": "Fair Grounds",
"FAIR GROUNDS": "Fair Grounds",
"FLAMBORO": "Flamboro",
"FLAMBORO DOWNS": "Flamboro",
"WESTERN FAIR DISTRICT": "Western Fair",
"WESTERN FAIR RACEWAY": "Western Fair",
"LONDON": "Western Fair",
"WESTERN FAIR": "Western Fair",
"WOODBINE MOHAWK PARK": "Mohawk",
"WOODBINE MOHAWK": "Mohawk",
"RIDEAU CARLETON": "Rideau",
"RIDEAU CARLETON RACEWAY": "Rideau",
"RIDEAU": "Rideau",
"WOODBINE RACETRACK": "Woodbine",
"DRESDEN RACEWAY": "Dresden",
"CLINTON RACEWAY": "Clinton",
"HANOVER RACEWAY": "Hanover",
"GRAND RIVER RACEWAY": "Grand River",
"GEORGETOWN": "Grand River",
"KAWARTHA DOWNS": "Kawartha",
"KEE": "Keeneland",
"KEENELAND": "Keeneland",
"LEAMINGTON RACEWAY": "Leamington",
"FONTWELL": "Fontwell Park",
"FONTWELL PARK": "Fontwell Park",
"GREAT YARMOUTH": "Great Yarmouth",
"GG": "Golden Gate Fields",
"GOLDEN GATE": "Golden Gate Fields",
"GP": "Gulfstream Park",
"GULFSTREAM": "Gulfstream Park",
"GULFSTREAM PARK": "Gulfstream Park",
"HAYDOCK": "Haydock Park",
"HAYDOCK PARK": "Haydock Park",
"HOOSIER PARK": "Hoosier Park",
"HOVE": "Hove",
"KEMPTON": "Kempton Park",
"KEMPTON PARK": "Kempton Park",
"LRL": "Laurel Park",
"LAUREL PARK": "Laurel Park",
"LINGFIELD": "Lingfield Park",
"LINGFIELD PARK": "Lingfield Park",
"LOS ALAMITOS": "Los Alamitos",
"MARONAS": "Maronas",
"MEADOWLANDS": "Meadowlands",
"MEYDAN": "Meydan",
"MTH": "Monmouth Park",
"MONMOUTH PARK": "Monmouth Park",
"MIAMI VALLEY": "Miami Valley",
"MIAMI VALLEY RACEWAY": "Miami Valley",
"MVR": "Mahoning Valley",
"MOHAWK": "Mohawk",
"MOHAWK PARK": "Mohawk",
"MUSSELBURGH": "Musselburgh",
"NORTHFIELD PARK": "Northfield Park",
"NAAS": "Naas",
"NEWCASTLE": "Newcastle",
"NEWMARKET": "Newmarket",
"OXFORD": "Oxford",
"PAU": "Pau",
"OP": "Oaklawn Park",
"PEN": "Penn National",
"PIM": "Pimlico",
"PIMLICO": "Pimlico",
"PRX": "Parx Racing",
"PARX RACING": "Parx Racing",
"POCONO DOWNS": "Pocono Downs",
"SAR": "Saratoga",
"SARATOGA": "Saratoga",
"SAM HOUSTON": "Sam Houston",
"SAM HOUSTON RACE PARK": "Sam Houston",
"HOU": "Sam Houston",
"SANDOWN": "Sandown Park",
"SANDOWN PARK": "Sandown Park",
"SA": "Santa Anita",
"SANTA ANITA": "Santa Anita",
"SARATOGA HARNESS": "Saratoga Harness",
"SCIOTO DOWNS": "Scioto Downs",
"SHEFFIELD": "Sheffield",
"STRATFORD": "Stratford-on-Avon",
"SUN": "Sunland Park",
"SUNLAND PARK": "Sunland Park",
"TAM": "Tampa Bay Downs",
"TAMPA BAY DOWNS": "Tampa Bay Downs",
"THURLES": "Thurles",
"TP": "Turfway Park",
"TUP": "Turf Paradise",
"TURF PARADISE": "Turf Paradise",
"TURFFONTEIN": "Turffontein",
"UTTOXETER": "Uttoxeter",
"VINCENNES": "Vincennes",
"WARWICK": "Warwick",
"WETHERBY": "Wetherby",
"WO": "Woodbine",
"WOODBINE": "Woodbine",
"YARMOUTH": "Great Yarmouth",
"YONKERS": "Yonkers",
"YONKERS RACEWAY": "Yonkers",
"HARLOW": "Harlow",
"KILKENNY": "Kilkenny",
"MONMORE": "Monmore",
"MONMORE GREEN": "Monmore",
"ROMFORD": "Romford",
"TOWCESTER": "Towcester",
"NOTTINGHAM": "Nottingham",
"VALLEY": "Valley",
"CRAYFORD": "Crayford",
"SUNDERLAND": "Sunderland",
"PERRY BARR": "Perry Barr",
"CAGNES SUR MER": "Cagnes Sur Mer",
"CAGNES-SUR-MER": "Cagnes Sur Mer",
"CAGNES SUR MER MIDI": "Cagnes Sur Mer",
"CAGNES SUR MER QUEYRAS": "Cagnes Sur Mer",
"CAGNES SUR MER MENTHE POIVREE": "Cagnes Sur Mer",
"CAGNES SUR MER ANTOINE CAPOZZI": "Cagnes Sur Mer",
"LYON": "Lyon",
"LYON LA SOIE": "Lyon",
"PORNICHET": "Pornichet",
"LAVAL": "Laval",
"SCOTTSVILLE": "Scottsville",
"MAHONING VALLEY": "Mahoning Valley",
"TURFWAY": "Turfway Park",
"TURFWAY PARK": "Turfway Park",
"SUNLAND": "Sunland Park",
"RP": "Remington Park",
"REMINGTON": "Remington Park",
"REMINGTON PARK": "Remington Park",
}

DRF_VENUE_MAP: Final[Dict[str, str]] = {
    "AQU": "Aqueduct",
    "BEL": "Belmont Park",
    "SAR": "Saratoga",
    "GP": "Gulfstream Park",
    "LRL": "Laurel Park",
    "PIM": "Pimlico",
    "TAM": "Tampa Bay Downs",
    "TP": "Turfway Park",
    "OP": "Oaklawn Park",
    "CD": "Churchill Downs",
    "SA": "Santa Anita",
    "DMR": "Del Mar",
    "GG": "Golden Gate Fields",
    "MTH": "Monmouth Park",
    "PRX": "Parx Racing",
    "KEE": "Keeneland",
    "CT": "Charles Town",
    "DED": "Delta Downs",
    "FG": "Fair Grounds",
    "RP": "Remington Park",
    "HOU": "Sam Houston",
    "FON": "Fonner Park",
    "MVR": "Mahoning Valley",
    "LA": "Los Alamitos",
    "LAD": "Louisiana Downs",
}

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
"SPEEDWAY", "DOWNS", "PARK", "HARNESS", " STANDARDBRED", "FORM GUIDE", "FULL FIELDS",
"SUZUKI", "MATCHBOOK", "STALLION", "GRADUATION", "QUALIFIER", "NOVICES", "HANDICAP"
]

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

DISCIPLINE_KEYWORDS: Final[Dict[str, List[str]]] = {
    "Harness": ["harness", "trotter", "pacer", "standardbred", "trot", "pace"],
    "Greyhound": ["greyhound", "dog", "dogs"],
    "Quarter Horse": ["quarter horse", "quarterhorse"],
}

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
    return clean_text(txt() if callable(txt) else str(txt))

@lru_cache(maxsize=1024)
def get_canonical_venue(name: Optional[str]) -> str:
    """
    Returns a URL-safe, lowercase, alphanumeric-only version of the venue name.
    Used for generating consistent race IDs.
    """
    if not name: return "unknown"
    norm = normalize_venue_name(name)
    # Remove everything in parentheses (extra safety)
    norm = re.sub(r"[\(\[（].*?[\)\]）]", "", norm)
    # Remove special characters, lowercase, strip
    res = re.sub(r"[^a-z0-9]", "", norm.lower())
    return res or "unknown"

def normalize_venue_name(name: Optional[str]) -> str:
    """
    Normalizes a racecourse name to a standard format.
    Aggressively strips race names, sponsorships, and country noise.
    """
    if not name:
        return "Unknown"

    # 1. Initial Cleaning
    name = str(name).replace("-", " ").replace("_", " ")
    name = re.sub(r"[\(\[\uff08].*?[\)\]\uff09]", " ", name)

    # Item 11: Standardize/Strip country prefixes (NYRA etc.)
    # Australian venues often prefixed with 'Au ' or 'Aus '
    # French venues often prefixed with 'fr'
    # NZ venues often prefixed with 'Nz '
    # SA venues often prefixed with 'Sa '
    name = re.sub(r"^(?:Au|Aus|fr|Nz|Sa|Aus|Zp|Jpn?|Hk|Uae)\s+", "", name, flags=re.I)
    # Special case for NYRA 'frfontainebleau'
    name = re.sub(r"^fr([a-z])", r"\1", name, flags=re.I)

    cleaned = clean_text(name)
    if not cleaned:
        return "Unknown"

    # 2. Aggressive Race/Meeting Name Stripping
    upper_name = cleaned.upper()
    earliest_idx = len(cleaned)
    for kw in RACING_KEYWORDS:
        idx = upper_name.find(" " + kw)
        if idx != -1:
            earliest_idx = min(earliest_idx, idx)

    # Strip from first digit preceded by space (e.g. "Ludlow 13:30")
    digit_match = re.search(r"\s\d", cleaned)
    if digit_match:
        earliest_idx = min(earliest_idx, digit_match.start())

    track_part = cleaned[:earliest_idx].strip()
    if not track_part:
        track_part = cleaned

    # Handle repetition (e.g., "Bahrain Bahrain" -> "Bahrain")
    words = track_part.split()
    if len(words) > 1 and words[0].lower() == words[1].lower():
        track_part = words[0]

    upper_track = track_part.upper()

    # 3. High-Confidence Mapping — direct match
    if upper_track in VENUE_MAP:
        return VENUE_MAP[upper_track]

    # 4. Word-boundary prefix match (longest known venue first)
    track_words = upper_track.split()
    for end in range(len(track_words), 0, -1):
        candidate = " ".join(track_words[:end])
        if candidate in VENUE_MAP:
            return VENUE_MAP[candidate]

    # 5. Same approach on the full (unstripped) cleaned name
    full_words = upper_name.split()
    for end in range(min(len(full_words), 4), 0, -1):  # max 4-word venue names
        candidate = " ".join(full_words[:end])
        if candidate in VENUE_MAP:
            return VENUE_MAP[candidate]

    # 6. Legacy prefix match (substring-based, for backward compat)
    for known_track in sorted(VENUE_MAP.keys(), key=len, reverse=True):
        if upper_name.startswith(known_track + " ") or upper_name == known_track:
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
        int_match = re.match(r"^(\d+)$", s)
        if int_match:
            val = int(int_match.group(1))
            if 1 <= val <= 50:
                return float(val + 1)

    except Exception: pass
    return None

class SmartOddsExtractor:
    """
    Advanced heuristics for extracting odds from noisy HTML or text.
    Scans for various patterns and returns the first plausible odds found.
    """
    @staticmethod
    def extract_from_text(text: str) -> Optional[float]:
        if not text: return None
        keyword_match = re.search(r"(?:odds|line|m/l|ml)[:\s]+(\d+[/-]\d+|\d+\.\d+)", text, re.I)
        if keyword_match:
            if val := parse_odds_to_decimal(keyword_match.group(1)):
                return val

        decimals = re.findall(r"(\d+\.\d+)", text)
        for d in decimals:
            val = float(d)
            if MIN_VALID_ODDS <= val < MAX_VALID_ODDS: return round(val, 2)

        fractions = re.findall(r"(\d+)\s*[/\-]\s*(\d+)", text)
        for num, den in fractions:
            n, d = int(num), int(den)
            if d > 0 and (n/d) > 0.1: return round((n / d) + 1.0, 2)

        return None

    @staticmethod
    def extract_from_node(node: Any) -> Optional[float]:
        """Scans a selectolax node for odds using multiple strategies."""
        if hasattr(node, 'text'):
            if val := SmartOddsExtractor.extract_from_text(node_text(node)):
                return val

        if hasattr(node, 'attributes'):
            for attr in ["data-odds", "data-price", "data-bestprice", "title"]:
                if val_str := node.attributes.get(attr):
                    if val := parse_odds_to_decimal(val_str):
                        return val

        return None

def is_placeholder_odds(value: Optional[Union[float, Decimal]]) -> bool:
    """Detects if odds value is a known placeholder or default."""
    if value is None:
        return True
    try:
        val_float = round(float(value), 2)
        return val_float in COMMON_PLACEHOLDERS
    except (ValueError, TypeError):
        return True

def is_valid_odds(odds: Any) -> bool:
    if odds is None: return False
    try:
        odds_float = float(odds)
        if not (MIN_VALID_ODDS <= odds_float < MAX_VALID_ODDS):
            return False
        return not is_placeholder_odds(odds_float)
    except Exception: return False

def scrape_available_bets(html_content: str) -> List[str]:
    """Extract exotic bet types mentioned in HTML content."""
    if not html_content: return []

    available_bets: List[str] = []
    html_lower = html_content.lower()

    for kw, bet_name in BET_TYPE_KEYWORDS.items():
        # Handle multi-word keywords properly (e.g., "pick 3", "daily double")
        # Split on spaces, escape each word, join with \s+ for flexible whitespace matching
        words = kw.split()
        pattern = r"\b" + r"\s+".join(re.escape(w) for w in words) + r"\b"

        if re.search(pattern, html_lower) and bet_name not in available_bets:
            available_bets.append(bet_name)

    return available_bets

def detect_discipline(html_content: str) -> str:
    if not html_content: return "Thoroughbred"
    html_lower = html_content.lower()
    for disc, keywords in DISCIPLINE_KEYWORDS.items():
        if any(kw in html_lower for kw in keywords): return disc
    return "Thoroughbred"

def now_eastern() -> datetime:
    """Returns the current time in US Eastern Time."""
    return datetime.now(EASTERN)

def to_eastern(dt: datetime) -> datetime:
    """Converts a datetime object to US Eastern Time."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=EASTERN)
    return dt.astimezone(EASTERN)

def ensure_eastern(dt: datetime) -> datetime:
    """Ensures datetime is timezone-aware and in Eastern time."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=EASTERN)
    # Note: identity check assumes ZoneInfo('America/New_York') is globally cached/singleton.
    if dt.tzinfo is not EASTERN:
        try:
            return dt.astimezone(EASTERN)
        except Exception:
            return dt.replace(tzinfo=EASTERN)
    return dt

def get_places_paid(field_size: int, is_handicap: Optional[bool] = None) -> int:
    """
    Return number of paid places for a given field size.
    UK Industry Standard Rules:
    - 1-4 runners: 1 place (win only)
    - 5-7 runners: 2 places
    - 8-15 runners: 3 places
    - 16+ runners (Non-Handicap): 3 places
    - 16+ runners (Handicap): 4 places
    """
    if field_size <= 4:
        return 1
    if field_size <= 7:
        return 2
    if field_size <= 15:
        return 3
    # 16+ runners
    if is_handicap is True:
        return 4
    return 3
