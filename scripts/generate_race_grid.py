import json
import os
import re
import glob
import sys
import argparse
from datetime import datetime, timedelta
import zoneinfo

# Ensure root is in path for fortuna_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from fortuna_utils import get_canonical_venue, detect_discipline, parse_odds_to_decimal
except ImportError:
    # This might happen if PYTHONPATH is not set correctly
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from fortuna_utils import get_canonical_venue, detect_discipline, parse_odds_to_decimal

def get_tz_for_country(country_name, location=""):
    """Returns a ZoneInfo object for the given country name."""
    c = country_name.lower()
    if "united states" in c or "usa" in c:
        return zoneinfo.ZoneInfo("America/New_York")
    if "england" in c or "wales" in c or "united kingdom" in c or "scotland" in c:
        return zoneinfo.ZoneInfo("Europe/London")
    if "eire" in c or "ireland" in c:
        return zoneinfo.ZoneInfo("Europe/Dublin")
    if "france" in c:
        return zoneinfo.ZoneInfo("Europe/Paris")
    if "south africa" in c or "saf" in c:
        return zoneinfo.ZoneInfo("Africa/Johannesburg")
    if "turkey" in c:
        return zoneinfo.ZoneInfo("Europe/Istanbul")
    if "australia" in c:
        return zoneinfo.ZoneInfo("Australia/Sydney")
    if "new zealand" in c:
        return zoneinfo.ZoneInfo("Pacific/Auckland")
    if "japan" in c:
        return zoneinfo.ZoneInfo("Asia/Tokyo")
    if "hong kong" in c:
        return zoneinfo.ZoneInfo("Asia/Hong_Kong")
    if "south korea" in c:
        return zoneinfo.ZoneInfo("Asia/Seoul")

    return zoneinfo.ZoneInfo("UTC")

def parse_distance_to_miles(dist_str):
    """Standardizes race distances to Miles in decimal format (3 decimal places)."""
    if not dist_str or dist_str == "?": return "?"
    s = str(dist_str).lower().strip()

    total_yards = 0.0
    found = False

    # Try meters first if it ends in 'm' and is a large number
    m_match = re.search(r'^(\d+)\s*m$', s)
    if m_match:
        val = float(m_match.group(1))
        if val > 100: # 1000m, 1200m etc
            total_yards = val * 1.09361
            found = True

    if not found:
        # Standard parts
        parts = re.findall(r'(\d+\.?\d*)\s*([mfyk])', s)
        for val, unit in parts:
            try:
                v = float(val)
                if unit == 'm':
                    if v < 10: # Miles are rarely > 4
                        total_yards += v * 1760
                    else:
                        total_yards += v * 1.09361
                    found = True
                elif unit == 'f':
                    total_yards += v * 220
                    found = True
                elif unit == 'y':
                    total_yards += v
                    found = True
                elif unit == 'k':
                    total_yards += v * 1093.61
                    found = True
            except: continue

    if not found:
        return dist_str

    if total_yards == 0: return dist_str
    miles = total_yards / 1760.0
    return f"{miles:.3f}"

def format_purse(purse_str):
    """Formats purse string to K format (e.g. 50000 -> 50K)."""
    if not purse_str or purse_str == "?": return "?"
    try:
        # Strip currency and commas
        val = re.sub(r'[^\d.]', '', str(purse_str))
        if not val: return purse_str
        f_val = float(val)
        if f_val >= 1000:
            return f"{int(f_val/1000)}K"
        return str(int(f_val))
    except:
        return purse_str

def parse_rpb2b_json(filepath):
    """Parses US race data from RPB2B JSON export."""
    if not os.path.exists(filepath): return []
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except: return []

    et_tz = zoneinfo.ZoneInfo("America/New_York")
    races = []

    meetings = data.get('meetings') if isinstance(data, dict) else data
    if not meetings: return []

    for meeting in meetings:
        location = meeting.get('name') or meeting.get('CourseName', 'Unknown')
        disc = meeting.get('Discipline', 'T')[0].upper()
        for race in meeting.get("races", []):
            try:
                dt_utc = datetime.fromisoformat(race.get("datetimeUtc").replace('Z', '+00:00'))
                dt_et = dt_utc.astimezone(et_tz)
                time_val = dt_et
            except:
                time_val = None

            purse = race.get("purse") or race.get("Prize", "?")

            races.append({
                "DateTime": time_val,
                "PostTime": time_val.strftime('%H:%M') if time_val else "Unknown",
                "FieldSize": str(race.get("numberOfRunners") or race.get("Runners", "?")),
                "Distance": parse_distance_to_miles(race.get("distance") or race.get("Distance", "?")),
                "RaceNum": str(race.get("raceNumber") or race.get("RaceNo", "?")),
                "Location": location,
                "Discipline": disc,
                "Purse": format_purse(purse),
                "ML": "?"
            })
    return races

def parse_sl_hard(filepath):
    """Parses international race data from Sporting Life HTML using __NEXT_DATA__."""
    if not os.path.exists(filepath): return []
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except: return []

    et_tz = zoneinfo.ZoneInfo("America/New_York")

    json_match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', content, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            page_props = data.get('props', {}).get('pageProps', {})
            meetings_list = page_props.get('meetings', []) or page_props.get('races', [])

            if not meetings_list and 'racecard' in page_props:
                meetings_list = [page_props['racecard'].get('meeting', {})]

            all_parsed_races = []
            for meeting in meetings_list:
                loc = meeting.get('name') or meeting.get('course_name') or "Unknown"
                country = meeting.get('country_name') or meeting.get('long_name') or "Unknown"
                local_tz = get_tz_for_country(country, loc)

                races_to_parse = meeting.get('races', [])
                if not races_to_parse and 'races' in page_props:
                    races_to_parse = page_props['races']

                for r in races_to_parse:
                    time_str = r.get('time')
                    date_str = r.get('date') or page_props.get('date') or datetime.now().strftime("%Y-%m-%d")

                    purse = r.get('purse') or r.get('prize') or "?"

                    runners = r.get('runners', [])
                    ml_str = "?"
                    if runners:
                        try:
                            fps = []
                            for runner in runners:
                                p = runner.get('forecast_price') or runner.get('odds')
                                if p: fps.append(p)
                            if fps:
                                fps.sort(key=lambda x: parse_odds_to_decimal(x) or 999)
                                if len(fps) >= 2:
                                    ml_str = f"{fps[0]}, {fps[1]}"
                                else:
                                    ml_str = f"{fps[0]}"
                        except: pass

                    try:
                        dt_local = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=local_tz)
                        dt_et = dt_local.astimezone(et_tz)
                        time_val = dt_et
                    except:
                        time_val = None

                    all_parsed_races.append({
                        "DateTime": time_val,
                        "PostTime": time_val.strftime('%H:%M') if time_val else time_str,
                        "FieldSize": str(r.get('ride_count') or r.get('number_of_runners') or "?"),
                        "Distance": parse_distance_to_miles(r.get('distance') or "?"),
                        "RaceNum": str(r.get('race_number') or "?"),
                        "Location": loc,
                        "Discipline": detect_discipline(str(r))[0].upper(),
                        "Purse": format_purse(purse),
                        "ML": ml_str
                    })
            if all_parsed_races: return all_parsed_races
        except: pass

    # Fallback to regex if JSON failed
    time_matches = list(re.finditer(r'"time":"([^"]+)"', content))
    races = []
    for tm in time_matches:
        time_str = tm.group(1)
        pos = tm.start()
        look_back = content[max(0, pos-1000):pos]
        course_matches = re.findall(r'"course_name":"([^"]+)"', look_back)
        location = course_matches[-1] if course_matches else "Unknown"
        country_match = re.search(r'"long_name":"([^"]+)"', look_back)
        country = country_match.group(1) if country_match else "Unknown"
        date_match = re.search(r'"date":"([^"]+)"', look_back)
        date_str = date_match.group(1) if date_match else "2026-03-26"

        look_forward = content[pos:pos+1500]
        runners_match = re.search(r'"ride_count":(\d+)', look_forward)
        dist_match = re.search(r'"distance":"([^"]+)"', look_forward)
        purse_match = re.search(r'"purse":"([^"]+)"', look_forward)

        try:
            local_tz = get_tz_for_country(country, location)
            dt_local = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=local_tz)
            dt_et = dt_local.astimezone(et_tz)
            time_val = dt_et
        except:
            time_val = None

        races.append({
            "DateTime": time_val,
            "PostTime": time_val.strftime('%H:%M') if time_val else time_str,
            "FieldSize": str(runners_match.group(1)) if runners_match else "?",
            "Distance": parse_distance_to_miles(dist_match.group(1) if dist_match else "?"),
            "RaceNum": "?",
            "Location": location,
            "Discipline": detect_discipline(look_forward[:2000])[0].upper(),
            "Purse": format_purse(purse_match.group(1) if purse_match else "?"),
            "ML": "?"
        })
    return races

def parse_equibase_html(filepath):
    """Parses US race data from Equibase Summary HTML."""
    if not os.path.exists(filepath): return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except: return []

    et_tz = zoneinfo.ZoneInfo("America/New_York")
    races = []
    venue_match = re.search(r'<h3>([^<]+)</h3>', content)
    location = venue_match.group(1).strip() if venue_match else "Unknown"

    tables = re.findall(r'<table[^>]*>(.*?)</table>', content, re.DOTALL)
    for table in tables:
        header_match = re.search(r'Race (\d+) - ([^<]+)', table)
        if not header_match: continue
        race_num = header_match.group(1)
        date_str = header_match.group(2).strip()

        purse_match = re.search(r'Purse:\s*\$([\d,]+)', table)
        purse = purse_match.group(1).replace(',', '') if purse_match else "?"

        tbody_match = re.search(r'<tbody>(.*?)</tbody>', table, re.DOTALL)
        field_size = 0
        if tbody_match:
            field_size = len(re.findall(r'<tr>', tbody_match.group(1)))

        try:
            dt = datetime.strptime(date_str, "%B %d, %Y").replace(hour=12, minute=0, tzinfo=et_tz)
            time_val = dt
        except:
            time_val = None

        races.append({
            "DateTime": time_val,
            "PostTime": time_val.strftime('%H:%M') if time_val else "12:00",
            "FieldSize": str(field_size) if field_size > 0 else "?",
            "Distance": "?",
            "RaceNum": race_num,
            "Location": location,
            "Discipline": "T",
            "Purse": format_purse(purse),
            "ML": "?"
        })
    return races

def parse_trackinfo_csv(filepath):
    """Parses race data from TrackInfo_Entries.csv."""
    if not os.path.exists(filepath): return []
    try:
        import pandas as pd
        df = pd.read_csv(filepath)
        races = []
        for _, row in df.iterrows():
            races.append({
                "DateTime": None,
                "PostTime": str(row.get('PostTime', '00:00')),
                "FieldSize": str(row.get('FieldSize', '?')),
                "Distance": parse_distance_to_miles(str(row.get('Distance', '?'))),
                "RaceNum": str(row.get('RaceNum', '?')),
                "Location": str(row.get('Location', 'Unknown')),
                "Discipline": "T",
                "Purse": format_purse(row.get('Purse', '?')),
                "ML": str(row.get('ML', '?'))
            })
        return races
    except: return []

def parse_drf_html(filepath, target_date):
    """Parses US race data from DRF HTML."""
    if not os.path.exists(filepath): return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except: return []

    races = []
    venue_match = re.search(r'<h1[^>]*>([^<]+)</h1>', content)
    location = venue_match.group(1).strip() if venue_match else "Unknown"
    race_blocks = re.findall(r'<div[^>]*class="[^"]*race-card[^"]*"[^>]*>(.*?)</div>', content, re.DOTALL)
    if not race_blocks:
        race_blocks = re.findall(r'<div[^>]*class="[^"]*race-header[^"]*"[^>]*>(.*?)</div>', content, re.DOTALL)

    et_tz = zoneinfo.ZoneInfo("America/New_York")
    for i, block in enumerate(race_blocks):
        race_num = str(i + 1)
        num_match = re.search(r'Race\s+(\d+)', block)
        if num_match: race_num = num_match.group(1)
        time_match = re.search(r'(\d{1,2}:\d{2}\s*[APM]*)', block)
        time_str = time_match.group(1).strip() if time_match else "12:00"
        runners = re.findall(r'<tr[^>]*class="[^"]*horse-row[^"]*"', block)
        field_size = len(runners)

        purse_match = re.search(r'Purse:\s*\$([\d,]+)', block)
        purse = purse_match.group(1).replace(',', '') if purse_match else "?"

        time_val = None
        try:
            dt_str = f"{target_date} {time_str}"
            if "PM" in time_str.upper() and not time_str.startswith("12"):
                dt = datetime.strptime(dt_str.replace("PM", "").replace("AM", "").strip(), "%Y-%m-%d %H:%M")
                dt = dt.replace(hour=dt.hour + 12)
            else:
                dt = datetime.strptime(dt_str.replace("PM", "").replace("AM", "").strip(), "%Y-%m-%d %H:%M")
            time_val = dt.replace(tzinfo=et_tz)
        except: pass

        races.append({
            "DateTime": time_val,
            "PostTime": time_val.strftime('%H:%M') if time_val else time_str,
            "FieldSize": str(field_size) if field_size > 0 else "?",
            "Distance": "?",
            "RaceNum": race_num,
            "Location": location,
            "Discipline": "T",
            "Purse": format_purse(purse),
            "ML": "?"
        })
    return races

def parse_ras_json(filepath):
    """Parses international race data from Racing & Sports JSON."""
    if not os.path.exists(filepath): return []
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except: return []

    races = []
    et_tz = zoneinfo.ZoneInfo("America/New_York")
    if isinstance(data, list):
        for disc_item in data:
            disc_code = disc_item.get("Discipline", "T")[0].upper()
            countries = disc_item.get("Countries") or disc_item.get("countries", [])
            for country in countries:
                country_name = country.get("Country") or country.get("CountryName", "Unknown")
                local_tz = get_tz_for_country(country_name)
                meetings = country.get("Meetings") or country.get("meetings", [])
                for meeting in meetings:
                    location = meeting.get("Meeting") or meeting.get("Course", "Unknown")
                    race_list = meeting.get("Races") or meeting.get("races", [])
                    for race in race_list:
                        time_str = race.get("StartTime") or race.get("startTime", "00:00")
                        date_str = race.get("Date") or race.get("date", "")
                        time_val = None
                        try:
                            if len(time_str) == 5:
                                dt_local = datetime.strptime(f"{date_str[:10]} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=local_tz)
                            else:
                                dt_local = datetime.fromisoformat(time_str).astimezone(local_tz)
                            time_val = dt_local.astimezone(et_tz)
                        except: pass
                        races.append({
                            "DateTime": time_val,
                            "PostTime": time_val.strftime('%H:%M') if time_val else time_str,
                            "FieldSize": str(race.get("Runners") or race.get("runners", "?")),
                            "Distance": parse_distance_to_miles(race.get("Distance") or race.get("distance", "?")),
                            "RaceNum": str(race.get("RaceNo") or race.get("raceNumber", "?")),
                            "Location": location,
                            "Discipline": disc_code,
                            "Purse": "?",
                            "ML": "?"
                        })
    return races

def parse_jra_html(filepath, target_date):
    """Parses Japan race data from JRA HTML."""
    if not os.path.exists(filepath): return []
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except: return []
    races = []
    venue = "Japan"
    venue_match = re.search(r'class="course">([^<]+)', content)
    if venue_match: venue = venue_match.group(1).strip()
    tables = re.findall(r'<table[^>]*class="[^"]*race_table[^"]*"[^>]*>(.*?)</table>', content, re.DOTALL)
    et_tz = zoneinfo.ZoneInfo("America/New_York")
    jp_tz = zoneinfo.ZoneInfo("Asia/Tokyo")
    for i, table in enumerate(tables):
        race_num = str(i + 1)
        time_match = re.search(r'(\d{1,2}:\d{2})', content)
        time_str = time_match.group(1) if time_match else "00:00"
        runners = re.findall(r'<tr>\s*<td[^>]*class="num"', table)
        field_size = len(runners)
        time_val = None
        try:
            dt_jp = datetime.strptime(f"{target_date} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=jp_tz)
            time_val = dt_jp.astimezone(et_tz)
        except: pass
        races.append({
            "DateTime": time_val,
            "PostTime": time_val.strftime('%H:%M') if time_val else time_str,
            "FieldSize": str(field_size) if field_size > 0 else "?",
            "Distance": "?",
            "RaceNum": race_num,
            "Location": venue,
            "Discipline": "T",
            "Purse": "?",
            "ML": "?"
        })
    return races

def parse_tf_html(filepath, target_date):
    """Parses international race data from Timeform HTML."""
    if not os.path.exists(filepath): return []
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except: return []
    races = []
    venue_match = re.search(r'class="rp-course-name">([^<]+)', content)
    if not venue_match:
        venue_match = re.search(r'class="course-name">([^<]+)', content)
    location = venue_match.group(1).strip() if venue_match else "Unknown"
    time_match = re.search(r'class="rp-race-time">(\d{1,2}:\d{2})', content)
    time_str = time_match.group(1) if time_match else "00:00"
    runners = re.findall(r'class="rp-horse-name"', content)
    field_size = len(runners)
    et_tz = zoneinfo.ZoneInfo("America/New_York")
    uk_tz = zoneinfo.ZoneInfo("Europe/London")
    time_val = None
    try:
        dt_uk = datetime.strptime(f"{target_date} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=uk_tz)
        time_val = dt_uk.astimezone(et_tz)
    except: pass
    if location != "Unknown":
        races.append({
            "DateTime": time_val,
            "PostTime": time_val.strftime('%H:%M') if time_val else time_str,
            "FieldSize": str(field_size) if field_size > 0 else "?",
            "Distance": "?",
            "RaceNum": "?",
            "Location": location,
            "Discipline": "T",
            "Purse": "?",
            "ML": "?"
        })
    return races

def parse_skysports_html(filepath, target_date):
    """Parses international race data from Sky Sports HTML."""
    if not os.path.exists(filepath): return []
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except: return []
    races = []
    et_tz = zoneinfo.ZoneInfo("America/New_York")
    uk_tz = zoneinfo.ZoneInfo("Europe/London")
    sections = re.split(r'class="sdc-site-concertina-block__header"', content)[1:]
    for section in sections:
        venue_match = re.search(r'class="sdc-site-concertina-block__title"[^>]*>([^<]+)</span>', section)
        if not venue_match: venue_match = re.search(r'<h3[^>]*>([^<]+)</h3>', section)
        location = venue_match.group(1).strip() if venue_match else "Unknown"
        location = re.sub(r'\s*\([^)]+\)', '', location).strip()
        events = re.findall(r'class="sdc-site-racing-meetings__event"(.*?)</a>', section, re.DOTALL)
        for i, event in enumerate(events):
            time_match = re.search(r'class="sdc-site-racing-meetings__event-time">(\d{1,2}:\d{2})', event)
            time_str = time_match.group(1).strip() if time_match else "00:00"
            details_match = re.search(r'class="sdc-site-racing-meetings__event-details">\(([^)]+)\)', event)
            details = details_match.group(1) if details_match else ""
            fs_match = re.search(r'(\d+)\s*runners', details)
            field_size = fs_match.group(1) if fs_match else "?"
            dist_match = re.search(r'(\d+[fmky]\s*\d*[ymk]*)', details)
            distance = parse_distance_to_miles(dist_match.group(1).strip() if dist_match else "?")
            time_val = None
            try:
                dt_uk = datetime.strptime(f"{target_date} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=uk_tz)
                time_val = dt_uk.astimezone(et_tz)
            except: pass
            races.append({
                "DateTime": time_val,
                "PostTime": time_val.strftime('%H:%M') if time_val else time_str,
                "FieldSize": field_size,
                "Distance": distance,
                "RaceNum": str(i + 1),
                "Location": location,
                "Discipline": "T",
                "Purse": "?",
                "ML": "?"
            })
    return races

def parse_hkjc_html(filepath, target_date):
    """Parses Hong Kong race data from HKJC HTML."""
    if not os.path.exists(filepath): return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except: return []
    races = []
    venue = "Hong Kong"
    if "Sha Tin" in content: venue = "Sha Tin"
    elif "Happy Valley" in content: venue = "Happy Valley"
    race_match = re.search(r'Race (\d+)', content)
    race_num = race_match.group(1) if race_match else "?"
    time_match = re.search(r'(\d{1,2}:\d{2})', content)
    time_str = time_match.group(1) if time_match else "00:00"
    runners = re.findall(r'<tr[^>]*>\s*<td>(\d+)</td>', content)
    field_size = len(runners)
    time_val = None
    try:
        hk_tz = zoneinfo.ZoneInfo("Asia/Hong_Kong")
        et_tz = zoneinfo.ZoneInfo("America/New_York")
        dt_hk = datetime.strptime(f"{target_date} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=hk_tz)
        time_val = dt_hk.astimezone(et_tz)
    except: pass
    if field_size > 0 or race_num != "?":
        races.append({
            "DateTime": time_val,
            "PostTime": time_val.strftime('%H:%M') if time_val else time_str,
            "FieldSize": str(field_size) if field_size > 0 else "?",
            "Distance": "?",
            "RaceNum": race_num,
            "Location": venue,
            "Discipline": "T",
            "Purse": "?",
            "ML": "?"
        })
    return races

def parse_rp_html(filepath, target_date):
    """Parses international race data from Racing Post HTML."""
    if not os.path.exists(filepath): return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except: return []
    races = []
    venue_match = re.search(r'data-test-selector="RC-courseHeader__name"[^>]*>([^<]+)', content)
    if not venue_match: venue_match = re.search(r'class="RC-courseHeader__name"[^>]*>([^<]+)', content)
    location = venue_match.group(1).strip() if venue_match else "Unknown"
    time_match = re.search(r'data-test-selector="RC-courseHeader__time"[^>]*>([^<]+)', content)
    time_str = time_match.group(1).strip() if time_match else "00:00"
    runners = re.findall(r'data-test-selector="RC-runnerCard"', content)
    field_size = len(runners)
    time_val = None
    try:
        uk_tz = zoneinfo.ZoneInfo("Europe/London")
        et_tz = zoneinfo.ZoneInfo("America/New_York")
        dt_uk = datetime.strptime(f"{target_date} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=uk_tz)
        time_val = dt_uk.astimezone(et_tz)
    except: pass
    if location != "Unknown":
        races.append({
            "DateTime": time_val,
            "PostTime": time_val.strftime('%H:%M') if time_val else time_str,
            "FieldSize": str(field_size) if field_size > 0 else "?",
            "Distance": "?",
            "RaceNum": "?",
            "Location": location,
            "Discipline": "T",
            "Purse": "?",
            "ML": "?"
        })
    return races

def find_files(pattern, target_date, date_suffix):
    """Finds files matching pattern in root or manual_fetch."""
    files = glob.glob(pattern.replace('{SUFFIX}', date_suffix))
    files += glob.glob(pattern.replace('{SUFFIX}', target_date))
    manual_pattern = os.path.join('manual_fetch', '*' + pattern.replace('{SUFFIX}', date_suffix))
    files += glob.glob(manual_pattern)
    manual_pattern_date = os.path.join('manual_fetch', '*' + pattern.replace('{SUFFIX}', target_date))
    files += glob.glob(manual_pattern_date)

    if 'equibase' in pattern:
        try:
            dt = datetime.strptime(target_date, "%Y-%m-%d")
            mmddyy = dt.strftime("%m%d%y")
            files += glob.glob(os.path.join('manual_fetch', f'*equibase*{mmddyy}sum*.html'))
        except: pass
    elif 'hkjc' in pattern:
        try:
            date_part = target_date.replace('-', '_')
            files += glob.glob(os.path.join('manual_fetch', f'*hkjc*{date_part}*.html'))
        except: pass
    elif 'skysports' in pattern:
        try:
            dt = datetime.strptime(target_date, "%Y-%m-%d")
            date_part = dt.strftime("%d_%m_%Y")
            files += glob.glob(os.path.join('manual_fetch', f'*skysports*{date_part}*.html'))
        except: pass
    elif 'TrackInfo' in pattern:
        files += glob.glob('TrackInfo_Entries.csv')
        files += glob.glob(os.path.join('manual_fetch', 'TrackInfo_Entries.csv'))

    return list(set(files))

def main():
    parser = argparse.ArgumentParser(description="Worldwide Race Grid Generator")
    parser.add_argument("--date", help="Race date (YYYY-MM-DD)", default=None)
    args = parser.parse_args()

    target_date = args.date or datetime.now().strftime("%Y-%m-%d")
    date_suffix = target_date.split("-")[-1]

    all_raw_races = []
    for f in find_files('rpb2b_{SUFFIX}.json', target_date, date_suffix): all_raw_races.extend(parse_rpb2b_json(f))
    for f in find_files('sportinglife_{SUFFIX}.html', target_date, date_suffix): all_raw_races.extend(parse_sl_hard(f))
    for f in find_files('equibase_{SUFFIX}.html', target_date, date_suffix): all_raw_races.extend(parse_equibase_html(f))
    for f in find_files('drf_{SUFFIX}.html', target_date, date_suffix): all_raw_races.extend(parse_drf_html(f, target_date))
    for f in find_files('hkjc_{SUFFIX}.html', target_date, date_suffix): all_raw_races.extend(parse_hkjc_html(f, target_date))
    for f in find_files('racingpost_{SUFFIX}.html', target_date, date_suffix): all_raw_races.extend(parse_rp_html(f, target_date))
    for f in find_files('ras_{SUFFIX}.json', target_date, date_suffix): all_raw_races.extend(parse_ras_json(f))
    for f in find_files('jra_{SUFFIX}.html', target_date, date_suffix): all_raw_races.extend(parse_jra_html(f, target_date))
    for f in find_files('tf_{SUFFIX}.html', target_date, date_suffix): all_raw_races.extend(parse_tf_html(f, target_date))
    for f in find_files('skysports_{SUFFIX}.html', target_date, date_suffix): all_raw_races.extend(parse_skysports_html(f, target_date))
    for f in find_files('TrackInfo_Entries.csv', target_date, date_suffix): all_raw_races.extend(parse_trackinfo_csv(f))

    # Assign Race Numbers for Sporting Life if missing
    meetings_data = {}
    for r in all_raw_races:
        if r['RaceNum'] == "?":
            loc = r['Location']
            if loc not in meetings_data: meetings_data[loc] = []
            meetings_data[loc].append(r)
    for loc, races in meetings_data.items():
        races.sort(key=lambda x: (x['DateTime'] if x['DateTime'] else datetime.min.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))))
        for i, r in enumerate(races): r['RaceNum'] = str(i + 1)

    # Merging
    merged_map = {}
    for r in all_raw_races:
        if r['Location'] == "Unknown": continue
        canon_loc = get_canonical_venue(r['Location'])
        key = (canon_loc, r['RaceNum'])
        if key not in merged_map:
            merged_map[key] = r
        else:
            existing = merged_map[key]
            if len(r['Location']) > len(existing['Location']): existing['Location'] = r['Location']
            if (existing['Distance'] == "?" or len(r['Distance']) > len(existing['Distance'])) and r['Distance'] != "?":
                existing['Distance'] = r['Distance']
            if (existing['FieldSize'] == "?" or existing['FieldSize'] == "0") and r['FieldSize'] != "?":
                existing['FieldSize'] = r['FieldSize']
            if not existing['DateTime'] and r['DateTime']:
                existing['DateTime'] = r['DateTime']
                existing['PostTime'] = r['PostTime']
            if existing['Purse'] == "?" and r['Purse'] != "?": existing['Purse'] = r['Purse']
            if existing['ML'] == "?" and r['ML'] != "?": existing['ML'] = r['ML']

    final_list = list(merged_map.values())
    final_list.sort(key=lambda x: (x['DateTime'] if x['DateTime'] else datetime.max.replace(tzinfo=zoneinfo.ZoneInfo("UTC")), x['Location']))

    et_tz = zoneinfo.ZoneInfo("America/New_York")
    now_et = datetime.now(et_tz)

    # Filtering for today: 90-minute crop
    is_today = target_date == now_et.strftime("%Y-%m-%d")
    if is_today:
        crop_time = now_et - timedelta(minutes=90)
        final_list = [r for r in final_list if not r['DateTime'] or r['DateTime'] >= crop_time]

    def print_grid(title, races_list):
        if not races_list: return
        print(f"\n=== {title} ===")
        header = f"{'Post (ET)':<10} | {'Field':<5} | {'Dist (mi)':<10} | {'Purse':<6} | {'ML (1/2)':<10} | {'Location':<25} | {'D'} | {'R#'}"
        print(header)
        print("-" * len(header))
        for r in races_list:
            p_time = r['PostTime']
            marker = "  "
            if r['DateTime'] and now_et - timedelta(minutes=5) <= r['DateTime'] <= now_et + timedelta(minutes=15):
                marker = ">>"
            fs = str(r.get('FieldSize', '?'))
            dst = str(r.get('Distance', '?'))
            prs = str(r.get('Purse', '?'))
            ml = str(r.get('ML', '?'))
            loc = str(r.get('Location', 'Unknown'))[:25]
            disc = str(r.get('Discipline', 'T'))[0]
            rn = str(r.get('RaceNum', '?'))
            print(f"{marker}{p_time:<8} | {fs:<5} | {dst:<10} | {prs:<6} | {ml:<10} | {loc:<25} | {disc} | {rn}")

    if final_list:
        print_grid("WORLDWIDE RACE GRID", final_list)
        concentrated = [r for r in final_list if r['FieldSize'] != "?" and r['FieldSize'].isdigit() and int(r['FieldSize']) < 7]
        if concentrated:
            print_grid("CONCENTRATED GRID (Field < 7)", concentrated)
    else:
        print("No race data found for date:", target_date)

if __name__ == "__main__":
    main()
