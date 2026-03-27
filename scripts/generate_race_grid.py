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
    from fortuna_utils import get_canonical_venue, detect_discipline
except ImportError:
    # Fallback if running from root without PYTHONPATH
    from fortuna_utils import get_canonical_venue, detect_discipline

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

def parse_rpb2b_json(filepath):
    """Parses US race data from RPB2B JSON export."""
    if not os.path.exists(filepath): return []
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except: return []

    et_tz = zoneinfo.ZoneInfo("America/New_York")
    races = []
    for meeting in data:
        location = meeting.get('name', 'Unknown')
        disc = "T"
        for race in meeting.get("races", []):
            try:
                dt_utc = datetime.fromisoformat(race.get("datetimeUtc").replace('Z', '+00:00'))
                dt_et = dt_utc.astimezone(et_tz)
                time_val = dt_et
            except:
                time_val = None

            races.append({
                "DateTime": time_val,
                "PostTime": time_val.strftime('%H:%M') if time_val else "Unknown",
                "FieldSize": str(race.get("numberOfRunners", "?")),
                "Distance": "?",
                "RaceNum": str(race.get("raceNumber", "?")),
                "Location": location,
                "Discipline": disc
            })
    return races

def parse_sl_hard(filepath):
    """Parses international race data from Sporting Life HTML."""
    if not os.path.exists(filepath): return []
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except: return []

    et_tz = zoneinfo.ZoneInfo("America/New_York")
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
        # Default date if not found - we'll try to infer from filename if needed
        date_str = date_match.group(1) if date_match else "2026-03-26"

        look_forward = content[pos:pos+1500]
        runners_match = re.search(r'"ride_count":(\d+)', look_forward)
        if not runners_match:
            runners_match = re.search(r'"number_of_runners":(\d+)', look_forward)
        dist_match = re.search(r'"distance":"([^"]+)"', look_forward)

        disc = detect_discipline(look_forward[:2000])
        d_code = disc[0].upper() if disc else "T"

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
            "Distance": dist_match.group(1) if dist_match else "?",
            "RaceNum": "?",
            "Location": location,
            "Discipline": d_code
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

    # Extract Venue
    venue_match = re.search(r'<h3>([^<]+)</h3>', content)
    location = venue_match.group(1).strip() if venue_match else "Unknown"

    # Find all tables which represent races
    tables = re.findall(r'<table[^>]*>(.*?)</table>', content, re.DOTALL)
    for table in tables:
        # Extract Race Number and Date
        header_match = re.search(r'Race (\d+) - ([^<]+)', table)
        if not header_match: continue

        race_num = header_match.group(1)
        date_str = header_match.group(2).strip()

        # Count rows in tbody to get field size
        tbody_match = re.search(r'<tbody>(.*?)</tbody>', table, re.DOTALL)
        field_size = 0
        if tbody_match:
            field_size = len(re.findall(r'<tr>', tbody_match.group(1)))

        try:
            # We don't have time in the summary table usually, just date.
            # Default to 12:00 PM if time is missing
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
            "Discipline": "T"
        })
    return races

def parse_drf_html(filepath):
    """Parses US race data from DRF HTML."""
    if not os.path.exists(filepath): return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except: return []

    # DRF often has JSON-like structures or specific table classes
    # This is a basic heuristic parser
    races = []
    # Placeholder for more complex DRF parsing if we had real samples
    return races

def parse_hkjc_html(filepath, target_date):
    """Parses Hong Kong race data from HKJC HTML."""
    if not os.path.exists(filepath): return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except: return []

    races = []
    # HKJC venue
    venue = "Hong Kong"
    if "Sha Tin" in content: venue = "Sha Tin"
    elif "Happy Valley" in content: venue = "Happy Valley"

    # Race number
    race_match = re.search(r'Race (\d+)', content)
    race_num = race_match.group(1) if race_match else "?"

    # Time
    time_match = re.search(r'(\d{1,2}:\d{2})', content)
    time_str = time_match.group(1) if time_match else "00:00"

    # Runners
    runners = re.findall(r'<tr[^>]*>\s*<td>(\d+)</td>', content)
    field_size = len(runners)

    time_val = None
    try:
        hk_tz = zoneinfo.ZoneInfo("Asia/Hong_Kong")
        et_tz = zoneinfo.ZoneInfo("America/New_York")
        dt_hk = datetime.strptime(f"{target_date} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=hk_tz)
        time_val = dt_hk.astimezone(et_tz)
    except:
        pass

    if field_size > 0 or race_num != "?":
        races.append({
            "DateTime": time_val,
            "PostTime": time_val.strftime('%H:%M') if time_val else time_str,
            "FieldSize": str(field_size) if field_size > 0 else "?",
            "Distance": "?",
            "RaceNum": race_num,
            "Location": venue,
            "Discipline": "T"
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
    # Venue
    venue_match = re.search(r'data-test-selector="RC-courseHeader__name"[^>]*>([^<]+)', content)
    if not venue_match:
        venue_match = re.search(r'class="RC-courseHeader__name"[^>]*>([^<]+)', content)

    location = venue_match.group(1).strip() if venue_match else "Unknown"

    # Race time
    time_match = re.search(r'data-test-selector="RC-courseHeader__time"[^>]*>([^<]+)', content)
    time_str = time_match.group(1).strip() if time_match else "00:00"

    # Runners
    runners = re.findall(r'data-test-selector="RC-runnerCard"', content)
    field_size = len(runners)

    time_val = None
    try:
        uk_tz = zoneinfo.ZoneInfo("Europe/London")
        et_tz = zoneinfo.ZoneInfo("America/New_York")
        dt_uk = datetime.strptime(f"{target_date} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=uk_tz)
        time_val = dt_uk.astimezone(et_tz)
    except:
        pass

    if location != "Unknown":
        races.append({
            "DateTime": time_val,
            "PostTime": time_val.strftime('%H:%M') if time_val else time_str,
            "FieldSize": str(field_size) if field_size > 0 else "?",
            "Distance": "?",
            "RaceNum": "?",
            "Location": location,
            "Discipline": "T"
        })

    return races

def find_files(pattern, target_date, date_suffix):
    """Finds files matching pattern in root or manual_fetch."""
    files = glob.glob(pattern.replace('{SUFFIX}', date_suffix))
    files += glob.glob(pattern.replace('{SUFFIX}', target_date))

    # Also look in manual_fetch
    manual_pattern = os.path.join('manual_fetch', '*' + pattern.replace('{SUFFIX}', date_suffix))
    files += glob.glob(manual_pattern)
    manual_pattern_date = os.path.join('manual_fetch', '*' + pattern.replace('{SUFFIX}', target_date))
    files += glob.glob(manual_pattern_date)

    # Special handling for equibase files with long slugs in manual_fetch
    if 'equibase' in pattern:
        # Example: www_equibase_com_static_chart_summary_gp032126sum_html.html
        # Extract MMDDYY from target_date YYYY-MM-DD
        try:
            dt = datetime.strptime(target_date, "%Y-%m-%d")
            mmddyy = dt.strftime("%m%d%y")
            files += glob.glob(os.path.join('manual_fetch', f'*equibase*{mmddyy}sum*.html'))
        except:
            pass

    return list(set(files))

def main():
    parser = argparse.ArgumentParser(description="Worldwide Race Grid Generator")
    parser.add_argument("--date", help="Race date (YYYY-MM-DD)", default=None)
    args = parser.parse_args()

    target_date = args.date or datetime.now().strftime("%Y-%m-%d")
    date_suffix = target_date.split("-")[-1] # e.g. "26"

    all_raw_races = []

    # RPB2B
    for f in find_files('rpb2b_{SUFFIX}.json', target_date, date_suffix):
        all_raw_races.extend(parse_rpb2b_json(f))

    # Sporting Life
    for f in find_files('sportinglife_{SUFFIX}.html', target_date, date_suffix):
        all_raw_races.extend(parse_sl_hard(f))

    # Equibase
    for f in find_files('equibase_{SUFFIX}.html', target_date, date_suffix):
        all_raw_races.extend(parse_equibase_html(f))

    # DRF
    for f in find_files('drf_{SUFFIX}.html', target_date, date_suffix):
        all_raw_races.extend(parse_drf_html(f))

    # HKJC
    for f in find_files('hkjc_{SUFFIX}.html', target_date, date_suffix):
        all_raw_races.extend(parse_hkjc_html(f, target_date))

    # Racing Post
    for f in find_files('racingpost_{SUFFIX}.html', target_date, date_suffix):
        all_raw_races.extend(parse_rp_html(f, target_date))

    if not all_raw_races:
        # Try any available files if target date yielded nothing
        # But filter by date suffix if possible to avoid excessive noise
        for f in glob.glob(f'rpb2b_*{date_suffix}.json'): all_raw_races.extend(parse_rpb2b_json(f))
        for f in glob.glob(f'sportinglife_*{date_suffix}.html'): all_raw_races.extend(parse_sl_hard(f))
        for f in glob.glob(f'manual_fetch/*equibase*{date_suffix}*.html'): all_raw_races.extend(parse_equibase_html(f))
        for f in glob.glob(f'manual_fetch/*hkjc*{date_suffix}*.html'): all_raw_races.extend(parse_hkjc_html(f, target_date))
        for f in glob.glob(f'manual_fetch/*racingpost*{date_suffix}*.html'): all_raw_races.extend(parse_rp_html(f, target_date))

    # Assign Race Numbers for Sporting Life
    meetings_data = {}
    for r in all_raw_races:
        if r['RaceNum'] == "?":
            loc = r['Location']
            if loc not in meetings_data: meetings_data[loc] = []
            meetings_data[loc].append(r)

    for loc, races in meetings_data.items():
        races.sort(key=lambda x: (x['DateTime'] if x['DateTime'] else datetime.min.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))))
        time_to_num = {}
        counter = 1
        for r in races:
            t_key = r['DateTime'] or r['PostTime']
            if t_key not in time_to_num:
                time_to_num[t_key] = str(counter)
                counter += 1
            r['RaceNum'] = time_to_num[t_key]

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

    final_list = list(merged_map.values())
    final_list.sort(key=lambda x: (x['DateTime'] if x['DateTime'] else datetime.max.replace(tzinfo=zoneinfo.ZoneInfo("UTC")), x['Location']))

    et_tz = zoneinfo.ZoneInfo("America/New_York")
    now_et = datetime.now(et_tz)

    if final_list:
        grid_lines = []
        grid_lines.append(f"{'PostTime (ET)':<14} | {'Field':<5} | {'Distance':<15} | {'Location':<30} | {'D'} | {'Race#'}")
        grid_lines.append("-" * 95)
        for r in final_list:
            display_time = r['PostTime']
            marker = "  "
            if r['DateTime']:
                if now_et - timedelta(minutes=5) <= r['DateTime'] <= now_et + timedelta(minutes=15):
                    marker = ">>"
            grid_lines.append(f"{marker}{display_time:<12} | {str(r['FieldSize']):<5} | {str(r['Distance']):<15} | {r['Location']:<30} | {r.get('Discipline', 'T')} | {r['RaceNum']}")

        grid_text = "\n".join(grid_lines)
        print(grid_text)
        return grid_text
    else:
        print("No race data files found for date:", target_date)
        return ""

if __name__ == "__main__":
    main()
