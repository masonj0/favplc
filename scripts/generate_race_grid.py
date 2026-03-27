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
    """Placeholder for Equibase HTML parsing."""
    if not os.path.exists(filepath): return []
    # Basic logic to extract tables or rows if possible
    return []

def parse_drf_html(filepath):
    """Placeholder for DRF HTML parsing."""
    if not os.path.exists(filepath): return []
    return []

def parse_hkjc_html(filepath):
    """Placeholder for HKJC HTML parsing."""
    if not os.path.exists(filepath): return []
    return []

def main():
    parser = argparse.ArgumentParser(description="Worldwide Race Grid Generator")
    parser.add_argument("--date", help="Race date (YYYY-MM-DD)", default=None)
    args = parser.parse_args()

    target_date = args.date or datetime.now().strftime("%Y-%m-%d")
    date_suffix = target_date.split("-")[-1] # e.g. "26"

    all_raw_races = []

    # RPB2B
    for f in glob.glob(f'rpb2b_{date_suffix}.json') + glob.glob(f'rpb2b_{target_date}.json'):
        all_raw_races.extend(parse_rpb2b_json(f))

    # Sporting Life
    for f in glob.glob(f'sportinglife_{date_suffix}.html') + glob.glob(f'sportinglife_{target_date}.html'):
        all_raw_races.extend(parse_sl_hard(f))

    # Equibase
    for f in glob.glob(f'equibase_{date_suffix}.html') + glob.glob(f'equibase_{target_date}.html'):
        all_raw_races.extend(parse_equibase_html(f))

    # DRF
    for f in glob.glob(f'drf_{date_suffix}.html') + glob.glob(f'drf_{target_date}.html'):
        all_raw_races.extend(parse_drf_html(f))

    # HKJC
    for f in glob.glob(f'hkjc_{date_suffix}.html') + glob.glob(f'hkjc_{target_date}.html'):
        all_raw_races.extend(parse_hkjc_html(f))

    if not all_raw_races:
        # Try any available files if target date yielded nothing
        for f in glob.glob('rpb2b_*.json'): all_raw_races.extend(parse_rpb2b_json(f))
        for f in glob.glob('sportinglife_*.html'): all_raw_races.extend(parse_sl_hard(f))

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
