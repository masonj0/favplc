import json
import os
import re
import glob
from datetime import datetime, timedelta
import zoneinfo
from fortuna_utils import get_canonical_venue

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
    """Parses US race data from RPB2B JSON export and converts to ET."""
    if not os.path.exists(filepath): return []
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except: return []

    et_tz = zoneinfo.ZoneInfo("America/New_York")
    races = []
    for meeting in data:
        location = meeting.get('name', 'Unknown')
        for race in meeting.get("races", []):
            try:
                dt_utc = datetime.fromisoformat(race.get("datetimeUtc").replace('Z', '+00:00'))
                dt_et = dt_utc.astimezone(et_tz)
                time_str = dt_et.strftime('%H:%M')
            except:
                time_str = "Unknown"

            races.append({
                "PostTime": time_str,
                "FieldSize": str(race.get("numberOfRunners", "?")),
                "Distance": "?",
                "RaceNum": str(race.get("raceNumber", "?")),
                "Location": location
            })
    return races

def parse_sl_hard(filepath):
    """Parses international race data from Sporting Life HTML and converts to ET."""
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

        # Look back for course_name
        look_back = content[max(0, pos-1000):pos]
        course_matches = re.findall(r'"course_name":"([^"]+)"', look_back)
        location = course_matches[-1] if course_matches else "Unknown"

        country_match = re.search(r'"long_name":"([^"]+)"', look_back)
        country = country_match.group(1) if country_match else "Unknown"

        look_forward = content[pos:pos+1500]
        runners_match = re.search(r'"ride_count":(\d+)', look_forward)
        if not runners_match:
            runners_match = re.search(r'"number_of_runners":(\d+)', look_forward)
        dist_match = re.search(r'"distance":"([^"]+)"', look_forward)

        # Adjust Time to ET
        try:
            local_tz = get_tz_for_country(country, location)
            dt_local = datetime.strptime("2026-03-26 " + time_str, "%Y-%m-%d %H:%M").replace(tzinfo=local_tz)
            dt_et = dt_local.astimezone(et_tz)
            time_et = dt_et.strftime('%H:%M')
        except:
            time_et = time_str

        races.append({
            "PostTime": time_et,
            "FieldSize": str(runners_match.group(1)) if runners_match else "?",
            "Distance": dist_match.group(1) if dist_match else "?",
            "RaceNum": "?",
            "Location": location
        })

    return races

def parse_ras_simple(filepath):
    """Parses RAS JSON for supplemental meeting locations."""
    if not os.path.exists(filepath): return []
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except: return []

    races = []
    for disc in data:
        for country in disc.get("Countries", []):
            cname = country.get("CountryName", "")
            for meeting in country.get("Meetings", []):
                loc = f"{meeting.get('Course')} ({cname})"
                races.append({
                    "PostTime": "See Guide",
                    "FieldSize": "?",
                    "Distance": "?",
                    "RaceNum": str(meeting.get("RaceNumber", "?")),
                    "Location": loc
                })
    return races

def main():
    """Aggregates, merges, and displays a worldwide race grid with all times in US ET."""
    all_raw_races = []

    rpb2b_files = glob.glob('rpb2b_*.json')
    for f in rpb2b_files:
        all_raw_races.extend(parse_rpb2b_json(f))

    sl_files = glob.glob('sportinglife_*.html')
    for f in sl_files:
        all_raw_races.extend(parse_sl_hard(f))

    ras_files = glob.glob('ras_*.json')
    ras_simple = []
    for f in ras_files:
        ras_simple.extend(parse_ras_simple(f))

    # 1. Assign Race Numbers for Sporting Life based on unique PostTime slots per meeting
    meetings_data = {}
    for r in all_raw_races:
        if r['RaceNum'] == "?":
            loc = r['Location']
            if loc not in meetings_data: meetings_data[loc] = []
            meetings_data[loc].append(r)

    for loc, races in meetings_data.items():
        unique_times = sorted(list(set(r['PostTime'] for r in races)))
        time_to_num = {t: str(i + 1) for i, t in enumerate(unique_times)}
        for r in races:
            r['RaceNum'] = time_to_num[r['PostTime']]

    # 2. Merging Subroutine
    merged_map = {}
    for r in all_raw_races:
        if r['Location'] == "Unknown" or r['PostTime'] == "Unknown": continue

        # User requested merging if PostTime, FieldSize, and RaceNum are identical.
        key = (r['PostTime'], r['FieldSize'], r['RaceNum'])

        if key not in merged_map:
            merged_map[key] = r
        else:
            existing = merged_map[key]
            # Use longer location name if they aren't the same
            if len(r['Location']) > len(existing['Location']):
                existing['Location'] = r['Location']
            # Merge distance
            if (existing['Distance'] == "?" or len(r['Distance']) > len(existing['Distance'])) and r['Distance'] != "?":
                existing['Distance'] = r['Distance']

    # 3. Handle identical tracks (CanonLoc + Num) that might have slight data differences
    final_merged = {}
    for r in merged_map.values():
        canon_loc = get_canonical_venue(r['Location'])
        key = (canon_loc, r['RaceNum'])

        if key not in final_merged:
            final_merged[key] = r
        else:
            existing = final_merged[key]
            if len(r['Location']) > len(existing['Location']):
                existing['Location'] = r['Location']
            if (existing['Distance'] == "?" or len(r['Distance']) > len(existing['Distance'])) and r['Distance'] != "?":
                existing['Distance'] = r['Distance']
            if (existing['FieldSize'] == "?" or existing['FieldSize'] == "0") and r['FieldSize'] != "?":
                existing['FieldSize'] = r['FieldSize']
            # Times might differ across feeds; prefer earliest ET if both valid
            if r['PostTime'] < existing['PostTime'] and r['PostTime'] != "Unknown":
                 existing['PostTime'] = r['PostTime']

    final_list = list(final_merged.values())

    # Sort primarily by post time, then by location
    final_list.sort(key=lambda x: (x['PostTime'], x['Location']))

    # Add unique locations from RAS that weren't in the others
    existing_canonical = {get_canonical_venue(x['Location']) for x in final_list}
    for r in ras_simple:
        canon = get_canonical_venue(r['Location'])
        if canon not in existing_canonical:
            final_list.append(r)
            existing_canonical.add(canon)

    # Extra Credit: Marker for races happening soon
    et_tz = zoneinfo.ZoneInfo("America/New_York")
    now_et = datetime.now(et_tz)

    if final_list:
        grid_lines = []
        grid_lines.append(f"{'PostTime (ET)':<14} | {'Field':<5} | {'Distance':<15} | {'Location':<30} | {'Race#'}")
        grid_lines.append("-" * 88)
        for r in final_list:
            display_time = r['PostTime']
            marker = "  "

            try:
                # Assuming current date 2026-03-26 for marker logic
                race_time = datetime.strptime("2026-03-26 " + r['PostTime'], "%Y-%m-%d %H:%M").replace(tzinfo=et_tz)
                # If race is in the next 15 minutes or was in the last 5 minutes
                if now_et - timedelta(minutes=5) <= race_time <= now_et + timedelta(minutes=15):
                    marker = ">>"
            except: pass

            grid_lines.append(f"{marker}{display_time:<12} | {str(r['FieldSize']):<5} | {str(r['Distance']):<15} | {r['Location']:<30} | {r['RaceNum']}")

        grid_text = "\n".join(grid_lines)
        print(grid_text)
        return grid_text
    else:
        msg = "No race data files found (expecting rpb2b_*.json, sportinglife_*.html, ras_*.json)"
        print(msg)
        return msg

if __name__ == "__main__":
    main()
