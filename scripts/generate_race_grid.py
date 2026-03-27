import json
import os
import re
import glob
from datetime import datetime, timedelta
import zoneinfo
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
        disc = "T"
        for race in meeting.get("races", []):
            try:
                dt_utc = datetime.fromisoformat(race.get("datetimeUtc").replace('Z', '+00:00'))
                dt_et = dt_utc.astimezone(et_tz)
                # Keep full datetime for sorting and marker
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

        # Look for date
        date_match = re.search(r'"date":"([^"]+)"', look_back)
        date_str = date_match.group(1) if date_match else "2026-03-26"

        # Look forward for race details
        look_forward = content[pos:pos+1500]
        runners_match = re.search(r'"ride_count":(\d+)', look_forward)
        if not runners_match:
            runners_match = re.search(r'"number_of_runners":(\d+)', look_forward)
        dist_match = re.search(r'"distance":"([^"]+)"', look_forward)

        disc = detect_discipline(look_forward[:2000])
        if not disc or disc == "Unknown":
             disc = detect_discipline(look_back + look_forward)
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

def parse_ras_simple(filepath):
    """Parses RAS JSON for supplemental meeting locations."""
    if not os.path.exists(filepath): return []
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except: return []

    races = []
    for disc_data in data:
        disc_full = disc_data.get("DisciplineFullText", "Thoroughbred")
        d_code = disc_full[0].upper() if disc_full else "T"

        for country in disc_data.get("Countries", []):
            cname = country.get("CountryName", "")
            for meeting in country.get("Meetings", []):
                loc = f"{meeting.get('Course')} ({cname})"
                races.append({
                    "DateTime": None,
                    "PostTime": "See Guide",
                    "FieldSize": "?",
                    "Distance": "?",
                    "RaceNum": str(meeting.get("RaceNumber", "?")),
                    "Location": loc,
                    "Discipline": d_code
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
        # Sort by DateTime if available, else PostTime
        races.sort(key=lambda x: (x['DateTime'] if x['DateTime'] else datetime.min.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))))
        time_to_num = {}
        counter = 1
        for r in races:
            t_key = r['DateTime'] or r['PostTime']
            if t_key not in time_to_num:
                time_to_num[t_key] = str(counter)
                counter += 1
            r['RaceNum'] = time_to_num[t_key]

    # 2. Merging Subroutine
    merged_map = {}
    for r in all_raw_races:
        if r['Location'] == "Unknown": continue
        canon_loc = get_canonical_venue(r['Location'])
        key = (canon_loc, r['RaceNum'])

        if key not in merged_map:
            merged_map[key] = r
        else:
            existing = merged_map[key]
            if len(r['Location']) > len(existing['Location']):
                existing['Location'] = r['Location']
            if (existing['Distance'] == "?" or len(r['Distance']) > len(existing['Distance'])) and r['Distance'] != "?":
                existing['Distance'] = r['Distance']
            if (existing['FieldSize'] == "?" or existing['FieldSize'] == "0") and r['FieldSize'] != "?":
                existing['FieldSize'] = r['FieldSize']
            # Prefer DateTime from RPB2B/better source
            if not existing['DateTime'] and r['DateTime']:
                 existing['DateTime'] = r['DateTime']
                 existing['PostTime'] = r['PostTime']

    # 3. Final consolidation by User Core Identity (DateTime, Field, Num)
    user_merged = {}
    for r in merged_map.values():
        if r['PostTime'] == "See Guide": continue
        # Use DateTime if available for precision
        ukey = (r['DateTime'] if r['DateTime'] else r['PostTime'], r['FieldSize'], r['RaceNum'])

        if ukey not in user_merged:
            user_merged[ukey] = r
        else:
            existing = user_merged[ukey]
            if len(r['Location']) > len(existing['Location']):
                existing['Location'] = r['Location']
            if existing['Distance'] == "?" and r['Distance'] != "?":
                existing['Distance'] = r['Distance']

    final_list = list(user_merged.values())
    # Sort primarily by DateTime
    final_list.sort(key=lambda x: (x['DateTime'] if x['DateTime'] else datetime.max.replace(tzinfo=zoneinfo.ZoneInfo("UTC")), x['Location']))

    existing_canonical = {get_canonical_venue(x['Location']) for x in final_list}
    for r in ras_simple:
        canon = get_canonical_venue(r['Location'])
        if canon not in existing_canonical:
            final_list.append(r)
            existing_canonical.add(canon)

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
                # Correct comparison using full date context
                if now_et - timedelta(minutes=5) <= r['DateTime'] <= now_et + timedelta(minutes=15):
                    marker = ">>"

            grid_lines.append(f"{marker}{display_time:<12} | {str(r['FieldSize']):<5} | {str(r['Distance']):<15} | {r['Location']:<30} | {r.get('Discipline', 'T')} | {r['RaceNum']}")

        grid_text = "\n".join(grid_lines)
        print(grid_text)
        return grid_text
    else:
        msg = "No race data files found"
        print(msg)
        return msg

if __name__ == "__main__":
    main()
