import json
import os
import re
import glob
from datetime import datetime
import zoneinfo

def parse_rpb2b_json(filepath):
    """Parses US race data from RPB2B JSON export."""
    if not os.path.exists(filepath): return []
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except: return []

    eastern = zoneinfo.ZoneInfo("America/New_York")
    races = []
    for meeting in data:
        location = meeting.get('name', 'Unknown')
        for race in meeting.get("races", []):
            try:
                dt_utc = datetime.fromisoformat(race.get("datetimeUtc").replace('Z', '+00:00'))
                dt_et = dt_utc.astimezone(eastern)
                time_str = dt_et.strftime('%H:%M ET')
            except:
                time_str = "Unknown"

            races.append({
                "PostTime": time_str,
                "FieldSize": race.get("numberOfRunners", "?"),
                "Distance": "?",
                "Purse": "?",
                "Location": location
            })
    return races

def parse_sl_hard(filepath):
    """Parses international race data from Sporting Life HTML using regex on the embedded JSON."""
    if not os.path.exists(filepath): return []
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except: return []

    time_matches = list(re.finditer(r'"time":"([^"]+)"', content))

    races = []
    for tm in time_matches:
        time_str = tm.group(1)
        pos = tm.start()

        # Look back for course_name
        look_back = content[max(0, pos-1000):pos]
        course_matches = re.findall(r'"course_name":"([^"]+)"', look_back)
        location = course_matches[-1] if course_matches else "Unknown"

        # Look forward for race details
        look_forward = content[pos:pos+1500]
        runners_match = re.search(r'"ride_count":(\d+)', look_forward)
        if not runners_match:
            runners_match = re.search(r'"number_of_runners":(\d+)', look_forward)

        dist_match = re.search(r'"distance":"([^"]+)"', look_forward)
        purse_match = re.search(r'"purse":"([^"]+)"', look_forward)
        if not purse_match:
            purse_match = re.search(r'"prize_money":"([^"]+)"', look_forward)
        if not purse_match:
            purse_match = re.search(r'"value":"([^"]+)"', look_forward)

        races.append({
            "PostTime": time_str,
            "FieldSize": runners_match.group(1) if runners_match else "?",
            "Distance": dist_match.group(1) if dist_match else "?",
            "Purse": purse_match.group(1).replace('\\u00a3', '£') if purse_match else "?",
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
                    "Purse": "?",
                    "Location": loc
                })
    return races

def main():
    """Aggregates and displays a worldwide race grid from cached data files."""
    all_races = []

    # Find all relevant data files in current directory
    rpb2b_files = glob.glob('rpb2b_*.json')
    for f in rpb2b_files:
        all_races.extend(parse_rpb2b_json(f))

    sl_files = glob.glob('sportinglife_*.html')
    for f in sl_files:
        all_races.extend(parse_sl_hard(f))

    ras_files = glob.glob('ras_*.json')
    ras_simple = []
    for f in ras_files:
        ras_simple.extend(parse_ras_simple(f))

    seen = set()
    final_list = []
    for r in all_races:
        if r['Location'] == "Unknown": continue
        key = (r['PostTime'], r['Location'])
        if key not in seen:
            seen.add(key)
            final_list.append(r)

    # Sort primarily by post time
    final_list.sort(key=lambda x: x['PostTime'])

    # Add unique locations from RAS that weren't in the others
    existing_locs = {x['Location'].lower() for x in final_list}
    for r in ras_simple:
        track_only = r['Location'].split(' (')[0].lower()
        if track_only not in existing_locs:
            final_list.append(r)
            existing_locs.add(track_only)

    if final_list:
        print(f"{'PostTime':<12} | {'Field':<5} | {'Distance':<15} | {'Purse':<15} | {'Location'}")
        print("-" * 100)
        for r in final_list:
            print(f"{r['PostTime']:<12} | {str(r['FieldSize']):<5} | {str(r['Distance']):<15} | {str(r['Purse']):<15} | {r['Location']}")
    else:
        print("No race data files found (expecting rpb2b_*.json, sportinglife_*.html, ras_*.json)")

if __name__ == "__main__":
    main()
