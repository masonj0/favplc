import json
import os
import re
import glob
import sys
import argparse
from datetime import datetime
import zoneinfo
from collections import defaultdict

# Ensure root is in path for fortuna_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from fortuna_utils import (
        get_canonical_venue, detect_discipline, parse_odds_to_decimal,
        parse_distance_to_miles, format_purse
    )
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from fortuna_utils import (
        get_canonical_venue, parse_distance_to_miles, format_purse
    )

from generate_hourly_trigger_sheet import parse_snapshot_json, evaluate_rules

def main():
    parser = argparse.ArgumentParser(description="Strategy-First Trigger Sheet Generator")
    parser.add_argument("--date", help="Race date (YYYY-MM-DD)", default=None)
    args = parser.parse_args()

    target_date = args.date or datetime.now().strftime("%Y-%m-%d")
    yymmdd = target_date[2:].replace('-', '')

    rules_path = os.path.join('scripts', 'consensus_ruleset.json')
    if not os.path.exists(rules_path):
        print(f"Error: Ruleset not found at {rules_path}")
        return
    with open(rules_path, 'r') as f:
        rules = json.load(f)

    snapshots = glob.glob(f'snapshots/*_{yymmdd}_races.json')
    all_races = []
    for s in snapshots: all_races.extend(parse_snapshot_json(s))

    if not all_races:
        print(f"No races found for {target_date} in snapshots/")
        return

    merged = {}
    for r in all_races:
        key = (get_canonical_venue(r['Location']), r['RaceNum'], r['Discipline'])
        if key not in merged or (not merged[key]['DateTime'] and r['DateTime']):
            merged[key] = r

    final_races = list(merged.values())
    final_races.sort(key=lambda x: (x['DateTime'] if x['DateTime'] else datetime.max.replace(tzinfo=zoneinfo.ZoneInfo("UTC")), x['Location']))

    by_strategy = defaultdict(list)

    for race in final_races:
        res = evaluate_rules(race, rules)
        if res['approved_strategies']:
            for strat in res['approved_strategies']:
                by_strategy[strat['name']].append({
                    "race": race,
                    "strat_details": strat
                })

    print(f"\n{'='*105}")
    print(f" STRATEGY-FIRST TRIGGER SHEET - {target_date} ".center(105, '='))
    print(f" (V3 Portfolio Grinder | {rules['_meta']['title']}) ".center(105, '='))
    print(f"{'='*105}\n")

    sorted_strategies = sorted(by_strategy.keys())

    for strat_name in sorted_strategies:
        matches = by_strategy[strat_name]
        family = matches[0]['strat_details']['family']
        mult = matches[0]['strat_details']['multiplier']
        cost = matches[0]['strat_details']['cost']

        print(f"\n>>> STRATEGY: {strat_name} ({family}) | Multiplier: {mult}x | Ticket: ${cost}")
        print("-" * 105)

        # Sort matches by time
        matches.sort(key=lambda x: x['race']['DateTime'] if x['race']['DateTime'] else datetime.max.replace(tzinfo=zoneinfo.ZoneInfo("UTC")))

        for m in matches:
            race = m['race']
            sd = m['strat_details']
            p_time = race['PostTime']
            loc = race['Location'][:20]
            rnum = race['RaceNum']
            fs = race['FieldSize']
            dist = race['Distance']
            purse = race['PurseFormatted']

            line = f"  [ ] {p_time} | {loc:<20} | R{rnum:<2} | F:{fs:<2} | D:{dist:<5} | P:{purse:<5} | SI:{race['SI']:>4.1f} | F2:{race['Fav2Exact']:>4.1f} | G2:{race['1GAP2']:>4.1f} | {race['Discipline']}"
            print(line)
        print()

if __name__ == "__main__":
    # Add scripts to path so we can import from generate_hourly_trigger_sheet
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
    main()
