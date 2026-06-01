import json
import os
import re
import glob
import sys
import argparse
from datetime import datetime, timedelta
import zoneinfo
from collections import defaultdict

# Ensure root is in path for fortuna_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from fortuna_utils import get_canonical_venue, detect_discipline, parse_odds_to_decimal
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from fortuna_utils import get_canonical_venue, detect_discipline, parse_odds_to_decimal

def parse_distance_to_miles(dist_str):
    if not dist_str or dist_str == "?": return "?"
    s = str(dist_str).lower().strip()
    total_yards = 0.0
    found = False
    m_match = re.search(r'^(\d+)\s*m$', s)
    if m_match:
        val = float(m_match.group(1))
        if val > 100:
            total_yards = val * 1.09361
            found = True
    if not found:
        parts = re.findall(r'(\d+\.?\d*)\s*([mfyk])', s)
        for val, unit in parts:
            try:
                v = float(val)
                if unit == 'm':
                    if v < 10: total_yards += v * 1760
                    else: total_yards += v * 1.09361
                    found = True
                elif unit == 'f': total_yards += v * 220; found = True
                elif unit == 'y': total_yards += v; found = True
                elif unit == 'k': total_yards += v * 1093.61; found = True
            except: continue
    if not found: return dist_str
    if total_yards == 0: return dist_str
    return f"{total_yards / 1760.0:.3f}"

def format_purse(purse_str):
    if not purse_str or purse_str == "?": return "?"
    try:
        val = re.sub(r'[^\d.]', '', str(purse_str))
        if not val: return purse_str
        f_val = float(val)
        if f_val >= 1000: return f"{int(f_val/1000)}K"
        return str(int(f_val))
    except: return purse_str

def parse_snapshot_json(filepath):
    if not os.path.exists(filepath): return []
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        races = []
        et_tz = zoneinfo.ZoneInfo("America/New_York")
        for r in data:
            st = r.get('start_time')
            dt = None
            if st:
                try:
                    dt = datetime.strptime(st, "%y%m%dT%H:%M:%S").replace(tzinfo=et_tz)
                except: pass

            purse_raw = r.get('metadata', {}).get('purse') or r.get('metadata', {}).get('Purse') or "?"

            races.append({
                "DateTime": dt,
                "PostTime": dt.strftime('%H:%M') if dt else "00:00",
                "FieldSize": len(r.get('runners', [])),
                "Distance": parse_distance_to_miles(r.get('distance', '?')),
                "RaceNum": r.get('race_number', 0),
                "Location": r.get('venue', 'Unknown'),
                "Discipline": r.get('discipline', 'T')[0].upper(),
                "Purse": purse_raw,
                "PurseFormatted": format_purse(purse_raw),
                "URL": r.get('metadata', {}).get('url')
            })
        return races
    except: return []

def evaluate_rules(race, rules):
    results = {
        "skip_reason": None,
        "approved_strategies": []
    }

    # Extract retired strategies array from JSON if it exists
    # Hardcoded fallback array based on Claude4's audit to protect the system immediately
    retired_strategies = rules.get('retired_strategies', {}).get('failed_2026_out_of_sample', [
        "XC12", "Tri123", "Tri322", "Tri321", "Tri132", "TriA22", "Tri1S2", "Tri2L1", "Trif333"
    ])

    fs = race['FieldSize']
    purse_val = 0
    try:
        # Extract digits from purse string (e.g. "$21,000" -> 21000)
        p_match = re.sub(r'[^\d]', '', str(race['Purse']))
        if p_match:
            purse_val = int(p_match)
    except: pass

    # PPR = Purse Per Runner (in thousands)
    ppr = (purse_val / 1000.0 / fs) if fs > 0 else 0

    # PPR Categories (Heuristic mapping)
    ppr_cat = "4_Medium"
    if ppr < 1.0: ppr_cat = "1_Lowest"
    elif ppr < 2.5: ppr_cat = "2_VeryLow"
    elif ppr < 5.0: ppr_cat = "3_Low"
    elif ppr > 20.0: ppr_cat = "8_Highest"
    elif ppr > 15.0: ppr_cat = "7_VeryHigh"
    elif ppr > 10.0: ppr_cat = "6_High"
    elif ppr > 5.0: ppr_cat = "5_AboveMedium"

    # 1. Evaluate Universal Gates
    gates = rules['live_bot_config']['universal_gates']['conditions']
    universal_si_floor = 2.0
    for cond in gates:
        field = cond.get('field')
        op = cond.get('operator')
        val = cond.get('value')

        if field == "WhichRace":
            if op == ">=" and race['RaceNum'] < val:
                results["skip_reason"] = f"Skip 1 and 2? (Race {race['RaceNum']})"
                return results

        if field == "Runners":
            if op == "<=" and fs > val:
                return results # Exceeds max runners

        if field == "SI" and op == ">=":
            universal_si_floor = max(universal_si_floor, val)

    # 2. Evaluate Execution Routing
    routing = rules['live_bot_config']['execution_routing']
    for route in routing:
        match = True
        route_si_floor = universal_si_floor
        route_chalk_req = "N"

        # Route-level trigger conditions
        for cond in route['trigger_conditions']:
            field = cond.get('field')
            op = cond.get('operator')
            val = cond.get('value')

            if field == "SI" and op == ">=":
                route_si_floor = max(route_si_floor, val)
                continue

            if field == "ChalkYN":
                route_chalk_req = val
                continue

            if field == "Runners":
                if op == "==" and fs != val: match = False
                elif op == "in" and fs not in val: match = False
            elif field == "Purse":
                if op == "<=" and purse_val > val: match = False
                elif op == ">" and purse_val <= val: match = False
                elif purse_val == 0: match = False
            elif field == "PPR_Half":
                if op == "<=" and ppr > val: match = False
            elif field == "PPR_Categ":
                if op == "==" and ppr_cat != val: match = False
                elif op == "not_in" and ppr_cat in val: match = False

            if not match: break

        if match:
            for strat in route['approved_strategies']:
                strat_name = strat['strategy']

                # RETIREMENT ENFORCEMENT: Loudly drop it if blacklisted
                if strat_name in retired_strategies:
                    continue

                strat_match = True
                strat_chalk_req = route_chalk_req

                # Chalk overrides
                if 'chalk_override' in strat:
                    ov = strat['chalk_override']
                    if ov['field'] == "Runners" and fs == ov['value']:
                        if ov['action'] == "allow_chalk":
                            strat_chalk_req = "Any"

                # Strategy-specific gates
                for s_gate in strat.get('strategy_specific_gates', []):
                    field = s_gate.get('field')
                    val = s_gate.get('value')
                    op = s_gate.get('operator')

                    if field == "Purse":
                        if op == "<=" and purse_val > val: strat_match = False
                    elif field == "ChalkYN":
                        strat_chalk_req = val

                if strat_match:
                    results["approved_strategies"].append({
                        "name": strat_name,
                        "cost": strat.get('ticket_cost', '?'),
                        "note": strat.get('note', ''),
                        "si_floor": route_si_floor,
                        "chalk_req": strat_chalk_req
                    })

    return results

def main():
    parser = argparse.ArgumentParser(description="Hourly Trigger Sheet Generator")
    parser.add_argument("--date", help="Race date (YYYY-MM-DD)", default=None)
    args = parser.parse_args()

    target_date = args.date or datetime.now().strftime("%Y-%m-%d")
    yymmdd = target_date[2:].replace('-', '')

    rules_path = os.path.join('scripts', 'consensus_ruleset.json')
    if not os.path.exists(rules_path):
        print(f"Error: {rules_path} not found.")
        return
    with open(rules_path, 'r') as f:
        rules = json.load(f)

    snapshots = glob.glob(f'snapshots/*_{yymmdd}_races.json')
    all_races = []
    for s in snapshots:
        all_races.extend(parse_snapshot_json(s))

    merged = {}
    for r in all_races:
        key = (get_canonical_venue(r['Location']), r['RaceNum'])
        if key not in merged or (not merged[key]['DateTime'] and r['DateTime']):
            merged[key] = r

    final_races = list(merged.values())
    final_races.sort(key=lambda x: (x['DateTime'] if x['DateTime'] else datetime.max.replace(tzinfo=zoneinfo.ZoneInfo("UTC")), x['Location']))

    hourly = defaultdict(list)
    for r in final_races:
        if r['DateTime']:
            hour = r['DateTime'].strftime("%I %p")
            hourly[hour].append(r)
        else:
            hourly["Unknown Time"].append(r)

    print(f"\n{'='*80}")
    print(f" HOURLY TRIGGER SHEET - {target_date} ".center(80, '='))
    print(f" (Grouped by Hour | Evaluation via Consensus Ruleset v{rules['_meta']['version']}) ".center(80, '='))
    print(f"{'='*80}\n")

    sorted_hours = sorted(hourly.keys(), key=lambda x: datetime.strptime(x, "%I %p") if x != "Unknown Time" else datetime.max)

    for hour in sorted_hours:
        print(f"\n--- {hour} " + "-" * (74 - len(hour)))
        for race in hourly[hour]:
            res = evaluate_rules(race, rules)

            p_time = race['PostTime']
            loc = race['Location'][:20]
            rnum = race['RaceNum']
            fs = race['FieldSize']
            purse = race['PurseFormatted']

            line = f"  {p_time} | {loc:<20} | R{rnum:<2} | Field:{fs:<2} | Purse:{purse:<5}"
            print(line)

            if res['skip_reason']:
                print(f"    >>> {res['skip_reason']}")
            elif res['approved_strategies']:
                for strat in res['approved_strategies']:
                    chalk_box = f"Chalk={strat['chalk_req']}"
                    si_box = f"SI >= {strat['si_floor']:.1f}"
                    print(f"    [ ] {strat['name']:<10} (${strat['cost']:<2}) [ ] {chalk_box:<9} [ ] {si_box}")
                    if strat['note']:
                        note = strat['note']
                        if len(note) > 65:
                            print(f"        Note: {note[:65]}...")
                        else:
                            print(f"        Note: {note}")
            else:
                print("    (No matching strategies)")
            print()

if __name__ == "__main__":
    main()
