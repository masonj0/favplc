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
        "abort_reason": None,
        "approved_strategies": []
    }

    # RETIREMENT ENFORCEMENT
    retired_strategies = rules.get('retired_strategies', {}).get('failed_2026_out_of_sample', [])
    for item in rules.get('retired_strategies', {}).get('catastrophic_exposure', []):
        retired_strategies.append(item['strategy'])
    retired_strategies = list(set(retired_strategies))

    fs = race['FieldSize']
    purse_val = 0
    try:
        p_match = re.sub(r'[^\d]', '', str(race['Purse']))
        if p_match: purse_val = int(p_match)
    except: pass

    ppr = (purse_val / 1000.0 / fs) if fs > 0 else 0
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
                results["skip_reason"] = f"Skip {val-1}? (Race {race['RaceNum']})"
                return results
        if field == "Runners":
            if op == "<=" and fs > val:
                note = cond.get('note', f"{fs} runners")
                results["abort_reason"] = f"ABORT: {note}"
                return results
        if field == "Purse":
            if op == ">=" and purse_val > 0 and purse_val < val:
                results["abort_reason"] = f"ABORT: Efficiency Floor ({race['PurseFormatted']} < {format_purse(val)})"
                return results
            if op == "<=" and purse_val > val:
                results["abort_reason"] = f"ABORT: Efficiency Ceiling ({race['PurseFormatted']} > {format_purse(val)})"
                return results
        if field == "SI" and op == ">=":
            universal_si_floor = max(universal_si_floor, val)

    # 2. Evaluate Execution Routing
    routing = rules['live_bot_config']['execution_routing']
    for route in routing:
        match = True
        route_si_floor = universal_si_floor
        route_si_cap = 99.0
        route_chalk_req = "N"
        engine = route.get('engine', 'Unknown')

        for cond in route['trigger_conditions']:
            field = cond.get('field')
            op = cond.get('operator')
            val = cond.get('value')

            if field == "SI" and op == ">=":
                route_si_floor = max(route_si_floor, val)
                continue
            if field == "SI" and op == "<":
                route_si_cap = val
                continue
            if field == "ChalkYN":
                route_chalk_req = val
                continue
            if field == "Runners":
                if op == "in" and fs not in val: match = False
                elif op == "==" and fs != val: match = False
            elif field == "Purse":
                if op == "<=" and purse_val > val: match = False
            elif field == "PPR_Half":
                if op == "<=" and ppr > val: match = False
            elif field in ["PPR_Target", "PPR_Categ"]:
                if op == "==" and ppr_cat != val: match = False
                elif op == "not_in" and ppr_cat in val: match = False

            if not match: break

        if match:
            for strat in route['approved_strategies']:
                strat_name = strat['strategy']
                if strat_name in retired_strategies: continue

                strat_match = True
                strat_chalk_req = route_chalk_req
                strat_fght_req = "Any"
                if 'chalk_override' in strat:
                    ov = strat['chalk_override']
                    if ov['field'] == "Runners" and fs == ov['value'] and ov['action'] == "allow_chalk":
                        strat_chalk_req = "Any"

                for s_gate in strat.get('strategy_specific_gates', []):
                    f = s_gate.get('field')
                    v = s_gate.get('value')
                    o = s_gate.get('operator')
                    if f == "Purse" and o == "<=" and purse_val > v: strat_match = False
                    elif f == "ChalkYN": strat_chalk_req = v
                    elif f == "Fight": strat_fght_req = v

                if strat_match:
                    results["approved_strategies"].append({
                        "engine": engine,
                        "name": strat_name,
                        "cost": strat.get('ticket_cost', '?'),
                        "note": strat.get('note', ''),
                        "si_floor": route_si_floor,
                        "si_cap": route_si_cap,
                        "chalk_req": strat_chalk_req,
                        "fght_req": strat_fght_req,
                        "skew": strat.get('skew')
                    })

    return results

def main():
    parser = argparse.ArgumentParser(description="Hourly Trigger Sheet Generator")
    parser.add_argument("--date", help="Race date (YYYY-MM-DD)", default=None)
    args = parser.parse_args()

    target_date = args.date or datetime.now().strftime("%Y-%m-%d")
    yymmdd = target_date[2:].replace('-', '')

    rules_path = os.path.join('scripts', 'consensus_ruleset.json')
    if not os.path.exists(rules_path): return
    with open(rules_path, 'r') as f:
        rules = json.load(f)

    snapshots = glob.glob(f'snapshots/*_{yymmdd}_races.json')
    all_races = []
    for s in snapshots: all_races.extend(parse_snapshot_json(s))

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
            hourly[r['DateTime'].strftime("%I %p")].append(r)
        else:
            hourly["Unknown Time"].append(r)

    print(f"\n{'='*80}")
    print(f" HOURLY TRIGGER SHEET - {target_date} ".center(80, '='))
    print(f" (Barbell Strategy | Dual-Engine Evaluation v{rules['_meta']['version']}) ".center(80, '='))
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
            line = f"  {p_time} | {loc:<20} | R{rnum:<2} | Field:{fs:<2} | Purse:{race['PurseFormatted']:<5}"
            print(line)

            if res['abort_reason']: print(f"    !!! {res['abort_reason']}")
            elif res['skip_reason']: print(f"    >>> {res['skip_reason']}")
            elif res['approved_strategies']:
                by_engine = defaultdict(list)
                for s in res['approved_strategies']: by_engine[s['engine']].append(s)
                for eng in sorted(by_engine.keys()):
                    print(f"    >> {eng}")
                    for strat in by_engine[eng]:
                        chalk_box = f"Chalk={strat['chalk_req']}"
                        si_box = f"SI >= {strat['si_floor']:.1f}"
                        if strat['si_cap'] < 90: si_box = f"SI {strat['si_floor']:.1f}-{strat['si_cap']:.1f}"
                        fght_box = f"Fght={strat['fght_req']}"

                        print(f"      [ ] {strat['name']:<10} (${strat['cost']:<2}) [ ] {chalk_box:<9} [ ] {fght_box:<8} [ ] {si_box}")
                        if strat['skew']:
                            s = strat['skew']
                            print(f"          Skew: Mean={s['overall_mean']:+.2f} | Upside={s['upside_skew_pct']}% | {s.get('skew_note', '')}")
                        if strat['note']:
                            note = strat['note']
                            if len(note) > 65: print(f"          Note: {note[:65]}...")
                            else: print(f"          Note: {note}")
            else: print("    (No matching strategies)")
            print()

if __name__ == "__main__": main()
