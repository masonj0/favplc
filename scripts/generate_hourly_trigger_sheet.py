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

def _get_best_win_odds(runner):
    """Helper to get win odds from runner object or dict."""
    if isinstance(runner, dict):
        if runner.get('win_odds') is not None:
            return float(runner['win_odds'])
        odds_dict = runner.get('odds', {})
        for src, data in odds_dict.items():
            val = data.get('win') if isinstance(data, dict) else getattr(data, 'win', None)
            if val: return float(val)
        return None

    if hasattr(runner, 'win_odds') and runner.win_odds is not None:
        return float(runner.win_odds)
    if hasattr(runner, 'odds') and runner.odds:
        for src, data in runner.odds.items():
            if hasattr(data, 'win') and data.win:
                return float(data.win)
    return None

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
                except:
                    try:
                        dt = datetime.fromisoformat(st.replace('Z', '+00:00')).astimezone(et_tz)
                    except: pass

            purse_raw = r.get('metadata', {}).get('purse') or r.get('metadata', {}).get('Purse') or "?"

            runners = r.get('runners', [])
            active_runners = [run for run in runners if not run.get('scratched', False)]

            runners_with_odds = []
            for run in active_runners:
                odds = _get_best_win_odds(run)
                if odds:
                    runners_with_odds.append((run, odds))

            runners_with_odds.sort(key=lambda x: x[1])

            fav_odds = runners_with_odds[0][1] if len(runners_with_odds) > 0 else None
            sec_fav_odds = runners_with_odds[1][1] if len(runners_with_odds) > 1 else None

            si = (fav_odds + sec_fav_odds) if fav_odds and sec_fav_odds else 0.0
            gap2 = (sec_fav_odds - fav_odds) if fav_odds and sec_fav_odds else 0.0

            chalk_yn = "N"
            if fav_odds and fav_odds < 0.90:
                chalk_yn = "Y"

            juv_yn = "N"
            race_type = (r.get('race_type') or "").upper()
            if "JUVENILE" in race_type or "2YO" in race_type:
                juv_yn = "Y"

            fight = r.get('metadata', {}).get('fight') or "fghtN"

            fs = len(active_runners)
            dist_val = parse_distance_to_miles(r.get('distance', '?'))
            u7f = False
            try:
                if dist_val != "?":
                    u7f = float(dist_val) < 0.875
            except: pass

            purse_val = 0
            try:
                p_str = str(purse_raw)
                p_match = re.sub(r'[^\d.]', '', p_str)
                if p_match: purse_val = float(p_match)
            except: pass

            ppr_half = (purse_val / 1000.0 / fs) / 2.0 if fs > 0 else 0

            races.append({
                "DateTime": dt,
                "PostTime": dt.strftime('%H:%M') if dt else "00:00",
                "FieldSize": fs,
                "Runners": fs,
                "Distance": dist_val,
                "RaceNum": r.get('race_number', 0),
                "WhichRace": r.get('race_number', 0),
                "Location": r.get('venue', 'Unknown'),
                "Discipline": r.get('discipline', 'T')[0].upper(),
                "Purse": purse_raw,
                "PurseVal": purse_val,
                "PurseFormatted": format_purse(purse_raw),
                "URL": r.get('metadata', {}).get('url'),
                "SI": si,
                "FavExact": fav_odds or 0.0,
                "Fav2Exact": sec_fav_odds or 0.0,
                "1GAP2": gap2,
                "ChalkYN": chalk_yn,
                "JuvYN": juv_yn,
                "Fight": fight,
                "u7f": u7f,
                "PPR_Half": ppr_half
            })
        return races
    except: return []

def evaluate_rules(race, rules):
    results = {
        "skip_reason": None,
        "abort_reason": None,
        "matches": []
    }

    # 1. Evaluate Universal Gates
    gates = rules['live_bot_config']['universal_gates']['conditions']
    for cond in gates:
        field = cond.get('field')
        op = cond.get('operator')
        val = cond.get('value')

        curr_val = race.get(field)
        if field == "Purse": curr_val = race['PurseVal']

        if curr_val is None: continue

        if field == "WhichRace":
            if op == ">=" and curr_val < val:
                results["skip_reason"] = f"Skip {val-1}? (Race {curr_val})"
                return results

        if op == "==":
            if curr_val != val:
                results["abort_reason"] = f"ABORT: {field} {curr_val} != {val}"
                return results
        elif op == ">=":
            if float(curr_val) < float(val):
                results["abort_reason"] = f"ABORT: {field} {curr_val} < {val}"
                return results
        elif op == "<=":
            if float(curr_val) > float(val):
                results["abort_reason"] = f"ABORT: {field} {curr_val} > {val}"
                return results

    # 2. Evaluate Execution Routing
    routing = rules['live_bot_config']['execution_routing']
    for route in routing:
        match = True
        route_multiplier = route.get('multiplier', 1.0)

        for cond in route['trigger_conditions']:
            field = cond.get('field')
            op = cond.get('operator')
            val = cond.get('value')

            curr_val = race.get(field)
            if field == "Purse": curr_val = race.get('PurseVal')

            if curr_val is None:
                match = False
                break

            if op == "==":
                if curr_val != val: match = False
            elif op == ">=":
                if float(curr_val) < float(val): match = False
            elif op == "<=":
                if float(curr_val) > float(val): match = False
            elif op == "in":
                if curr_val not in val: match = False

            if not match: break

        if match:
            for strat in route['approved_strategies']:
                strat_name = strat['strategy']
                strat_match = True

                for s_gate in strat.get('strategy_specific_gates', []):
                    f = s_gate.get('field')
                    v = s_gate.get('value')
                    o = s_gate.get('operator')

                    c_val = race.get(f)
                    if c_val is None:
                        strat_match = False
                        break

                    if o == "==":
                        if c_val != v: strat_match = False
                    elif o == ">=":
                        if float(c_val) < float(v): strat_match = False
                    elif o == "<=":
                        if float(c_val) > float(v): strat_match = False

                    if not strat_match: break

                if strat_match:
                    mult = route_multiplier * strat.get('multiplier_override', 1.0)
                    results["matches"].append({
                        "id": route.get('id', '?'),
                        "family": route.get('family', 'Unknown'),
                        "group": route.get('group', route.get('id', '?')[0:1]), # Fallback to first char of ID if group missing
                        "priority": route.get('priority', 99),
                        "multiplier": round(mult, 3),
                        "engine": route.get('engine', route.get('name', 'Unknown')),
                        "name": route.get('name', strat_name),
                        "cost": strat.get('ticket_cost', '?'),
                        "si": race['SI'],
                        "fav": race['FavExact'],
                        "fav2": race['Fav2Exact'],
                        "gap2": race['1GAP2'],
                        "chalk": race['ChalkYN'],
                        "juv": race['JuvYN']
                    })

    if results["matches"]:
        # Implement Deduplication Logic: Group Priority first, then internal Priority
        group_rank = {'C': 1, 'X': 2}
        results["matches"].sort(key=lambda x: (group_rank.get(x['group'], 99), x['priority']))
        # Keep only the best match
        results["approved_strategies"] = [results["matches"][0]]
    else:
        results["approved_strategies"] = []

    return results

def main():
    parser = argparse.ArgumentParser(description="Hourly Trigger Sheet Generator")
    parser.add_argument("--date", help="Race date (YYYY-MM-DD)", default=None)
    args = parser.parse_args()

    et_tz = zoneinfo.ZoneInfo("America/New_York")
    target_date = args.date or datetime.now(et_tz).strftime("%Y-%m-%d")
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

    hourly = defaultdict(list)
    for r in final_races:
        if r['DateTime']:
            hourly[r['DateTime'].strftime("%I %p")].append(r)
        else:
            hourly["Unknown Time"].append(r)

    output = []
    def emit(s=""):
        print(s)
        output.append(s)

    emit(f"\n{'='*115}")
    emit(f" HOURLY TRIGGER SHEET - {target_date} ".center(115, '='))
    emit(f" (V3 Portfolio Grinder | {rules['_meta']['title']}) ".center(115, '='))
    emit(f"{'='*115}\n")

    sorted_hours = sorted(hourly.keys(), key=lambda x: datetime.strptime(x, "%I %p") if x != "Unknown Time" else datetime.max)

    for hour in sorted_hours:
        emit(f"\n--- {hour} " + "-" * (109 - len(hour)))
        for race in hourly[hour]:
            res = evaluate_rules(race, rules)
            p_time = race['PostTime']
            loc = race['Location'][:20]
            rnum = race['RaceNum']
            fs = race['FieldSize']
            dist = race['Distance']
            line = f"  {p_time} | {loc:<20} | R{rnum:<2} | F:{fs:<2} | D:{dist:<5} | P:{race['PurseFormatted']:<5} | SI:{race['SI']:>4.1f} | F1:{race['FavExact']:>4.1f} | F2:{race['Fav2Exact']:>4.1f} | G2:{race['1GAP2']:>4.1f} | {race['Discipline']}"
            emit(line)

            if res['abort_reason']: emit(f"    !!! {res['abort_reason']}")
            elif res['skip_reason']: emit(f"    >>> {res['skip_reason']}")
            elif res['approved_strategies']:
                s = res['approved_strategies'][0]
                emit(f"    >> {s['family']} ({s['engine']}) | Mult: {s['multiplier']}x")
                emit(f"      [ ] {s['name']:<40} (${s['cost']:<3}) [ ] Chalk={s['chalk']} [ ] Juv={s['juv']}")
            else: emit("    (No matching strategies)")
            emit()

    mmdd = datetime.strptime(target_date, "%Y-%m-%d").strftime("%m%d")
    filename = f"v3_hourly_sheet_{mmdd}.txt"
    with open(filename, 'w') as f:
        f.write("\n".join(output))
    print(f"\nTrigger sheet saved to: {filename}")

if __name__ == "__main__": main()
