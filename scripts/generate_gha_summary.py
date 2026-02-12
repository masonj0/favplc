import json
import sys
import os
import re
from datetime import datetime

def get_summary_file():
    return os.environ.get('GITHUB_STEP_SUMMARY')

def write_to_summary(text):
    summary_file = get_summary_file()
    if summary_file:
        with open(summary_file, 'a', encoding='utf-8') as f:
            f.write(text + '\n')
    else:
        print(text)

def build_harvest_table(summary, title):
    if not summary:
        return f"#### {title}\n| Adapter | Races | Max Odds | Status |\n| --- | --- | --- | --- |\n| N/A | 0 | 0.0 | ⚠️ No harvest data |\n"

    lines = [f"#### {title}", "", "| Adapter | Races | Max Odds | Status |", "| --- | --- | --- | --- |"]

    def sort_key(item):
        adapter, data = item
        count = data.get('count', 0) if isinstance(data, dict) else data
        return (-count, adapter)

    sorted_adapters = sorted(summary.items(), key=sort_key)

    for adapter, data in sorted_adapters:
        if isinstance(data, dict):
            count = data.get('count', 0)
            max_odds = data.get('max_odds', 0.0)
        else:
            count = data
            max_odds = 0.0

        status = '✅' if count > 0 else '⚠️ No Data'
        lines.append(f"| {adapter} | {count} | {max_odds:.1f} | {status} |")
    return "\n".join(lines) + "\n"

def generate_summary():
    # Header
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    write_to_summary(f"## 🔔 Fortuna Intelligence Job Summary")
    write_to_summary(f"*Executive Intelligence Briefing - {now_str} (UTC)*")
    write_to_summary("")

    # 1. Predictions & Proof
    write_to_summary("### 🔮 Fortuna Predictions & Proof")

    # 1a. Predictions Table
    if os.path.exists('race_data.json'):
        try:
            with open('race_data.json', 'r', encoding='utf-8') as f:
                d = json.load(f)

            races = d.get('bet_now_races', []) + d.get('you_might_like_races', [])

            if races:
                write_to_summary("| Venue | Race# | Selection | Odds | Goldmine? | Pred Top 5 | Payout Proof |")
                write_to_summary("| --- | --- | --- | --- | --- | --- | --- |")

                # Take top 10
                for r in races[:10]:
                    odds = r.get('second_fav_odds') or 0.0
                    gold = '✅' if odds >= 4.5 else '—'
                    selection = r.get('second_fav_name') or f"#{r.get('selection_number', '?')}"
                    top5 = r.get('top_five_numbers') or 'TBD'

                    # Try to find payout info if available (merged from analytics maybe)
                    payout_text = 'Awaiting Results'

                    write_to_summary(f"| {r['track']} | {r['race_number']} | {selection} | {odds:.2f} | {gold} | {top5} | {payout_text} |")
            else:
                write_to_summary("No immediate Goldmine predictions available for this run.")
        except Exception as e:
            write_to_summary(f"⚠️ Error parsing race_data.json: {e}")
    else:
        write_to_summary("Awaiting discovery predictions.")

    # 1b. Audited Proof
    write_to_summary("")
    write_to_summary("#### 💰 Recent Audited Proof")

    proof_found = False
    if os.path.exists('analytics_report.txt'):
        try:
            with open('analytics_report.txt', 'r', encoding='utf-8') as f:
                content = f.read()
                # Extract the "RECENT PERFORMANCE PROOF" section
                if "💰 RECENT PERFORMANCE PROOF" in content:
                    proof_found = True
                    section = content.split("💰 RECENT PERFORMANCE PROOF")[1].split("\n\n")[0]
                    # Convert to table if possible or just code block
                    write_to_summary("```text")
                    write_to_summary("RECENT PERFORMANCE PROOF" + section)
                    write_to_summary("```")
        except Exception: pass

    if not proof_found:
        write_to_summary("Awaiting race results; nothing audited in this cycle.")

    # 2. Harvest Performance
    write_to_summary("")
    write_to_summary("### 🛰️ Harvest Performance & Adapter Health")

    discovery_harvest = {}
    if os.path.exists('discovery_harvest.json'):
        try:
            with open('discovery_harvest.json', 'r') as f:
                discovery_harvest = json.load(f)
        except Exception: pass

    results_harvest = {}
    if os.path.exists('results_harvest.json'):
        try:
            with open('results_harvest.json', 'r') as f:
                results_harvest = json.load(f)
        except Exception: pass

    write_to_summary(build_harvest_table(discovery_harvest, "Discovery Harvest"))
    write_to_summary(build_harvest_table(results_harvest, "Results Harvest"))

    # 3. Intelligence Grids
    if os.path.exists('summary_grid.txt') or os.path.exists('field_matrix.txt'):
        write_to_summary("### 📋 Intelligence Grids")

        if os.path.exists('summary_grid.txt'):
            write_to_summary("#### 🏁 Race Analysis Grid")
            with open('summary_grid.txt', 'r', encoding='utf-8') as f:
                write_to_summary(f.read())
            write_to_summary("")

        if os.path.exists('field_matrix.txt'):
            write_to_summary("#### 📊 Field Matrix (3-11 Runners)")
            with open('field_matrix.txt', 'r', encoding='utf-8') as f:
                write_to_summary(f.read())
            write_to_summary("")

    # 4. Report Artifacts
    write_to_summary("### 📁 Report Artifacts")
    write_to_summary("- [Summary Grid](summary_grid.txt)")
    write_to_summary("- [Field Matrix](field_matrix.txt)")
    write_to_summary("- [Goldmine Report](goldmine_report.txt)")
    write_to_summary("- [HTML Report](fortuna_report.html)")
    write_to_summary("- [Analytics Log](analytics_report.txt)")

    write_to_summary("")
    write_to_summary("---")
    write_to_summary(f"*Fortuna Intelligence Monolith - Optimized Briefing*")

if __name__ == "__main__":
    generate_summary()
