import json
import sys
import os
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

def generate_summary():
    # Header
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    write_to_summary(f"## 🎯 List of Best Bets - Intelligence Report")
    write_to_summary(f"**Run Date:** {now_str} (UTC)")
    write_to_summary("")

    # 1. Favorite-to-Place Monitor (from race_data.json)
    if os.path.exists('race_data.json'):
        write_to_summary("### 🎯 Favorite-to-Place Monitor")
        try:
            with open('race_data.json', 'r', encoding='utf-8') as f:
                d = json.load(f)

            total = d.get('total_races', 0)
            bet_now = d.get('bet_now_count', 0)
            might_like = d.get('you_might_like_count', 0)

            write_to_summary(f"**Total Races:** {total}")
            write_to_summary(f"**BET NOW:** {bet_now}")
            write_to_summary(f"**You Might Like:** {might_like}")
            write_to_summary("")

            if d.get('bet_now_races'):
                write_to_summary("#### 🎯 BET NOW OPPORTUNITIES")
                write_to_summary("| SUP | MTP | DISC | TRACK | R# | FIELD | ODDS | TOP 5 |")
                write_to_summary("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
                for r in d['bet_now_races']:
                    sup = '✅' if r.get('superfecta_offered') else '❌'
                    mtp = r.get('mtp', 'N/A')
                    # Leading zero formatting for MTP if needed (should already be in text reports but here it is in JSON)
                    try:
                        m_int = int(mtp)
                        mtp_str = f"{m_int:02d}m" if 0 <= m_int < 10 else f"{m_int}m"
                    except:
                        mtp_str = f"{mtp}m"

                    disc = r.get('discipline', 'N/A')
                    track = r.get('track', 'N/A')
                    race_num = r.get('race_number', 'N/A')
                    field = r.get('field_size', 'N/A')
                    fav = f"{r['favorite_odds']:.2f}" if r.get('favorite_odds') else 'N/A'
                    sec = f"{r['second_fav_odds']:.2f}" if r.get('second_fav_odds') else 'N/A'
                    odds = f"{fav}, {sec}"
                    top5 = f"`{r['top_five_numbers']}`" if r.get('top_five_numbers') else 'N/A'
                    write_to_summary(f"| {sup} | {mtp_str} | {disc} | {track} | {race_num} | {field} | {odds} | {top5} |")
                write_to_summary("")

            if d.get('you_might_like_races'):
                write_to_summary("#### 🌟 YOU MIGHT LIKE")
                write_to_summary("| SUP | MTP | DISC | TRACK | R# | FIELD | ODDS | TOP 5 |")
                write_to_summary("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
                for r in d['you_might_like_races']:
                    sup = '✅' if r.get('superfecta_offered') else '❌'
                    mtp = r.get('mtp', 'N/A')
                    try:
                        m_int = int(mtp)
                        mtp_str = f"{m_int:02d}m" if 0 <= m_int < 10 else f"{m_int}m"
                    except:
                        mtp_str = f"{mtp}m"

                    disc = r.get('discipline', 'N/A')
                    track = r.get('track', 'N/A')
                    race_num = r.get('race_number', 'N/A')
                    field = r.get('field_size', 'N/A')
                    fav = f"{r['favorite_odds']:.2f}" if r.get('favorite_odds') else 'N/A'
                    sec = f"{r['second_fav_odds']:.2f}" if r.get('second_fav_odds') else 'N/A'
                    odds = f"{fav}, {sec}"
                    top5 = f"`{r['top_five_numbers']}`" if r.get('top_five_numbers') else 'N/A'
                    write_to_summary(f"| {sup} | {mtp_str} | {disc} | {track} | {race_num} | {field} | {odds} | {top5} |")
                write_to_summary("")
        except Exception as e:
            write_to_summary(f"❌ Error parsing race_data.json: {e}")
    else:
        write_to_summary("⚠️ race_data.json not found - Discovery may have failed")

    # 2. Race Analysis Grid
    if os.path.exists('summary_grid.txt'):
        write_to_summary("### 📋 Race Analysis Grid")
        with open('summary_grid.txt', 'r', encoding='utf-8') as f:
            write_to_summary(f.read())
        write_to_summary("")

    # 3. Goldmine Intelligence
    if os.path.exists('goldmine_report.txt'):
        write_to_summary("### 💰 Goldmine Intelligence")
        write_to_summary("```text")
        with open('goldmine_report.txt', 'r', encoding='utf-8') as f:
            write_to_summary(f.read())
        write_to_summary("```")
        write_to_summary("")

    # 4. Performance Analytics Audit
    if os.path.exists('analytics_report.txt'):
        write_to_summary("### 📊 Performance Analytics Audit")
        write_to_summary("```text")
        with open('analytics_report.txt', 'r', encoding='utf-8') as f:
            # We skip the harvest summary if it's already at the bottom
            content = f.read()
            if "🔎 LIVE ADAPTER HARVEST PROOF" in content:
                content = content.split("🔎 LIVE ADAPTER HARVEST PROOF")[0]
            write_to_summary(content.strip())
        write_to_summary("```")
        write_to_summary("")

    # 5. LIVE ADAPTER HARVEST PROOF (Consolidated)
    write_to_summary("### 🔎 LIVE ADAPTER HARVEST PROOF")
    write_to_summary("-" * 40)

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

    if discovery_harvest:
        write_to_summary("#### 📋 Entries Adapters")
        for adapter in sorted(discovery_harvest.keys()):
            data = discovery_harvest[adapter]
            if isinstance(data, dict):
                count = data.get("count", 0)
                max_odds = data.get("max_odds", 0.0)
            else:
                count = data
                max_odds = 0.0

            status = "✅ SUCCESS" if count > 0 else "⏳ PENDING/NO DATA"
            odds_str = f" | MaxOdds: {max_odds:>5.1f}" if max_odds > 0 else ""
            write_to_summary(f"{adapter:<25} | {status:<15} | Records Found: {count}{odds_str}")
        write_to_summary("")

    if results_harvest:
        write_to_summary("#### 🏁 Results Adapters")
        for adapter in sorted(results_harvest.keys()):
            data = results_harvest[adapter]
            if isinstance(data, dict):
                count = data.get("count", 0)
                max_odds = data.get("max_odds", 0.0)
            else:
                count = data
                max_odds = 0.0

            status = "✅ SUCCESS" if count > 0 else "⏳ PENDING/NO DATA"
            odds_str = f" | MaxOdds: {max_odds:>5.1f}" if max_odds > 0 else ""
            write_to_summary(f"{adapter:<25} | {status:<15} | Records Found: {count}{odds_str}")
        write_to_summary("")

    if not discovery_harvest and not results_harvest:
        write_to_summary("No harvest data available.")
    write_to_summary("")

    # Footer
    write_to_summary("---")
    write_to_summary(f"*Generated by List of Best Bets at {now_str} (UTC)*")

if __name__ == "__main__":
    generate_summary()
