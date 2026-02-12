#!/usr/bin/env python3
"""
Enhanced GitHub Actions Job Summary Generator for Fortuna
Builds on existing structure with rich predictions, adapter performance, and verified results
"""
import json
import sys
import os
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

def get_summary_file():
    return os.environ.get('GITHUB_STEP_SUMMARY')

def write_to_summary(text):
    summary_file = get_summary_file()
    if summary_file:
        with open(summary_file, 'a', encoding='utf-8') as f:
            f.write(text + '\n')
    else:
        print(text)

def get_venue_emoji(venue, discipline=None):
    """Get emoji based on venue or discipline."""
    venue_lower = venue.lower() if venue else ""

    if discipline == "Greyhound":
        return "🐕"
    elif discipline == "Harness":
        return "🏇"

    # Geographic emojis
    if any(x in venue_lower for x in ['kentucky', 'churchill', 'oaklawn', 'tampa', 'gulfstream', 'santa', 'golden', 'belmont']):
        return "🇺🇸"
    elif any(x in venue_lower for x in ['ascot', 'cheltenham', 'newmarket', 'york', 'aintree', 'doncaster']):
        return "🇬🇧"
    elif any(x in venue_lower for x in ['longchamp', 'chantilly', 'deauville']):
        return "🇫🇷"
    elif any(x in venue_lower for x in ['flemington', 'randwick', 'moonee', 'caulfield']):
        return "🇦🇺"
    elif any(x in venue_lower for x in ['woodbine', 'mohawk']):
        return "🇨🇦"

    return "🏇"

def parse_time_to_minutes(start_time_str):
    """Parse start_time to minutes from now."""
    try:
        # Handle ISO format
        if 'T' in start_time_str:
            st = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
        else:
            # Try simple date parsing
            st = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')

        now = datetime.utcnow()
        delta = (st - now).total_seconds() / 60
        return delta
    except Exception:
        return 9999.0

def format_time_remaining(minutes):
    """Format minutes into readable time remaining."""
    if minutes < 0:
        return "🏁"
    elif minutes < 15:
        return f"⚡{int(minutes)}m"
    elif minutes < 60:
        return f"🕐{int(minutes)}m"
    else:
        hours = int(minutes / 60)
        mins = int(minutes % 60)
        return f"🕐{hours}h{mins}m"

def get_db_stats():
    """Get statistics from the fortuna.db database."""
    stats = {
        'total_tips': 0,
        'cashed': 0,
        'burned': 0,
        'pending': 0,
        'total_profit': 0.0,
        'recent_tips': []
    }

    if not os.path.exists('fortuna.db'):
        return stats

    try:
        conn = sqlite3.connect('fortuna.db')
        cursor = conn.cursor()

        # Get overall stats
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN verdict = 'CASHED' THEN 1 ELSE 0 END) as cashed,
                SUM(CASE WHEN verdict = 'BURNED' THEN 1 ELSE 0 END) as burned,
                SUM(CASE WHEN audit_completed = 0 THEN 1 ELSE 0 END) as pending,
                SUM(COALESCE(net_profit, 0.0)) as profit
            FROM tips
        """)
        row = cursor.fetchone()
        if row:
            stats['total_tips'] = row[0] or 0
            stats['cashed'] = row[1] or 0
            stats['burned'] = row[2] or 0
            stats['pending'] = row[3] or 0
            stats['total_profit'] = row[4] or 0.0

        # Get recent audited tips (last 15)
        cursor.execute("""
            SELECT
                venue, race_number, selection_number,
                predicted_2nd_fav_odds, verdict, net_profit,
                selection_position, actual_top_5, actual_2nd_fav_odds,
                superfecta_payout, trifecta_payout, top1_place_payout,
                discipline
            FROM tips
            WHERE audit_completed = 1
            ORDER BY audit_timestamp DESC
            LIMIT 15
        """)
        stats['recent_tips'] = cursor.fetchall()

        conn.close()
    except Exception as e:
        print(f"Error reading database: {e}")

    return stats

def build_enhanced_harvest_table(summary, title):
    """Enhanced harvest table with quality scores."""
    if not summary:
        return f"#### {title}\n| Adapter | 🏇 Races | 💰 Max Odds | 📊 Quality | ✅ Status |\n| :--- | ---: | ---: | ---: | :---: |\n| *No data* | 0 | 0.0 | — | ⚠️ |\n"

    lines = [
        f"#### {title}",
        "",
        "| Adapter | 🏇 Races | 💰 Max Odds | 📊 Quality | ✅ Status |",
        "| :--- | ---: | ---: | ---: | :---: |"
    ]

    def sort_key(item):
        adapter, data = item
        count = data.get('count', 0) if isinstance(data, dict) else data
        return (-count, adapter)

    sorted_adapters = sorted(summary.items(), key=sort_key)

    total_races = 0
    total_max_odds = 0.0

    for adapter, data in sorted_adapters:
        if isinstance(data, dict):
            count = data.get('count', 0)
            max_odds = data.get('max_odds', 0.0)
        else:
            count = data
            max_odds = 0.0

        total_races += count
        if max_odds > total_max_odds:
            total_max_odds = max_odds

        # Quality assessment
        if count == 0:
            quality = "—"
            status = "⚠️ No Data"
        elif count < 5:
            quality = "⚠️ Low"
            status = "🟡 Partial"
        elif max_odds < 10:
            quality = "🟢 Good"
            status = "✅ Active"
        else:
            quality = "🔥 Excel"
            status = "✅ Active"

        lines.append(f"| **{adapter}** | {count} | {max_odds:.1f} | {quality} | {status} |")

    # Add totals
    if total_races > 0:
        lines.extend([
            "| | | | | |",
            f"| **TOTAL** | **{total_races}** | **{total_max_odds:.1f}** | — | — |"
        ])

    return "\n".join(lines) + "\n"

def format_enhanced_predictions():
    """Enhanced predictions table with time-to-post and emojis."""
    lines = [
        "### 🔮 Top Goldmine Predictions",
        "",
        "*Sorted by time to post - Imminent races first!*",
        "",
        "| ⏰ MTP | 🏇 Venue | R# | 🎯 Selection | 💰 Odds | 📊 Gap | ⭐ Type | 🔝 Top 5 |",
        "| :---: | :--- | :---: | :--- | ---: | ---: | :---: | :--- |"
    ]

    if not os.path.exists('race_data.json'):
        lines.append("| | | | *Awaiting discovery predictions* | | | | |")
        return "\n".join(lines)

    try:
        with open('race_data.json', 'r', encoding='utf-8') as f:
            d = json.load(f)

        races = d.get('bet_now_races', []) + d.get('you_might_like_races', [])

        if not races:
            lines.append("| | | | *No goldmine predictions available* | | | | |")
            return "\n".join(lines)

        # Sort by time to post
        races_with_time = []
        for r in races:
            mtp = parse_time_to_minutes(r.get('start_time', ''))
            races_with_time.append((mtp, r))

        races_with_time.sort(key=lambda x: x[0])

        # Take top 12
        for mtp, r in races_with_time[:12]:
            venue = r.get('track', 'Unknown')
            discipline = r.get('discipline', 'Thoroughbred')
            emoji = get_venue_emoji(venue, discipline)

            race_num = r.get('race_number', '?')

            # Time formatting
            time_str = format_time_remaining(mtp)

            # Selection
            sel_name = r.get('second_fav_name', '')
            sel_num = r.get('selection_number', '?')
            if len(sel_name) > 18:
                sel_name = sel_name[:15] + "..."
            selection = f"**#{sel_num}** {sel_name}" if sel_name else f"**#{sel_num}**"

            # Odds and gap
            odds = r.get('second_fav_odds', 0.0)
            odds_str = f"**{odds:.2f}**" if odds else "N/A"

            gap = r.get('gap12', 0.0)
            gap_str = f"{gap:.2f}" if gap else "—"

            # Type
            is_gold = r.get('is_goldmine', False)
            type_str = "💎" if is_gold else "🎯"

            # Top 5
            top5 = r.get('top_five_numbers', 'TBD')
            if isinstance(top5, str) and len(top5) > 15:
                top5 = top5[:12] + "..."

            lines.append(
                f"| {time_str} | {emoji} {venue} | **{race_num}** | {selection} | {odds_str} | {gap_str} | {type_str} | `{top5}` |"
            )

    except Exception as e:
        lines.append(f"| | | | *Error loading predictions: {e}* | | | | |")

    return "\n".join(lines)

def format_enhanced_audit_results(stats):
    """Enhanced audit results with detailed breakdown."""
    lines = [
        "",
        "### 💰 Verified Performance Results",
        ""
    ]

    if stats['total_tips'] == 0:
        lines.append("⏳ *No tips audited yet. Results will appear after races complete.*")
        return "\n".join(lines)

    # Summary statistics
    win_rate = (stats['cashed'] / stats['total_tips'] * 100) if stats['total_tips'] > 0 else 0
    profit_color = "🟢" if stats['total_profit'] > 0 else "🔴" if stats['total_profit'] < 0 else "⚪"

    lines.extend([
        "#### 📊 Overall Statistics",
        "",
        f"- **Total Bets:** {stats['total_tips']}",
        f"- **Cashed:** ✅ {stats['cashed']} ({win_rate:.1f}%)",
        f"- **Burned:** ❌ {stats['burned']}",
        f"- **Pending:** ⏳ {stats['pending']}",
        f"- **Net P/L:** {profit_color} **${stats['total_profit']:+.2f}**",
        ""
    ])

    # Recent results detail
    if stats['recent_tips']:
        lines.extend([
            "#### 🎯 Recent Results (Last 15 Audited)",
            "",
            "| Verdict | 💵 P/L | 🏇 Venue | R# | 🎯 Pick | 🏁 Finish | 💰 Payouts |",
            "| :---: | ---: | :--- | :---: | :---: | :--- | :--- |"
        ])

        for tip in stats['recent_tips']:
            venue, race_num, sel_num, pred_odds, verdict, profit, sel_pos, actual_top5, actual_odds, sf_payout, tri_payout, pl_payout, discipline = tip

            # Verdict emoji
            if verdict == 'CASHED':
                v_emoji = "✅"
                v_text = "WIN"
            elif verdict == 'BURNED':
                v_emoji = "❌"
                v_text = "LOSS"
            else:
                v_emoji = "⏳"
                v_text = "PEND"

            # Profit
            profit = profit or 0.0
            p_color = "🟢" if profit > 0 else "🔴" if profit < 0 else "⚪"
            p_str = f"${profit:+.2f}"

            # Venue
            emoji = get_venue_emoji(venue, discipline)

            # Pick
            pick_str = f"**#{sel_num}**"
            if pred_odds:
                pick_str += f" @ {pred_odds:.2f}"

            # Finish
            if sel_pos:
                pos_emojis = {1: "🥇", 2: "🥈", 3: "🥉", 4: "4️⃣", 5: "5️⃣"}
                pos_emoji = pos_emojis.get(sel_pos, "❌")
                finish_str = f"{pos_emoji} {sel_pos}"
                if actual_top5:
                    finish_str += f" of `{actual_top5}`"
            else:
                finish_str = f"`{actual_top5}`" if actual_top5 else "—"

            # Payouts
            payouts = []
            if sf_payout:
                payouts.append(f"SF ${sf_payout:.2f}")
            if tri_payout:
                payouts.append(f"Tri ${tri_payout:.2f}")
            if pl_payout:
                payouts.append(f"Pl ${pl_payout:.2f}")
            payout_str = " • ".join(payouts) if payouts else "—"

            lines.append(
                f"| {v_emoji} **{v_text}** | {p_color} {p_str} | {emoji} {venue} | **{race_num}** | {pick_str} | {finish_str} | {payout_str} |"
            )

    return "\n".join(lines)

def format_performance_insights(stats):
    """Add performance insights section."""
    if stats['total_tips'] < 3:
        return ""

    lines = [
        "",
        "### 📈 Performance Insights",
        ""
    ]

    # Calculate recent form (last 10 bets)
    recent_tips = stats['recent_tips'][:10]
    if recent_tips:
        recent_profit = sum(tip[5] or 0.0 for tip in recent_tips)
        recent_cashed = sum(1 for tip in recent_tips if tip[4] == 'CASHED')
        recent_wr = (recent_cashed / len(recent_tips) * 100) if recent_tips else 0

        trend_emoji = "📈" if recent_profit > 0 else "📉"

        lines.extend([
            "#### 🔥 Recent Form (Last 10 Bets)",
            "",
            f"- {trend_emoji} **Trend:** ${recent_profit:+.2f}",
            f"- **Hit Rate:** {recent_wr:.0f}% ({recent_cashed}/{len(recent_tips)})",
            ""
        ])

    # Average profit per bet
    if stats['total_tips'] > 0:
        avg_profit = stats['total_profit'] / stats['total_tips']
        lines.append(f"**Average per Bet:** ${avg_profit:+.2f}")

    return "\n".join(lines)

def generate_summary():
    """Main summary generation function."""
    # Header
    now_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    write_to_summary("# 🎯 Fortuna Intelligence Report")
    write_to_summary("")
    write_to_summary(f"*Executive Intelligence Briefing - {now_str} UTC*")
    write_to_summary("")

    # 1. Enhanced Predictions
    write_to_summary(format_enhanced_predictions())
    write_to_summary("")

    # 2. Harvest Performance with Enhanced Tables
    write_to_summary("### 🛰️ Harvest Performance & Adapter Health")
    write_to_summary("")

    discovery_harvest = {}
    # Merge all discovery harvest files
    for hf in ['discovery_harvest.json', 'discovery_harvest_usa.json', 'discovery_harvest_int.json']:
        if os.path.exists(hf):
            try:
                with open(hf, 'r') as f:
                    data = json.load(f)
                    for k, v in data.items():
                        if k not in discovery_harvest:
                            discovery_harvest[k] = v
                        else:
                            if isinstance(v, dict) and isinstance(discovery_harvest[k], dict):
                                discovery_harvest[k]['count'] = max(discovery_harvest[k].get('count', 0), v.get('count', 0))
                                discovery_harvest[k]['max_odds'] = max(discovery_harvest[k].get('max_odds', 0.0), v.get('max_odds', 0.0))
            except Exception:
                pass

    results_harvest = {}
    # Merge all results harvest files
    for hf in ['results_harvest.json', 'results_harvest_audit.json']:
        if os.path.exists(hf):
            try:
                with open(hf, 'r') as f:
                    data = json.load(f)
                    for k, v in data.items():
                        if k not in results_harvest:
                            results_harvest[k] = v
                        else:
                            if isinstance(v, dict) and isinstance(results_harvest[k], dict):
                                results_harvest[k]['count'] = max(results_harvest[k].get('count', 0), v.get('count', 0))
                                results_harvest[k]['max_odds'] = max(results_harvest[k].get('max_odds', 0.0), v.get('max_odds', 0.0))
            except Exception:
                pass

    write_to_summary(build_enhanced_harvest_table(discovery_harvest, "🔍 Discovery Adapters"))

    if results_harvest:
        write_to_summary(build_enhanced_harvest_table(results_harvest, "🏁 Results Adapters"))

    # 3. Enhanced Audit Results
    stats = get_db_stats()
    write_to_summary(format_enhanced_audit_results(stats))

    # 4. Performance Insights
    write_to_summary(format_performance_insights(stats))

    # 5. Intelligence Grids (if available)
    if os.path.exists('summary_grid.txt') or os.path.exists('field_matrix.txt'):
        write_to_summary("")
        write_to_summary("### 📋 Intelligence Grids")
        write_to_summary("")

        if os.path.exists('summary_grid.txt'):
            write_to_summary("<details>")
            write_to_summary("<summary><b>🏁 Race Analysis Grid</b> (click to expand)</summary>")
            write_to_summary("")
            write_to_summary("```")
            with open('summary_grid.txt', 'r', encoding='utf-8') as f:
                write_to_summary(f.read())
            write_to_summary("```")
            write_to_summary("</details>")
            write_to_summary("")

        if os.path.exists('field_matrix.txt'):
            write_to_summary("<details>")
            write_to_summary("<summary><b>📊 Field Matrix</b> (3-11 Runners, click to expand)</summary>")
            write_to_summary("")
            with open('field_matrix.txt', 'r', encoding='utf-8') as f:
                content = f.read()
                # Check if it's markdown table or plain text
                if '|' in content:
                    write_to_summary(content)
                else:
                    write_to_summary("```")
                    write_to_summary(content)
                    write_to_summary("```")
            write_to_summary("</details>")
            write_to_summary("")

    # 6. Report Artifacts
    write_to_summary("### 📁 Detailed Reports")
    write_to_summary("")
    write_to_summary("Download full reports for deeper analysis:")
    write_to_summary("")
    write_to_summary("- 📊 [Summary Grid](summary_grid.txt)")
    write_to_summary("- 🎯 [Field Matrix](field_matrix.txt)")
    write_to_summary("- 💎 [Goldmine Report](goldmine_report.txt)")
    write_to_summary("- 🌐 [HTML Report](fortuna_report.html)")
    write_to_summary("- 📈 [Analytics Log](analytics_report.txt)")
    write_to_summary("- 🗄️ [Database](fortuna.db)")

    # Footer
    write_to_summary("")
    write_to_summary("---")
    write_to_summary("")
    write_to_summary("*🤖 Fortuna Intelligence Engine - Automated Racing Analysis System*")
    write_to_summary("")
    write_to_summary("💡 Artifacts retained for 30 days")

if __name__ == "__main__":
    generate_summary()
