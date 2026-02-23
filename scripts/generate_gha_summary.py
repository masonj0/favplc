#!/usr/bin/env python3
import json
import logging
import os
import sqlite3
import sys
import re
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple

# Constants
EASTERN = ZoneInfo("America/New_York")
STANDARD_BET = 2.00
DB_PATH = "fortuna.db"

# File patterns
DISCOVERY_HARVEST_FILES = ["discovery_harvest_entry-solid.json", "discovery_harvest_entry-lousy.json"]
RESULTS_HARVEST_FILES = ["results_harvest_audit.json"]

logger = logging.getLogger("generate_gha_summary")

def _now_et() -> datetime:
    return datetime.now(EASTERN)

def _trunc(s: Any, length: int) -> str:
    s = str(s)
    if len(s) <= length:
        return s
    return s[:length-1] + "…"

def _mtp_str(start_time_iso: str) -> str:
    try:
        # Handle various ISO formats
        if 'Z' in start_time_iso:
            st = datetime.fromisoformat(start_time_iso.replace('Z', '+00:00'))
        else:
            st = datetime.fromisoformat(start_time_iso)
            if st.tzinfo is None:
                st = st.replace(tzinfo=ZoneInfo("UTC"))

        now = datetime.now(ZoneInfo("UTC"))
        mtp = int((st - now).total_seconds() / 60)
        return str(mtp)
    except Exception:
        return "?"

class SummaryWriter:
    def __init__(self):
        self.lines = []
        self.output_path = os.environ.get("GITHUB_STEP_SUMMARY", "/dev/stdout")

    def write(self, text: str = ""):
        self.lines.append(text)

    def flush(self):
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")

def get_db_conn():
    if not Path(DB_PATH).exists():
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception:
        return None

# ── Section 01: Header ───────────────────────────────────────────────────────
def section_header(out: SummaryWriter):
    now = _now_et()
    day_name = now.strftime("%A")
    mon = now.strftime("%b")
    dd = now.strftime("%d")
    hh_mm = now.strftime("%I:%M %p")

    hour = now.hour
    if hour < 12: greeting = "☀️ Morning"
    elif hour < 17: greeting = "🌤️ Afternoon"
    elif hour < 21: greeting = "🌆 Evening"
    else: greeting = "🌙 Night"

    out.write(f"# 🎯 Fortuna — {day_name} {mon} {dd}, {hh_mm} ET")
    out.write()
    out.write(f"*{greeting}*")
    out.write()
    out.write("---")

# ── Section 02: Scoreboard ───────────────────────────────────────────────────
def section_scoreboard(out: SummaryWriter, conn: sqlite3.Connection):
    out.write("## 💰 Scoreboard")
    out.write()

    if not conn:
        out.write("🔴 Database not found. First run?")
        return

    # Lifetime metrics
    rows = conn.execute("""
        SELECT verdict, net_profit
        FROM tips
        WHERE audit_completed=1 AND verdict IN ('CASHED','CASHED_ESTIMATED','BURNED')
        ORDER BY audit_timestamp DESC
    """).fetchall()

    decided = len(rows)
    if decided == 0:
        out.write("```text\n  No decided bets yet.\n```")
        return

    wins = [r for r in rows if r['verdict'] in ('CASHED', 'CASHED_ESTIMATED')]
    w_count = len(wins)
    hit_pct = (w_count / decided * 100)
    net_profit = sum(r['net_profit'] or 0.0 for r in rows)
    roi = (net_profit / (decided * STANDARD_BET)) * 100

    # Last 10
    last_10 = rows[:10]
    w10 = len([r for r in last_10 if r['verdict'] in ('CASHED', 'CASHED_ESTIMATED')])
    p10 = sum(r['net_profit'] or 0.0 for r in last_10)

    # Streak
    streak_val = 0
    streak_type = None
    for r in rows:
        v = "W" if r['verdict'] in ('CASHED', 'CASHED_ESTIMATED') else "L"
        if streak_type is None:
            streak_type = v
            streak_val = 1
        elif streak_type == v:
            streak_val += 1
        else:
            break

    streak_emoji = "➡️"
    if streak_type == "W" and streak_val >= 3: streak_emoji = "🔥"
    elif streak_type == "L" and streak_val >= 3: streak_emoji = "🧊"
    streak_desc = f"{streak_type}{streak_val}" if streak_type else "None"

    # Payout analysis
    avg_pay = 0.0
    if w_count > 0:
        avg_pay = sum((r['net_profit'] or 0.0) + STANDARD_BET for r in wins) / w_count

    be = (STANDARD_BET / avg_pay * 100) if avg_pay > 0 else 0
    margin = hit_pct - be

    out.write("```text")
    out.write(f"  LIFETIME   {w_count}/{decided} ({hit_pct:.0f}%)    ${net_profit:>+8.2f}    ROI {roi:>+.1f}%")
    out.write(f"  LAST 10    {w10}/{len(last_10):<2}                ${p10:>+8.2f}")
    out.write(f"  STREAK     {streak_emoji} {streak_desc}")
    out.write(f"  PAYOUT     avg ${avg_pay:.2f}    breakeven {be:.0f}%    margin {margin:>+.0f}pp")
    out.write("```")

# ── Section 03: Coming Up ────────────────────────────────────────────────────
def section_coming_up(out: SummaryWriter, conn: sqlite3.Connection):
    if not conn: return

    # Check for upcoming unaudited tips
    rows = conn.execute("""
        SELECT start_time, venue, race_number, field_size, selection_number, selection_name,
               predicted_2nd_fav_odds, gap12, is_goldmine, is_best_bet, is_superfecta_key, qualification_grade
        FROM tips
        WHERE audit_completed=0
          AND start_time > datetime('now', '-10 minutes')
        ORDER BY start_time ASC LIMIT 15
    """).fetchall()

    out.write("## ⚡ Coming Up")
    out.write()

    if not rows:
        out.write("No plays discovered yet — check back next hour.")
        return

    out.write("```text")
    out.write(f"  MTP   VENUE                R#   FLD  SELECTION              ODDS   GAP    FLAGS")
    out.write(f"  ────  ───────────────────  ──   ───  ─────────────────────  ─────  ─────  ─────")

    for r in rows:
        mtp = _mtp_str(r['start_time'])
        venue = _trunc(r['venue'], 19)
        rn = r['race_number']
        fld = r['field_size'] or "?"
        num = r['selection_number'] or "?"
        name = _trunc(r['selection_name'] or "Unknown", 18)
        sel_str = f"#{num} {name}"
        odds = r['predicted_2nd_fav_odds'] or 0.0
        gap = float(r['gap12'] or 0.0)

        flags = []
        if r['is_goldmine']: flags.append("GOLD")
        if r['is_superfecta_key']: flags.append("KEY")
        if r['is_best_bet'] and r['qualification_grade'] in ('A+', 'A'):
            flags.append(r['qualification_grade'])

        flag_str = " ".join(flags)

        out.write(f"  {mtp:>3}m  {venue:<19.19}  {rn:>2}   {fld:>3}  {sel_str:<21.21}  {odds:>5.2f}  {gap:>5.2f}  {flag_str}")
    out.write("```")

# ── Section 04: Keybox ───────────────────────────────────────────────────────
def section_keybox(out: SummaryWriter, conn: sqlite3.Connection):
    if not conn: return

    rows = conn.execute("""
        SELECT start_time, venue, race_number, selection_number, selection_name,
               gap12, is_superfecta_key, superfecta_key_number, superfecta_key_name,
               superfecta_box_numbers, top_five
        FROM tips
        WHERE is_superfecta_key=1
          AND audit_completed=0
          AND start_time > datetime('now', '-10 minutes')
        ORDER BY start_time ASC LIMIT 10
    """).fetchall()

    if not rows: return

    out.write("## 🗝️ Superfecta Keybox")
    out.write()
    out.write("```text")
    out.write(f"  MTP   VENUE                R#   KEY                    BOX (2-3-4)              GAP")
    out.write(f"  ────  ───────────────────  ──   ─────────────────────  ─────────────────────  ─────")

    for r in rows:
        mtp = _mtp_str(r['start_time'])
        venue = _trunc(r['venue'], 19)
        rn = r['race_number']
        kn = r['superfecta_key_number'] or "?"
        kname = _trunc(r['superfecta_key_name'] or "Unknown", 18)
        key_str = f"#{kn} {kname}"

        # Try to get box numbers
        box_str = r['superfecta_box_numbers'] or ""
        if not box_str and r['top_five']:
            t5 = [x.strip() for x in r['top_five'].split(',')]
            if len(t5) >= 4:
                box_str = f"#{t5[1]}, #{t5[2]}, #{t5[3]}"

        gap = float(r['gap12'] or 0.0)
        out.write(f"  {mtp:>3}m  {venue:<19.19}  {rn:>2}   {key_str:<21.21}  {box_str:<21.21}  {gap:>5.2f}")
    out.write("```")

# ── Section 05: Recent Results ───────────────────────────────────────────────
def section_recent_results(out: SummaryWriter, conn: sqlite3.Connection):
    if not conn: return

    rows = conn.execute("""
        SELECT venue, race_number, selection_number, selection_name, predicted_2nd_fav_odds,
               actual_2nd_fav_odds, verdict, net_profit, selection_position, actual_top_5
        FROM tips
        WHERE audit_completed=1
        ORDER BY audit_timestamp DESC LIMIT 15
    """).fetchall()

    if not rows: return

    out.write("## 📊 Recent Results")
    out.write()
    out.write("```text")
    out.write(f"       VENUE                R#   PICK              ODDS  →  ACTUAL   P/L       FIN")
    out.write(f"  ───  ───────────────────  ──   ────────────────  ─────    ──────  ────────  ────")

    total_pl = 0.0
    drifts = []

    for r in rows:
        v = r['verdict']
        e = "⚪"
        if v == "CASHED": e = "✅"
        elif v == "CASHED_ESTIMATED": e = "~✅"
        elif v == "BURNED": e = "❌"

        venue = _trunc(r['venue'], 19)
        rn = r['race_number']
        sn = r['selection_number'] or "?"
        sname = _trunc(r['selection_name'] or "", 12)
        pick = f"#{sn} {sname}"

        pred = r['predicted_2nd_fav_odds'] or 0.0
        act = r['actual_2nd_fav_odds'] or 0.0
        pl = r['net_profit'] or 0.0
        total_pl += pl

        if pred > 0 and act > 0:
            drifts.append(act - pred)

        pos = r['selection_position']
        fin = "—"
        if pos:
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(pos, "")
            fin = f"{medal}P{pos}"

        out.write(f"  {e:<2}  {venue:<19.19}  {rn:>2}   {pick:<16.16}  {pred:>5.2f}  →  {act:>5.2f}  ${pl:>+7.2f}  {fin}")

    avg_drift = sum(drifts)/len(drifts) if drifts else 0.0
    d_label = "stable"
    if avg_drift < -0.5: d_label = "market tightened"
    elif avg_drift > 0.5: d_label = "market softened"

    out.write(f"  ───  ───────────────────  ──   ────────────────  ─────    ──────  ────────  ────")
    out.write(f"  DRIFT: avg {avg_drift:>+5.2f} ({d_label:<16}){' ': <14}  ${total_pl:>+7.2f}")
    out.write("```")

# ── Section 06: Harvest ──────────────────────────────────────────────────────
def section_harvest(out: SummaryWriter, conn: sqlite3.Connection):
    out.write("## 🛰️ Adapter Harvest")
    out.write()

    def load_json(path):
        try:
            if Path(path).exists():
                return json.loads(Path(path).read_text())
        except Exception: pass
        return {}

    discovery = {}
    for f in DISCOVERY_HARVEST_FILES:
        data = load_json(f)
        for k, v in data.items():
            if k not in discovery: discovery[k] = v
            else:
                discovery[k]['count'] = discovery[k].get('count', 0) + v.get('count', 0)
                discovery[k]['max_odds'] = max(discovery[k].get('max_odds', 0), v.get('max_odds', 0))

    results = load_json(RESULTS_HARVEST_FILES[0])
    if not results and conn:
        # Fallback to DB
        try:
            db_logs = conn.execute("""
                SELECT adapter_name, race_count, max_odds
                FROM harvest_logs
                WHERE timestamp >= datetime('now', '-2 hours')
                ORDER BY race_count DESC
            """).fetchall()
            for l in db_logs:
                results[l['adapter_name']] = {'count': l['race_count'], 'max_odds': l['max_odds']}
        except Exception: pass

    if not discovery and not results:
        out.write("No harvest data available for this run.")
        return

    out.write("```text")
    out.write(f"  PHASE       ADAPTER                         RACES   MAX ODDS   STATUS")
    out.write(f"  ──────────  ──────────────────────────────  ─────  ─────────  ──────")

    for phase, data in [("Discovery", discovery), ("Results", results)]:
        if not data: continue
        sorted_adapters = sorted(data.items(), key=lambda x: x[1].get('count', 0), reverse=True)
        for name, d in sorted_adapters:
            cnt = d.get('count', 0)
            mx = d.get('max_odds', 0.0)
            status = "✅" if cnt > 0 else "⚠️"
            out.write(f"  {phase:<10.10}  {_trunc(name, 30):<30}  {cnt:>5}  {mx:>9.1f}  {status}")
    out.write("```")

# ── Section 07: Goldmine vs Standard ─────────────────────────────────────────
def section_goldmine_vs_standard(out: SummaryWriter, conn: sqlite3.Connection):
    if not conn: return
    rows = conn.execute("""
        SELECT is_goldmine, verdict, net_profit
        FROM tips
        WHERE audit_completed=1 AND verdict IN ('CASHED','CASHED_ESTIMATED','BURNED')
    """).fetchall()

    if len(rows) < 5: return

    out.write("## ⛏️ Goldmine vs Standard")
    out.write()
    out.write("```text")
    out.write(f"  CATEGORY      BETS    W    L   HIT%    NET P&L     ROI")
    out.write(f"  ────────────  ────  ───  ───  ─────  ─────────  ──────")

    for label, is_gm in [("🏆 Goldmine", 1), ("📊 Standard", 0)]:
        subset = [r for r in rows if r['is_goldmine'] == is_gm]
        b = len(subset)
        if b == 0: continue
        w = len([r for r in subset if r['verdict'] in ('CASHED', 'CASHED_ESTIMATED')])
        l = b - w
        h = (w / b * 100)
        pl = sum(r['net_profit'] or 0.0 for r in subset)
        roi = (pl / (b * STANDARD_BET)) * 100
        out.write(f"  {label:<12}  {b:>4}  {w:>3}  {l:>3}  {h:>4.0f}%  ${pl:>+8.2f}  {roi:>+5.1f}%")
    out.write("```")

# ── Section 08: By Discipline ────────────────────────────────────────────────
def section_by_discipline(out: SummaryWriter, conn: sqlite3.Connection):
    if not conn: return
    rows = conn.execute("""
        SELECT discipline, verdict, net_profit, actual_2nd_fav_odds
        FROM tips
        WHERE audit_completed=1 AND verdict IN ('CASHED','CASHED_ESTIMATED','BURNED')
    """).fetchall()

    discs = set(r['discipline'] for r in rows if r['discipline'])
    if len(discs) < 2: return

    out.write("## 🏁 By Discipline")
    out.write()
    out.write("```text")
    out.write(f"  DISCIPLINE      BETS    W    L   HIT%   AVG PAY   B/E%   MARGIN    NET P&L")
    out.write(f"  ──────────────  ────  ───  ───  ─────  ───────  ─────  ────────  ────────")

    for d in sorted(discs):
        subset = [r for r in rows if r['discipline'] == d]
        b = len(subset)
        if b == 0: continue
        wins = [r for r in subset if r['verdict'] in ('CASHED', 'CASHED_ESTIMATED')]
        w = len(wins)
        l = b - w
        h = (w / b * 100)
        pl = sum(r['net_profit'] or 0.0 for r in subset)

        ap = sum((r['net_profit'] or 0.0) + STANDARD_BET for r in wins) / w if w > 0 else 0.0
        be = (STANDARD_BET / ap * 100) if ap > 0 else 0
        m = h - be

        out.write(f"  {d:<14.14}  {b:>4}  {w:>3}  {l:>3}  {h:>4.0f}%  ${ap:>5.2f}  {be:>4.0f}%  {m:>+6.0f}pp  ${pl:>+7.2f}")
    out.write("```")

# ── Section 09: Exotic Payouts ───────────────────────────────────────────────
def section_exotic_payouts(out: SummaryWriter, conn: sqlite3.Connection):
    if not conn: return
    rows = conn.execute("""
        SELECT venue, race_number, DATE(start_time) as dt, trifecta_payout, trifecta_combination,
               superfecta_payout, superfecta_combination
        FROM tips
        WHERE audit_completed=1 AND (trifecta_payout IS NOT NULL OR superfecta_payout IS NOT NULL)
        ORDER BY COALESCE(superfecta_payout,0)+COALESCE(trifecta_payout,0) DESC LIMIT 5
    """).fetchall()

    if not rows: return

    out.write("## 🎰 Exotic Payouts")
    out.write()
    out.write("```text")
    out.write(f"  TYPE         PAYOUT      VENUE                R#   DATE        COMBO")
    out.write(f"  ───────────  ──────────  ───────────────────  ──   ──────────  ──────────────")

    for r in rows:
        if r['superfecta_payout']:
            out.write(f"  {'Superfecta':<11}  ${r['superfecta_payout']:>9.2f}  {_trunc(r['venue'], 19):<19.19}  {r['race_number']:>2}   {r['dt']}  {_trunc(r['superfecta_combination'] or '', 14)}")
        if r['trifecta_payout']:
            out.write(f"  {'Trifecta':<11}  ${r['trifecta_payout']:>9.2f}  {_trunc(r['venue'], 19):<19.19}  {r['race_number']:>2}   {r['dt']}  {_trunc(r['trifecta_combination'] or '', 14)}")
    out.write("```")

# ── Section 10: Data Quality ─────────────────────────────────────────────────
def section_data_quality(out: SummaryWriter, conn: sqlite3.Connection):
    out.write("## 🔬 Data Quality")
    out.write()
    if not conn:
        out.write("🔴 No database to check.")
        return

    cols = ["qualification_grade", "composite_score", "is_best_bet", "place_prob", "market_depth", "predicted_ev", "match_confidence"]

    out.write("```text")
    out.write(f"  COLUMN               POPULATED         SAMPLE")
    out.write(f"  ───────────────────  ────────────────  ──────────")

    for col in cols:
        try:
            stats = conn.execute(f"SELECT COUNT(*) as t, SUM(CASE WHEN {col} IS NOT NULL AND {col} != '' THEN 1 ELSE 0 END) as n FROM tips").fetchone()
            sample = conn.execute(f"SELECT {col} FROM tips WHERE {col} IS NOT NULL AND {col} != '' LIMIT 1").fetchone()

            t, n = stats['t'], stats['n']
            pct = (n / t * 100) if t > 0 else 0
            s_val = _trunc(sample[0], 10) if sample else "—"
            out.write(f"  {col:<19.19}  {n:>3}/{t:>3} ({pct:>3.0f}%)  {s_val}")
        except Exception:
            out.write(f"  {col:<19.19}  {'error':<16}  —")
    out.write("```")

# ── Section 11: System Status ────────────────────────────────────────────────
def section_system_status(out: SummaryWriter, conn: sqlite3.Connection):
    status = "HEALTHY"
    emoji = "🟢"
    findings = []

    # Discovery check
    total_discovery_races = 0
    adapters_found = 0
    for f in [Path(p) for p in DISCOVERY_HARVEST_FILES]:
        if f.exists():
            try:
                data = json.loads(f.read_text())
                for d in data.values():
                    if d.get('count', 0) > 0:
                        total_discovery_races += d['count']
                        adapters_found += 1
            except Exception: pass

    if not Path(DB_PATH).exists():
        status, emoji = "CRITICAL", "🔴"
        findings.append("🔴 DB file missing or unreadable")
    elif adapters_found == 0:
        status, emoji = "CRITICAL", "🔴"
        findings.append("🔴 No discovery data — all adapters failed.")
    else:
        findings.append(f"🟢 {total_discovery_races} races discovered from {adapters_found} adapters.")
        if adapters_found == 1 and total_discovery_races < 20:
            status, emoji = "WARNING", "🟡"
            findings.append("🟡 Only 1 adapter returned data — single source dependency.")

    # Freshness check
    if conn:
        try:
            fresh = conn.execute("SELECT MAX(report_date) FROM tips").fetchone()[0]
            if fresh:
                last_seen = datetime.fromisoformat(fresh.replace('Z', '+00:00'))
                if datetime.now(ZoneInfo("UTC")) - last_seen > timedelta(hours=24):
                    if status != "CRITICAL": status, emoji = "WARNING", "🟡"
                    findings.append(f"🟡 No new tips in {int((datetime.now(ZoneInfo('UTC'))-last_seen).total_seconds()/3600)}h.")

            sig_check = conn.execute("SELECT SUM(CASE WHEN qualification_grade IS NOT NULL THEN 1 ELSE 0 END) FROM tips").fetchone()[0]
            if sig_check == 0:
                if status != "CRITICAL": status, emoji = "WARNING", "🟡"
                findings.append("🟡 All scoring columns still NULL — pipeline issue.")
            else:
                findings.append("🟢 Scoring signals populating.")
        except Exception: pass

    if not findings:
        findings.append("🟢 All systems nominal.")

    out.write(f"## {emoji} System: **{status}**")
    out.write()
    for f in findings:
        out.write(f"- {f}")
    out.write()
    out.write("---")
    out.write()

    artifacts = [
        ("📊", "Summary Grid",    "summary_grid.txt"),
        ("💎", "Goldmine Report", "goldmine_report.txt"),
        ("🏁", "Field Matrix",    "field_matrix.txt"),
        ("🖼️", "HTML Dashboard",  "fortuna_report.html"),
        ("📈", "Analytics Log",   "analytics_report.txt"),
        ("🗄️", "Database",        "fortuna.db"),
    ]
    existing = [f"{e} [{l}]({p})" for e, l, p in artifacts if Path(p).exists()]
    if existing:
        out.write("📦 " + " · ".join(existing))
    out.write()
    out.write("*Refreshes every hour*")

def main():
    out = SummaryWriter()
    conn = get_db_conn()

    section_header(out)
    out.write()
    section_scoreboard(out, conn)
    out.write()
    section_coming_up(out, conn)
    out.write()
    section_keybox(out, conn)
    out.write()
    section_recent_results(out, conn)
    out.write()
    section_harvest(out, conn)
    out.write()
    section_goldmine_vs_standard(out, conn)
    out.write()
    section_by_discipline(out, conn)
    out.write()
    section_exotic_payouts(out, conn)
    out.write()
    section_data_quality(out, conn)
    out.write()
    section_system_status(out, conn)

    out.flush()
    if conn: conn.close()

if __name__ == "__main__":
    main()
