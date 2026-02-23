#!/usr/bin/env python3
import json
import logging
import os
import sqlite3
import re
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
EASTERN = ZoneInfo("America/New_York")
STANDARD_BET = 2.00
HERO_VISIBLE = 5
MAX_PREDICTIONS = 15
MAX_RECENT_RESULTS = 15
RECENT_FORM_COUNT = 10
NAME_TRUNCATE = 15
DB_PATH = "fortuna.db"

DISCOVERY_HARVEST_FILES = ["discovery_harvest_entry-solid.json", "discovery_harvest_entry-lousy.json"]
RESULTS_HARVEST_FILES = ["results_harvest_audit.json"]

CASHED_VERDICTS = ["CASHED", "CASHED_ESTIMATED"]
VERDICT_EMOJI = {
    "CASHED": "✅",
    "CASHED_ESTIMATED": "~✅",
    "BURNED": "❌",
    "VOID": "⚪"
}
POSITION_EMOJI = {
    "1": "🥇", "2": "🥈", "3": "🥉", "4": "4️⃣", "5": "5️⃣"
}

# Country flag derivation keywords
VENUE_FLAGS: List[Tuple[List[str], str]] = [
    (["ZA", "TURFFONTEIN", "VAAL", "KENILWORTH", "DURBANVILLE", "GREYVILLE", "SCOTTSVILLE"], "🇿🇦"),
    (["GB", "UK", "LINGFIELD", "KEMPTON", "SOUTHWELL", "WOLVERHAMPTON", "CHELMSFORD", "NEWCASTLE", "LUDLOW", "TAUNTON", "WARWICK"], "🇬🇧"),
    (["IE", "IRE", "PUNCHESTOWN", "NAAS", "DUNDALK", "FAIRYHOUSE", "LEOPARDSTOWN", "CURRAGH"], "🇮🇪"),
    (["FR", "PARIS", "CHANTILLY", "DEAUVILLE", "SAINT-CLOUD", "LONGCHAMP", "CAGNES-SUR-MER"], "🇫🇷"),
    (["AU", "AUS", "FLEMINGTON", "RANDWICK", "CAULFIELD", "MOONEE VALLEY", "ROSEHILL", "BALAKLAVA", "BALLARAT"], "🇦🇺"),
    (["NZ", "ELLERSLIE", "TRENTHAM", "RICCARTON", "ASHBURTON"], "🇳🇿"),
    (["US", "USA", "AQUEDUCT", "TURFWAY", "GULFSTREAM", "OAKLAWN", "SANTA ANITA", "CHURCHILL", "KEENELAND", "TAMPA"], "🇺🇸"),
]

DISCIPLINE_EMOJI = {
    "greyhound": "🐕",
    "harness": "🏇",
    "thoroughbred": "🏇"
}

_JSON_CACHE: Dict[str, Any] = {}

# ═══════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TipStats:
    total_tips: int = 0
    cashed: int = 0
    burned: int = 0
    voided: int = 0
    pending: int = 0
    total_profit: float = 0.0
    recent_tips: list[tuple] = field(default_factory=list)

    @property
    def decided(self) -> int:
        return self.cashed + self.burned

    @property
    def win_rate(self) -> float:
        return (self.cashed / self.decided * 100) if self.decided > 0 else 0.0

    @property
    def avg_profit(self) -> float:
        return (self.total_profit / self.decided) if self.decided > 0 else 0.0

    @property
    def avg_payout(self) -> float:
        """Mean gross payout per winning bet."""
        wins = [r for r in self.recent_tips if r[4] in CASHED_VERDICTS]
        if not wins: return 0.0
        # profit column is index 5
        return sum((r[5] or 0.0) + STANDARD_BET for r in wins) / len(wins)

    @property
    def breakeven_pct(self) -> float:
        ap = self.avg_payout
        return (STANDARD_BET / ap * 100) if ap > 0 else 100.0

    @property
    def margin(self) -> float:
        return self.win_rate - self.breakeven_pct

    @property
    def roi(self) -> float:
        return (self.total_profit / (self.decided * STANDARD_BET) * 100) if self.decided > 0 else 0.0

    @property
    def profit_icon(self) -> str:
        if self.total_profit > 0.01: return "🟢"
        if self.total_profit < -0.01: return "🔴"
        return "⚪"

    @property
    def streak_bar(self) -> str:
        decided_tips = [r for r in self.recent_tips if r[4] in list(VERDICT_EMOJI.keys())]
        return "".join(VERDICT_EMOJI.get(r[4], "❓") for r in decided_tips[:RECENT_FORM_COUNT])

    @property
    def current_streak(self) -> Tuple[str, int]:
        decided_tips = [r for r in self.recent_tips if r[4] in list(VERDICT_EMOJI.keys())]
        if not decided_tips: return ("None", 0)

        kind = "W" if decided_tips[0][4] in CASHED_VERDICTS else "L"
        count = 0
        for r in decided_tips:
            v = "W" if r[4] in CASHED_VERDICTS else "L"
            if v == kind: count += 1
            else: break
        return kind, count

    @property
    def streak_message(self) -> str:
        kind, length = self.current_streak
        if kind == "W" and length >= 3: return f"🔥 {length}-bet winning streak!"
        if kind == "L" and length >= 3: return f"🧊 {length} losses in a row"
        return ""

# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def _now_et() -> datetime:
    return datetime.now(EASTERN)

def _read_json(path: str) -> Any:
    if path in _JSON_CACHE: return _JSON_CACHE[path]
    p = Path(path)
    if not p.exists(): return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        _JSON_CACHE[path] = data
        return data
    except Exception:
        return None

def _read_text(path: str) -> Optional[str]:
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else None

def _venue_flag(venue: str, discipline: Optional[str] = None) -> str:
    v_up = str(venue).upper()
    for keywords, flag in VENUE_FLAGS:
        if any(k in v_up for k in keywords):
            return flag

    if discipline:
        return DISCIPLINE_EMOJI.get(discipline.lower(), "🏇")
    return "🏇"

def _mtp(start_time_str: Any) -> float:
    if not start_time_str: return 9999.0
    try:
        s = str(start_time_str)
        if 'Z' in s:
            st = datetime.fromisoformat(s.replace('Z', '+00:00'))
        else:
            # Try common format
            try: st = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo("UTC"))
            except ValueError: st = datetime.fromisoformat(s)
            if st.tzinfo is None: st = st.replace(tzinfo=ZoneInfo("UTC"))

        now = datetime.now(ZoneInfo("UTC"))
        return (st - now).total_seconds() / 60
    except Exception:
        return 9999.0

def _mtp_str(minutes: float) -> str:
    if minutes < 0: return "OFF"
    if minutes < 60: return f"{int(minutes)}m"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}h{mins}m"

def _trunc(text: Any, limit: int = 18) -> str:
    s = str(text)
    return s[:limit-1] + "…" if len(s) > limit else s

def _time_context() -> str:
    h = _now_et().hour
    if 0 <= h < 6: return "🌙 Overnight — AUS/NZ/Asian racing active"
    if 6 <= h < 9: return "🌅 Early morning — UK/IRE cards underway"
    if 9 <= h < 12: return "☀️ Morning — Early US cards starting up"
    if 12 <= h < 17: return "🏇 Afternoon — Peak US racing window"
    if 17 <= h < 21: return "🌆 Evening — US twilight + International cards"
    return "🌙 Night — International overnight racing"

def _merge_harvests(paths: List[str]) -> Dict[str, Dict]:
    merged = {}
    for path in paths:
        data = _read_json(path)
        if not isinstance(data, dict): continue
        for adapter, stats in data.items():
            if adapter not in merged:
                merged[adapter] = dict(stats)
            else:
                merged[adapter]['count'] = merged[adapter].get('count', 0) + stats.get('count', 0)
                merged[adapter]['max_odds'] = max(merged[adapter].get('max_odds', 0), stats.get('max_odds', 0))
    return merged

def _get_stats() -> TipStats:
    stats = TipStats()
    if not Path(DB_PATH).exists(): return stats

    try:
        with sqlite3.connect(DB_PATH) as conn:
            now_utc = datetime.now(ZoneInfo("UTC")).isoformat()

            # Main counts
            res = conn.execute("""
                SELECT
                    COUNT(*),
                    SUM(CASE WHEN verdict LIKE 'CASHED%' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN verdict = 'BURNED' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN verdict = 'VOID' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN audit_completed = 0 AND start_time < ? THEN 1 ELSE 0 END),
                    SUM(COALESCE(net_profit, 0.0))
                FROM tips
            """, (now_utc,)).fetchone()

            if res:
                stats.total_tips, stats.cashed, stats.burned, stats.voided, stats.pending, stats.total_profit = (
                    res[0] or 0, res[1] or 0, res[2] or 0, res[3] or 0, res[4] or 0, res[5] or 0.0
                )

            # Recent tips
            stats.recent_tips = conn.execute("""
                SELECT venue, race_number, selection_number, predicted_2nd_fav_odds, verdict,
                       net_profit, selection_position, actual_top_5, actual_2nd_fav_odds,
                       superfecta_payout, trifecta_payout, top1_place_payout, discipline,
                       selection_name
                FROM tips WHERE audit_completed = 1
                ORDER BY audit_timestamp DESC LIMIT 30
            """).fetchall()
    except Exception as e:
        print(f"Stats Error: {e}")
    return stats

class SummaryWriter:
    def __init__(self):
        self.lines = []
        self.output_path = os.environ.get("GITHUB_STEP_SUMMARY", "/dev/stdout")

    def write(self, text: str = ""):
        self.lines.append(text)

    def flush(self):
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")

# ═══════════════════════════════════════════════════════════════════
# SECTION BUILDERS
# ═══════════════════════════════════════════════════════════════════

def _build_header(out: SummaryWriter, now: datetime):
    out.write(f"# 🎯 Fortuna — {now.strftime('%A %b %d, %I:%M %p')} ET")
    out.write()
    out.write(f"*{_time_context()}*")
    out.write()
    out.write("---")
    out.write()

def _build_plays(out: SummaryWriter):
    # 1. Try race_data.json (monitor output)
    data = _read_json("race_data.json")
    races = []
    if data:
        races = data.get("bet_now_races", []) + data.get("you_might_like_races", [])

    # 2. Fallback to DB if empty
    if not races and Path(DB_PATH).exists():
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                db_races = conn.execute("""
                    SELECT * FROM tips WHERE audit_completed=0
                    AND start_time > datetime('now', '-10 minutes')
                    ORDER BY start_time ASC LIMIT 15
                """).fetchall()
                races = [dict(r) for r in db_races]
        except Exception: pass

    if not races:
        out.write("No plays discovered yet — discovery runs every hour. Check back soon!")
        out.write()
        return

    # Process & Sort
    for r in races:
        r['_mtp_val'] = _mtp(r.get('start_time'))

    upcoming = sorted([r for r in races if r['_mtp_val'] > -5], key=lambda x: x['_mtp_val'])
    upcoming = upcoming[:MAX_PREDICTIONS]

    if not upcoming:
        out.write("No imminent plays found.")
        out.write()
        return

    # Dynamic Header
    soon = len([r for r in upcoming if 0 < r['_mtp_val'] <= 15])
    if soon > 0: out.write(f"## 🔥 {soon} Plays Going Off Soon!")
    elif any(r.get('is_goldmine') for r in upcoming if r['_mtp_val'] > 0): out.write("## 🔥 Live Plays")
    else: out.write("## ⚡ Coming Up")
    out.write()

    # Best Bet Callout
    best_bet = next((r for r in upcoming if r.get('is_goldmine') and 0 < r['_mtp_val'] < 120), None)
    if best_bet:
        flag = _venue_flag(best_bet.get('venue') or best_bet.get('track'))
        vname = best_bet.get('venue') or best_bet.get('track')
        sel = best_bet.get('selection_number') or "?"
        name = best_bet.get('selection_name') or best_bet.get('second_fav_name') or "Unknown"
        odds = float(best_bet.get('predicted_2nd_fav_odds') or best_bet.get('second_fav_odds') or 0)
        mtp_s = _mtp_str(best_bet['_mtp_val'])
        gap = float(best_bet.get('gap12', 0))
        out.write(f"> 🏆 **Best Bet:** {flag} {vname} R{best_bet.get('race_number')} — **#{sel} {name}** @ {odds:.2f}")
        out.write(f"> *{mtp_s} to post · Gap: {gap:.2f}*")
        out.write()

    # Monospace Table
    def _render_table(rows_list):
        out.write("```text")
        out.write(f"  MTP    VENUE                R#   FLD  PICK                   ODDS    GAP   FLAGS")
        out.write(f"  ─────  ───────────────────  ──   ───  ─────────────────────  ──────  ─────  ─────")
        for r in rows_list:
            mtp_s = _mtp_str(r['_mtp_val'])
            v = r.get('venue') or r.get('track') or "?"
            flag = _venue_flag(v, r.get('discipline'))
            venue = _trunc(v, 18)
            rn = r.get('race_number', "?")
            fld = r.get('field_size') or "?"
            sn = r.get('selection_number') or "?"
            sname = _trunc(r.get('selection_name') or r.get('second_fav_name') or "", 17)
            odds = float(r.get('predicted_2nd_fav_odds') or r.get('second_fav_odds') or 0)
            gap = float(r.get('gap12') or 0)

            flags = []
            if r.get('is_goldmine'): flags.append("GOLD")
            if r.get('is_superfecta_key'): flags.append("KEY")
            grade = r.get('qualification_grade')
            if grade in ('A+', 'A'): flags.append(grade)
            flag_str = " ".join(flags)

            out.write(f"  {mtp_s:>4}  {flag}{venue:<18}  {rn:>2}   {fld:>3}  #{sn:<2} {sname:<17}  {odds:>6.2f}  {gap:>5.2f}  {flag_str}")
        out.write("```")

    _render_table(upcoming[:HERO_VISIBLE])

    if len(upcoming) > HERO_VISIBLE:
        out.write(f"<details><summary>📋 See all {len(upcoming)} plays</summary>")
        out.write()
        _render_table(upcoming[HERO_VISIBLE:])
        out.write("</details>")
        out.write()

def _build_keybox(out: SummaryWriter):
    data = _read_json("race_data.json")
    races = []
    if data:
        races = data.get("bet_now_races", []) + data.get("you_might_like_races", [])

    keys = [r for r in races if r.get('is_superfecta_key')]
    if not keys: return

    out.write("## 🗝️ Superfecta Keybox")
    out.write()
    out.write("```text")
    out.write(f"  MTP    VENUE                R#   KEY                     BOX (2-3-4)              GAP")
    out.write(f"  ─────  ───────────────────  ──   ──────────────────────  ─────────────────────  ─────")
    for r in keys:
        mtp_s = _mtp_str(_mtp(r.get('start_time')))
        v = r.get('venue') or r.get('track') or "?"
        venue = _trunc(v, 19)
        rn = r.get('race_number', "?")
        kn = r.get('superfecta_key_number') or "?"
        kname = _trunc(r.get('superfecta_key_name') or "", 18)

        box = r.get('superfecta_box_numbers', [])
        if isinstance(box, str): box = [b.strip() for b in box.split(',')]
        box_str = ", ".join([f"#{b}" for b in box]) if box else "—"

        gap = float(r.get('gap12', 0))
        out.write(f"  {mtp_s:>4}  {venue:<19}  {rn:>2}   #{kn:<2} {kname:<18}  {box_str:<21}  {gap:>5.2f}")
    out.write("```")
    out.write()

def _build_scoreboard(out: SummaryWriter, stats: TipStats):
    out.write("## 💰 Scoreboard")
    out.write()

    if stats.decided == 0:
        out.write("```text\n  No decided bets yet.\n```")
        return

    out.write("```text")
    out.write(f"  LIFETIME   {stats.cashed}/{stats.decided} ({stats.win_rate:.0f}%)    ${stats.total_profit:>+8.2f}    ROI {stats.roi:>+.1f}%")

    # Last 10
    l10 = [r for r in stats.recent_tips if r[4] in list(VERDICT_EMOJI.keys())][:10]
    w10 = len([r for r in l10 if r[4] in CASHED_VERDICTS])
    p10 = sum(r[5] or 0.0 for r in l10)
    out.write(f"  LAST 10    {w10}/{len(l10):<2}                ${p10:>+8.2f}")

    kind, length = stats.current_streak
    streak_desc = f"{kind}{length}-bet streak" if kind != "None" else "None"
    out.write(f"  STREAK     {stats.profit_icon} {streak_desc}")

    out.write(f"  PAYOUT     avg ${stats.avg_payout:.2f}    breakeven {stats.breakeven_pct:.0f}%    margin {stats.margin:>+.0f}pp")
    out.write("```")
    out.write()

    out.write(f"**Recent:** {stats.streak_bar} {stats.streak_message}")
    out.write()

    recent_p = sum(r[5] or 0.0 for r in stats.recent_tips[:RECENT_FORM_COUNT])
    trend = "📈" if recent_p > 0 else "📉"
    out.write(f"{trend} Last {len(stats.recent_tips[:RECENT_FORM_COUNT])}: **${recent_p:+.2f}** · Avg/bet: **${stats.avg_profit:+.2f}**")
    out.write()

def _build_recent_results(out: SummaryWriter, stats: TipStats):
    if not stats.recent_tips: return

    out.write("## 📊 Recent Results")
    out.write()

    # Drift Summary
    drifts = [(r[8] - r[3]) for r in stats.recent_tips if r[8] and r[3]]
    avg_drift = sum(drifts)/len(drifts) if drifts else 0.0
    drift_label = "stable"
    if avg_drift < -0.5: drift_label = "market tightened"
    elif avg_drift > 0.5: drift_label = "market softened"

    def _render_res_table(rows, is_first=False):
        out.write("```text")
        out.write(f"       VENUE                R#   PICK                   ODDS  →  ACTUAL   P/L       FIN")
        out.write(f"  ───  ───────────────────  ──   ─────────────────────  ─────    ──────  ────────  ─────")
        for r in rows:
            v = r[4]
            e = VERDICT_EMOJI.get(v, "⚪")
            venue_name = r[0] or "?"
            flag = _venue_flag(venue_name, r[12])
            venue = _trunc(venue_name, 18)
            rn = r[1]
            sn = r[2] or "?"
            sname = _trunc(r[13] or "", 15)
            pick = f"#{sn} {sname}"

            pred = r[3] or 0.0
            act = r[8] or 0.0
            pl = r[5] or 0.0

            pos = r[6]
            fin = "—"
            if pos:
                medal = POSITION_EMOJI.get(str(pos), "")
                fin = f"{medal}P{pos}"

            # Add exotic info if present
            if r[9]: fin = f"SF${int(r[9])}"
            elif r[10]: fin = f"T${int(r[10])}"

            out.write(f"  {e:<2}  {flag}{venue:<18}  {rn:>2}   {pick:<21}  {pred:>5.2f}  →  {act:>5.2f}  ${pl:>+7.2f}  {fin}")

        if is_first:
            out.write(f"  ────────────────────────────────────────────────────────────────────────────────────────")
            out.write(f"  DRIFT: avg {avg_drift:>+.2f} ({drift_label})")
        out.write("```")

    visible = stats.recent_tips[:5]
    _render_res_table(visible, is_first=True)

    if len(stats.recent_tips) > 5:
        out.write(f"<details><summary>📋 All {len(stats.recent_tips)} recent results</summary>")
        out.write()
        _render_res_table(stats.recent_tips[5:])
        out.write("</details>")
        out.write()

def _build_harvest(out: SummaryWriter):
    discovery = _merge_harvests(DISCOVERY_HARVEST_FILES)
    results = _merge_harvests(RESULTS_HARVEST_FILES)

    if not discovery and not results and Path(DB_PATH).exists():
        # Fallback to DB harvest logs
        try:
            with sqlite3.connect(DB_PATH) as conn:
                db_logs = conn.execute("SELECT adapter_name, race_count, max_odds FROM harvest_logs WHERE timestamp >= datetime('now', '-4 hours')").fetchall()
                results = {l[0]: {'count': l[1], 'max_odds': l[2]} for l in db_logs}
        except Exception: pass

    if not discovery and not results:
        out.write("## 🛰️ Adapter Harvest\n\nNo adapter data available yet.\n")
        return

    total_adapters = len(set(discovery.keys()) | set(results.keys()))
    active = len([d for d in (list(discovery.values()) + list(results.values())) if d.get('count', 0) > 0])
    total_races = sum(d.get('count', 0) for d in (list(discovery.values()) + list(results.values())))

    out.write("## 🛰️ Adapter Harvest")
    out.write()
    out.write(f"{active}/{total_adapters} adapters active · {total_races} races harvested")
    out.write()

    failed = [n for n, d in discovery.items() if d.get('count', 0) == 0] + \
             [n for n, d in results.items() if d.get('count', 0) == 0]
    if failed:
        for f in failed[:5]: out.write(f"- ⚠️ **{f}** — 0 races")
        out.write()

    out.write("<details><summary>📋 Adapter details</summary>")
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

    def _render_harvest_table(data, label):
        if not data: return
        out.write(f"**{label} Phase**")
        out.write("```text")
        out.write(f"  ADAPTER                          RACES   MAX ODDS   STATUS")
        out.write(f"  ────────────────────────────────  ─────  ─────────  ──────")
        tot = 0
        sorted_data = sorted(data.items(), key=lambda x: x[1].get('count', 0), reverse=True)
        for name, d in sorted_data:
            cnt = d.get('count', 0)
            mx = d.get('max_odds', 0.0)
            tot += cnt
            status = '✅' if cnt > 0 else '⚠️ No data'
            out.write(f"  {_trunc(name, 32):<32}  {cnt:>5}  {mx:>9.1f}  {status}")
        out.write(f"  ────────────────────────────────  ─────  ─────────  ──────")
        out.write(f"  TOTAL                             {tot:>5}")
        out.write("```")
        out.write()

    _render_harvest_table(discovery, "Discovery")
    _render_harvest_table(results, "Results")
    out.write("</details>")
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

def _build_goldmine_vs_standard(out: SummaryWriter):
    if not Path(DB_PATH).exists(): return
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("SELECT is_goldmine, verdict, net_profit FROM tips WHERE audit_completed=1 AND verdict IN ('CASHED','CASHED_ESTIMATED','BURNED')").fetchall()

        if len(rows) < 5: return

        out.write("## ⛏️ Goldmine vs Standard")
        out.write()
        out.write("```text")
        out.write(f"  CATEGORY      BETS    W    L   HIT%    NET P&L     ROI")
        out.write(f"  ────────────  ────  ───  ───  ─────  ─────────  ──────")
        for label, is_gm in [("🏆 Goldmine", 1), ("📊 Standard", 0)]:
            subset = [r for r in rows if r[0] == is_gm]
            b = len(subset)
            if b == 0: continue
            w = len([r for r in subset if r[1] in CASHED_VERDICTS])
            l = b - w
            h = (w / b * 100)
            pl = sum(r[2] or 0.0 for r in subset)
            roi = (pl / (b * STANDARD_BET)) * 100
            out.write(f"  {label:<12}  {b:>4}  {w:>3}  {l:>3}  {h:>4.0f}%  ${pl:>+8.2f}  {roi:>+5.1f}%")
        out.write("```")
        out.write()
    except Exception: pass

def _build_by_discipline(out: SummaryWriter):
    if not Path(DB_PATH).exists(): return
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("SELECT discipline, verdict, net_profit FROM tips WHERE audit_completed=1 AND verdict IN ('CASHED','CASHED_ESTIMATED','BURNED')").fetchall()

        discs = set(r[0] for r in rows if r[0])
        if len(discs) < 2: return

        out.write("## 🏁 By Discipline")
        out.write()
        out.write("```text")
        out.write(f"  DISCIPLINE      BETS    W    L   HIT%   AVG PAY   B/E%   MARGIN    NET P&L")
        out.write(f"  ──────────────  ────  ───  ───  ─────  ───────  ─────  ────────  ────────")
        for d in sorted(discs):
            subset = [r for r in rows if r[0] == d]
            b = len(subset)
            wins = [r for r in subset if r[1] in CASHED_VERDICTS]
            w = len(wins)
            h = (w / b * 100) if b > 0 else 0
            pl = sum(r[2] or 0.0 for r in subset)
            ap = sum((r[2] or 0.0) + STANDARD_BET for r in wins) / w if w > 0 else 0.0
            be = (STANDARD_BET / ap * 100) if ap > 0 else 100
            m = h - be
            out.write(f"  {d:<14.14}  {b:>4}  {w:>3}  {(b-w):>3}  {h:>4.0f}%  ${ap:>5.2f}  {be:>4.0f}%  {m:>+6.0f}pp  ${pl:>+7.2f}")
        out.write("```")
        out.write()
    except Exception: pass

def _build_exotic_payouts(out: SummaryWriter):
    if not Path(DB_PATH).exists(): return
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("""
                SELECT venue, race_number, DATE(start_time), trifecta_payout, trifecta_combination,
                       superfecta_payout, superfecta_combination
                FROM tips WHERE audit_completed=1
                AND (trifecta_payout > 0 OR superfecta_payout > 0)
                ORDER BY COALESCE(superfecta_payout,0)+COALESCE(trifecta_payout,0) DESC LIMIT 5
            """).fetchall()

        if not rows: return

        out.write("## 🎰 Exotic Payouts")
        out.write()
        out.write("```text")
        out.write(f"  TYPE         PAYOUT      VENUE                R#   DATE        COMBO")
        out.write(f"  ───────────  ──────────  ───────────────────  ──   ──────────  ──────────────")
        for r in rows:
            if r[5]: # Superfecta
                out.write(f"  {'Superfecta':<11}  ${r[5]:>9.2f}  {_trunc(r[0], 19):<19}  {r[1]:>2}   {r[2]}  {_trunc(r[6] or '', 14)}")
            if r[3]: # Trifecta
                out.write(f"  {'Trifecta':<11}  ${r[3]:>9.2f}  {_trunc(r[0], 19):<19}  {r[1]:>2}   {r[2]}  {_trunc(r[4] or '', 14)}")
        out.write("```")
        out.write()
    except Exception: pass

def _build_data_quality(out: SummaryWriter):
    out.write("## 🔬 Data Quality")
    out.write()
    if not Path(DB_PATH).exists(): return

    cols = ["qualification_grade", "composite_score", "is_best_bet", "place_prob", "market_depth", "predicted_ev", "match_confidence"]
    out.write("```text")
    out.write(f"  COLUMN               POPULATED          SAMPLE")
    out.write(f"  ───────────────────  ────────────────  ──────────")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            for col in cols:
                # Security: column names are from a hardcoded whitelist
                res = conn.execute(f"SELECT COUNT(*), SUM(CASE WHEN {col} IS NOT NULL AND CAST({col} AS TEXT) != '' THEN 1 ELSE 0 END) FROM tips").fetchone()
                sample = conn.execute(f"SELECT {col} FROM tips WHERE {col} IS NOT NULL AND CAST({col} AS TEXT) != '' LIMIT 1").fetchone()
                t, n = res[0], res[1] or 0
                pct = (n / t * 100) if t > 0 else 0
                s_val = _trunc(sample[0], 10) if sample else "—"
                out.write(f"  {col:<19}  {n:>3}/{t:>3} ({pct:>3.0f}%)  {s_val}")
    except Exception: pass
    out.write("```")
    out.write()

def _build_intelligence_grids(out: SummaryWriter):
    files = [
        ("summary_grid.txt", "🏁 Race Analysis Grid"),
        ("field_matrix.txt", "📊 Field Matrix"),
        ("goldmine_report.txt", "💎 Goldmine Report")
    ]
    for path, label in files:
        content = _read_text(path)
        if content:
            out.write(f"<details><summary>{label}</summary>")
            out.write()
            out.write("```text")
            out.write(content)
            out.write("```")
            out.write("</details>")
            out.write()

def _build_system_status(out: SummaryWriter, stats: TipStats):
    status = "HEALTHY"
    emoji = "🟢"
    findings = []

    discovery = _merge_harvests(DISCOVERY_HARVEST_FILES)
    r_count = sum(d.get('count', 0) for d in discovery.values())
    a_count = len([d for d in discovery.values() if d.get('count', 0) > 0])

    if not Path(DB_PATH).exists():
        status, emoji = "CRITICAL", "🔴"
        findings.append("🔴 DB file missing or unreadable")
    elif a_count == 0:
        status, emoji = "CRITICAL", "🔴"
        findings.append("🔴 No discovery data — all adapters failed.")
    else:
        findings.append(f"🟢 {r_count} races discovered from {a_count} adapters.")
        if a_count == 1 and r_count < 20:
            status, emoji = "WARNING", "🟡"
            findings.append("🟡 Only 1 adapter returned data — single source risk.")

    if Path(DB_PATH).exists():
        try:
            with sqlite3.connect(DB_PATH) as conn:
                # Freshness
                fresh = conn.execute("SELECT MAX(report_date) FROM tips").fetchone()[0]
                if fresh:
                    last_seen = datetime.fromisoformat(fresh.replace('Z', '+00:00'))
                    if datetime.now(ZoneInfo("UTC")) - last_seen > timedelta(hours=24):
                        if status != "CRITICAL": status, emoji = "WARNING", "🟡"
                        findings.append(f"🟡 No new tips in last 24 hours.")

                # Quality
                q_res = conn.execute("SELECT COUNT(*) FROM tips WHERE qualification_grade IS NOT NULL").fetchone()[0]
                if q_res == 0:
                    if status != "CRITICAL": status, emoji = "WARNING", "🟡"
                    findings.append("🟡 All scoring columns still NULL — check VFIX_01.")
                else:
                    findings.append("🟢 Scoring signals populating.")

                if stats.margin > 0: findings.append(f"🟢 Margin: +{stats.margin:.0f}pp above breakeven.")
        except Exception: pass

    out.write("---")
    out.write()
    out.write(f"## {emoji} System: **{status}**")
    out.write()
    for f in findings: out.write(f"- {f}")
    out.write()
    out.write("---")
    out.write()

    artifacts = [
        ("📊", "Summary Grid", "summary_grid.txt"),
        ("💎", "Goldmine Report", "goldmine_report.txt"),
        ("📈", "Analytics Log", "analytics_report.txt"),
        ("🗄️", "Database", "fortuna.db"),
    ]
    links = [f"{e} [{l}]({p})" for e, l, p in artifacts if Path(p).exists()]
    if links: out.write("📦 " + " · ".join(links))
    out.write()
    out.write("*Refreshes every hour*")

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    now = _now_et()
    stats = _get_stats()
    out = SummaryWriter()

    try: _build_header(out, now)
    except Exception as e: out.write(f"<!-- header failed: {e} -->")

    try: _build_plays(out)
    except Exception as e: out.write(f"<!-- plays failed: {e} -->")

    try: _build_keybox(out)
    except Exception as e: out.write(f"<!-- keybox failed: {e} -->")

    out.write("---")
    out.write()

    try: _build_scoreboard(out, stats)
    except Exception as e: out.write(f"<!-- scoreboard failed: {e} -->")

    try: _build_recent_results(out, stats)
    except Exception as e: out.write(f"<!-- results failed: {e} -->")

    out.write("---")
    out.write()

    try: _build_harvest(out)
    except Exception as e: out.write(f"<!-- harvest failed: {e} -->")

    try: _build_goldmine_vs_standard(out)
    except Exception as e: out.write(f"<!-- goldmine failed: {e} -->")

    try: _build_by_discipline(out)
    except Exception as e: out.write(f"<!-- discipline failed: {e} -->")

    try: _build_exotic_payouts(out)
    except Exception as e: out.write(f"<!-- exotics failed: {e} -->")

    try: _build_data_quality(out)
    except Exception as e: out.write(f"<!-- quality failed: {e} -->")

    try: _build_intelligence_grids(out)
    except Exception as e: out.write(f"<!-- grids failed: {e} -->")

    try: _build_system_status(out, stats)
    except Exception as e: out.write(f"<!-- status failed: {e} -->")

    out.flush()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
