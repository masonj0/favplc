#!/usr/bin/env python3
import json
import logging
import os
import asyncio
import re
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field

# Import Fortuna infrastructure
# Ensure PYTHONPATH=. is set when running this script
try:
    from fortuna import FortunaDB
    from fortuna_utils import (
        EASTERN, STORAGE_FORMAT, DATE_FORMAT,
        now_eastern, from_storage_format, to_storage_format,
        normalize_venue_name, get_canonical_venue
    )
except ImportError:
    # Fallback for local development if PYTHONPATH not set
    import sys
    sys.path.append(os.getcwd())
    from fortuna import FortunaDB
    from fortuna_utils import (
        EASTERN, STORAGE_FORMAT, DATE_FORMAT,
        now_eastern, from_storage_format, to_storage_format,
        normalize_venue_name, get_canonical_venue
    )

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
import glob

STANDARD_BET = 2.00
HERO_VISIBLE = 5
MAX_PREDICTIONS = 15
MAX_RECENT_RESULTS = 15
RECENT_FORM_COUNT = 10
NAME_TRUNCATE = 15

def _discover_harvest_files() -> List[str]:
    """Find all harvest JSON files from any workflow era.
    Handles both old naming (entry-solid/entry-lousy) and new daypart naming.
    """
    patterns = [
        "discovery_harvest_*.json",
        "results_harvest_*.json",
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))

    # Also check inside artifact directories (GHA download-artifact creates subdirs)
    for pattern in patterns:
        files.extend(glob.glob(f"*/{pattern}"))
        files.extend(glob.glob(f"**/{pattern}", recursive=True))

    # Deduplicate by resolved path
    seen = set()
    unique = []
    for f in files:
        resolved = str(Path(f).resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(f)

    return unique

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

VENUE_FLAGS: List[Tuple[List[str], str]] = [
    (["ZA", "TURFFONTEIN", "VAAL", "KENILWORTH", "DURBANVILLE", "GREYVILLE", "SCOTTSVILLE", "FAIRVIEW"], "🇿🇦"),
    (["GB", "UK", "LINGFIELD", "KEMPTON", "SOUTHWELL", "WOLVERHAMPTON", "CHELMSFORD", "NEWCASTLE", "LUDLOW", "TAUNTON", "WARWICK", "PLUMPTON", "AYR", "HEREFORD", "FONTWELL", "CHEPSTOW", "FFOS LAS", "DONCASTER", "MARKET RASEN", "HUNTINGDON", "WINCANTON", "LEICESTER"], "🇬🇧"),
    (["IE", "IRE", "PUNCHESTOWN", "NAAS", "DUNDALK", "FAIRYHOUSE", "LEOPARDSTOWN", "CURRAGH", "THURLES", "GOWRAN", "NAVAN", "TRAMORE", "WEXFORD"], "🇮🇪"),
    (["FR", "PAU", "AUTEUIL", "PARIS", "CHANTILLY", "DEAUVILLE", "SAINT-CLOUD", "LONGCHAMP", "CAGNES-SUR-MER", "LE CROISE", "VINCENNES", "ENGHIEN", "FONTAINEBLEAU", "MARSEILLE", "TOULOUSE", "LYON", "STRASBOURG", "PORNICHET"], "🇫🇷"),
    (["AU", "AUS", "FLEMINGTON", "RANDWICK", "CAULFIELD", "MOONEE VALLEY", "ROSEHILL", "BALAKLAVA", "BALLARAT", "CANTERBURY", "WARWICK FARM", "SANDOWN", "SUNSHINE COAST", "DOOMBEN", "EAGLE FARM"], "🇦🇺"),
    (["NZ", "ELLERSLIE", "TRENTHAM", "RICCARTON", "ASHBURTON"], "🇳🇿"),
    (["US", "USA", "AQUEDUCT", "TURFWAY", "GULFSTREAM", "OAKLAWN", "SANTA ANITA", "CHURCHILL", "KEENELAND", "TAMPA", "SUNLAND", "FAIR GROUNDS", "FINGER LAKES"], "🇺🇸"),
    (["BR", "GAVEA", "MARONAS"], "🇧🇷"),
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
    recent_tips: List[Dict[str, Any]] = field(default_factory=list)

    # Tiered stats (Phase 3)
    best_bet_count: int = 0
    best_bet_cashed: int = 0
    best_bet_profit: float = 0.0

    @property
    def decided(self) -> int:
        return self.cashed + self.burned

    @property
    def win_rate(self) -> float:
        return (self.cashed / self.decided * 100) if self.decided > 0 else 0.0

    @property
    def avg_profit(self) -> float:
        return (self.total_profit / self.decided) if self.decided > 0 else 0.0

    # Lifetime average values set by _get_stats
    lifetime_avg_payout: float = 0.0

    @property
    def avg_payout(self) -> float:
        return self.lifetime_avg_payout if self.lifetime_avg_payout > 0 else 0.0

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
        decided_tips = [r for r in self.recent_tips if r.get('verdict') in VERDICT_EMOJI]
        return "".join(VERDICT_EMOJI.get(r.get('verdict'), "❓") for r in decided_tips[:RECENT_FORM_COUNT])

    @property
    def current_streak(self) -> Tuple[str, int]:
        decided_tips = [r for r in self.recent_tips if r.get('verdict') in VERDICT_EMOJI]
        if not decided_tips: return ("None", 0)
        kind = "W" if decided_tips[0].get('verdict') in CASHED_VERDICTS else "L"
        count = 0
        for r in decided_tips:
            v = "W" if r.get('verdict') in CASHED_VERDICTS else "L"
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
    """Minutes to post. Returns 9999.0 for unparseable or missing times (Fix 14)."""
    if not start_time_str:
        return 9999.0
    try:
        st = from_storage_format(str(start_time_str))
    except (ValueError, TypeError, AttributeError):
        return 9999.0
    if not st:
        return 9999.0
    now = now_eastern()
    return (st - now).total_seconds() / 60

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
    h = now_eastern().hour
    if 0 <= h < 6: return "🌙 Overnight — AUS/NZ/Asian racing active"
    if 6 <= h < 9: return "🌅 Early morning — UK/IRE cards underway"
    if 9 <= h < 12: return "☀️ Morning — Early US cards starting up"
    if 12 <= h < 17: return "🏇 Afternoon — Peak US racing window"
    if 17 <= h < 21: return "🌆 Evening — US twilight + International cards"
    return "🌙 Night — International overnight racing"

def _merge_harvests(paths: List[str]) -> Dict[str, Dict]:
    """Merge harvest data from multiple JSON files.
    Uses MAX for counts (not SUM) to avoid double-counting the same adapter
    across multiple harvest files from retries or overlapping runs.
    """
    merged = {}
    for path in paths:
        data = _read_json(path)
        if not isinstance(data, dict): continue
        for adapter, stats in data.items():
            if adapter not in merged:
                merged[adapter] = dict(stats)
            else:
                # MAX avoids double-counting same races from retried/overlapping runs
                merged[adapter]['count'] = max(
                    merged[adapter].get('count', 0),
                    stats.get('count', 0)
                )
                merged[adapter]['max_odds'] = max(
                    merged[adapter].get('max_odds', 0),
                    stats.get('max_odds', 0)
                )
    return merged

async def _get_stats(db: FortunaDB) -> TipStats:
    stats = TipStats()
    try:
        db_stats = await db.get_stats()
        stats.total_tips = db_stats.get('total_tips', 0)
        stats.cashed = db_stats.get('cashed', 0)
        stats.burned = db_stats.get('burned', 0)
        stats.voided = db_stats.get('voided', 0)
        stats.total_profit = db_stats.get('total_profit', 0.0)
        stats.lifetime_avg_payout = db_stats.get('lifetime_avg_payout', 0.0)

        # Get recent audited tips for form/scoreboard
        recent_audited = await db.get_tips(audited=True, limit=30)
        stats.recent_tips = recent_audited

        # Tiered stats
        for r in recent_audited:
            tier = r.get('tip_tier') or 'best_bet'
            v = r.get('verdict') or ""
            if tier == 'best_bet' and v in (CASHED_VERDICTS + ['BURNED']):
                stats.best_bet_count += 1
                if v in CASHED_VERDICTS:
                    stats.best_bet_cashed += 1
                stats.best_bet_profit += (r.get('net_profit') or 0.0)

        # Count pending (unaudited but started)
        now = now_eastern()
        unaudited = await db.get_tips(audited=False)
        for r in unaudited:
            st = from_storage_format(r.get('start_time'))
            if st and st < now:
                stats.pending += 1

    except Exception as e:
        print(f"Stats Error: {e}")
    return stats

class GHASummaryWriter:
    """GHA-specific summary writer. Writes to GITHUB_STEP_SUMMARY or stdout. (Fix 15)
    See also: fortuna.SummaryWriter for the stream-based variant.
    """
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

def _build_header(out: GHASummaryWriter, now: datetime):
    out.write(f"# 🏇 Favourite to Place — Simply Success Picks")
    out.write(f"### {now.strftime('%y%m%d %A %b %d, %I:%M %p')} ET")
    out.write()
    out.write(f"*{_time_context()}*")
    out.write()

async def _build_action_plan(out: GHASummaryWriter, stats: TipStats, db: FortunaDB):
    status = "HEALTHY"
    emoji = "🟢"
    findings = []
    actions = []

    all_harvest_files = _discover_harvest_files()
    discovery = _merge_harvests(all_harvest_files)
    r_count = sum(d.get('count', 0) for d in discovery.values())
    a_count = len([d for d in discovery.values() if d.get('count', 0) > 0])

    if not Path(db.db_path).exists():
        status, emoji = "CRITICAL", "🔴"
        findings.append("🔴 DB file missing or unreadable.")
        actions.append("Check DB deployment and cache restore.")
    elif a_count == 0:
        # Check harvest logs in DB as fallback
        db_harvest = await db.get_harvest_logs(hours=4)
        if not db_harvest:
            status, emoji = "CRITICAL", "🔴"
            findings.append("🔴 No discovery data — all adapters failed.")
            actions.append("Re-deploy discovery adapters. Check adapter logs in Phase 3.")
        else:
            findings.append(f"🟢 {len(db_harvest)} harvest logs found in DB.")
    else:
        findings.append(f"🟢 {r_count} races discovered from {a_count} adapters.")

    try:
        db_meta = await db.get_stats()
        fresh = db_meta.get('max_report_date')
        if fresh:
            last_seen = from_storage_format(fresh)
            if last_seen and (now_eastern() - last_seen > timedelta(hours=24)):
                if status != "CRITICAL": status, emoji = "WARNING", "🟡"
                findings.append("🟡 No new tips in last 24 hours.")
                actions.append("Check discovery pipeline schedule and adapter health.")

        pop_count = db_meta.get('populated_scoring_count', 0)
        if pop_count == 0 and stats.total_tips > 0:
            if status != "CRITICAL": status, emoji = "WARNING", "🟡"
            findings.append("🟡 All scoring columns still NULL — check log_tips.")
            actions.append("Verify scoring columns are included in log_tips.")
        elif pop_count > 0:
            findings.append("🟢 Scoring signals populating.")

        if stats.margin > 0:
            findings.append(f"🟢 Margin: +{stats.margin:.0f}pp above breakeven.")
        elif stats.decided >= 10 and stats.margin < -5:
            if status == "HEALTHY": status, emoji = "WARNING", "🟡"
            findings.append(f"🟡 Margin: {stats.margin:.0f}pp below breakeven.")
            actions.append("Review qualification thresholds and gap12 minimum.")
    except Exception:
        pass

    out.write(f"## {emoji} System: **{status}**")
    out.write()
    for f in findings:
        out.write(f"- {f}")
    out.write()
    if actions:
        out.write("**Action Plan:**")
        out.write()
        for i, a in enumerate(actions, 1):
            out.write(f"{i}. {a}")
        out.write()
    out.write("---")
    out.write()

async def _build_plays(out: GHASummaryWriter, db: FortunaDB):
    """Build the upcoming plays section.
    Fetches future tips directly, avoiding the stuck-tip ordering problem.
    """
    races = []
    try:
        # FIX-08: Use public DB method instead of private method access
        db_races = await db.get_upcoming_tips(past_minutes=10, future_hours=18, limit=50)
        for r in db_races:
            r['_mtp_val'] = _mtp(r.get('start_time'))
            races.append(r)
    except Exception:
        # Fallback to general query
        try:
            db_races = await db.get_tips(audited=False, limit=50)
            now = now_eastern()
            for r in db_races:
                st = from_storage_format(r.get('start_time'))
                if st and st > now - timedelta(minutes=10):
                    r['_mtp_val'] = _mtp(r.get('start_time'))
                    races.append(r)
        except Exception:
            pass

    if not races:
        out.write("No plays discovered yet.")
        out.write()
        return

    upcoming = sorted([r for r in races if r['_mtp_val'] > -10], key=lambda x: x['_mtp_val'])
    best_bets = [r for r in upcoming if r.get('tip_tier') == 'best_bet' or r.get('is_best_bet')]
    yml_plays = [r for r in upcoming if r.get('tip_tier') == 'you_might_like']

    if not best_bets and not yml_plays:
        out.write("No imminent plays found.")
        out.write()
        return

    soon = len([r for r in upcoming if 0 < r['_mtp_val'] <= 15])
    if soon > 0: out.write(f"## 🔥 {soon} Plays Going Off Soon!")
    else: out.write("## ⚡ Coming Up")
    out.write()

    def _render_table(rows_list):
        out.write("```text")
        out.write(f"  MTP    VENUE                R#   FLD  FAVOURITE              FAV@    GAP   FLAGS")
        out.write(f"  ─────  ───────────────────  ──   ───  ─────────────────────  ──────  ─────  ─────")
        for r in rows_list:
            mtp_s = _mtp_str(r['_mtp_val'])
            v = r.get('venue') or "?"
            flag = _venue_flag(v, r.get('discipline'))
            venue = _trunc(v, 18)
            rn = r.get('race_number', "?")
            fld = r.get('field_size') or "?"
            sn = r.get('selection_number') or "?"
            sname = _trunc(r.get('selection_name') or "Unknown", 17)
            odds = float(r.get('predicted_fav_odds') or 0)
            gap = float(r.get('gap12') or 0)

            flags = []
            if r.get('is_goldmine'): flags.append("GOLD")
            if r.get('is_superfecta_key'): flags.append("KEY")
            grade = r.get('qualification_grade')
            if grade: flags.append(grade)
            flag_str = " ".join(flags)

            out.write(f"  {mtp_s:>4}  {flag}{venue:<18}  {rn:>2}   {fld:>3}  #{sn:<2} {sname:<17}  {odds:>6.2f}  {gap:>5.2f}  {flag_str}")
        out.write("```")

    if best_bets:
        out.write("### 🏆 Best Bets")
        _render_table(best_bets[:HERO_VISIBLE])
        if len(best_bets) > HERO_VISIBLE:
            out.write(f"<details><summary>📋 See all {len(best_bets)} Best Bets</summary>")
            out.write()
            _render_table(best_bets[HERO_VISIBLE:])
            out.write("</details>")
        out.write()

    if yml_plays:
        out.write("---")
        out.write("### 👀 You Might Like")
        _render_table(yml_plays[:HERO_VISIBLE])
        if len(yml_plays) > HERO_VISIBLE:
            out.write(f"<details><summary>📋 See all {len(yml_plays)} supplemental plays</summary>")
            out.write()
            _render_table(yml_plays[HERO_VISIBLE:])
            out.write("</details>")
        out.write()

def _build_keybox(out: GHASummaryWriter):
    # This still reads from the text/json artifact if present for immedate feedback
    content = _read_text("summary_grid.txt")
    if not content: return

    # We can try to extract keys from the grid or just rely on the detailed intelligence grid section
    pass

def _build_scoreboard(out: GHASummaryWriter, stats: TipStats):
    out.write("## 💰 Scoreboard")
    out.write()

    if stats.decided == 0:
        out.write("```text\n  No decided bets yet.\n```")
        return

    out.write("```text")
    bb_decided = stats.best_bet_count
    bb_hit = (stats.best_bet_cashed / bb_decided * 100) if bb_decided > 0 else 0.0
    bb_roi = (stats.best_bet_profit / (bb_decided * STANDARD_BET) * 100) if bb_decided > 0 else 0.0
    out.write(f"  🏆 BEST BETS  {stats.best_bet_cashed}/{bb_decided} ({bb_hit:.0f}%)    ${stats.best_bet_profit:>+8.2f}    ROI {bb_roi:>+.1f}%")
    out.write(f"  👀 ALL PICKS   {stats.cashed}/{stats.decided} ({stats.win_rate:.0f}%)    ${stats.total_profit:>+8.2f}    ROI {stats.roi:>+.1f}%")

    l10 = [r for r in stats.recent_tips if r.get('verdict') in VERDICT_EMOJI][:10]
    w10 = len([r for r in l10 if r.get('verdict') in CASHED_VERDICTS])
    p10 = sum(r.get('net_profit') or 0.0 for r in l10)
    out.write(f"  LAST 10    {w10}/{len(l10):<2}                ${p10:>+8.2f}")

    kind, length = stats.current_streak
    streak_desc = f"{kind}{length}-bet streak" if kind != "None" else "None"
    out.write(f"  STREAK     {stats.profit_icon} {streak_desc}")
    out.write(f"  PAYOUT     avg ${stats.avg_payout:.2f}    breakeven {stats.breakeven_pct:.0f}%    margin {stats.margin:>+.0f}pp")
    out.write("```")
    out.write()
    out.write(f"**Recent:** {stats.streak_bar} {stats.streak_message}")
    out.write()

def _build_recent_results(out: GHASummaryWriter, stats: TipStats):
    if not stats.recent_tips: return
    out.write("## 📊 Recent Results")
    out.write()

    drifts = [( (r.get('actual_fav_odds') or 0) - (r.get('predicted_fav_odds') or 0) )
              for r in stats.recent_tips if r.get('actual_fav_odds') and r.get('predicted_fav_odds')]
    avg_drift = sum(drifts)/len(drifts) if drifts else 0.0
    drift_label = "stable"
    if avg_drift < -0.5: drift_label = "market tightened"
    elif avg_drift > 0.5: drift_label = "market softened"

    def _render_res_table(rows, is_first=False):
        out.write("```text")
        out.write(f"       VENUE                R#   PICK                   FAV@  →  ACTUAL   P/L       FIN")
        out.write(f"  ───  ───────────────────  ──   ─────────────────────  ─────    ──────  ────────  ─────")
        for r in rows:
            v = r.get('verdict')
            e = VERDICT_EMOJI.get(v, "⚪")
            venue_name = r.get('venue') or "?"
            flag = _venue_flag(venue_name, r.get('discipline'))
            venue = _trunc(venue_name, 18)
            rn = r.get('race_number')
            sn = r.get('selection_number') or "?"
            sname = _trunc(r.get('selection_name') or "", 15)
            pick = f"#{sn} {sname}"
            pred = r.get('predicted_fav_odds') or 0.0
            act = r.get('actual_fav_odds') or 0.0
            pl = r.get('net_profit') or 0.0
            pos = r.get('selection_position')
            fin = "—"
            if pos:
                medal = POSITION_EMOJI.get(str(pos), "")
                fin = f"{medal}P{pos}"
            if r.get('superfecta_payout'): fin = f"SF${int(r['superfecta_payout'])}"
            elif r.get('trifecta_payout'): fin = f"T${int(r['trifecta_payout'])}"
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

async def _build_harvest(out: GHASummaryWriter, db: FortunaDB):
    # Dynamic discovery instead of hardcoded filenames
    all_harvest_files = _discover_harvest_files()
    merged = _merge_harvests(all_harvest_files)

    # Also merge DB harvest logs
    db_logs = await db.get_harvest_logs(hours=4)
    db_harvest = {
        l['adapter_name']: {'count': l['race_count'], 'max_odds': l['max_odds']}
        for l in db_logs
    }

    for k, v in db_harvest.items():
        if k not in merged:
            merged[k] = v
        else:
            # Prefer higher counts from DB if available
            merged[k]['count'] = max(merged[k]['count'], v['count'])

    if not merged:
        out.write("## 🛰️ Adapter Harvest\n\nNo adapter data available yet.\n")
        return

    active = len([d for d in merged.values() if d.get('count', 0) > 0])
    total_races = sum(d.get('count', 0) for d in merged.values())

    out.write("## 🛰️ Adapter Harvest")
    out.write()
    out.write(f"{active}/{len(merged)} adapters active · {total_races} races harvested")
    out.write()

    failed = [n for n, d in merged.items() if d.get('count', 0) == 0]
    if failed:
        for f in failed[:5]: out.write(f"- ⚠️ **{f}** — 0 races")
        out.write()

    out.write("<details><summary>📋 Adapter details</summary>")
    out.write()
    out.write("```text")
    out.write(f"  ADAPTER                          RACES   MAX ODDS   STATUS")
    out.write(f"  ────────────────────────────────  ─────  ─────────  ──────")
    sorted_merged = sorted(merged.items(), key=lambda x: x[1].get('count', 0), reverse=True)
    for name, d in sorted_merged:
        cnt = d.get('count', 0)
        mx = d.get('max_odds', 0.0)
        status = '✅' if cnt > 0 else '⚠️ No data'
        out.write(f"  {_trunc(name, 32):<32}  {cnt:>5}  {mx:>9.1f}  {status}")
    out.write("```")
    out.write("</details>")
    out.write()

async def _build_data_quality(out: GHASummaryWriter, db: FortunaDB):
    out.write("## 🔬 Data Quality")
    out.write()
    cols = ["qualification_grade", "composite_score", "is_best_bet", "place_prob", "market_depth", "predicted_ev", "match_confidence"]
    out.write("```text")
    out.write(f"  COLUMN               POPULATED          SAMPLE")
    out.write(f"  ───────────────────  ────────────────  ──────────")

    # FortunaDB uses synchronous sqlite3 + ThreadPoolExecutor, not aiosqlite.
    # Must use _run_in_executor with synchronous DB calls.
    def _query_quality():
        conn = db._get_conn()
        total_row = conn.execute("SELECT COUNT(*) FROM tips").fetchone()
        total = total_row[0] if total_row else 0

        results = []
        for col in cols:
            # Check column exists before querying
            # GPT5 Fix: Use pragma_table_info correctly
            cursor = conn.execute(f"PRAGMA table_info('tips')")
            cols_in_db = [row['name'] for row in cursor.fetchall()]

            if col not in cols_in_db:
                results.append((col, 0, total, "[missing]"))
                continue

            count_row = conn.execute(
                f"SELECT COUNT(*) FROM tips WHERE {col} IS NOT NULL AND CAST({col} AS TEXT) != ''"
            ).fetchone()
            n = count_row[0] if count_row else 0

            sample_row = conn.execute(
                f"SELECT {col} FROM tips WHERE {col} IS NOT NULL AND CAST({col} AS TEXT) != '' ORDER BY id DESC LIMIT 1"
            ).fetchone()
            s_val = str(sample_row[0])[:10] if sample_row and sample_row[0] is not None else "—"
            results.append((col, n, total, s_val))

        return results

    try:
        quality_data = await db._run_in_executor(_query_quality)
        for col, n, total, s_val in quality_data:
            pct = (n / total * 100) if total > 0 else 0
            out.write(f"  {col:<19}  {n:>3}/{total:>3} ({pct:>3.0f}%)  {s_val}")
    except Exception as e:
        out.write(f"  [Error querying data quality: {e}]")
    out.write("```")
    out.write()

async def main():
    now = now_eastern()
    db = FortunaDB()
    await db.initialize()
    stats = await _get_stats(db)
    out = GHASummaryWriter()

    _build_header(out, now)
    await _build_action_plan(out, stats, db)
    await _build_plays(out, db)
    _build_keybox(out)
    out.write("---")
    out.write()
    _build_scoreboard(out, stats)
    _build_recent_results(out, stats)
    out.write("---")
    out.write()
    await _build_harvest(out, db)
    await _build_data_quality(out, db)

    # Intelligence Grids (existing files)
    for path, label in [("summary_grid.txt", "🏁 Race Analysis Grid"), ("field_matrix.txt", "📊 Field Matrix"), ("goldmine_report.txt", "💎 Goldmine Report")]:
        content = _read_text(path)
        if content:
            out.write(f"<details><summary>{label}</summary>\n\n```text\n{content}\n```\n</details>\n")

    # Footer
    out.write("---\n")
    artifacts = [("📊", "Summary Grid", "summary_grid.txt"), ("💎", "Goldmine Report", "goldmine_report.txt"), ("🗄️", "Database", "fortuna.db")]
    links = [f"{e} [{l}]({p})" for e, l, p in artifacts if Path(p).exists()]
    if links: out.write("📦 " + " · ".join(links) + "\n")
    out.write("*Refreshes every hour*")

    out.flush()
    await db.close()

if __name__ == "__main__":
    asyncio.run(main())
