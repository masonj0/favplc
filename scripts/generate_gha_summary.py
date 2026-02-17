#!/usr/bin/env python3
"""
Fortuna Dashboard — Personal race intelligence summary.
Generates a GitHub Actions Job Summary optimized for quick checking.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO
from zoneinfo import ZoneInfo

# ── Constants ──────────────────────────────────────────────────────────────────

EASTERN = ZoneInfo("America/New_York")

HERO_VISIBLE      = 5      # always-visible rows in the plays table
MAX_PREDICTIONS    = 12
MAX_RECENT_TIPS    = 15
MIN_INSIGHTS_TIPS  = 3
RECENT_FORM_COUNT  = 10
NAME_TRUNCATE      = 15
NAME_MAX_DISPLAY   = 18

CASHED_VERDICTS = frozenset({"CASHED", "CASHED_ESTIMATED"})

VERDICT_ICON = {
    "CASHED":           "✅",
    "CASHED_ESTIMATED": "✅",
    "BURNED":           "❌",
    "VOID":             "⚪",
}

VENUE_FLAGS: list[tuple[list[str], str]] = [
    (["kentucky", "churchill", "oaklawn", "tampa", "gulfstream",
      "santa", "golden", "belmont", "aqueduct", "parx", "turfway",
      "delta", "fair grounds", "laurel", "sam houston", "penn",
      "charles town", "sunland", "mahoning", "turf paradise",
      "saratoga", "monmouth", "woodbine",
      "meadowlands", "yonkers", "mohawk", "flamboro", "northfield",
      "scioto", "hoosier", "pocono", "dover", "miami valley"], "🇺🇸"),
    (["turffontein", "kenilworth", "greyville", "vaal",
      "hollywoodbet"], "🇿🇦"),
    (["ascot", "cheltenham", "newmarket", "york", "aintree",
      "doncaster", "lingfield", "kempton", "sandown", "haydock",
      "curragh", "leopardstown", "fairyhouse", "galway",
      "tipperary", "dundalk", "wolverhampton", "catterick",
      "sedgefield", "plumpton", "exeter", "musselburgh",
      "wetherby", "fontwell"], "🇬🇧"),
    (["longchamp", "chantilly", "deauville"], "🇫🇷"),
    (["flemington", "randwick", "moonee", "caulfield",
      "rosehill", "canterbury", "morphettville"], "🇦🇺"),
]

DISCIPLINE_EMOJI = {"greyhound": "🐕", "harness": "🏇"}

POSITION_EMOJI = {1: "🥇", 2: "🥈", 3: "🥉", 4: "4️⃣", 5: "5️⃣"}

# Region mapping for sorting (Global > Intl > USA)
GLOBAL_ADAPTERS = {
    "SkyRacingWorld", "AtTheRaces", "AtTheRacesGreyhound", "RacingPost",
    "Oddschecker", "Timeform", "BoyleSports", "SportingLife", "SkySports",
    "RacingAndSports"
}
INT_ADAPTERS = {
    "TAB", "BetfairDataScientist", "RacingPostResults", "RacingPostTote",
    "AtTheRacesResults", "SkySportsResults", "RacingAndSportsResults",
    "TimeformResults"
}
USA_ADAPTERS = {
    "Equibase", "TwinSpires", "RacingPostB2B", "StandardbredCanada",
    "EquibaseResults", "StandardbredCanadaResults", "SportingLifeResults"
}

def _get_region_rank(adapter_name: str) -> int:
    """Returns 0 for Global, 1 for Intl, 2 for USA, 3 for Other."""
    # Strip 'Adapter' suffix if present for matching
    name = adapter_name.replace("Adapter", "")
    if name in GLOBAL_ADAPTERS: return 0
    if name in INT_ADAPTERS: return 1
    if name in USA_ADAPTERS: return 2
    return 3

DISCOVERY_HARVEST_FILES = [
    "discovery_harvest.json",
    "discovery_harvest_usa.json",
    "discovery_harvest_int.json",
    "discovery_harvest_global.json",
]
RESULTS_HARVEST_FILES = [
    "results_harvest.json",
    "results_harvest_audit.json",
]

logger = logging.getLogger(__name__)

# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class TipStats:
    total_tips: int = 0
    cashed: int = 0
    burned: int = 0
    pending: int = 0
    total_profit: float = 0.0
    recent_tips: list[tuple] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        decided = self.cashed + self.burned
        return (self.cashed / decided * 100) if decided > 0 else 0.0

    @property
    def avg_profit(self) -> float:
        return (self.total_profit / self.total_tips) if self.total_tips > 0 else 0.0

    @property
    def profit_icon(self) -> str:
        if self.total_profit > 0:
            return "🟢"
        return "🔴" if self.total_profit < 0 else "⚪"

    @property
    def streak_bar(self) -> str:
        """``✅✅❌✅❌`` for last 10 results."""
        return "".join(
            VERDICT_ICON.get(tip[4], "⚪") for tip in self.recent_tips[:10]
        )

    @property
    def current_streak(self) -> tuple[str, int]:
        """(category, length) of the current run — ``('win', 3)``."""
        if not self.recent_tips:
            return ("", 0)
        first_won = self.recent_tips[0][4] in CASHED_VERDICTS
        count = 0
        for tip in self.recent_tips:
            is_win = tip[4] in CASHED_VERDICTS
            if is_win == first_won:
                count += 1
            else:
                break
        return ("win" if first_won else "loss", count)

    @property
    def streak_message(self) -> str:
        kind, length = self.current_streak
        if length < 2:
            return ""
        if kind == "win":
            return f"🔥 *{length}-bet winning streak!*"
        return f"😤 *{length} losses in a row — bounce incoming*"


# ── Writer ─────────────────────────────────────────────────────────────────────

class SummaryWriter:
    """A simple wrapper for GHA Step Summary writes with auto-flush."""
    def __init__(self, stream: TextIO) -> None:
        self._s = stream

    def write(self, text: str = "") -> None:
        self._s.write(text + "\n")
        self._s.flush()

    def lines(self, rows: list[str]) -> None:
        self._s.write("\n".join(rows) + "\n")
        self._s.flush()


@contextmanager
def open_summary():
    """Context manager for writing to GHA Job Summary with fallback to stdout."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            yield SummaryWriter(fh)
    else:
        import sys
        yield SummaryWriter(sys.stdout)


# ── Helpers ────────────────────────────────────────────────────────────────────

_JSON_CACHE: dict[str, Any] = {}

def _now_et() -> datetime:
    return datetime.now(EASTERN)


def _read_json(path: str | Path) -> dict | list | None:
    path_str = str(path)
    if path_str in _JSON_CACHE:
        return _JSON_CACHE[path_str]

    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        _JSON_CACHE[path_str] = data
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None


def _read_text(path: str | Path) -> str | None:
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else None


def _venue_flag(venue: str | None, discipline: str | None = None) -> str:
    if discipline:
        e = DISCIPLINE_EMOJI.get(discipline.lower())
        if e:
            return e
    v = (venue or "").lower()
    for keywords, flag in VENUE_FLAGS:
        if any(kw in v for kw in keywords):
            return flag
    return "🏇"


def _mtp(start_time_str: str) -> float:
    """Minutes to post (negative = already started)."""
    try:
        if "T" in start_time_str:
            st = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
        else:
            st = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc,
            )
        return (st - datetime.now(timezone.utc)).total_seconds() / 60.0
    except (ValueError, TypeError):
        return 9999.0


def _mtp_str(minutes: float) -> str:
    if minutes < 0:
        return "OFF"
    if minutes < 60:
        return f"{int(minutes)}m"
    h, m = divmod(int(minutes), 60)
    return f"{h}h{m}m"


def _trunc(text: str, limit: int = 18) -> str:
    return text[: limit - 3] + "..." if len(text) > limit else text


def _time_context() -> str:
    hour = _now_et().hour
    if hour < 6:
        return "🌙 Overnight — AUS/Asian racing active"
    if hour < 9:
        return "🌅 Early morning — International cards from UK/IRE"
    if hour < 12:
        return "☀️ Morning — Early US cards starting up"
    if hour < 17:
        return "🏇 Afternoon — Peak US racing window"
    if hour < 21:
        return "🌆 Evening — US twilight + International cards"
    return "🌙 Night — International overnight racing"


# ── Database ───────────────────────────────────────────────────────────────────

def _get_stats(db_path: str = "fortuna.db") -> TipStats:
    stats = TipStats()
    if not Path(db_path).exists():
        return stats
    try:
        now_et = _now_et().isoformat()
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            # GPT5 Improvement: Only count 'Pending' if the race has actually run
            cur.execute("""
                SELECT COUNT(*),
                       SUM(CASE WHEN verdict LIKE 'CASHED%' THEN 1 ELSE 0 END),
                       SUM(CASE WHEN verdict = 'BURNED' THEN 1 ELSE 0 END),
                       SUM(CASE WHEN audit_completed = 0 AND start_time < ? THEN 1 ELSE 0 END),
                       SUM(COALESCE(net_profit, 0.0))
                FROM tips
            """, (now_et,))
            row = cur.fetchone()
            if row:
                stats.total_tips   = row[0] or 0
                stats.cashed       = row[1] or 0
                stats.burned       = row[2] or 0
                stats.pending      = row[3] or 0
                stats.total_profit = row[4] or 0.0

            cur.execute("""
                SELECT venue, race_number, selection_number,
                       predicted_2nd_fav_odds, verdict, net_profit,
                       selection_position, actual_top_5, actual_2nd_fav_odds,
                       superfecta_payout, trifecta_payout, top1_place_payout,
                       discipline
                FROM tips
                WHERE audit_completed = 1
                ORDER BY audit_timestamp DESC
                LIMIT ?
            """, (MAX_RECENT_TIPS,))
            stats.recent_tips = cur.fetchall()
    except sqlite3.Error as exc:
        logger.error("DB error: %s", exc)
    return stats


# ── Harvest merging ────────────────────────────────────────────────────────────

def _merge_harvests(paths: list[str]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for path in paths:
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        for adapter, value in data.items():
            inc = value if isinstance(value, dict) else {"count": value, "max_odds": 0.0}
            if adapter not in merged:
                merged[adapter] = dict(inc)
            else:
                ex = merged[adapter]
                ex["count"]    = max(ex.get("count", 0),    inc.get("count", 0))
                ex["max_odds"] = max(ex.get("max_odds", 0), inc.get("max_odds", 0))
    return merged


# ── Section: Plays ─────────────────────────────────────────────────────────────

def _load_races() -> list[dict]:
    data = _read_json("race_data.json")
    if not isinstance(data, dict):
        return []
    races = data.get("bet_now_races", []) + data.get("you_might_like_races", [])
    for r in races:
        r["_mtp"] = _mtp(r.get("start_time", ""))
    races.sort(key=lambda r: r["_mtp"])
    return races


def _build_plays(out: SummaryWriter) -> None:
    races = _load_races()

    if not races:
        out.write("## ⚡ Coming Up")
        out.write()
        out.write(
            "No plays discovered yet — discovery runs every 30 minutes. "
            "Check back soon!"
        )
        out.write()
        return

    # Find best bet (first goldmine that hasn't started, or first race)
    best = next(
        (r for r in races if r.get("is_goldmine") and r["_mtp"] > 0),
        next((r for r in races if r["_mtp"] > 0), None),
    )

    imminent = [r for r in races if 0 < r["_mtp"] <= 15]

    # Header
    has_gold = any(r.get("is_goldmine") and r["_mtp"] > 0 for r in races)
    if imminent:
        out.write(f"## 🔥 {len(imminent)} Play{'s' if len(imminent) != 1 else ''} Going Off Soon!")
    elif has_gold:
        out.write("## 🔥 Live Plays")
    else:
        out.write("## ⚡ Coming Up")
    out.write()

    # Best bet callout
    if best and best["_mtp"] < 120:
        flag  = _venue_flag(best.get("track"), best.get("discipline"))
        venue = best.get("track", "?")
        rnum  = best.get("race_number", "?")
        sel   = best.get("selection_number", "?")
        name  = _trunc(best.get("second_fav_name", ""), 20)
        odds  = best.get("second_fav_odds", 0)
        gap   = best.get("gap12", 0)
        mtp_s = _mtp_str(best["_mtp"])

        out.write(
            f"> 🏆 **Best Bet:** {flag} {venue} R{rnum} — "
            f"**#{sel} {name}** @ {odds:.2f}"
        )
        out.write(f"> *{mtp_s} to post • Odds gap: {gap:.2f}*")
        out.write()

    # Plays table
    upcoming = [r for r in races if r["_mtp"] > -5][:MAX_PREDICTIONS]
    if not upcoming:
        out.write("All races have started — waiting for next card.")
        out.write()
        return

    # Sort upcoming by Region Rank then MTP (GPT5 Requirement)
    upcoming.sort(key=lambda r: (_get_region_rank(r.get("adapter", "")), r["_mtp"]))

    visible = upcoming[:HERO_VISIBLE]
    overflow = upcoming[HERO_VISIBLE:]

    out.write("```text")
    out.lines(_plays_table(visible))
    out.write("```")
    out.write()

    if overflow:
        out.write("<details>")
        out.write(f"<summary>📋 See all {len(upcoming)} plays</summary>")
        out.write()
        out.write("```text")
        out.lines(_plays_table(overflow, continued=True))
        out.write("```")
        out.write()
        out.write("</details>")
        out.write()


def _plays_table(races: list[dict], *, continued: bool = False) -> list[str]:
    """Generates a text-based monospace table for plays."""
    rows: list[str] = []
    # Fixed-width columns for monospace alignment
    # MTP (5) | Race (25) | Pick (20) | Odds (6) | Gap (5)
    header = f"{'MTP':<5} | {'Race':<25} | {'Pick':<20} | {'Odds':<6} | {'Gap':<5}"
    sep    = "-" * len(header)

    if not continued:
        rows.extend([header, sep])

    for r in races:
        venue = _trunc(r.get("track", "?"), 17)
        rnum  = r.get("race_number", "?")
        gold  = "*" if r.get("is_goldmine") else ""
        race_str = f"{gold}{venue} R{rnum}"
        mtp_s = _mtp_str(r["_mtp"])

        sel_num  = r.get("selection_number", "?")
        sel_name = _trunc(r.get("second_fav_name", ""), 15)
        pick     = f"#{sel_num} {sel_name}".strip()

        odds = r.get("second_fav_odds", 0)
        gap  = r.get("gap12", 0)

        rows.append(
            f"{mtp_s:<5} | {race_str:<25} | {pick:<20} | {odds:>6.2f} | {gap:>5.2f}"
        )
    return rows


# ── Section: Scoreboard ───────────────────────────────────────────────────────

def _build_scoreboard(out: SummaryWriter, stats: TipStats) -> None:
    out.write("## 💰 Scoreboard")
    out.write()

    if stats.total_tips == 0:
        if Path("fortuna.db").exists():
            out.write("No results yet — check back after today's races finish.")
        else:
            out.write(
                "🆕 First run! No data yet — predictions will appear "
                "after the first discovery cycle."
            )
        out.write()
        return

    decided = stats.cashed + stats.burned
    wr_str  = f"{stats.win_rate:.0f}% ({stats.cashed}/{decided})" if decided else "—"

    out.write(
        f"**${stats.total_profit:+.2f}** {stats.profit_icon} profit "
        f"• Win rate **{wr_str}** "
        f"• Pending: {stats.pending}"
    )
    out.write()

    # Streak
    bar = stats.streak_bar
    if bar:
        msg = stats.streak_message
        streak_line = f"**Recent:** {bar}"
        if msg:
            streak_line += f" — {msg}"
        out.write(streak_line)
        out.write()

    # Insights
    if stats.total_tips >= MIN_INSIGHTS_TIPS:
        recent = stats.recent_tips[:RECENT_FORM_COUNT]
        if recent:
            r_profit = sum(tip[5] or 0.0 for tip in recent)
            trend = "📈" if r_profit > 0 else "📉"
            out.write(
                f"{trend} Last {len(recent)}: **${r_profit:+.2f}** "
                f"• Avg/bet: **${stats.avg_profit:+.2f}**"
            )
            out.write()

    # Detailed results (collapsed)
    if stats.recent_tips:
        out.write("<details>")
        out.write(
            f"<summary>📋 Recent results ({len(stats.recent_tips)} bets)</summary>"
        )
        out.write()
        out.write("```text")
        out.lines(_results_table(stats.recent_tips))
        out.write("```")
        out.write()
        out.write("</details>")
        out.write()


def _results_table(tips: list[tuple]) -> list[str]:
    """Generates a text-based monospace table for results."""
    # Result | P/L | Race | Pick | Finish
    header = f"{'Res':<4} | {'P/L':<8} | {'Race':<20} | {'Pick':<12} | {'Finish':<15}"
    sep    = "-" * len(header)
    rows = [header, sep]

    for tip in tips:
        # (venue, race_num, sel_num, pred_odds, verdict, profit,
        #  sel_pos, actual_top5, actual_2nd_odds,
        #  sf_payout, tri_payout, pl_payout, discipline)
        venue      = _trunc(tip[0] or "?", 14)
        race_num   = tip[1]
        sel_num    = tip[2]
        pred_odds  = tip[3]
        verdict    = tip[4] or "?"
        profit     = tip[5] or 0.0
        sel_pos    = tip[6]
        actual_t5  = tip[7] or ""
        sf_pay     = tip[9]
        tri_pay    = tip[10]

        res_map = {"CASHED": "WIN", "CASHED_ESTIMATED": "WIN", "BURNED": "LOSS", "VOID": "VOID"}
        res_text = res_map.get(verdict, "???")

        race_str = f"{venue} R{race_num}"
        pick = f"#{sel_num}"
        if pred_odds:
            pick += f" @{pred_odds:.1f}"

        finish_parts = []
        if sel_pos:
            pos_icon = POSITION_EMOJI.get(sel_pos, "")
            finish_parts.append(f"{pos_icon}P{sel_pos}" if pos_icon else f"P{sel_pos}")
        if sf_pay:
            finish_parts.append(f"SF${sf_pay:.0f}")
        elif tri_pay:
            finish_parts.append(f"T${tri_pay:.0f}")
        elif actual_t5:
            finish_parts.append(f"[{_trunc(actual_t5, 8)}]")
        finish = " ".join(finish_parts) or "-"

        rows.append(
            f"{res_text:<4} | ${profit:>7.2f} | {race_str:<20} | {pick:<12} | {finish:<15}"
        )
    return rows


# ── Section: System Health ─────────────────────────────────────────────────────

def _build_system(out: SummaryWriter) -> None:
    discovery = _merge_harvests(DISCOVERY_HARVEST_FILES)
    results   = _merge_harvests(RESULTS_HARVEST_FILES)
    combined  = {**discovery, **results}

    total_adapters = len(combined)
    active  = sum(1 for d in combined.values() if d.get("count", 0) > 0)
    races   = sum(d.get("count", 0) for d in combined.values())
    failed  = [n for n, d in combined.items() if d.get("count", 0) == 0]

    # One-line status
    if not combined:
        out.write("## 🔧 System")
        out.write()
        out.write("No adapter data available yet.")
        out.write()
        return

    if not failed:
        out.write(f"## 🔧 System: ✅ All Good")
        out.write()
        out.write(f"{active} adapters active • {races} races harvested")
    else:
        out.write(f"## 🔧 System: ⚠️ {len(failed)} Issue{'s' if len(failed) != 1 else ''}")
        out.write()
        out.write(
            f"{active}/{total_adapters} adapters active • {races} races harvested"
        )
        out.write()
        for name in failed[:5]:
            out.write(f"- **{name}** — returned 0 races")
    out.write()

    # Collapsed detail
    out.write("<details>")
    out.write("<summary>📋 Adapter details</summary>")
    out.write()

    if discovery:
        out.lines(_adapter_table(discovery, "Discovery"))
        out.write()
    if results:
        out.lines(_adapter_table(results, "Results"))
        out.write()

    out.write("</details>")
    out.write()

def _adapter_table(adapters: dict[str, dict], label: str) -> list[str]:
    """Generates a text-based monospace table for adapters, sorted by region rank (GPT5)."""
    header = f"{'Adapter':<20} | {'Races':>5} | {'MaxOdds':>8} | {'Status'}"
    sep    = "-" * len(header)
    rows = [f"**{label} Adapters**", "", "```text", header, sep]

    total = 0
    # Sort by Region Rank (Global > Intl > USA) then count
    sorted_adapters = sorted(
        adapters.items(),
        key=lambda x: (_get_region_rank(x[0]), -x[1].get("count", 0))
    )

    for name, data in sorted_adapters:
        count = data.get("count", 0)
        odds  = data.get("max_odds", 0.0)
        total += count
        status = "Active" if count > 0 else "No data"
        rows.append(f"{name[:20]:<20} | {count:>5} | {odds:>8.1f} | {status}")

    if len(adapters) > 1:
        rows.append(f"{'Total':<20} | {total:>5} | {'':>8} |")

    rows.append("```")
    return rows


# ── Section: Intelligence Grids ────────────────────────────────────────────────

def _build_grids(out: SummaryWriter) -> None:
    files = {
        "summary_grid.txt": "🏁 Race Analysis Grid",
        "field_matrix.txt":  "📊 Field Matrix",
    }
    any_exist = any(Path(p).exists() for p in files)
    if not any_exist:
        return

    for path, label in files.items():
        content = _read_text(path)
        if not content:
            continue
        out.write("<details>")
        out.write(f"<summary>{label}</summary>")
        out.write()
        out.write("```")
        out.write(content)
        out.write("```")
        out.write("</details>")
        out.write()


# ── Main ───────────────────────────────────────────────────────────────────────

def generate_summary() -> None:
    now = _now_et()
    date_str = now.strftime("%a %b %d, %I:%M %p ET")

    with open_summary() as out:
        # Title
        out.write(f"# 🎯 Fortuna — {date_str}")
        out.write()
        out.write(f"*{_time_context()}*")
        out.write()
        out.write("---")
        out.write()

        # 1. What to bet on
        _build_plays(out)

        out.write("---")
        out.write()

        # 2. How you're doing
        stats = _get_stats()
        _build_scoreboard(out, stats)

        out.write("---")
        out.write()

        # 3. Is the system working
        _build_system(out)

        # 4. Deep-dive data (collapsed)
        _build_grids(out)

        # 5. Artifact links
        out.write("---")
        out.write()
        artifacts = [
            ("📊", "Summary Grid",    "summary_grid.txt"),
            ("💎", "Goldmine Report", "goldmine_report.txt"),
            ("📈", "Analytics Log",   "analytics_report.txt"),
            ("🗄️", "Database",        "fortuna.db"),
        ]
        # Only list files that actually exist
        existing = [
            (e, l, f) for e, l, f in artifacts if Path(f).exists()
        ]
        if existing:
            links = " • ".join(f"{e} [{l}]({f})" for e, l, f in existing)
            out.write(f"📦 {links}")
            out.write()

        out.write("*Refreshes every 30 minutes*")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_summary()
