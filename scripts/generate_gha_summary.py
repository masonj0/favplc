#!/usr/bin/env python3
"""
Enhanced GitHub Actions Job Summary Generator for Fortuna.
Builds on existing structure with rich predictions, adapter performance, and verified results.
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

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_PREDICTIONS    = 12
MAX_RECENT_TIPS    = 15
MIN_INSIGHTS_TIPS  = 3
RECENT_FORM_COUNT  = 10
NAME_TRUNCATE      = 15
NAME_MAX_DISPLAY   = 18
TOP5_TRUNCATE      = 12
TOP5_MAX_DISPLAY   = 15

VENUE_FLAGS: list[tuple[list[str], str]] = [
    (['kentucky', 'churchill', 'oaklawn', 'tampa', 'gulfstream',
      'santa', 'golden', 'belmont', 'aqueduct', 'parx', 'turfway',
      'delta', 'fair grounds', 'laurel', 'santa anita', 'sam houston',
      'charles town', 'penn national', 'sunland', 'mahoning', 'turf paradise'], '🇺🇸'),
    (['ascot', 'cheltenham', 'newmarket', 'york', 'aintree', 'doncaster'], '🇬🇧'),
    (['longchamp', 'chantilly', 'deauville'], '🇫🇷'),
    (['flemington', 'randwick', 'moonee', 'caulfield'], '🇦🇺'),
    (['woodbine', 'mohawk'], '🇨🇦'),
]

DISCIPLINE_EMOJIS = {
    'greyhound': '🐕',
    'harness':   '🏇',
}

POSITION_EMOJIS = {1: '🥇', 2: '🥈', 3: '🥉', 4: '4️⃣', 5: '5️⃣'}

DISCOVERY_HARVEST_FILES = [
    'discovery_harvest.json',
    'discovery_harvest_usa.json',
    'discovery_harvest_int.json',
    'discovery_harvest_global.json',
]

RESULTS_HARVEST_FILES = [
    'results_harvest.json',
    'results_harvest_audit.json',
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
        return (self.cashed / self.total_tips * 100) if self.total_tips > 0 else 0.0

    @property
    def avg_profit(self) -> float:
        return (self.total_profit / self.total_tips) if self.total_tips > 0 else 0.0

    @property
    def profit_emoji(self) -> str:
        if self.total_profit > 0:
            return '🟢'
        return '🔴' if self.total_profit < 0 else '⚪'


# ── Summary writer ─────────────────────────────────────────────────────────────

class SummaryWriter:
    """Holds an open file handle (or falls back to stdout) for the entire run."""

    def __init__(self, stream: TextIO):
        self._stream = stream

    def write(self, text: str = '') -> None:
        self._stream.write(text + '\n')

    def write_lines(self, lines: list[str]) -> None:
        self._stream.write('\n'.join(lines) + '\n')


@contextmanager
def open_summary():
    """Open the summary file once for the whole generation pass."""
    path = os.environ.get('GITHUB_STEP_SUMMARY')
    if path:
        with open(path, 'a', encoding='utf-8') as fh:
            yield SummaryWriter(fh)
    else:
        import sys
        yield SummaryWriter(sys.stdout)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _read_json(path: str | Path) -> dict | None:
    """Return parsed JSON or None on any failure."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning('Failed to read %s: %s', path, exc)
        return None


def _read_text(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return p.read_text(encoding='utf-8')
    except OSError as exc:
        logger.warning('Failed to read %s: %s', path, exc)
        return None


def get_venue_emoji(venue: str | None, discipline: str | None = None) -> str:
    if discipline:
        emoji = DISCIPLINE_EMOJIS.get(discipline.lower())
        if emoji:
            return emoji

    venue_lower = (venue or '').lower()
    for keywords, flag in VENUE_FLAGS:
        if any(kw in venue_lower for kw in keywords):
            return flag
    return '🏇'


def parse_time_to_minutes(start_time_str: str) -> float:
    try:
        if 'T' in start_time_str:
            st = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
        else:
            st = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S').replace(
                tzinfo=timezone.utc,
            )
        return (st - _now_utc()).total_seconds() / 60.0
    except (ValueError, TypeError):
        return 9999.0


def format_time_remaining(minutes: float) -> str:
    if minutes < 0:
        return '🏁'
    if minutes < 15:
        return f'⚡{int(minutes)}m'
    if minutes < 60:
        return f'🕐{int(minutes)}m'
    hours, mins = divmod(int(minutes), 60)
    return f'🕐{hours}h{mins}m'


def _truncate(text: str, limit: int, display_max: int) -> str:
    return text[:limit] + '...' if len(text) > display_max else text


def _pad(text: Any, width: int, align: str = 'left') -> str:
    s = str(text)
    if align == 'right':
        return s.rjust(width)
    if align == 'center':
        return s.center(width)
    return s.ljust(width)


# ── Database ───────────────────────────────────────────────────────────────────

def get_db_stats(db_path: str = 'fortuna.db') -> TipStats:
    stats = TipStats()
    if not Path(db_path).exists():
        return stats

    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()

            cur.execute("""
                SELECT COUNT(*),
                       SUM(CASE WHEN verdict = 'CASHED' THEN 1 ELSE 0 END),
                       SUM(CASE WHEN verdict = 'BURNED' THEN 1 ELSE 0 END),
                       SUM(CASE WHEN audit_completed = 0 THEN 1 ELSE 0 END),
                       SUM(COALESCE(net_profit, 0.0))
                FROM tips
            """)
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
        logger.error('Error reading database: %s', exc)

    return stats


# ── Harvest merging ────────────────────────────────────────────────────────────

def _merge_harvest_files(paths: list[str]) -> dict[str, dict]:
    """Merge multiple harvest JSON files, keeping max counts/odds per adapter."""
    merged: dict[str, dict] = {}
    for path in paths:
        data = _read_json(path)
        if data is None:
            continue
        for adapter, value in data.items():
            incoming = (
                value if isinstance(value, dict) else {'count': value, 'max_odds': 0.0}
            )
            if adapter not in merged:
                merged[adapter] = dict(incoming)
            else:
                existing = merged[adapter]
                existing['count']    = max(existing.get('count', 0),    incoming.get('count', 0))
                existing['max_odds'] = max(existing.get('max_odds', 0), incoming.get('max_odds', 0.0))
                existing['trust_ratio'] = max(existing.get('trust_ratio', 0.0), incoming.get('trust_ratio', 0.0))
    return merged


# ── Section builders ───────────────────────────────────────────────────────────

def build_harvest_table(summary: dict[str, dict], title: str) -> list[str]:
    header = [
        f'#### {title}',
        "<details>",
        f"<summary><b>View {title}</b> (click to expand)</summary>",
        "",
        "```",
        f"{_pad('Adapter', 25)} | {_pad('Races', 6, 'right')} | {_pad('MaxOdds', 8, 'right')} | {_pad('Quality', 15)} | {_pad('Status', 12)}",
        "-" * 75,
    ]

    if not summary:
        return header + [f"{_pad('*No data*', 25)} | {_pad(0, 6, 'right')} | {_pad('0.0', 8, 'right')} | {_pad('—', 15)} | {_pad('⚠️', 12)}", "```", ""]

    sorted_adapters = sorted(
        summary.items(),
        key=lambda item: (-item[1].get('count', 0), item[0]),
    )

    rows: list[str] = []
    total_races = 0
    total_max_odds = 0.0

    for adapter, data in sorted_adapters:
        count    = data.get('count', 0)
        max_odds = data.get('max_odds', 0.0)
        trust    = data.get('trust_ratio', 0.0)
        total_races += count
        total_max_odds = max(total_max_odds, max_odds)

        if count == 0:
            quality, status = '—', '⚠️ No Data'
        elif trust < 0.3:
            quality, status = 'Poor', 'Garbage'
        elif trust < 0.7:
            quality, status = 'Fair', 'Mixed'
        elif max_odds < 5 and trust > 0.9:
             quality, status = 'Clean', 'Active'
        else:
            quality, status = 'High', 'Active'

        q_str = f"{quality} ({int(trust*100)}%)"
        rows.append(
            f"{_pad(adapter, 25)} | {_pad(count, 6, 'right')} | {_pad(f'{max_odds:.1f}', 8, 'right')} | {_pad(q_str, 15)} | {_pad(status, 12)}"
        )

    if total_races > 0:
        rows.append("-" * 75)
        rows.append(
            f"{_pad('TOTAL', 25)} | {_pad(total_races, 6, 'right')} | {_pad(f'{total_max_odds:.1f}', 8, 'right')} | {_pad('—', 15)} | {_pad('—', 12)}"
        )

    rows.append("```")
    rows.append("</details>")
    return header + rows + ['']


def build_predictions_section(region_filter: str | None = None) -> list[str]:
    title = "🔮 Top Goldmine Predictions"
    if region_filter == "USA":
        title = "🇺🇸 USA Goldmine Predictions"
    elif region_filter == "INT":
        title = "🌐 International Goldmine Predictions"
    elif region_filter == "GLOBAL":
        title = "🌍 Global Goldmine Predictions"

    lines = [
        f"### {title}",
        "<details>",
        f"<summary><b>View {title}</b> (click to expand)</summary>",
        "",
        "*Sorted by time to post — imminent races first!*",
        "",
        "```",
        f"{_pad('MTP', 8)} | {_pad('Date', 5)} | {_pad('Venue', 20)} | {_pad('R#', 3, 'center')} | {_pad('Selection', 20)} | {_pad('Odds', 6, 'right')} | {_pad('Gap', 6, 'right')} | {_pad('Type', 4, 'center')} | {_pad('Top 5', 15)}",
        "-" * 105,
    ]

    data = _read_json('race_data.json')
    if data is None:
        lines.append(f"{_pad('', 8)} | {_pad('', 5)} | {_pad('Awaiting discovery', 20)} | {_pad('', 3)} | {_pad('', 20)} | {_pad('', 6)} | {_pad('', 6)} | {_pad('', 4)} | {_pad('', 15)}")
        lines.append("```")
        lines.append("</details>")
        return lines

    races = data.get('bet_now_races', []) + data.get('you_might_like_races', [])

    # Filter by region if requested
    filtered_races = []
    for r in races:
        emoji = get_venue_emoji(r.get('track'), r.get('discipline'))
        if region_filter == "USA" and emoji != "🇺🇸":
            continue
        if region_filter == "INT" and emoji == "🇺🇸":
            continue
        filtered_races.append(r)

    if not filtered_races:
        lines.append(f"{_pad('', 8)} | {_pad('', 5)} | {_pad('No predictions', 20)} | {_pad('', 3)} | {_pad('', 20)} | {_pad('', 6)} | {_pad('', 6)} | {_pad('', 4)} | {_pad('', 15)}")
        lines.append("```")
        lines.append("</details>")
        return lines

    races_sorted = sorted(
        filtered_races,
        key=lambda r: parse_time_to_minutes(r.get('start_time', '')),
    )

    for race in races_sorted[:MAX_PREDICTIONS]:
        start_time_str = race.get('start_time', '')
        mtp        = parse_time_to_minutes(start_time_str)

        # Extract date from start_time
        try:
            if 'T' in start_time_str:
                dt = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
            else:
                dt = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
            date_str = dt.strftime('%m/%d')
        except Exception:
            date_str = '??/??'

        venue      = race.get('track', 'Unknown')
        # Remove emojis for code block alignment
        # emoji      = get_venue_emoji(venue, discipline)
        race_num   = race.get('race_number', '?')
        # Remove emojis from time_str
        if mtp < 0: time_str = 'OFF'
        elif mtp < 60: time_str = f'{int(mtp)}m'
        else: time_str = f'{int(mtp//60)}h{int(mtp%60)}m'

        sel_name = race.get('second_fav_name', '')
        sel_num  = race.get('selection_number', '?')
        sel_name = _truncate(sel_name, NAME_TRUNCATE, NAME_MAX_DISPLAY)
        selection = f'#{sel_num} {sel_name}'.strip()

        odds     = race.get('second_fav_odds', 0.0)
        odds_str = f'{odds:.2f}' if odds else 'N/A'

        gap      = race.get('gap12', 0.0)
        gap_str  = f'{gap:.2f}' if gap else '—'

        type_str = 'GOLD' if race.get('is_goldmine') else 'BET'

        top5 = race.get('top_five_numbers', 'TBD')
        if isinstance(top5, str):
            top5 = _truncate(top5, TOP5_TRUNCATE, TOP5_MAX_DISPLAY)

        lines.append(
            f"{_pad(time_str, 8)} | {_pad(date_str, 5)} | {_pad(venue, 20)} | {_pad(race_num, 3, 'center')} | {_pad(selection, 20)} | {_pad(odds_str, 6, 'right')} | {_pad(gap_str, 6, 'right')} | {_pad(type_str, 4, 'center')} | {_pad(top5, 15)}"
        )

    lines.append("```")
    lines.append("</details>")
    return lines


def build_audit_section(stats: TipStats) -> list[str]:
    lines = ['', '### 💰 Verified Performance Results', '', '<details>', '<summary><b>View Verified Results</b> (click to expand)</summary>', '']

    if stats.total_tips == 0:
        lines.append('⏳ *No tips audited yet. Results will appear after races complete.*')
        lines.append('</details>')
        return lines

    lines.extend([
        '#### 📊 Overall Statistics',
        '',
        f'- **Total Bets:** {stats.total_tips}',
        f'- **Cashed:** {stats.cashed} ({stats.win_rate:.1f}%)',
        f'- **Burned:** {stats.burned}',
        f'- **Pending:** {stats.pending}',
        f'- **Net P/L:** **${stats.total_profit:+.2f}**',
        '',
    ])

    if not stats.recent_tips:
        return lines

    lines.extend([
        f'#### 🎯 Recent Results (Last {MAX_RECENT_TIPS} Audited)',
        '',
        "```",
        f"{_pad('Verdict', 8)} | {_pad('P/L', 8, 'right')} | {_pad('Venue', 20)} | {_pad('R#', 3, 'center')} | {_pad('Pick', 12)} | {_pad('Finish', 15)} | {_pad('Payouts', 20)}",
        "-" * 95,
    ])

    for tip in stats.recent_tips:
        (venue, race_num, sel_num, pred_odds, verdict, profit,
         sel_pos, actual_top5, _actual_odds,
         sf_payout, tri_payout, pl_payout, discipline) = tip

        v_text = {
            'CASHED': 'WIN',
            'BURNED': 'LOSS',
        }.get(verdict, 'PEND')

        profit = profit or 0.0

        pick_str = f'#{sel_num}'
        if pred_odds:
            pick_str += f' @{pred_odds:.1f}'

        if sel_pos:
            finish_str = f'Pos {sel_pos}'
            if actual_top5:
                finish_str += f' [{actual_top5[:7]}]'
        else:
            finish_str = f'[{actual_top5[:10]}]' if actual_top5 else '—'

        payouts = []
        if sf_payout:
            payouts.append(f'SF${sf_payout:.0f}')
        if tri_payout:
            payouts.append(f'T${tri_payout:.0f}')
        if pl_payout:
            payouts.append(f'P${pl_payout:.0f}')
        payout_str = ' • '.join(payouts) or '—'

        lines.append(
            f"{_pad(v_text, 8)} | {_pad(f'${profit:+.2f}', 8, 'right')} | {_pad(venue[:20], 20)} | {_pad(race_num, 3, 'center')} | {_pad(pick_str, 12)} | {_pad(finish_str, 15)} | {_pad(payout_str, 20)}"
        )

    lines.append("```")
    lines.append('</details>')
    return lines


def build_insights_section(stats: TipStats) -> list[str]:
    if stats.total_tips < MIN_INSIGHTS_TIPS:
        return []

    lines = ['', '### 📈 Performance Insights', '']

    recent = stats.recent_tips[:RECENT_FORM_COUNT]
    if recent:
        recent_profit = sum(tip[5] or 0.0 for tip in recent)
        recent_cashed = sum(1 for tip in recent if tip[4] == 'CASHED')
        recent_wr     = (recent_cashed / len(recent) * 100) if recent else 0
        trend         = '📈' if recent_profit > 0 else '📉'

        lines.extend([
            f'#### 🔥 Recent Form (Last {len(recent)} Bets)',
            '',
            f'- {trend} **Trend:** ${recent_profit:+.2f}',
            f'- **Hit Rate:** {recent_wr:.0f}% ({recent_cashed}/{len(recent)})',
            '',
        ])

    lines.append(f'**Average per Bet:** ${stats.avg_profit:+.2f}')
    return lines


def build_intelligence_grids() -> list[str]:
    grid_files = {
        'summary_grid.txt': ('🏁 Race Analysis Grid', True),
        'field_matrix.txt':  ('📊 Field Matrix (3–11 Runners)', False),
    }
    lines: list[str] = []
    any_exist = any(Path(p).exists() for p in grid_files)
    if not any_exist:
        return lines

    lines.extend(['', '### 📋 Intelligence Grids', ''])

    for path, (label, always_code) in grid_files.items():
        content = _read_text(path)
        if content is None:
            continue
        lines.append('<details>')
        lines.append(f'<summary><b>{label}</b> (click to expand)</summary>')
        lines.append('')
        use_code = always_code or '|' not in content
        if use_code:
            lines.append('```')
        lines.append(content)
        if use_code:
            lines.append('```')
        lines.extend(['</details>', ''])

    return lines


# ── Main ───────────────────────────────────────────────────────────────────────

def generate_summary() -> None:
    now_str = _now_utc().strftime('%Y-%m-%d %H:%M:%S')

    with open_summary() as out:
        out.write(f'*Executive Intelligence Briefing — {now_str} UTC*')
        out.write()

        # --- USA SECTION ---
        out.write("## 🇺🇸 USA Intelligence")
        out.write()
        out.write_lines(build_predictions_section(region_filter="USA"))
        out.write()

        discovery_usa = _merge_harvest_files(['discovery_harvest_usa.json'])
        out.write_lines(build_harvest_table(discovery_usa, '🛰️ USA Discovery Health'))
        out.write()

        # --- INT SECTION ---
        out.write("## 🌐 International Intelligence")
        out.write()
        out.write_lines(build_predictions_section(region_filter="INT"))
        out.write()

        discovery_int = _merge_harvest_files(['discovery_harvest_int.json'])
        out.write_lines(build_harvest_table(discovery_int, '🛰️ International Discovery Health'))
        out.write()

        # --- GLOBAL SECTION ---
        out.write("## 🌍 Global Intelligence")
        out.write()
        out.write_lines(build_predictions_section(region_filter="GLOBAL"))
        out.write()

        discovery_global = _merge_harvest_files(['discovery_harvest_global.json'])
        out.write_lines(build_harvest_table(discovery_global, '🛰️ Global Discovery Health'))
        out.write()

        # --- Common Sections ---
        out.write("## 📊 Global Metrics & Audit")
        out.write()

        results = _merge_harvest_files(RESULTS_HARVEST_FILES)
        if results:
            out.write_lines(build_harvest_table(results, '🏁 Results Adapters Performance'))
        else:
            out.write('#### 🏁 Results Adapters Performance')
            out.write()
            out.write('⏳ *No results harvested in this cycle.*')
            out.write()

        # Audit results
        stats = get_db_stats()
        out.write_lines(build_audit_section(stats))

        # Performance insights
        out.write_lines(build_insights_section(stats))

        # Intelligence grids
        out.write_lines(build_intelligence_grids())

        # Report artifacts
        out.write('### Detailed Reports')
        out.write()
        out.write('Download full reports for deeper analysis:')
        out.write()
        for emoji, label, fname in [
            ('📊', 'Summary Grid',    'summary_grid.txt'),
            ('🎯', 'Field Matrix',    'field_matrix.txt'),
            ('💎', 'Goldmine Report', 'goldmine_report.txt'),
            ('🌐', 'HTML Report',     'fortuna_report.html'),
            ('📈', 'Analytics Log',   'analytics_report.txt'),
            ('🗄️', 'Database',        'fortuna.db'),
        ]:
            out.write(f'- {emoji} [{label}]({fname})')

        # Footer
        out.write()
        out.write('---')
        out.write()
        out.write('💡 Artifacts retained for 30 days')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    generate_summary()
