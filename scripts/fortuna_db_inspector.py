#!/usr/bin/env python3
"""
fortuna_db_inspector.py

Connects to a Fortuna SQLite database and produces a comprehensive
structural and statistical report for auditing and evaluation.

Usage:
    python fortuna_db_inspector.py [path_to_db]

If no path is given, defaults to FORTUNA_DB_PATH env var or "fortuna.db".
"""

import os
import sys
import sqlite3
import json
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = os.environ.get("FORTUNA_DB_PATH", "fortuna.db")

SEPARATOR = "═" * 80
SUBSEP = "─" * 60
INDENT = "  "


# ── Formatting Helpers ───────────────────────────────────────────────────────

def fmt_number(n: Any) -> str:
    """Format a number with commas, or return 'N/A'."""
    if n is None:
        return "N/A"
    if isinstance(n, float):
        return f"{n:,.2f}"
    return f"{n:,}"


def fmt_pct(numerator: float, denominator: float) -> str:
    if denominator == 0:
        return "N/A"
    return f"{numerator / denominator * 100:.1f}%"


def fmt_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def section(title: str) -> str:
    return f"\n{SEPARATOR}\n  {title}\n{SEPARATOR}"


def subsection(title: str) -> str:
    return f"\n{INDENT}{SUBSEP}\n{INDENT}{title}\n{INDENT}{SUBSEP}"


def table_rows(headers: List[str], rows: List[List[Any]], col_widths: Optional[List[int]] = None) -> str:
    """Render a simple ASCII table."""
    if not rows:
        return f"{INDENT}(no data)\n"

    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(str(h))
            for row in rows:
                if i < len(row):
                    max_w = max(max_w, len(str(row[i])))
            col_widths.append(min(max_w + 1, 50))

    lines = []
    header_line = INDENT + "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append(INDENT + "".join("─" * w for w in col_widths))
    for row in rows:
        line = INDENT + "".join(str(v).ljust(w) for v, w in zip(row, col_widths))
        lines.append(line)
    return "\n".join(lines) + "\n"


# ── Core Inspector ───────────────────────────────────────────────────────────

class FortunaDBInspector:
    """Read-only inspector for a Fortuna SQLite database."""

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            print(f"ERROR: Database not found at '{self.db_path}'")
            sys.exit(1)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.cur = self.conn.cursor()
        self.output: List[str] = []

    def close(self) -> None:
        self.conn.close()

    def log(self, text: str = "") -> None:
        self.output.append(text)

    def print_report(self) -> None:
        print("\n".join(self.output))

    def run_full_report(self) -> None:
        self.log(section("FORTUNA DATABASE INSPECTION REPORT"))
        self.log(f"{INDENT}Generated: {datetime.now().strftime('%y%m%dT%H:%M:%S')}")
        self.log(f"{INDENT}Database:  {self.db_path.resolve()}")

        self._file_metadata()
        self._schema_overview()
        self._per_table_analysis()
        self._tips_analysis()
        self._audit_analysis()
        self._results_analysis()
        self._races_analysis()
        self._runners_analysis()
        self._data_quality_checks()
        self._temporal_analysis()
        self._pnl_summary()
        self._matching_diagnostic()

        self.log(section("END OF REPORT"))

    # ── File-level metadata ──────────────────────────────────────────────────

    def _file_metadata(self) -> None:
        self.log(section("1. FILE METADATA"))

        stat = self.db_path.stat()
        self.log(f"{INDENT}File size:     {fmt_bytes(stat.st_size)}")
        self.log(f"{INDENT}Last modified: {datetime.fromtimestamp(stat.st_mtime).strftime('%y%m%dT%H:%M:%S')}")

        # SQLite metadata
        self.cur.execute("PRAGMA page_size;")
        page_size = self.cur.fetchone()[0]
        self.cur.execute("PRAGMA page_count;")
        page_count = self.cur.fetchone()[0]
        self.cur.execute("PRAGMA freelist_count;")
        free_pages = self.cur.fetchone()[0]
        self.cur.execute("PRAGMA journal_mode;")
        journal = self.cur.fetchone()[0]
        self.cur.execute("PRAGMA integrity_check;")
        integrity = self.cur.fetchone()[0]

        self.log(f"{INDENT}Page size:     {fmt_number(page_size)} bytes")
        self.log(f"{INDENT}Total pages:   {fmt_number(page_count)}")
        self.log(f"{INDENT}Free pages:    {fmt_number(free_pages)} ({fmt_pct(free_pages, page_count)} fragmentation)")
        self.log(f"{INDENT}Journal mode:  {journal}")
        self.log(f"{INDENT}Integrity:     {integrity}")

    # ── Schema overview ──────────────────────────────────────────────────────

    def _schema_overview(self) -> None:
        self.log(section("2. SCHEMA OVERVIEW"))

        self.cur.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY type, name;")
        objects = self.cur.fetchall()

        tables = [o for o in objects if o["type"] == "table"]
        views = [o for o in objects if o["type"] == "view"]

        self.log(f"{INDENT}Tables: {len(tables)}    Views: {len(views)}")

        # Index summary
        self.cur.execute("SELECT name, tbl_name FROM sqlite_master WHERE type='index' ORDER BY tbl_name, name;")
        indexes = self.cur.fetchall()
        self.log(f"{INDENT}Indexes: {len(indexes)}")

        if indexes:
            self.log(subsection("Indexes"))
            idx_rows = [[idx["name"], idx["tbl_name"]] for idx in indexes]
            self.log(table_rows(["Index Name", "Table"], idx_rows))

        # Table structure
        self.log(subsection("Table Structures"))

        for tbl in tables:
            name = tbl["name"]
            self.cur.execute(f"PRAGMA table_info({self._safe_name(name)});")
            columns = self.cur.fetchall()
            self.cur.execute(f"SELECT COUNT(*) as cnt FROM {self._safe_name(name)};")
            row_count = self.cur.fetchone()["cnt"]

            self.log(f"\n{INDENT}╔══ {name}  ({fmt_number(row_count)} rows)")

            col_rows = []
            for col in columns:
                pk_marker = " [PK]" if col["pk"] else ""
                nn_marker = " NOT NULL" if col["notnull"] else ""
                default = f" DEFAULT {col['dflt_value']}" if col["dflt_value"] is not None else ""
                col_rows.append([
                    f"  {col['name']}",
                    col["type"] or "ANY",
                    f"{pk_marker}{nn_marker}{default}",
                ])

            self.log(table_rows(["Column", "Type", "Constraints"], col_rows))

    # ── Per-table row counts and sample data ─────────────────────────────────

    def _per_table_analysis(self) -> None:
        self.log(section("3. TABLE ROW COUNTS & SAMPLE DATA"))

        tables = self._get_table_names()

        summary_rows = []
        for name in tables:
            self.cur.execute(f"SELECT COUNT(*) as cnt FROM {self._safe_name(name)};")
            count = self.cur.fetchone()["cnt"]
            summary_rows.append([name, fmt_number(count)])

        self.log(table_rows(["Table", "Row Count"], summary_rows))

        # Show 3 sample rows from each table
        for name in tables:
            self.cur.execute(f"SELECT COUNT(*) as cnt FROM {self._safe_name(name)};")
            count = self.cur.fetchone()["cnt"]
            if count == 0:
                continue

            self.log(subsection(f"Sample rows from '{name}' (up to 3)"))
            self.cur.execute(f"SELECT * FROM {self._safe_name(name)} LIMIT 3;")
            rows = self.cur.fetchall()
            if rows:
                headers = rows[0].keys()
                data = []
                for row in rows:
                    data.append([self._truncate(str(row[h]), 40) for h in headers])
                self.log(table_rows(list(headers), data))

    # ── Tips analysis ────────────────────────────────────────────────────────

    def _tips_analysis(self) -> None:
        self.log(section("4. TIPS ANALYSIS"))

        tips_table = self._find_table_like("tip")
        if not tips_table:
            self.log(f"{INDENT}No tips table found.")
            return

        self.cur.execute(f"SELECT COUNT(*) as cnt FROM {self._safe_name(tips_table)};")
        total = self.cur.fetchone()["cnt"]
        self.log(f"{INDENT}Total tips: {fmt_number(total)}")

        if total == 0:
            return

        columns = self._get_columns(tips_table)

        # Venue breakdown
        if "venue" in columns:
            self.log(subsection("Tips by Venue (top 20)"))
            self.cur.execute(f"SELECT venue, COUNT(*) as cnt FROM {self._safe_name(tips_table)} GROUP BY venue ORDER BY cnt DESC LIMIT 20;")
            rows = [[r["venue"], fmt_number(r["cnt"])] for r in self.cur.fetchall()]
            self.log(table_rows(["Venue", "Count"], rows))

        # Source breakdown
        if "source" in columns:
            self.log(subsection("Tips by Source"))
            self.cur.execute(f"SELECT source, COUNT(*) as cnt FROM {self._safe_name(tips_table)} GROUP BY source ORDER BY cnt DESC;")
            rows = [[r["source"], fmt_number(r["cnt"])] for r in self.cur.fetchall()]
            self.log(table_rows(["Source", "Count"], rows))

        # Discipline breakdown
        if "discipline" in columns:
            self.log(subsection("Tips by Discipline"))
            self.cur.execute(f"SELECT discipline, COUNT(*) as cnt FROM {self._safe_name(tips_table)} GROUP BY discipline ORDER BY cnt DESC;")
            rows = [[r["discipline"] or "(null)", fmt_number(r["cnt"])] for r in self.cur.fetchall()]
            self.log(table_rows(["Discipline", "Count"], rows))

        # Verified vs unverified
        verified_col = self._find_column_like(tips_table, "verified", "audited", "audit")
        if verified_col:
            self.log(subsection("Verification Status"))
            self.cur.execute(f"SELECT {self._safe_name(verified_col)}, COUNT(*) as cnt FROM {self._safe_name(tips_table)} GROUP BY {self._safe_name(verified_col)};")
            rows = [[str(r[verified_col]), fmt_number(r["cnt"])] for r in self.cur.fetchall()]
            self.log(table_rows(["Status", "Count"], rows))

        # Date range
        date_col = self._find_column_like(tips_table, "start_time", "date", "created", "timestamp")
        if date_col:
            self.log(subsection("Date Range"))
            self.cur.execute(f"SELECT MIN({self._safe_name(date_col)}) as earliest, MAX({self._safe_name(date_col)}) as latest FROM {self._safe_name(tips_table)};")
            r = self.cur.fetchone()
            self.log(f"{INDENT}Earliest: {r['earliest']}")
            self.log(f"{INDENT}Latest:   {r['latest']}")

        # Null analysis for key columns
        self.log(subsection("Null / Empty Value Analysis"))
        null_rows = []
        for col in columns:
            self.cur.execute(f"SELECT COUNT(*) as cnt FROM {self._safe_name(tips_table)} WHERE {self._safe_name(col)} IS NULL OR CAST({self._safe_name(col)} AS TEXT) = '';")
            null_count = self.cur.fetchone()["cnt"]
            if null_count > 0:
                null_rows.append([col, fmt_number(null_count), fmt_pct(null_count, total)])
        if null_rows:
            self.log(table_rows(["Column", "Null/Empty", "% of Total"], null_rows))
        else:
            self.log(f"{INDENT}No null or empty values found in any column.")

    # ── Audit analysis ───────────────────────────────────────────────────────

    def _audit_analysis(self) -> None:
        self.log(section("5. AUDIT / VERDICT ANALYSIS"))

        # Look for verdict data in tips table or a separate audit table
        tips_table = self._find_table_like("tip")
        audit_table = self._find_table_like("audit")

        target_table = None
        verdict_col = None

        for candidate in [tips_table, audit_table]:
            if candidate:
                vc = self._find_column_like(candidate, "verdict")
                if vc:
                    target_table = candidate
                    verdict_col = vc
                    break

        if not target_table or not verdict_col:
            self.log(f"{INDENT}No verdict/audit data found.")
            return

        columns = self._get_columns(target_table)

        self.cur.execute(f"SELECT COUNT(*) as cnt FROM {self._safe_name(target_table)} WHERE {self._safe_name(verdict_col)} IS NOT NULL AND {self._safe_name(verdict_col)} != '';")
        audited_total = self.cur.fetchone()["cnt"]
        self.log(f"{INDENT}Audited tips: {fmt_number(audited_total)}")

        if audited_total == 0:
            return

        # Verdict distribution
        self.log(subsection("Verdict Distribution"))
        self.cur.execute(f"""
            SELECT {self._safe_name(verdict_col)}, COUNT(*) as cnt
            FROM {self._safe_name(target_table)}
            WHERE {self._safe_name(verdict_col)} IS NOT NULL AND {self._safe_name(verdict_col)} != ''
            GROUP BY {self._safe_name(verdict_col)}
            ORDER BY cnt DESC;
        """)
        verdict_rows = self.cur.fetchall()
        rows = [[r[verdict_col], fmt_number(r["cnt"]), fmt_pct(r["cnt"], audited_total)] for r in verdict_rows]
        self.log(table_rows(["Verdict", "Count", "% of Audited"], rows))

        # Win rate (CASHED)
        cashed = sum(r["cnt"] for r in verdict_rows if r[verdict_col] and "CASH" in str(r[verdict_col]).upper())
        burned = sum(r["cnt"] for r in verdict_rows if r[verdict_col] and "BURN" in str(r[verdict_col]).upper())
        decided = cashed + burned
        if decided > 0:
            self.log(f"\n{INDENT}Strike rate (CASHED / decided): {fmt_pct(cashed, decided)}")
            self.log(f"{INDENT}  CASHED: {fmt_number(cashed)}  |  BURNED: {fmt_number(burned)}  |  Total decided: {fmt_number(decided)}")

        # Profit/loss column
        profit_col = self._find_column_like(target_table, "net_profit", "profit", "pnl", "pl")
        if profit_col:
            self.log(subsection("Profit / Loss Summary"))
            self.cur.execute(f"""
                SELECT
                    SUM({self._safe_name(profit_col)}) as total_pnl,
                    AVG({self._safe_name(profit_col)}) as avg_pnl,
                    MIN({self._safe_name(profit_col)}) as worst,
                    MAX({self._safe_name(profit_col)}) as best,
                    COUNT(*) as cnt
                FROM {self._safe_name(target_table)}
                WHERE {self._safe_name(profit_col)} IS NOT NULL;
            """)
            r = self.cur.fetchone()
            self.log(f"{INDENT}Total P&L:   {fmt_number(r['total_pnl'])}")
            self.log(f"{INDENT}Average P&L: {fmt_number(r['avg_pnl'])}")
            self.log(f"{INDENT}Best single:  {fmt_number(r['best'])}")
            self.log(f"{INDENT}Worst single: {fmt_number(r['worst'])}")
            self.log(f"{INDENT}Tips with P&L data: {fmt_number(r['cnt'])}")

            # P&L by verdict
            self.log(subsection("P&L by Verdict"))
            self.cur.execute(f"""
                SELECT {self._safe_name(verdict_col)},
                    COUNT(*) as cnt,
                    SUM({self._safe_name(profit_col)}) as total,
                    AVG({self._safe_name(profit_col)}) as avg
                FROM {self._safe_name(target_table)}
                WHERE {self._safe_name(profit_col)} IS NOT NULL AND {self._safe_name(verdict_col)} IS NOT NULL
                GROUP BY {self._safe_name(verdict_col)}
                ORDER BY total DESC;
            """)
            rows = [
                [r[verdict_col], fmt_number(r["cnt"]), fmt_number(r["total"]), fmt_number(r["avg"])]
                for r in self.cur.fetchall()
            ]
            self.log(table_rows(["Verdict", "Count", "Total P&L", "Avg P&L"], rows))

            # P&L by venue (top 15)
            if "venue" in columns:
                self.log(subsection("P&L by Venue (top 15 by volume)"))
                self.cur.execute(f"""
                    SELECT venue,
                        COUNT(*) as cnt,
                        SUM({self._safe_name(profit_col)}) as total,
                        AVG({self._safe_name(profit_col)}) as avg,
                        SUM(CASE WHEN {self._safe_name(verdict_col)} LIKE '%CASH%' THEN 1 ELSE 0 END) as wins
                    FROM {self._safe_name(target_table)}
                    WHERE {self._safe_name(profit_col)} IS NOT NULL
                    GROUP BY venue
                    ORDER BY cnt DESC
                    LIMIT 15;
                """)
                rows = [
                    [r["venue"], fmt_number(r["cnt"]), fmt_number(r["total"]),
                     fmt_number(r["avg"]), fmt_pct(r["wins"], r["cnt"])]
                    for r in self.cur.fetchall()
                ]
                self.log(table_rows(["Venue", "Tips", "Total P&L", "Avg P&L", "Strike%"], rows))

            # P&L by source
            if "source" in columns:
                self.log(subsection("P&L by Source"))
                self.cur.execute(f"""
                    SELECT source,
                        COUNT(*) as cnt,
                        SUM({self._safe_name(profit_col)}) as total,
                        AVG({self._safe_name(profit_col)}) as avg,
                        SUM(CASE WHEN {self._safe_name(verdict_col)} LIKE '%CASH%' THEN 1 ELSE 0 END) as wins
                    FROM {self._safe_name(target_table)}
                    WHERE {self._safe_name(profit_col)} IS NOT NULL
                    GROUP BY source
                    ORDER BY cnt DESC;
                """)
                rows = [
                    [r["source"], fmt_number(r["cnt"]), fmt_number(r["total"]),
                     fmt_number(r["avg"]), fmt_pct(r["wins"], r["cnt"])]
                    for r in self.cur.fetchall()
                ]
                self.log(table_rows(["Source", "Tips", "Total P&L", "Avg P&L", "Strike%"], rows))

    # ── Results analysis ─────────────────────────────────────────────────────

    def _results_analysis(self) -> None:
        self.log(section("6. RESULTS DATA ANALYSIS"))

        results_table = self._find_table_like("result")
        if not results_table:
            self.log(f"{INDENT}No results table found.")
            return

        self.cur.execute(f"SELECT COUNT(*) as cnt FROM {self._safe_name(results_table)};")
        total = self.cur.fetchone()["cnt"]
        self.log(f"{INDENT}Total result records: {fmt_number(total)}")

        if total == 0:
            return

        columns = self._get_columns(results_table)

        # Venue breakdown
        if "venue" in columns:
            self.log(subsection("Results by Venue (top 15)"))
            self.cur.execute(f"SELECT venue, COUNT(*) as cnt FROM {self._safe_name(results_table)} GROUP BY venue ORDER BY cnt DESC LIMIT 15;")
            rows = [[r["venue"], fmt_number(r["cnt"])] for r in self.cur.fetchall()]
            self.log(table_rows(["Venue", "Count"], rows))

        # Exotic payouts summary
        for exotic in ["trifecta_payout", "exacta_payout", "superfecta_payout"]:
            if exotic in columns:
                self.cur.execute(f"""
                    SELECT COUNT(*) as cnt, AVG({self._safe_name(exotic)}) as avg, MAX({self._safe_name(exotic)}) as mx
                    FROM {self._safe_name(results_table)}
                    WHERE {self._safe_name(exotic)} IS NOT NULL AND {self._safe_name(exotic)} > 0;
                """)
                r = self.cur.fetchone()
                if r["cnt"] > 0:
                    self.log(f"{INDENT}{exotic}: {fmt_number(r['cnt'])} records, avg={fmt_number(r['avg'])}, max={fmt_number(r['mx'])}")

    # ── Races analysis ───────────────────────────────────────────────────────

    def _races_analysis(self) -> None:
        self.log(section("7. RACES DATA ANALYSIS"))

        races_table = self._find_table_like("race")
        if not races_table:
            self.log(f"{INDENT}No races table found.")
            return

        self.cur.execute(f"SELECT COUNT(*) as cnt FROM {self._safe_name(races_table)};")
        total = self.cur.fetchone()["cnt"]
        self.log(f"{INDENT}Total race records: {fmt_number(total)}")

        if total == 0:
            return

        columns = self._get_columns(races_table)

        if "venue" in columns:
            self.log(subsection("Races by Venue (top 15)"))
            self.cur.execute(f"SELECT venue, COUNT(*) as cnt FROM {self._safe_name(races_table)} GROUP BY venue ORDER BY cnt DESC LIMIT 15;")
            rows = [[r["venue"], fmt_number(r["cnt"])] for r in self.cur.fetchall()]
            self.log(table_rows(["Venue", "Count"], rows))

        if "discipline" in columns:
            self.log(subsection("Races by Discipline"))
            self.cur.execute(f"SELECT discipline, COUNT(*) as cnt FROM {self._safe_name(races_table)} GROUP BY discipline ORDER BY cnt DESC;")
            rows = [[r["discipline"] or "(null)", fmt_number(r["cnt"])] for r in self.cur.fetchall()]
            self.log(table_rows(["Discipline", "Count"], rows))

        if "source" in columns:
            self.log(subsection("Races by Source"))
            self.cur.execute(f"SELECT source, COUNT(*) as cnt FROM {self._safe_name(races_table)} GROUP BY source ORDER BY cnt DESC;")
            rows = [[r["source"] or "(null)", fmt_number(r["cnt"])] for r in self.cur.fetchall()]
            self.log(table_rows(["Source", "Count"], rows))

    # ── Runners analysis ─────────────────────────────────────────────────────

    def _runners_analysis(self) -> None:
        self.log(section("8. RUNNERS DATA ANALYSIS"))

        runners_table = self._find_table_like("runner")
        if not runners_table:
            self.log(f"{INDENT}No runners table found.")
            return

        self.cur.execute(f"SELECT COUNT(*) as cnt FROM {self._safe_name(runners_table)};")
        total = self.cur.fetchone()["cnt"]
        self.log(f"{INDENT}Total runner records: {fmt_number(total)}")

        if total == 0:
            return

        columns = self._get_columns(runners_table)

        if "scratched" in columns:
            self.cur.execute(f"SELECT COUNT(*) as cnt FROM {self._safe_name(runners_table)} WHERE scratched = 1;")
            scratched = self.cur.fetchone()["cnt"]
            self.log(f"{INDENT}Scratched: {fmt_number(scratched)} ({fmt_pct(scratched, total)})")

        # Odds distribution
        odds_col = self._find_column_like(runners_table, "win_odds", "odds", "final_win_odds")
        if odds_col:
            self.log(subsection("Odds Distribution"))
            self.cur.execute(f"""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN {self._safe_name(odds_col)} IS NULL THEN 1 ELSE 0 END) as null_odds,
                    AVG({self._safe_name(odds_col)}) as avg_odds,
                    MIN({self._safe_name(odds_col)}) as min_odds,
                    MAX({self._safe_name(odds_col)}) as max_odds
                FROM {self._safe_name(runners_table)};
            """)
            r = self.cur.fetchone()
            self.log(f"{INDENT}With odds: {fmt_number(r['total'] - r['null_odds'])}  |  Missing odds: {fmt_number(r['null_odds'])}")
            self.log(f"{INDENT}Avg: {fmt_number(r['avg_odds'])}  |  Min: {fmt_number(r['min_odds'])}  |  Max: {fmt_number(r['max_odds'])}")

            # Odds brackets
            self.log(subsection("Odds Brackets"))
            self.cur.execute(f"""
                SELECT
                    CASE
                        WHEN {self._safe_name(odds_col)} < 2 THEN 'Heavy fav (<2.0)'
                        WHEN {self._safe_name(odds_col)} < 4 THEN 'Short (2.0-3.9)'
                        WHEN {self._safe_name(odds_col)} < 8 THEN 'Mid (4.0-7.9)'
                        WHEN {self._safe_name(odds_col)} < 15 THEN 'Long (8.0-14.9)'
                        WHEN {self._safe_name(odds_col)} < 30 THEN 'Very long (15-29.9)'
                        ELSE 'Outsider (30+)'
                    END as bracket,
                    COUNT(*) as cnt
                FROM {self._safe_name(runners_table)}
                WHERE {self._safe_name(odds_col)} IS NOT NULL AND {self._safe_name(odds_col)} > 0
                GROUP BY bracket
                ORDER BY MIN({self._safe_name(odds_col)});
            """)
            rows = [[r["bracket"], fmt_number(r["cnt"])] for r in self.cur.fetchall()]
            self.log(table_rows(["Bracket", "Count"], rows))

    # ── Data quality checks ──────────────────────────────────────────────────

    def _data_quality_checks(self) -> None:
        self.log(section("9. DATA QUALITY CHECKS"))

        tables = self._get_table_names()
        issues_found = 0

        for tbl in tables:
            columns = self._get_columns(tbl)
            self.cur.execute(f"SELECT COUNT(*) as cnt FROM {self._safe_name(tbl)};")
            total = self.cur.fetchone()["cnt"]

            if total == 0:
                self.log(f"{INDENT}⚠  Table '{tbl}' is empty")
                issues_found += 1
                continue

            # Check for completely null columns
            for col in columns:
                self.cur.execute(f"SELECT COUNT(*) as cnt FROM {self._safe_name(tbl)} WHERE {self._safe_name(col)} IS NOT NULL;")
                non_null = self.cur.fetchone()["cnt"]
                if non_null == 0:
                    self.log(f"{INDENT}⚠  '{tbl}.{col}' is entirely NULL ({fmt_number(total)} rows)")
                    issues_found += 1

            # Check for potential duplicate race_ids
            if "race_id" in columns:
                self.cur.execute(f"""
                    SELECT race_id, COUNT(*) as cnt
                    FROM {self._safe_name(tbl)}
                    GROUP BY race_id
                    HAVING cnt > 1
                    ORDER BY cnt DESC
                    LIMIT 5;
                """)
                dupes = self.cur.fetchall()
                if dupes:
                    self.log(f"{INDENT}⚠  '{tbl}' has duplicate race_ids:")
                    for d in dupes:
                        self.log(f"{INDENT}     {d['race_id']} appears {d['cnt']} times")
                    issues_found += 1

        # Check for orphaned references
        tips_table = self._find_table_like("tip")
        results_table = self._find_table_like("result")
        if tips_table and results_table:
            tips_cols = self._get_columns(tips_table)
            results_cols = self._get_columns(results_table)

            if "race_id" in tips_cols and "race_id" in results_cols:
                self.cur.execute(f"""
                    SELECT COUNT(*) as cnt
                    FROM {self._safe_name(tips_table)} t
                    LEFT JOIN {self._safe_name(results_table)} r ON t.race_id = r.race_id
                    WHERE r.race_id IS NULL;
                """)
                orphaned = self.cur.fetchone()["cnt"]
                if orphaned > 0:
                    self.log(f"{INDENT}ℹ  {fmt_number(orphaned)} tips have no matching result record")

        if issues_found == 0:
            self.log(f"{INDENT}✓  No critical data quality issues detected.")
        else:
            self.log(f"\n{INDENT}Total issues found: {issues_found}")

    # ── Temporal analysis ────────────────────────────────────────────────────

    def _temporal_analysis(self) -> None:
        self.log(section("10. TEMPORAL ANALYSIS"))

        tips_table = self._find_table_like("tip")
        if not tips_table:
            self.log(f"{INDENT}No tips table for temporal analysis.")
            return

        columns = self._get_columns(tips_table)
        date_col = self._find_column_like(tips_table, "start_time", "date", "created", "timestamp")
        if not date_col:
            self.log(f"{INDENT}No date column found for temporal analysis.")
            return

        # Tips per day (last 14 days)
        self.log(subsection("Tips per Day (last 14 days with data)"))
        self.cur.execute(f"""
            SELECT DATE({self._safe_name(date_col)}) as day, COUNT(*) as cnt
            FROM {self._safe_name(tips_table)}
            WHERE {self._safe_name(date_col)} IS NOT NULL
            GROUP BY day
            ORDER BY day DESC
            LIMIT 14;
        """)
        rows = [[r["day"], fmt_number(r["cnt"])] for r in self.cur.fetchall()]
        if rows:
            self.log(table_rows(["Date", "Tips"], rows))
        else:
            self.log(f"{INDENT}No temporal data available.")

        # Tips per day-of-week
        self.log(subsection("Tips by Day of Week"))
        self.cur.execute(f"""
            SELECT
                CASE CAST(strftime('%w', {self._safe_name(date_col)}) AS INTEGER)
                    WHEN 0 THEN 'Sunday'
                    WHEN 1 THEN 'Monday'
                    WHEN 2 THEN 'Tuesday'
                    WHEN 3 THEN 'Wednesday'
                    WHEN 4 THEN 'Thursday'
                    WHEN 5 THEN 'Friday'
                    WHEN 6 THEN 'Saturday'
                END as dow,
                COUNT(*) as cnt
            FROM {self._safe_name(tips_table)}
            WHERE {self._safe_name(date_col)} IS NOT NULL
            GROUP BY strftime('%w', {self._safe_name(date_col)})
            ORDER BY strftime('%w', {self._safe_name(date_col)});
        """)
        rows = [[r["dow"], fmt_number(r["cnt"])] for r in self.cur.fetchall()]
        if rows:
            self.log(table_rows(["Day", "Tips"], rows))

        # Tips per hour
        self.log(subsection("Tips by Hour (ET / local)"))
        self.cur.execute(f"""
            SELECT
                CAST(strftime('%H', {self._safe_name(date_col)}) AS INTEGER) as hour,
                COUNT(*) as cnt
            FROM {self._safe_name(tips_table)}
            WHERE {self._safe_name(date_col)} IS NOT NULL
            GROUP BY hour
            ORDER BY hour;
        """)
        rows = [[f"{r['hour']:02d}:00", fmt_number(r["cnt"])] for r in self.cur.fetchall()]
        if rows:
            self.log(table_rows(["Hour", "Tips"], rows))

        # Verdict trends: strike rate by week
        verdict_col = self._find_column_like(tips_table, "verdict")
        if verdict_col:
            self.log(subsection("Weekly Strike Rate (recent 8 weeks with data)"))
            self.cur.execute(f"""
                SELECT
                    strftime('%Y-W%W', {self._safe_name(date_col)}) as week,
                    COUNT(*) as total,
                    SUM(CASE WHEN {self._safe_name(verdict_col)} LIKE '%CASH%' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN {self._safe_name(verdict_col)} LIKE '%BURN%' THEN 1 ELSE 0 END) as losses
                FROM {self._safe_name(tips_table)}
                WHERE {self._safe_name(verdict_col)} IS NOT NULL AND {self._safe_name(verdict_col)} != ''
                    AND {self._safe_name(date_col)} IS NOT NULL
                GROUP BY week
                ORDER BY week DESC
                LIMIT 8;
            """)
            weekly = self.cur.fetchall()
            rows = [
                [r["week"], fmt_number(r["total"]),
                 fmt_number(r["wins"]), fmt_number(r["losses"]),
                 fmt_pct(r["wins"], r["wins"] + r["losses"])]
                for r in weekly
            ]
            if rows:
                self.log(table_rows(["Week", "Audited", "Wins", "Losses", "Strike%"], rows))

    # ── P&L summary ──────────────────────────────────────────────────────────

    def _pnl_summary(self) -> None:
        self.log(section("11. CUMULATIVE P&L SUMMARY"))

        tips_table = self._find_table_like("tip")
        if not tips_table:
            self.log(f"{INDENT}No tips table for P&L analysis.")
            return

        columns = self._get_columns(tips_table)
        profit_col = self._find_column_like(tips_table, "net_profit", "profit", "pnl", "pl")
        verdict_col = self._find_column_like(tips_table, "verdict")

        if not profit_col:
            self.log(f"{INDENT}No profit column found.")
            return

        self.cur.execute(f"""
            SELECT
                COUNT(*) as total_tips,
                SUM(CASE WHEN {self._safe_name(profit_col)} IS NOT NULL THEN 1 ELSE 0 END) as with_pnl,
                SUM({self._safe_name(profit_col)}) as total_pnl,
                AVG({self._safe_name(profit_col)}) as avg_pnl,
                SUM(CASE WHEN {self._safe_name(profit_col)} > 0 THEN {self._safe_name(profit_col)} ELSE 0 END) as total_winnings,
                SUM(CASE WHEN {self._safe_name(profit_col)} < 0 THEN ABS({self._safe_name(profit_col)}) ELSE 0 END) as total_losses,
                SUM(CASE WHEN {self._safe_name(profit_col)} > 0 THEN 1 ELSE 0 END) as winning_bets,
                SUM(CASE WHEN {self._safe_name(profit_col)} < 0 THEN 1 ELSE 0 END) as losing_bets,
                SUM(CASE WHEN {self._safe_name(profit_col)} = 0 THEN 1 ELSE 0 END) as breakeven_bets,
                MAX({self._safe_name(profit_col)}) as best_bet,
                MIN({self._safe_name(profit_col)}) as worst_bet
            FROM {self._safe_name(tips_table)}
            WHERE {self._safe_name(profit_col)} IS NOT NULL;
        """)
        r = self.cur.fetchone()

        if r["with_pnl"] == 0:
            self.log(f"{INDENT}No P&L data recorded yet.")
            return

        self.log(f"{INDENT}Tips with P&L data:  {fmt_number(r['with_pnl'])}")
        self.log(f"{INDENT}Total P&L:           {fmt_number(r['total_pnl'])}")
        self.log(f"{INDENT}Average P&L per tip: {fmt_number(r['avg_pnl'])}")
        self.log(f"{INDENT}Total winnings:      {fmt_number(r['total_winnings'])}")
        self.log(f"{INDENT}Total losses:        {fmt_number(r['total_losses'])}")
        self.log(f"{INDENT}Winning bets:        {fmt_number(r['winning_bets'])}")
        self.log(f"{INDENT}Losing bets:         {fmt_number(r['losing_bets'])}")
        self.log(f"{INDENT}Breakeven bets:      {fmt_number(r['breakeven_bets'])}")
        self.log(f"{INDENT}Best single bet:     {fmt_number(r['best_bet'])}")
        self.log(f"{INDENT}Worst single bet:    {fmt_number(r['worst_bet'])}")

        total_losses = r["total_losses"] or 0
        total_pnl = r["total_pnl"] or 0
        if total_losses > 0:
            roi = total_pnl / total_losses * 100
            self.log(f"{INDENT}ROI (P&L / outlay):  {roi:.1f}%")

        # Cumulative P&L over time
        date_col = self._find_column_like(tips_table, "start_time", "date", "created", "timestamp")
        if date_col:
            self.log(subsection("Cumulative P&L Over Time (by date)"))
            self.cur.execute(f"""
                SELECT
                    DATE({self._safe_name(date_col)}) as day,
                    SUM({self._safe_name(profit_col)}) as daily_pnl,
                    COUNT(*) as tips
                FROM {self._safe_name(tips_table)}
                WHERE {self._safe_name(profit_col)} IS NOT NULL AND {self._safe_name(date_col)} IS NOT NULL
                GROUP BY day
                ORDER BY day;
            """)
            daily_rows = self.cur.fetchall()
            if daily_rows:
                cumulative = 0.0
                display_rows = []
                for dr in daily_rows:
                    cumulative += dr["daily_pnl"]
                    display_rows.append([
                        dr["day"],
                        fmt_number(dr["tips"]),
                        fmt_number(dr["daily_pnl"]),
                        fmt_number(cumulative),
                    ])
                # Show last 20 days
                if len(display_rows) > 20:
                    self.log(f"{INDENT}(showing last 20 of {len(display_rows)} days)")
                    display_rows = display_rows[-20:]
                self.log(table_rows(["Date", "Tips", "Day P&L", "Cumulative"], display_rows))

                # Simple sparkline-style bar chart
                self.log(subsection("Cumulative P&L Trend"))
                cumulative = 0.0
                values = []
                for dr in daily_rows:
                    cumulative += dr["daily_pnl"]
                    values.append(cumulative)

                if values:
                    min_val = min(values)
                    max_val = max(values)
                    span = max_val - min_val if max_val != min_val else 1
                    bar_width = 40

                    # Sample if too many days
                    step = max(1, len(values) // 30)
                    sampled_days = daily_rows[::step]
                    sampled_vals = values[::step]

                    for day_row, val in zip(sampled_days, sampled_vals):
                        normalized = int((val - min_val) / span * bar_width)
                        bar = "█" * normalized
                        prefix = "+" if val >= 0 else ""
                        self.log(f"{INDENT}{day_row['day']}  {bar.ljust(bar_width)}  {prefix}{val:.2f}")

    def _matching_diagnostic(self) -> None:
        """WHY are tips unverified? Diagnose the prediction→result gap."""
        self.log(section("12. MATCHING DIAGNOSTIC — WHY TIPS ARE UNVERIFIED"))

        tips_table = self._find_table_like("tip")
        if not tips_table:
            self.log(f"{INDENT}No tips table found.")
            return

        columns = self._get_columns(tips_table)
        verdict_col = self._find_column_like(tips_table, "verdict")

        # ── Stage 1: How many tips have never been audited? ──

        if verdict_col:
            self.cur.execute(f"""
                SELECT COUNT(*) as cnt FROM {self._safe_name(tips_table)}
                WHERE {self._safe_name(verdict_col)} IS NULL OR {self._safe_name(verdict_col)} = '';
            """)
            unverified_count = self.cur.fetchone()["cnt"]
            self.cur.execute(f"SELECT COUNT(*) as cnt FROM {self._safe_name(tips_table)};")
            total = self.cur.fetchone()["cnt"]

            self.log(f"{INDENT}Unverified tips: {fmt_number(unverified_count)} / {fmt_number(total)} "
                     f"({fmt_pct(unverified_count, total)})")

            if unverified_count == 0:
                self.log(f"{INDENT}✓ All tips have been audited.")
                return
        else:
            self.log(f"{INDENT}⚠  No verdict column found — cannot determine verification status.")
            return

        # ── Stage 2: Do unverified tips have the fields needed for matching? ──

        self.log(subsection("Required Fields for Key Matching"))

        key_fields = ["venue", "race_number", "start_time", "discipline"]
        for field in key_fields:
            if field not in columns:
                self.log(f"{INDENT}❌ Column '{field}' MISSING from tips table — matching impossible")
                continue

            self.cur.execute(f"""
                SELECT COUNT(*) as cnt FROM {self._safe_name(tips_table)}
                WHERE ({self._safe_name(verdict_col)} IS NULL OR {self._safe_name(verdict_col)} = '')
                  AND ({self._safe_name(field)} IS NULL OR CAST({self._safe_name(field)} AS TEXT) = '');
            """)
            missing = self.cur.fetchone()["cnt"]
            status = "✓" if missing == 0 else "⚠"
            self.log(f"{INDENT}{status} '{field}': {fmt_number(missing)} unverified tips missing this value")

        # ── Stage 3: What venues are unverified tips waiting on? ──

        if "venue" in columns:
            self.log(subsection("Unverified Tips by Venue"))
            self.cur.execute(f"""
                SELECT venue, COUNT(*) as cnt
                FROM {self._safe_name(tips_table)}
                WHERE {self._safe_name(verdict_col)} IS NULL OR {self._safe_name(verdict_col)} = ''
                GROUP BY venue
                ORDER BY cnt DESC
                LIMIT 20;
            """)
            rows = [[r["venue"], fmt_number(r["cnt"])] for r in self.cur.fetchall()]
            self.log(table_rows(["Venue (awaiting results)", "Count"], rows))

        # ── Stage 4: Do we have ANY results for those venues? ──

        results_table = self._find_table_like("result")
        if results_table and "venue" in columns:
            results_cols = self._get_columns(results_table)
            if "venue" in results_cols:
                self.log(subsection("Venue Coverage: Tips vs Results"))
                self.cur.execute(f"""
                    SELECT
                        t.venue,
                        COUNT(DISTINCT t.rowid) as tip_count,
                        COUNT(DISTINCT r.rowid) as result_count
                    FROM {self._safe_name(tips_table)} t
                    LEFT JOIN {self._safe_name(results_table)} r ON LOWER(t.venue) = LOWER(r.venue)
                    WHERE t.{self._safe_name(verdict_col)} IS NULL OR t.{self._safe_name(verdict_col)} = ''
                    GROUP BY t.venue
                    ORDER BY tip_count DESC
                    LIMIT 20;
                """)
                rows = [
                    [r["venue"], fmt_number(r["tip_count"]),
                     fmt_number(r["result_count"]),
                     "❌ NO RESULTS" if r["result_count"] == 0 else "✓"]
                    for r in self.cur.fetchall()
                ]
                self.log(table_rows(["Venue", "Unverified Tips", "Result Records", "Status"], rows))

        # ── Stage 5: Time zone / key alignment check ──

        if "start_time" in columns:
            self.log(subsection("Start Time Format Samples (unverified tips)"))
            self.cur.execute(f"""
                SELECT start_time, venue, race_number
                FROM {self._safe_name(tips_table)}
                WHERE ({self._safe_name(verdict_col)} IS NULL OR {self._safe_name(verdict_col)} = '')
                  AND start_time IS NOT NULL
                LIMIT 10;
            """)
            rows = [[r["start_time"], r["venue"], r["race_number"]]
                    for r in self.cur.fetchall()]
            self.log(table_rows(["start_time (raw)", "venue", "race#"], rows))

            # Check if results table has comparable times
            if results_table:
                date_col_r = self._find_column_like(results_table, "start_time", "date", "timestamp")
                if date_col_r:
                    self.log(subsection("Start Time Format Samples (results)"))
                    self.cur.execute(f"""
                        SELECT {self._safe_name(date_col_r)}, venue, race_number
                        FROM {self._safe_name(results_table)}
                        WHERE {self._safe_name(date_col_r)} IS NOT NULL
                        LIMIT 10;
                    """)
                    rows = [[r[date_col_r], r["venue"], r["race_number"]]
                            for r in self.cur.fetchall()]
                    self.log(table_rows([f"{date_col_r} (raw)", "venue", "race#"], rows))

        # ── Stage 6: Simulated key matching ──

        if results_table and all(f in columns for f in ["venue", "race_number", "start_time"]):
            results_cols = self._get_columns(results_table)
            if all(f in results_cols for f in ["venue", "race_number"]):
                self.log(subsection("Simulated Key Match Attempts"))

                date_col_r = self._find_column_like(
                    results_table, "start_time", "date", "timestamp"
                )
                if date_col_r:
                    # Try exact date+venue+race_number match
                    self.cur.execute(f"""
                        SELECT
                            COUNT(*) as total_unverified,
                            SUM(CASE WHEN r.rowid IS NOT NULL THEN 1 ELSE 0 END) as matched
                        FROM {self._safe_name(tips_table)} t
                        LEFT JOIN {self._safe_name(results_table)} r
                            ON LOWER(TRIM(t.venue)) = LOWER(TRIM(r.venue))
                            AND t.race_number = r.race_number
                            AND DATE(t.start_time) = DATE(r.{self._safe_name(date_col_r)})
                        WHERE t.{self._safe_name(verdict_col)} IS NULL OR t.{self._safe_name(verdict_col)} = '';
                    """)
                    r = self.cur.fetchone()
                    self.log(f"{INDENT}Exact match (venue+race#+date): "
                             f"{fmt_number(r['matched'])} / {fmt_number(r['total_unverified'])}")

                    # Try relaxed match: just venue + date
                    self.cur.execute(f"""
                        SELECT
                            COUNT(*) as total_unverified,
                            SUM(CASE WHEN r.rowid IS NOT NULL THEN 1 ELSE 0 END) as matched
                        FROM {self._safe_name(tips_table)} t
                        LEFT JOIN {self._safe_name(results_table)} r
                            ON LOWER(TRIM(t.venue)) = LOWER(TRIM(r.venue))
                            AND DATE(t.start_time) = DATE(r.{self._safe_name(date_col_r)})
                        WHERE t.{self._safe_name(verdict_col)} IS NULL OR t.{self._safe_name(verdict_col)} = '';
                    """)
                    r = self.cur.fetchone()
                    self.log(f"{INDENT}Relaxed match (venue+date only): "
                             f"{fmt_number(r['matched'])} / {fmt_number(r['total_unverified'])}")

                    # Try very relaxed: just venue
                    self.cur.execute(f"""
                        SELECT
                            COUNT(*) as total_unverified,
                            SUM(CASE WHEN r.rowid IS NOT NULL THEN 1 ELSE 0 END) as matched
                        FROM {self._safe_name(tips_table)} t
                        LEFT JOIN {self._safe_name(results_table)} r
                            ON LOWER(TRIM(t.venue)) = LOWER(TRIM(r.venue))
                        WHERE t.{self._safe_name(verdict_col)} IS NULL OR t.{self._safe_name(verdict_col)} = '';
                    """)
                    r = self.cur.fetchone()
                    self.log(f"{INDENT}Venue-only match (any date):     "
                             f"{fmt_number(r['matched'])} / {fmt_number(r['total_unverified'])}")

                    # Show the unmatched venues
                    self.cur.execute(f"""
                        SELECT DISTINCT t.venue
                        FROM {self._safe_name(tips_table)} t
                        LEFT JOIN {self._safe_name(results_table)} r
                            ON LOWER(TRIM(t.venue)) = LOWER(TRIM(r.venue))
                        WHERE (t.{self._safe_name(verdict_col)} IS NULL OR t.{self._safe_name(verdict_col)} = '')
                          AND r.rowid IS NULL
                        LIMIT 20;
                    """)
                    orphan_venues = [r["venue"] for r in self.cur.fetchall()]
                    if orphan_venues:
                        self.log(f"\n{INDENT}Venues with tips but ZERO results ever fetched:")
                        for v in orphan_venues:
                            self.log(f"{INDENT}  ❌ {v}")

    # ── Utility methods ──────────────────────────────────────────────────────

    def _get_table_names(self) -> List[str]:
        self.cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        return [r["name"] for r in self.cur.fetchall()]

    def _get_columns(self, table_name: str) -> List[str]:
        self.cur.execute(f"PRAGMA table_info({self._safe_name(table_name)});")
        return [r["name"] for r in self.cur.fetchall()]

    def _find_table_like(self, *fragments: str) -> Optional[str]:
        """Find a table whose name contains any of the given fragments (case-insensitive)."""
        tables = self._get_table_names()
        # Prefer exact or plural match first (Bug #4 Fix)
        for frag in fragments:
            fl = frag.lower()
            for t in tables:
                if t.lower() in (fl, fl + "s"):
                    return t

        # Substring match fallback
        for frag in fragments:
            fl = frag.lower()
            for t in tables:
                if fl in t.lower():
                    return t
        return None

    def _find_column_like(self, table: str, *fragments: str) -> Optional[str]:
        """Find a column in a table whose name contains any of the given fragments."""
        columns = self._get_columns(table)
        for frag in fragments:
            fl = frag.lower()
            for c in columns:
                if fl in c.lower():
                    return c
        return None

    def _safe_name(self, name: str) -> str:
        """Double-quote an identifier, escaping any embedded quotes (Bug #4 Fix)."""
        return '"' + name.replace('"', '""') + '"'

    @staticmethod
    def _truncate(s: str, max_len: int = 40) -> str:
        if len(s) <= max_len:
            return s
        return s[:max_len - 3] + "..."

    # ── Full SQL dump (optional, for raw inspection) ─────────────────────────

    def dump_schema_sql(self) -> None:
        self.log(section("APPENDIX: FULL SCHEMA SQL"))
        self.cur.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name;")
        for row in self.cur.fetchall():
            self.log(f"{INDENT}{row['sql']};\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect and analyze a Fortuna SQLite database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python fortuna_db_inspector.py
    python fortuna_db_inspector.py fortuna.db
    python fortuna_db_inspector.py /path/to/fortuna.db --schema-only
    python fortuna_db_inspector.py fortuna.db --output report.txt
        """,
    )
    parser.add_argument(
        "db_path",
        nargs="?",
        default=DEFAULT_DB_PATH,
        help=f"Path to the SQLite database (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Only dump the schema SQL, skip analysis.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Write report to file instead of stdout.",
    )

    args = parser.parse_args()

    inspector = FortunaDBInspector(args.db_path)

    try:
        if args.schema_only:
            inspector.dump_schema_sql()
        else:
            inspector.run_full_report()
            inspector.dump_schema_sql()

        if args.output:
            Path(args.output).write_text("\n".join(inspector.output), encoding="utf-8")
            print(f"Report written to {args.output}")
        else:
            inspector.print_report()
    finally:
        inspector.close()


if __name__ == "__main__":
    main()
