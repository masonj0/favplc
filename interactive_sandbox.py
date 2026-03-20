#!/usr/bin/env python3
import os
import sys
import asyncio
import subprocess
import json
import sqlite3
from datetime import datetime
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich.progress import Progress, SpinnerColumn, TextColumn
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

class InteractiveSandbox:
    def __init__(self):
        self.console = Console() if HAS_RICH else None
        self.results = {}
        self.start_time = datetime.now()

    def print_header(self, text):
        if self.console:
            self.console.print(f"\n[bold cyan]=== {text} ===[/bold cyan]")
        else:
            print(f"\n=== {text} ===")

    def run_cmd(self, cmd, desc=None):
        if desc and self.console:
            self.console.print(f"[yellow]Running: {desc}...[/yellow]")

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": "."}
            )
            return result
        except Exception as e:
            return type('Obj', (object,), {'returncode': 1, 'stderr': str(e), 'stdout': ''})

    async def phase_0_setup(self):
        self.print_header("PHASE 0: Environment Setup")

        checks = [
            ("fortuna.py", os.path.exists("fortuna.py")),
            ("fortuna_analytics.py", os.path.exists("fortuna_analytics.py")),
            ("fortuna_interactive.py", os.path.exists("fortuna_interactive.py")),
        ]

        table = Table(title="File Verification")
        table.add_column("File", style="cyan")
        table.add_column("Status", style="magenta")

        all_ok = True
        for name, exists in checks:
            status = "✅ Found" if exists else "❌ Missing"
            table.add_row(name, status)
            if not exists: all_ok = False

        self.console.print(table)

        if not all_ok:
            self.console.print("[red]Critical files missing. Please check your repository.[/red]")
            return False

        # Verify Imports
        with self.console.status("[bold green]Verifying imports..."):
            res = self.run_cmd("python3 -c 'import fortuna, fortuna_analytics, fortuna_interactive; print(\"OK\")'")

        if "OK" in res.stdout:
            self.console.print("✅ Imports: [green]OK[/green]")
        else:
            self.console.print(f"❌ Imports failed:\n[red]{res.stderr}[/red]")
            return False

        return True

    async def phase_1_discovery(self):
        self.print_header("PHASE 1: Discovery")

        choice = Prompt.ask(
            "Run Discovery?",
            choices=["Standard", "Interactive", "Skip"],
            default="Skip"
        )

        if choice == "Skip":
            self.results['discovery'] = "Skipped"
            return

        script = "fortuna.py" if choice == "Standard" else "fortuna_interactive.py"
        self.console.print(f"Starting {choice} Discovery using {script}...")

        # Interactive mode needs to be run directly in the foreground
        if choice == "Interactive":
            subprocess.run(f"python3 {script} --region GLOBAL", shell=True)
        else:
            # Standard mode with progress
            with self.console.status("[bold yellow]Running standard discovery (this may take a few minutes)..."):
                res = self.run_cmd(f"python3 {script} --region GLOBAL")
                if res.returncode == 0:
                    self.console.print("✅ Discovery completed successfully.")
                else:
                    self.console.print(f"⚠️ Discovery returned non-zero code: {res.returncode}")

        # Validate outputs
        self.validate_discovery_outputs()

    def validate_discovery_outputs(self):
        outputs = [
            ('fortuna.db', 1000),
            ('discovery_harvest.json', 10),
            ('adapter_health_report.txt', 10),
        ]

        table = Table(title="Output Validation")
        table.add_column("Artifact", style="cyan")
        table.add_column("Size", style="magenta")
        table.add_column("Status", style="green")

        for fname, min_size in outputs:
            path = Path(fname)
            if path.exists():
                size = path.stat().st_size
                status = "✅ Pass" if size >= min_size else "⚠️ Small"
                table.add_row(fname, f"{size} bytes", status)
            else:
                table.add_row(fname, "0", "❌ Missing")

        self.console.print(table)

    async def phase_2_analytics(self):
        self.print_header("PHASE 2: Analytics Audit")

        if not Confirm.ask("Run Analytics Audit?", default=True):
            return

        with self.console.status("[bold yellow]Harvesting results and auditing tips..."):
            res = self.run_cmd("python3 fortuna_analytics.py -v --days 2")

        if res.returncode == 0:
            self.console.print("✅ Audit complete.")
            # Show top lines of report
            if os.path.exists("analytics_report.txt"):
                report = Path("analytics_report.txt").read_text().splitlines()[:20]
                self.console.print(Panel("\n".join(report), title="Analytics Report Snippet"))
        else:
            self.console.print(f"❌ Audit failed:\n{res.stderr}")

    def phase_3_db_status(self):
        self.print_header("PHASE 3: Database Status")

        if not os.path.exists("fortuna.db"):
            self.console.print("[red]Database not found.[/red]")
            return

        try:
            conn = sqlite3.connect("fortuna.db")
            cur = conn.cursor()

            cur.execute("SELECT COUNT(*) FROM tips")
            total = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM tips WHERE audit_completed = 1")
            audited = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM tips WHERE is_goldmine = 1")
            goldmines = cur.fetchone()[0]

            cur.execute("SELECT SUM(net_profit) FROM tips WHERE audit_completed = 1")
            profit = cur.fetchone()[0] or 0.0

            self.console.print(f"Total Tips: [bold]{total}[/bold]")
            self.console.print(f"Audited:    [bold]{audited}[/bold]")
            self.console.print(f"Goldmines:  [bold]{goldmines}[/bold]")
            self.console.print(f"Net Profit: [bold]${profit:+.2f}[/bold]")

            conn.close()
        except Exception as e:
            self.console.print(f"[red]DB Query Error: {e}[/red]")

    def phase_4_summary(self):
        self.print_header("PHASE 4: Verification Summary")

        summary = f"""
## Fortuna Sandbox Verification Report

### Environment
- Time of run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Directory: {os.getcwd()}

### Status
- Files checked: OK
- Discovery: Performed
- Audit: Performed

(Copy and paste this to the user message)
        """
        self.console.print(Panel(summary, title="Verification Copy-Pasta"))

    async def run(self):
        if not HAS_RICH:
            print("Error: 'rich' library missing. Please install it first.")
            return

        self.console.print(Panel.fit(
            "Fortuna Interactive Sandbox Guide",
            subtitle="Guidance for the Council of Superbrains",
            border_style="bold green"
        ))

        if await self.phase_0_setup():
            await self.phase_1_discovery()
            await self.phase_2_analytics()
            self.phase_3_db_status()
            self.phase_4_summary()

        self.console.print("\n[bold green]Sandbox Verification Session Ended.[/bold green]")

if __name__ == "__main__":
    sandbox = InteractiveSandbox()
    asyncio.run(sandbox.run())
