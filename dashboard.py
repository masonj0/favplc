from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Any, Optional
import asyncio

EASTERN = ZoneInfo("America/New_York")


class FortunaDashboard:
    """Live-updating terminal dashboard for Fortuna."""

    def __init__(self):
        self.console = Console()
        self.goldmine_races = []
        self.stats = {
            "tips": 0,
            "cashed": 0,
            "profit": 0.0
        }

    def create_layout(self) -> Layout:
        """Build the dashboard layout."""
        layout = Layout()

        # Split into sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="goldmines", ratio=2),
            Layout(name="stats", size=8),
            Layout(name="footer", size=1)
        )

        return layout

    def render_header(self) -> Panel:
        """Render the header section."""
        title = Text("🏇 FORTUNA LIVE DASHBOARD", style="bold cyan", justify="center")

        # Calculate next race time
        next_race_time = self._get_next_race_time()

        subtitle = Text()
        if self.goldmine_races:
            subtitle.append(f"🔥 GOLDMINE OPPORTUNITIES ({len(self.goldmine_races)})", style="bold yellow")
            if next_race_time:
                subtitle.append("        ")
                subtitle.append(f"⏰ Next: {next_race_time}", style="bold green")
        else:
            subtitle.append("No upcoming races", style="dim")

        return Panel(
            subtitle,
            title=title,
            border_style="cyan"
        )

    def render_goldmines(self) -> Table:
        """Render the goldmine races table."""
        table = Table(
            show_header=True,
            header_style="bold magenta",
            border_style="cyan",
            expand=True
        )

        table.add_column("Track", style="cyan", width=15)
        table.add_column("Race", justify="center", width=5)
        table.add_column("Selection", style="yellow", width=15)
        table.add_column("Odds", justify="right", width=6)
        table.add_column("Gap12", justify="right", width=6)
        table.add_column("Confidence", width=20)

        # Add rows for each goldmine
        sorted_races = sorted(self.goldmine_races, key=lambda r: self._get_start_time(r))
        for race in sorted_races[:10]:
            # Get goldmine selection
            selection = self._get_goldmine_selection(race)
            if not selection:
                continue

            st = self._get_start_time(race)
            if not st: continue

            # Time to post
            time_to_post = (st - datetime.now(EASTERN)).total_seconds() / 60

            # Color code by urgency
            if time_to_post < 0:
                style = "dim"
                icon = "🏁"
            elif time_to_post < 10:
                style = "bold red"
                icon = "🚨"
            elif time_to_post < 30:
                style = "bold yellow"
                icon = "⚠️"
            else:
                style = "green"
                icon = "✓"

            # Get confidence and gap12
            meta = getattr(race, 'metadata', {})
            confidence = meta.get("confidence", 0.7)
            gap12 = meta.get("1Gap2", 0.0)

            # Render confidence bar
            conf_bars = int(confidence * 10)
            conf_display = "█" * conf_bars + "░" * (10 - conf_bars)
            conf_text = f"{conf_display} {confidence*100:.0f}%"

            venue = str(getattr(race, 'venue', 'Unknown'))
            race_num = getattr(race, 'race_number', '?')
            sel_name = str(getattr(selection, 'name', 'Unknown'))
            sel_num = getattr(selection, 'number', '?')
            odds = getattr(selection, 'win_odds', 0.0) or 0.0

            table.add_row(
                f"{icon} {venue[:13]}",
                f"R{race_num}",
                f"#{sel_num} {sel_name[:8]}",
                f"{odds:.2f}",
                f"{gap12:.2f}",
                conf_text,
                style=style
            )

        return table

    def render_stats(self) -> Panel:
        """Render performance statistics."""
        # Calculate derived stats
        tips = self.stats.get("tips", 0)
        cashed = self.stats.get("cashed", 0)
        strike_rate = (cashed / tips * 100) if tips > 0 else 0
        roi = self.stats.get("roi", 0.0)

        # Create stats display
        stats_text = Text()
        stats_text.append("📊 PERFORMANCE SUMMARY\n\n", style="bold cyan")

        stats_text.append(f"Tips: {tips}", style="white")
        stats_text.append("  │  ")
        stats_text.append(f"Cashed: {cashed}", style="green" if cashed > 0 else "white")
        stats_text.append("  │  ")
        stats_text.append(f"Strike Rate: {strike_rate:.1f}%", style="green" if strike_rate > 60 else "yellow")
        stats_text.append("  │  ")
        stats_text.append(f"ROI: {roi:.1f}%", style="green" if roi > 0 else "red" if roi < 0 else "yellow")
        stats_text.append("\n\n")

        # Profit display
        profit = self.stats.get("profit", 0.0)
        profit_style = "bold green" if profit > 0 else "bold red" if profit < 0 else "white"
        stats_text.append(f"Net Profit: ${profit:.2f}", style=profit_style)
        stats_text.append("\n\n")

        # Progress bar for win rate
        stats_text.append("Win Rate Progress:\n")
        progress_bars = int((strike_rate / 100) * 30)
        stats_text.append("█" * progress_bars, style="green")
        stats_text.append("░" * (30 - progress_bars), style="dim")

        return Panel(stats_text, border_style="cyan")

    def render_footer(self) -> Text:
        """Render the footer with controls."""
        now = datetime.now(EASTERN).strftime("%I:%M:%S %p %Z")

        footer = Text()
        footer.append(f"Last Updated: {now}", style="dim")
        footer.append("  │  ", style="dim")
        footer.append("Press Ctrl+C to quit", style="yellow")

        return footer

    def _get_start_time(self, race: Any) -> Optional[datetime]:
        st = getattr(race, 'start_time', None)
        if isinstance(st, str):
            try: st = datetime.fromisoformat(st.replace('Z', '+00:00'))
            except Exception: return None
        if st and st.tzinfo is None:
            st = st.replace(tzinfo=EASTERN)
        return st

    def _get_next_race_time(self) -> str:
        """Calculate time to next race."""
        if not self.goldmine_races:
            return ""

        now = datetime.now(EASTERN)
        upcoming = []
        for r in self.goldmine_races:
            st = self._get_start_time(r)
            if st and st > now:
                upcoming.append(st)

        if not upcoming:
            return ""

        next_time = min(upcoming)
        delta = (next_time - now).total_seconds()

        minutes = int(delta / 60)
        seconds = int(delta % 60)

        if minutes > 60:
            hours = minutes // 60
            minutes = minutes % 60
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m {seconds}s"

    def _get_goldmine_selection(self, race: Any) -> Optional[Any]:
        """Extract goldmine selection from race."""
        # Find runner with 2nd lowest odds (goldmine logic often targets 2nd fav)
        runners = getattr(race, 'runners', [])
        if not runners: return None

        active = [r for r in runners if not getattr(r, 'scratched', False)]
        if len(active) < 2: return None

        # We need a stable way to identify the intended selection.
        # Often it is the 2nd favorite.
        active.sort(key=lambda x: getattr(x, 'win_odds', 999.0) or 999.0)
        return active[1]

    def update(self, goldmine_races: List, stats: dict):
        """Update dashboard data."""
        self.goldmine_races = goldmine_races
        self.stats.update(stats)

    def render(self):
        """Render the complete dashboard."""
        layout = self.create_layout()

        layout["header"].update(self.render_header())
        layout["goldmines"].update(self.render_goldmines())
        layout["stats"].update(self.render_stats())
        layout["footer"].update(self.render_footer())

        return layout


def print_dashboard(goldmine_races: List, stats: dict = None):
    """Print a static dashboard (non-live version)."""
    console = Console()

    # Header
    console.print(Panel(
        Text("🏇 FORTUNA DISCOVERY COMPLETE", style="bold cyan", justify="center"),
        border_style="cyan"
    ))

    # Goldmines table
    table = Table(title="🔥 GOLDMINE OPPORTUNITIES", border_style="yellow", expand=True)
    table.add_column("Track", style="cyan")
    table.add_column("Race", justify="center")
    table.add_column("Time", style="green")
    table.add_column("Selection", style="yellow")
    table.add_column("Odds", justify="right")
    table.add_column("Gap12", justify="right")

    db = FortunaDashboard()

    for race in goldmine_races:
        selection = db._get_goldmine_selection(race)
        if not selection: continue

        st = db._get_start_time(race)
        time_str = st.strftime("%I:%M %p") if st else "Unknown"
        gap12 = getattr(race, 'metadata', {}).get('1Gap2', 0.0)

        table.add_row(
            getattr(race, 'venue', 'Unknown'),
            f"R{getattr(race, 'race_number', '?')}",
            time_str,
            f"#{getattr(selection, 'number', '?')} {getattr(selection, 'name', 'Unknown')}",
            f"{getattr(selection, 'win_odds', 0.0):.2f}",
            f"{gap12:.2f}"
        )

    console.print(table)

    # Stats panel if provided
    if stats:
        strike_rate = (stats['cashed'] / stats['tips'] * 100) if stats.get('tips', 0) > 0 else 0
        stats_text = (
            f"📊 Tips: {stats.get('tips', 0)}  │  "
            f"Cashed: {stats.get('cashed', 0)}  │  "
            f"Strike Rate: {strike_rate:.1f}%  │  "
            f"Profit: ${stats.get('profit', 0.0):.2f}"
        )
        console.print(Panel(stats_text, title="Today's Performance", border_style="green"))
