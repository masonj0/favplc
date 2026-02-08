import asyncio
import structlog
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo
from notifications import DesktopNotifier

EASTERN = ZoneInfo("America/New_York")

class LiveOddsTracker:
    """
    Tracks live odds movements for qualified races.
    Sends notifications when significant movements occur.
    """

    def __init__(self, qualified_races: List[Any], adapter_classes: List[type]):
        self.qualified_races = qualified_races
        self.adapter_classes = adapter_classes
        self.odds_history: Dict[str, List[Tuple[datetime, Dict[int, float]]]] = {}
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.notifier = DesktopNotifier()
        self._running = False

    async def start_tracking(self):
        """Start monitoring odds for all qualified races."""
        self._running = True
        tasks = []
        now = datetime.now(EASTERN)

        for race in self.qualified_races:
            st = self._get_start_time(race)
            if not st: continue

            # Only track races in next 2 hours or recently started
            time_to_post = (st - now).total_seconds() / 60
            if -5 < time_to_post < 120:
                tasks.append(self.track_race(race))

        if tasks:
            self.logger.info("Starting live odds tracking", race_count=len(tasks))
            await asyncio.gather(*tasks)

    def stop(self):
        self._running = False

    async def track_race(self, race: Any):
        """Monitor a single race until post time."""
        st = self._get_start_time(race)
        if not st: return

        adapter = self._get_adapter_for_race(race)
        if not adapter:
            self.logger.warning("No adapter found for live tracking", venue=getattr(race, 'venue', '?'))
            return

        while self._running and datetime.now(EASTERN) < st:
            try:
                # Fetch current odds
                current_odds = await self.fetch_current_odds(adapter, race)
                if not current_odds:
                    await asyncio.sleep(60)
                    continue

                # Check for significant movements
                alerts = self.analyze_movements(race, current_odds)

                # Send notifications
                for alert in alerts:
                    self.notifier.send(alert)

                # Wait 60 seconds before next check
                await asyncio.sleep(60)

            except Exception as e:
                self.logger.error("Tracking error", race_id=getattr(race, 'id', '?'), error=str(e))
                await asyncio.sleep(60)

    async def fetch_current_odds(self, adapter: Any, race: Any) -> Dict[int, float]:
        """Fetch current odds from live source."""
        try:
            # Re-fetch the race data
            races = await adapter.get_races(datetime.now(EASTERN).strftime("%Y-%m-%d"))
            target_num = getattr(race, 'race_number', None)

            match = next((r for r in races if getattr(r, 'race_number', None) == target_num), None)
            if not match: return {}

            odds_map = {}
            for runner in getattr(match, 'runners', []):
                num = getattr(runner, 'number', None)
                # Try to get win odds. Adapter might have them in .win_odds or .odds dict
                win_odds = getattr(runner, 'win_odds', None)
                if win_odds and num:
                    odds_map[num] = float(win_odds)
            return odds_map
        except Exception as e:
            self.logger.error("Failed to fetch fresh odds", error=str(e))
            return {}

    def analyze_movements(self, race: Any, current_odds: Dict[int, float]) -> List[dict]:
        """Detect significant odds movements."""
        alerts = []
        race_key = getattr(race, 'id', str(id(race)))

        # Initialize history if needed
        if race_key not in self.odds_history:
            self.odds_history[race_key] = []

        # Store current snapshot
        timestamp = datetime.now(EASTERN)
        self.odds_history[race_key].append((timestamp, current_odds))

        # We only keep last 10 snapshots to save memory
        if len(self.odds_history[race_key]) > 10:
            self.odds_history[race_key].pop(0)

        # Check for target selection (usually 2nd favorite for goldmines)
        target_selection = self._get_goldmine_selection(race)
        if not target_selection:
            return alerts

        # Get historical odds for this selection
        history = self.odds_history[race_key]
        if len(history) < 2:
            return alerts

        sel_num = getattr(target_selection, 'number', None)
        if sel_num is None: return alerts

        current = current_odds.get(sel_num)
        previous = history[-2][1].get(sel_num)

        if not current or not previous:
            return alerts

        # Calculate drift
        drift_pct = ((current - previous) / previous) * 100
        venue = getattr(race, 'venue', 'Unknown')
        r_num = getattr(race, 'race_number', '?')
        sel_name = getattr(target_selection, 'name', 'Unknown')
        st = self._get_start_time(race)

        # Alert on favorable drift (odds going up = better value if we suspect it's overpriced)
        # OR alert on shortening (odds dropping = smart money moving in)
        if abs(drift_pct) >= 15:
            direction = "DRIFTING 📈" if drift_pct > 0 else "SHORTENING 📉"
            title = f"🚨 ODDS {direction}"

            time_to_post = ""
            if st:
                mins = int((st - timestamp).total_seconds() / 60)
                time_to_post = f"\nPost in {mins}m"

            alerts.append({
                "title": title,
                "message": (
                    f"{venue} R{r_num}\n"
                    f"#{sel_num} {sel_name}\n"
                    f"{previous:.2f} → {current:.2f} ({drift_pct:+.1f}%){time_to_post}"
                ),
                "urgency": "high" if abs(drift_pct) > 25 else "normal",
                "race_id": race_key
            })

        return alerts

    def _get_goldmine_selection(self, race: Any) -> Optional[Any]:
        runners = getattr(race, 'runners', [])
        active = [r for r in runners if not getattr(r, 'scratched', False)]
        if len(active) < 2: return None
        active.sort(key=lambda x: getattr(x, 'win_odds', 999.0) or 999.0)
        return active[1] # 2nd favorite

    def _get_start_time(self, race: Any) -> Optional[datetime]:
        st = getattr(race, 'start_time', None)
        if isinstance(st, str):
            try: st = datetime.fromisoformat(st.replace('Z', '+00:00'))
            except: return None
        if st and st.tzinfo is None:
            st = st.replace(tzinfo=EASTERN)
        return st

    def _get_adapter_for_race(self, race: Any) -> Optional[Any]:
        source = getattr(race, 'source', None)
        if not source: return None

        # Match source string to adapter class
        for cls in self.adapter_classes:
            if hasattr(cls, 'SOURCE_NAME') and cls.SOURCE_NAME in source:
                return cls()
        return None
