import pytest
from unittest.mock import AsyncMock, patch
from fortuna_analytics import EquibaseResultsAdapter, ResultRace
from datetime import datetime

@pytest.mark.asyncio
async def test_equibase_results_adapter_parsing():
    adapter = EquibaseResultsAdapter()

    # Mock HTML response for index - MUST BE > 1000 chars to satisfy the check
    index_html = '<html><table class="display"><tr><td><a href="AQU020826sum.html">Aqueduct</a></td></tr></table>' + (' ' * 1000) + '</html>'

    # Mock HTML response for track page (minimal chart)
    track_html = """
    <html>
        <body>
            <h3>Aqueduct</h3>
            <table>
                <thead><tr><th>Race 1 - Post Time: 1:30 PM</th></tr></thead>
                <tbody>
                    <tr>
                        <td>1st</td><td>4</td><td>Horse A</td><td>2.5</td><td>5.00</td><td>3.00</td><td>2.10</td>
                    </tr>
                    <tr>
                        <td>2nd</td><td>1</td><td>Horse B</td><td>10.0</td><td>0.00</td><td>8.00</td><td>5.00</td>
                    </tr>
                </tbody>
            </table>
            <table>
                <tr><td>Trifecta</td><td>4-1-5</td><td>$125.50</td></tr>
                <tr><td>Superfecta</td><td>4-1-5-2</td><td>$1,250.00</td></tr>
            </table>
        </body>
    </html>
    """

    class MockResponse:
        def __init__(self, text, status, url=""):
            self.text = text
            self.status = status
            self.url = url
        def json(self):
            import json
            return json.loads(self.text)

    with patch("fortuna.SmartFetcher.fetch", new_callable=AsyncMock) as mock_fetch:
        # 1. Index page fetch, 2. Track page fetch
        mock_fetch.side_effect = [
            MockResponse(index_html, 200, "/static/chart/summary/index.html"),
            MockResponse(track_html, 200, "/static/chart/summary/AQU020826sum.html")
        ]

        races = await adapter.get_races("2026-02-08")

        assert len(races) > 0
        race = races[0]
        assert "Aqueduct" in race.venue
        assert race.race_number == 1
        assert len(race.runners) == 2

        runner1 = next(r for r in race.runners if r.name == "Horse A")
        assert runner1.position_numeric == 1
        assert runner1.win_payout == 5.0

        assert race.trifecta_payout == 125.50
        assert race.superfecta_payout == 1250.00
