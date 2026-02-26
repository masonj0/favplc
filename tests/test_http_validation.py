import pytest
import httpx
import respx
from datetime import datetime
from fortuna import fetch_json, BaseAdapterV3, FetchStrategy, Race, Runner, EASTERN
from typing import List, Any, Optional

class MockAdapter(BaseAdapterV3):
    def _configure_fetch_strategy(self) -> FetchStrategy:
        return FetchStrategy(primary_engine="httpx")
    async def _fetch_data(self, date: str) -> Optional[Any]:
        return await self.make_request("GET", f"http://test.com/{date}")
    def _parse_races(self, raw_data: Any) -> List[Race]:
        return []

@pytest.mark.asyncio
async def test_fetch_json_raises_for_status():
    async with httpx.AsyncClient() as client:
        with respx.mock:
            respx.get("http://test.com/error").respond(status_code=500)
            with pytest.raises(httpx.HTTPStatusError):
                await fetch_json("http://test.com/error", client=client, adapter_name="test")

@pytest.mark.asyncio
async def test_adapter_make_request_raises_for_status():
    adapter = MockAdapter(source_name="Mock", base_url="http://test.com")
    with respx.mock:
        respx.get("http://test.com/500").respond(status_code=500)
        # In our implementation, get_races catches the exception and returns []
        # but logs the error and records failure in circuit breaker
        races = await adapter.get_races("500")
        assert races == []
        assert adapter.last_response_status == 500
    await adapter.close()

@pytest.mark.asyncio
async def test_adapter_retries_on_failure():
    adapter = MockAdapter(source_name="Mock", base_url="http://test.com")
    with respx.mock:
        # First two fail, third succeeds
        route = respx.get("http://test.com/retry")
        route.side_effect = [
            httpx.Response(500),
            httpx.Response(500),
            httpx.Response(200, json={"ok": True})
        ]

        # We need to mock _fetch_data to return the json if success
        async def _mock_fetch_data(date):
            resp = await adapter.make_request("GET", f"http://test.com/{date}")
            return resp.json() if resp else None

        adapter._fetch_data = _mock_fetch_data

        # We also need to mock _parse_races to not fail
        # Provide at least 2 runners to satisfy RaceValidator
        runners = [Runner(name="R1", number=1), Runner(name="R2", number=2)]
        now = datetime.now(EASTERN)
        adapter._parse_races = lambda x: [Race(id="1", venue="T", race_number=1, start_time=now, source="M", runners=runners)]

        races = await adapter.get_races("retry")
        assert len(races) == 1
        assert route.call_count == 3
    await adapter.close()
