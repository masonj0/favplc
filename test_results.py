import asyncio
import httpx
from selectolax.parser import HTMLParser
import re
from datetime import datetime

async def test_rp():
    url = "https://www.racingpost.com/results/2026-02-05"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        try:
            r = await client.get(url, timeout=30)
            print(f"RP Status: {r.status_code}")
            parser = HTMLParser(r.text)
            links = []
            for a in parser.css("a[href*='/results/']"):
                href = a.attributes.get("href", "")
                if re.search(r"/results/\d+/", href):
                    links.append(href)
            print(f"RP Links found: {len(links)}")
        except Exception as e:
            print(f"RP Error: {e}")

async def test_atr():
    # Test both YYYY-MM-DD and DD-Month-YYYY
    date_str = "2026-02-05"
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    alt_date = dt.strftime("%d-%B-%Y")

    for d in [date_str, alt_date]:
        url = f"https://www.attheraces.com/results/{d}"
        headers = {"User-Agent": "Mozilla/5.0"}
        async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
            try:
                r = await client.get(url, timeout=30)
                print(f"ATR Status ({d}): {r.status_code}")
                parser = HTMLParser(r.text)
                links = []
                for a in parser.css("a[href*='/results/']"):
                    href = a.attributes.get("href") or ""
                    if re.search(r"/results/[^/]+/.+-\d{4}/\d{4}", href):
                        links.append(href)
                print(f"ATR Links found ({d}): {len(links)}")
            except Exception as e:
                print(f"ATR Error ({d}): {e}")

async def test_eqb():
    url = "https://www.equibase.com/static/chart/summary/index.html?date=02/05/2026"
    headers = {"User-Agent": "Mozilla/5.0"}
    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        try:
            r = await client.get(url, timeout=30)
            print(f"EQB Status: {r.status_code}")
            if "Incapsula" in r.text:
                 print("EQB Blocked by Incapsula")
        except Exception as e:
            print(f"EQB Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_rp())
    asyncio.run(test_atr())
    asyncio.run(test_eqb())
