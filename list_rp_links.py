import asyncio
from fortuna import RacingPostAdapter
from selectolax.parser import HTMLParser

async def main():
    adapter = RacingPostAdapter()
    resp = await adapter.make_request("GET", "/racecards/")
    if not resp or not resp.text:
        print("No response")
        return

    parser = HTMLParser(resp.text)
    links = []
    for a in parser.css('a[href*="/racecards/"]'):
        href = a.attributes.get("href")
        if href:
            links.append(href)

    for l in sorted(list(set(links))):
        print(l)
    await adapter.close()

if __name__ == "__main__":
    asyncio.run(main())
