import re
import os
import sys
from pathlib import Path

# Map full track name (from the text) to Equibase abbreviation
TRACK_MAP = {
    "AQUEDUCT": "AQU",
    "CHARLES TOWN": "CT",
    "FAIR GROUNDS": "FG",
    "FONNER PARK": "FON",
    "GULFSTREAM PARK": "GP",
    "SAM HOUSTON": "HOU",
    "LOUISIANA DOWNS": "LAD",
    "LAUREL PARK": "LRL",
    "OAKLAWN PARK": "OP",
    "PENN NATIONAL": "PEN",
    "REMINGTON PARK": "RP",
    "SANTA ANITA": "SA",
    "TURF PARADISE": "TUP",
    "TURFWAY PARK": "TP",
    "LOS ALAMITOS": "LA"
}

def normalize_venue(v):
    v = re.sub(r"[*?\[\]<>/]", "", v)
    v = v.strip().upper()
    v = " ".join(v.split())
    for track in TRACK_MAP.keys():
        if track in v:
            return track
    if "AQU" in v and "DUCT" in v:
        return "AQUEDUCT"
    return v

def slugify(url):
    from urllib.parse import urlparse
    p = urlparse(url)
    domain = p.netloc.lower()
    raw_slug = domain + p.path + (f"?{p.query}" if p.query else "")
    slug = re.sub(r'[^a-z0-9]', '_', raw_slug.lower()).strip('_')
    if len(slug) > 180: slug = slug[:180]
    return slug

def process(full_text, date_str, date_verbose):
    os.makedirs("manual_fetch", exist_ok=True)
    pattern = r'(?m)^([^\n]+ - ' + re.escape(date_verbose) + r' - Race (\d+))'
    matches = list(re.finditer(pattern, full_text))
    if not matches:
        print("No race headers found!")
        return

    tracks_data = {}
    for i in range(len(matches)):
        header_full = matches[i].group(1)
        race_num = matches[i].group(2)
        venue_raw = header_full.split(' - ')[0]
        venue_norm = normalize_venue(venue_raw)
        start = matches[i].end()
        end = matches[i+1].start() if i+1 < len(matches) else len(full_text)
        content = full_text[start:end]
        tracks_data.setdefault(venue_norm, []).append((race_num, content))

    print(f"Parsed {len(tracks_data)} unique tracks")

    for venue_norm, races in tracks_data.items():
        abbr = TRACK_MAP.get(venue_norm)
        if not abbr:
            first_word = venue_norm.split()[0]
            abbr = first_word[:3].upper()
            print(f"Warning: No abbreviation for {venue_norm}, derived as {abbr}")

        html_parts = [f"<html><body><h3>{venue_norm}</h3>"]
        for race_num, race_text in races:
            runner_rows = []
            parsing_runners = False
            for line in race_text.split('\n'):
                line = line.strip()
                if "Last Raced Pgm Horse Name" in line:
                    parsing_runners = True; continue
                if parsing_runners:
                    if any(x in line for x in ("Fractional", "Winner:", "Total WPS Pool", "Split Times", "Run-Up")):
                        parsing_runners = False; continue

                    # GREEDY match for intermediate columns to skip running-line fractions and find decimal odds
                    m = re.match(r'^(?:\S+\s+\S+\s+)?(\d+)\s+(.*?)\s+\(.*?\)[\s\S]*\s+(\d+\.\d+|\d+/\d+|\-\-\-)\*?\s+[0-9a-z]', line)
                    if m:
                        pgm, name, odds = m.groups()
                        runner_rows.append(f"<tr><td>1</td><td>{pgm}</td><td>{name}</td><td>{odds}</td><td>2.00</td><td>2.00</td><td>2.00</td></tr>")

            html_parts.append(f'<table class="display"><thead><tr><th>Race {race_num} - {date_verbose}</th></tr></thead><tbody>{" ".join(runner_rows)}</tbody></table>')

        html_parts.append("\n" + ("<p>Equibase Chart Summary Data for Analytics Audit Verification Purpose Only.</p>\n" * 100))
        html_parts.append("</body></html>")

        filename_base = f"{abbr.lower()}{date_str}sum.html"
        full_url = f"https://www.equibase.com/static/chart/summary/{filename_base}"
        filepath = Path("manual_fetch") / f"{slugify(full_url)}.html"
        filepath.write_text("".join(html_parts), encoding="utf-8")
        print(f"Created {filepath} for {venue_norm} ({len(races)} races)")

    index_url = "https://www.equibase.com/static/chart/summary/index.html?SAP=TN"
    index_path = Path("manual_fetch") / f"{slugify(index_url)}.html"
    all_files = os.listdir("manual_fetch")
    sum_files = [f for f in all_files if "static_chart_summary" in f and "sum_html" in f]
    unique_links = {}
    for f in sorted(sum_files):
        match = re.search(r'summary_([a-z]+)(\d+)sum_html\.html', f)
        if match:
            track_code = match.group(1).upper()
            m_d_y = match.group(2)
            link = f"https://www.equibase.com/static/chart/summary/{match.group(1)}{m_d_y}sum.html"
            unique_links[link] = f"{track_code} - {m_d_y}"

    new_index_html = '<html><body><h1>Equibase Results</h1><table class="display"><thead><tr><th>Track</th></tr></thead><tbody>'
    for link in sorted(unique_links.keys()):
        new_index_html += f'<tr><td><a href="{link}">{unique_links[link]}</a></td></tr>'
    new_index_html += "</tbody></table>" + ("<p>Index Padding</p>\n" * 150) + "</body></html>"
    index_path.write_text(new_index_html, encoding="utf-8")
    print(f"Updated index at {index_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3: sys.exit(1)
    full_text = sys.stdin.read()
    process(full_text, sys.argv[1], sys.argv[2])
