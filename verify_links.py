import re
import sys
import os
from datetime import datetime

def extract_urls(html_content, date_obj=None):
    """
    Extracts URLs from <a> tags and data-pattern attributes.
    Replaces placeholders like {YYYY-MM-DD} with values from date_obj.
    """
    if date_obj is None:
        date_obj = datetime.now()

    Y = date_obj.strftime('%Y')
    M = date_obj.strftime('%m')
    D = date_obj.strftime('%d')
    Mno = str(date_obj.month)
    Dno = str(date_obj.day)
    YY = Y[2:]

    patterns = set()
    # Find all data-pattern attributes
    patterns.update(re.findall(r'data-pattern=[\'"]([^\'"]+)[\'"]', html_content))
    # Find all href attributes
    patterns.update(re.findall(r'href=[\'"]([^\'"]+)[\'"]', html_content))

    urls = set()
    for p in patterns:
        if not p.startswith('http'):
            continue
        if 'view-source:' in p:
            continue

        url = p.replace('{YYYY-MM-DD}', f"{Y}-{M}-{D}")
        url = url.replace('{YYYY/MM/DD}', f"{Y}/{M}/{D}")
        url = url.replace('{DD-MM-YYYY}', f"{D}-{M}-{Y}")
        url = url.replace('{YYMMDD}', f"{YY}{M}{D}")
        url = url.replace('{YYYY}', Y)
        url = url.replace('{MM}', M)
        url = url.replace('{DD}', D)
        url = url.replace('{M}', Mno)
        url = url.replace('{D}', Dno)
        urls.add(url)

    return sorted(list(urls))

def verify_url(url):
    """
    Verifies a URL using curl, checking for HTTP 200 and bot blocks.
    """
    import subprocess

    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

    try:
        # Get HTTP code
        cmd_code = ['curl', '-s', '-L', '-o', '/dev/null', '-w', '%{http_code}', '-H', f'User-Agent: {user_agent}', '--max-time', '10', url]
        result = subprocess.run(cmd_code, capture_output=True, text=True)
        code = result.stdout.strip()

        if code == "200":
            # Check content for bot blocks
            cmd_content = ['curl', '-s', '-L', '-H', f'User-Agent: {user_agent}', '--max-time', '10', url]
            content_result = subprocess.run(cmd_content, capture_output=True, text=True)
            content = content_result.stdout[:2000].lower()

            blocks = ["incapsula", "cloudflare", "access denied", "distil networks", "sucuri"]
            for block in blocks:
                if block in content:
                    return False, f"Bot Block ({block})"
            return True, "Success"
        else:
            return False, f"HTTP {code}"
    except Exception as e:
        return False, str(e)

def main():
    html_file = "discovery_v2.html"
    if not os.path.exists(html_file):
        print(f"Error: {html_file} not found.")
        return

    with open(html_file, 'r') as f:
        html = f.read()

    # Use 2026-03-26 as the target date for consistency with previous fetches
    target_date = datetime(2026, 3, 26)
    urls = extract_urls(html, target_date)

    print(f"Extracted {len(urls)} unique URLs. Verifying...")

    success_list = []
    for url in urls:
        success, reason = verify_url(url)
        if success:
            print(f"[OK] {url}")
            success_list.append(url)
        else:
            print(f"[FAIL] {url} - {reason}")

    print("\n--- Successful Hyperlinks ---")
    for s in success_list:
        print(s)

if __name__ == "__main__":
    main()
