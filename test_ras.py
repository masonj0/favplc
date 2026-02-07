from curl_cffi import requests
import sys

def test_ras():
    url = "https://www.racingandsports.com.au/racing-index"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }

    try:
        # Use curl_cffi to impersonate a real browser
        response = requests.get(url, headers=headers, impersonate="chrome120")
        print(f"Status Code: {response.status_code}")
        print(f"Content Length: {len(response.text)}")
        if response.status_code == 200:
            with open("ras_test.html", "w") as f:
                f.write(response.text)
            print("Successfully saved ras_test.html")
            if "Access Denied" in response.text:
                print("Detected 'Access Denied' in content.")
            else:
                print("Content looks promising!")
        else:
            print("Failed to fetch.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_ras()
