from browserforge.headers import HeaderGenerator
from curl_cffi import requests
import json

def test_ras_advanced():
    gen = HeaderGenerator()
    headers = gen.generate()
    url = "https://www.racingandsports.com.au/racing-index"

    print(f"Using headers: {json.dumps(headers, indent=2)}")

    try:
        # Use impersonate='chrome120' which is very stable
        response = requests.get(url, headers=headers, impersonate="chrome120", timeout=30)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("SUCCESS!")
            with open("ras_final_test.html", "w") as f:
                f.write(response.text)
        else:
            print(f"FAILED: {response.text[:100]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_ras_advanced()
