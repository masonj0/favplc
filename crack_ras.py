from scrapling import Fetcher
import logging

logging.basicConfig(level=logging.INFO)

def crack_ras():
    url = "https://www.racingandsports.com.au/racing-index"
    # Scrapling Fetcher uses curl_cffi by default if available
    # We'll try chrome impersonation
    fetcher = Fetcher()
    try:
        # RAS might be checking for specific headers or behavior
        response = fetcher.get(
            url,
            impersonate="chrome120",
        )
        print(f"Status: {response.status}")
        print(f"Content Length: {len(response.text)}")

        if response.status == 200:
            with open("ras_success.html", "w") as f:
                f.write(response.text)
            if "Access Denied" in response.text:
                print("STILL BLOCKED: Access Denied")
            else:
                print("CRACKED! Content saved to ras_success.html")
        else:
            print(f"BLOCKED: Status {response.status}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    crack_ras()
