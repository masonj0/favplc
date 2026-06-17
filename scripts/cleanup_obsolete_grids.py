import os
import glob
import zoneinfo
from datetime import datetime

def cleanup_obsolete_files():
    """Removes tactical reports and grids from previous calendar days."""
    et_tz = zoneinfo.ZoneInfo("America/New_York")
    now_et = datetime.now(et_tz)
    today_mmdd = now_et.strftime("%m%d")
    today_iso = now_et.strftime("%Y-%m-%d")

    patterns = [
        "v3_hourly_sheet_*.txt",
        "v3_strategy_sheet_*.txt",
        "race_grid_*.html",
        "concentrated_grid_*.html"
    ]

    count = 0
    for pattern in patterns:
        for f in glob.glob(pattern):
            # Keep today's files
            if today_mmdd in f or today_iso in f:
                continue
            try:
                os.remove(f)
                count += 1
            except:
                pass
    print(f"Cleanup complete: Removed {count} obsolete tactical files.")

if __name__ == "__main__":
    cleanup_obsolete_files()
