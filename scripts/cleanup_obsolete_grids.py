import os
import glob
from datetime import datetime

def cleanup_obsolete_files():
    """Removes tactical reports and grids from previous calendar days."""
    today_mmdd = datetime.now().strftime("%m%d")
    today_iso = datetime.now().strftime("%Y-%m-%d")

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
