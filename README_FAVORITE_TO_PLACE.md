# Fortuna Favorite-to-Place Betting Monitor

## 🎯 Overview

This script monitors live racing data and identifies **favorite-to-place betting opportunities** based on:

1. **Second favorite odds ≥ 5.0 decimal** (longshot second choice)
2. **Races under 20 minutes to post (MTP)**
3. **Superfecta availability** (preferred, sorted first)

## 📋 Features

### 1. Full Race List
Displays ALL fetched races with:
- Discipline (T/H/G)
- Track name
- Race number
- Field size
- Superfecta offered (Yes/No)
- Adapter source
- Start time

### 2. BET NOW List (Auto-Refreshing)
Shows only races meeting betting criteria:
- **Sorted by**: Superfecta available → MTP (soonest first)
- **Updates every 30 seconds** (configurable)
- Displays favorite and 2nd favorite with odds

### 3. JSON Export
Saves complete data to `race_data.json`:
```json
{
  "generated_at": "2026-02-03T14:30:00",
  "target_date": "2026-02-03",
  "total_races": 142,
  "bet_now_count": 8,
  "all_races": [...],
  "bet_now_races": [...]
}
```

## 🚀 Installation

### Prerequisites
```bash
# Ensure you have fortuna.py in the same directory
# Install required packages:
pip install httpx structlog pandas selectolax tenacity pydantic scrapling
```

### Quick Start
```bash
# Make executable
chmod +x favorite_to_place_monitor.py

# Run with defaults (today's date, continuous monitoring)
./favorite_to_place_monitor.py

# Or with Python
python favorite_to_place_monitor.py
```

## 📖 Usage Examples

### Basic Usage (Continuous Monitoring)
```bash
python favorite_to_place_monitor.py
```
**Output:**
- Shows full race list once
- Refreshes BET NOW list every 30 seconds
- Press `Ctrl+C` to stop

### Run Once (No Continuous Updates)
```bash
python favorite_to_place_monitor.py --once
```
**Output:**
- Fetches races
- Displays full list
- Displays BET NOW list
- Saves JSON and exits

### Specify Date
```bash
# Tomorrow's races
python favorite_to_place_monitor.py --date 2026-02-04

# Specific date
python favorite_to_place_monitor.py --date 2026-02-14
```

### Custom Refresh Interval
```bash
# Refresh every 60 seconds
python favorite_to_place_monitor.py --refresh-interval 60

# Refresh every 10 seconds (faster updates)
python favorite_to_place_monitor.py --refresh-interval 10
```

### Combined Options
```bash
python favorite_to_place_monitor.py \
  --date 2026-02-05 \
  --refresh-interval 45 \
  --once
```

## 📊 Output Format

### Full Race List
```
================================================================================
                           FULL RACE LIST
================================================================================
DISC  TRACK                     R#   FIELD  SUPER  ADAPTER                   START TIME
--------------------------------------------------------------------------------
T     Aqueduct                  1    8      Yes    RacingPostB2BAdapter      2026-02-03 13:00
T     Aqueduct                  2    10     Yes    RacingPostB2BAdapter      2026-02-03 13:30
H     Northfield Park           1    7      No     StandardbredCanadaAdapter 2026-02-03 13:15
...
```

### BET NOW List
```
================================================================================
              🎯 BET NOW - FAVORITE TO PLACE OPPORTUNITIES
================================================================================
Updated: 2026-02-03 14:30:15
Criteria: MTP < 20 minutes AND 2nd Favorite Odds >= 5.0
--------------------------------------------------------------------------------
SUPER  MTP   DISC  TRACK                R#   FIELD  2ND FAV              2ND ODDS   FAVORITE             FAV ODDS   ADAPTER
--------------------------------------------------------------------------------
✅     5     T     Gulfstream Park      3    9      Lightning Strike     8.50       Rocket Man           2.10       AtTheRacesAdapter
✅     12    T     Santa Anita          7    8      Dark Thunder         6.20       Speed Demon          1.85       TwinSpiresAdapter
❌     8     H     Meadowlands          4    7      Pacer Supreme        5.40       Quick Step           2.40       StandardbredCanada
...
```

## 🔧 Configuration

### Adapters Used
The script uses these adapters (from `fortuna.py`):
- RacingPostB2BAdapter
- SportingLifeAdapter
- SkySportsAdapter
- StandardbredCanadaAdapter
- AtTheRacesAdapter
- TwinSpiresAdapter

### Betting Criteria (Customize in Code)

Edit these constants in the script:

```python
# In get_bet_now_races() method:
if race.mtp is not None and race.mtp < 20:  # Change 20 to adjust MTP threshold
    if race.second_fav_odds is not None and race.second_fav_odds >= 5.0:  # Change 5.0 for odds threshold
```

### Sort Order (Customize in Code)

Default sort: Superfecta first, then MTP

```python
# In get_bet_now_races() method:
bet_now.sort(key=lambda r: (not r.superfecta_offered, r.mtp))

# Alternative sorts:
# By MTP only:
bet_now.sort(key=lambda r: r.mtp)

# By odds only (highest first):
bet_now.sort(key=lambda r: -r.second_fav_odds)

# By field size (largest first):
bet_now.sort(key=lambda r: -r.field_size)
```

## 📁 Output Files

### race_data.json
Complete race data in JSON format:
```json
{
  "generated_at": "2026-02-03T14:30:00.123456",
  "target_date": "2026-02-03",
  "total_races": 142,
  "bet_now_count": 8,
  "all_races": [
    {
      "discipline": "T",
      "track": "Gulfstream Park",
      "race_number": 3,
      "field_size": 9,
      "superfecta_offered": true,
      "adapter": "AtTheRacesAdapter",
      "start_time": "2026-02-03T18:35:00+00:00",
      "mtp": 5,
      "second_fav_odds": 8.5,
      "second_fav_name": "Lightning Strike",
      "favorite_odds": 2.1,
      "favorite_name": "Rocket Man"
    }
  ],
  "bet_now_races": [...]
}
```

## 🎨 Display Legend

### Discipline Codes
- **T** = Thoroughbred
- **H** = Harness (Standardbred)
- **G** = Greyhound

### Superfecta Column
- **✅** = Superfecta offered
- **❌** = No Superfecta

## 🐛 Troubleshooting

### No races fetched
```bash
# Check if adapters are working
python favorite_to_place_monitor.py --once

# Check the output - you should see:
# ✅ RacingPostB2BAdapter
# If you see ❌, that adapter failed
```

### Import errors
```bash
# Make sure fortuna.py is in the same directory
ls -la fortuna.py

# Install missing packages
pip install httpx structlog pandas selectolax tenacity pydantic scrapling
```

### No BET NOW races
This is normal! The criteria are strict:
- MTP < 20 minutes (races must be starting soon)
- 2nd favorite odds ≥ 5.0 (longshot second choice)

**Solutions:**
- Wait for races to get closer to post time
- Lower the odds threshold in the code (change `5.0` to `4.0`)
- Increase MTP threshold (change `20` to `30`)

### Screen clearing issues
If the screen-clearing is annoying, comment out this line:
```python
# In print_bet_now_list() method:
# print("\033[H\033[J", end="")  # Comment this out
```

## 📈 Advanced Usage

### Run in Background
```bash
# Redirect output to file
python favorite_to_place_monitor.py > fortuna.log 2>&1 &

# View live updates
tail -f fortuna.log
```

### Run on a Schedule (cron)
```bash
# Edit crontab
crontab -e

# Run every hour at :00
0 * * * * cd /path/to/fortuna && python favorite_to_place_monitor.py --once >> fortuna.log 2>&1
```

### Parse JSON with jq
```bash
# Get count of BET NOW races
jq '.bet_now_count' race_data.json

# Get all Superfecta races
jq '.bet_now_races[] | select(.superfecta_offered == true)' race_data.json

# Get races under 10 MTP
jq '.bet_now_races[] | select(.mtp < 10)' race_data.json
```

## 🔍 Strategy Notes

### Why Second Favorite ≥ 5.0?
- Favorite-to-place bets profit when the favorite places (1st, 2nd, or 3rd)
- If the second choice is heavily favored (low odds), place payouts are small
- A longshot second choice (≥5.0) means more value if favorite places

### Why Superfecta Preferred?
- Superfecta races typically have:
  - Larger fields (more betting interest)
  - Better odds on exotic bets
  - More liquidity in the pools

### Optimal MTP Window
- **Too early (>30 MTP)**: Odds will change significantly
- **Too late (<5 MTP)**: Not enough time to place bet
- **Sweet spot (10-20 MTP)**: Odds stabilizing, time to bet

## 📞 Support

### Common Questions

**Q: Why are some fields empty (N/A)?**
A: Not all adapters provide complete odds data. This is normal.

**Q: Can I add more adapters?**
A: Yes! Edit the `ADAPTER_CLASSES` list in the script.

**Q: How do I change the criteria?**
A: Edit the `get_bet_now_races()` method in the script.

**Q: Does this place bets automatically?**
A: No! This is a monitoring tool only. You must place bets manually.

## ⚖️ Disclaimer

**This tool is for informational purposes only.** 

- Does not provide betting advice
- Does not guarantee profits
- Requires manual bet placement
- Use at your own risk
- Gamble responsibly

---

**Version:** 1.0  
**Last Updated:** February 3, 2026  
**License:** MIT
