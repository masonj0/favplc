# Tactical Guide: Extracting Parimutuel Data from Official Track Websites

## Overview

You now have:
1. **Historical datasets** (Kaggle, horseracingdatasets.com)
2. **Official track websites** (comprehensive directory)
3. **Central portals** (Equibase, USTA, HRI, JRA, etc.)

This guide shows you **how to extract data** from these sources for parimutuel opportunity testing.

---

## 📋 Phase 1: Identify Your Research Hypothesis

Before scraping/collecting, be clear:

### Example Hypotheses

**Hypothesis A: "Favorites are overvalued in exacta pools"**
- **Data needed:** Final odds + exacta payouts
- **Best source:** Kaggle datasets (historical) + official tracks (current validation)
- **Metric:** Compare favorite win rate vs. exacta payout ratio

**Hypothesis B: "Trifecta pools undervalue long-odds combinations"**
- **Data needed:** All finishing orders + trifecta payouts
- **Best source:** Detailed results from HKJC or JRA (sectional times, all trifecta prices)
- **Metric:** Identify systematically underpriced combinations

**Hypothesis C: "Morning line odds indicate mispricing"**
- **Data needed:** Morning line odds + final odds + results
- **Best source:** Kaggle + horseracingdatasets.com (both include morning line)
- **Metric:** Find races where morning line diverged most from final odds

**Hypothesis D: "Track surface changes affect payout margins"**
- **Data needed:** Race results + payout data + track condition (fast/muddy/sloppy)
- **Best source:** Official track sites (conditions recorded) + Kaggle (structured data)
- **Metric:** Compare payout margins on fast vs. sloppy tracks

---

## 🛠️ Phase 2: Collect Historical Data

### Step 1: Download Kaggle Datasets (Easiest)

```bash
# Install Kaggle API
pip install kaggle

# Set up credentials (https://kaggle.com/settings/account)
# Create ~/.kaggle/kaggle.json with your API token

# Download horse racing datasets
kaggle datasets download -d hwaitt/horse-racing
kaggle datasets download -d jpmiller/big-data-derby
unzip -d ./data horse-racing.zip
unzip -d ./data big-data-derby.zip
```

**What you'll get:**
- CSV files with historical races, odds, results, payouts
- Dates, tracks, jockeys, horse names, field sizes
- Win/place/show odds and payouts
- Some datasets include exacta/trifecta data

### Step 2: Get Specialized Derby Data (horseracingdatasets.com)

```bash
# Visit: https://www.horseracingdatasets.com/
# Download "Kentucky Derby Historical Payouts" (free)
# Includes: All winning odds, all exotic payouts, all years
# Format: CSV or Excel

# Example structure:
# Year | Winner | Winning Odds | Exacta $2 | Trifecta $2 | Superfecta $2 | Field Size
# 2024 | Mystik Dan | 26.00 | 128.00 | 1,451.20 | 32,893.40 | 17
```

### Step 3: Combine into Analysis Database

```python
import pandas as pd
import sqlite3

# Load Kaggle data
races_df = pd.read_csv('./data/races.csv')
results_df = pd.read_csv('./data/results.csv')

# Load Derby data
derby_df = pd.read_csv('./data/kentucky_derby_historical.csv')

# Create SQLite database
conn = sqlite3.connect('parimutuel_analysis.db')

races_df.to_sql('races', conn, if_exists='replace', index=False)
results_df.to_sql('results', conn, if_exists='replace', index=False)
derby_df.to_sql('derby_historical', conn, if_exists='replace', index=False)

conn.close()

print("Historical data loaded. Ready for analysis.")
```

---

## 🔍 Phase 3: Extract Current Data from Official Tracks

### Option A: Simple Browser Approach (Most Legal)

**For quick spot-checking:**

```
1. Visit official track website (e.g., NYRA.com)
2. Find "Results" section
3. Click on past race card
4. Screenshot or copy:
   - Race #, distance, surface, conditions
   - Finishing order
   - Payouts (Win, Place, Show, Exacta, Trifecta, Superfecta)
   - Field size
5. Paste into spreadsheet or database
```

**Best tracks for this approach:**
- NYRA (aqueduct.com, belmont.com) — Clean UI
- Official track websites — Usually readable
- OffTrackBetting.com — Aggregated in simple format

### Option B: Automated Collection (Python + Playwright)

**For systematic data collection:**

```python
from playwright.sync_api import sync_playwright
import pandas as pd
import re
from datetime import datetime, timedelta

def scrape_track_results(track_url, date):
    """
    Scrape results from official track website
    Safe approach: Use Playwright (renders JavaScript properly)
    """
    results = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        
        # Example: NYRA
        url = f"{track_url}/racing/results/{date}"
        page.goto(url)
        
        # Wait for results to load
        page.wait_for_selector('.race-results', timeout=10000)
        
        # Extract all races
        races = page.query_selector_all('.race-card')
        
        for race in races:
            race_num = race.query_selector('.race-number').text_content()
            
            # Get winning payouts
            win_payout = float(
                race.query_selector('.win-payout').text_content().replace('$', '')
            )
            exacta_payout = float(
                race.query_selector('.exacta-payout').text_content().replace('$', '')
            )
            trifecta_payout = float(
                race.query_selector('.trifecta-payout').text_content().replace('$', '')
            )
            superfecta_payout = float(
                race.query_selector('.superfecta-payout').text_content().replace('$', '')
            )
            
            # Get field info
            field_size = int(race.query_selector('.field-size').text_content())
            
            # Get winning combination (odds)
            winner_odds = float(
                race.query_selector('.winner-odds').text_content().replace('$', '')
            )
            
            results.append({
                'date': date,
                'race_number': race_num,
                'field_size': field_size,
                'winner_odds': winner_odds,
                'win_payout': win_payout,
                'exacta_payout': exacta_payout,
                'trifecta_payout': trifecta_payout,
                'superfecta_payout': superfecta_payout,
            })
        
        browser.close()
    
    return pd.DataFrame(results)

# Usage
nyra_results = scrape_track_results('https://www.nyra.com/aqueduct', '2026-02-24')
print(nyra_results)
```

**Important considerations:**
- Check the track's robots.txt first
- Use delays between requests (2-3 seconds)
- Consider "last 30 days" worth of data
- Focus on recent/live results for validation

### Option C: Use Official APIs (Best Long-Term)

**If available through official track or aggregator:**

```python
import requests
import pandas as pd

# Example: The Racing API (free trial)
# https://theracingapi.com

BASE_URL = "https://theracingapi.com/api"
PARAMS = {
    'action': 'getPastResults',
    'date': '2026-02-24',
    'track': 'Aqueduct',
}

response = requests.get(BASE_URL, params=PARAMS, headers={
    'Authorization': f'Bearer {YOUR_API_KEY}'
})

if response.status_code == 200:
    data = response.json()
    results_df = pd.DataFrame(data['races'])
    print(results_df[['race_num', 'winning_odds', 'exacta_payout', 'superfecta_payout']])
else:
    print(f"API error: {response.status_code}")
```

---

## 📊 Phase 4: Analyze for Parimutuel Opportunities

### Analysis Template (Hypothesis Testing)

```python
import pandas as pd
import numpy as np
from scipy import stats

# Load your combined dataset
df = pd.read_csv('parimutuel_combined.csv')

# ============================================
# HYPOTHESIS: Favorites are overvalued in exacta
# ============================================

# Filter: races with field size 8+
significant_races = df[df['field_size'] >= 8].copy()

# Define "favorite" as odds < 3.0
significant_races['is_favorite'] = significant_races['winner_odds'] < 3.0

# Group by favorite/non-favorite
fav_stats = significant_races[significant_races['is_favorite']].agg({
    'winner_odds': 'mean',
    'exacta_payout': 'mean',
    'exacta_payout': ['min', 'max', 'std']
})

nonfav_stats = significant_races[~significant_races['is_favorite']].agg({
    'winner_odds': 'mean',
    'exacta_payout': 'mean',
})

print("FAVORITE HORSES (odds < 3.0):")
print(f"  Avg winning odds: {fav_stats['winner_odds']['mean']:.2f}")
print(f"  Avg exacta payout: ${fav_stats['exacta_payout']['mean']:.2f}")
print(f"  Win rate: {(df['is_favorite'].sum() / len(df) * 100):.1f}%")

print("\nNON-FAVORITE HORSES (odds >= 3.0):")
print(f"  Avg winning odds: {nonfav_stats['winner_odds']['mean']:.2f}")
print(f"  Avg exacta payout: ${nonfav_stats['exacta_payout']['mean']:.2f}")

# Calculate margin
margin = fav_stats['exacta_payout']['mean'] - nonfav_stats['exacta_payout']['mean']
print(f"\nExacta payout margin: ${margin:.2f}")
print(f"Interpretation: Favorites undervalued by ${abs(margin):.2f}" if margin < 0 else f"Favorites overvalued by ${margin:.2f}")

# Statistical significance
t_stat, p_value = stats.ttest_ind(
    significant_races[significant_races['is_favorite']]['exacta_payout'],
    significant_races[~significant_races['is_favorite']]['exacta_payout']
)
print(f"\nStatistical significance (p-value): {p_value:.4f}")
if p_value < 0.05:
    print("✅ SIGNIFICANT (p < 0.05) — Not just random variation")
else:
    print("❌ NOT SIGNIFICANT — Could be random variation")
```

### Key Analyses by Hypothesis

**A. Odds Valuation Analysis**
```python
# Create odds brackets
df['odds_bracket'] = pd.cut(df['winner_odds'], 
    bins=[0, 2, 3, 5, 10, 50],
    labels=['Fav <2', '2-3', '3-5', '5-10', '10+'])

# Compare win rate vs. payout
comparison = df.groupby('odds_bracket').agg({
    'winner_odds': 'mean',
    'win': 'sum',  # If you have actual wins
    'exacta_payout': 'mean',
    'trifecta_payout': 'mean',
}).reset_index()

print(comparison)
# Find brackets where actual wins >> odds suggest
```

**B. Track Condition Impact**
```python
# Group by track condition
conditions = df.groupby('track_condition').agg({
    'exacta_payout': ['mean', 'median', 'std'],
    'trifecta_payout': ['mean', 'median', 'std'],
    'win': 'count'
}).reset_index()

print(conditions)
# Check if muddy/sloppy tracks have different payout margins
```

**C. Field Size Impact**
```python
# Larger fields → more exotic combinations → different pricing?
df['field_bracket'] = pd.cut(df['field_size'], 
    bins=[0, 6, 8, 10, 15, 20],
    labels=['4-6', '6-8', '8-10', '10-15', '15+'])

field_analysis = df.groupby('field_bracket').agg({
    'superfecta_payout': 'mean',
    'exacta_payout': 'mean',
    'win': 'count'
}).reset_index()

print(field_analysis)
# Larger fields → smaller payouts (pool divided more ways)?
```

---

## 🎯 Phase 5: Validate on Current Data

Once you identify a pattern in historical data, test it on **current races**:

```python
# Today's race data (collect from official track)
today_races = pd.read_csv('today_aqueduct.csv')

# Apply your hypothesis filter
hypothesis_races = today_races[
    (today_races['field_size'] >= 8) &
    (today_races['odds_bracket'] == 'Fav <2')  # Your pattern
]

print(f"Today's races matching your hypothesis: {len(hypothesis_races)}")
print(hypothesis_races[['race_num', 'winner_odds', 'exacta_payout']])

# Now monitor these races in real-time
# Do the payouts match your historical prediction?
```

---

## 🚀 Quick Start: Next 24 Hours

```
Hour 1:  Download Kaggle datasets + horseracingdatasets.com Kentucky Derby data
Hour 2:  Load into SQLite, run quick summary statistics
Hour 3:  Identify your hypothesis (overvalued favorites? undervalued long shots?)
Hour 4:  Run hypothesis test on historical data
Hour 5:  Measure statistical significance (p-value)
Hour 6:  Collect today's/this week's results from 2-3 official tracks
Hour 7:  Test hypothesis on current data
Hour 8+: Refine and plan next round of testing
```

---

## 📝 Key Metrics to Track

For any parimutuel hypothesis:

```
Pool depth        = Total money wagered on a bet type
Payout ratio      = Amount returned / amount wagered
Field efficiency  = Better combinations / total combinations
Favorite bias     = How much favorites overpay relative to odds
Odds movement     = Change from morning line to post time
Win rate by odds  = % of favorites (by odds bracket) that actually win
```

---

## ⚠️ Important Notes

- **Historical data quality varies** — Kaggle datasets may have errors or incomplete payout data
- **Rules change** — Payout calculations differ by era, jurisdiction, and parimutuel rules
- **Sample size matters** — Need minimum 100+ races per hypothesis to see true patterns
- **Validation is critical** — Test on 2026 data (out of historical dataset) before betting
- **Track variation** — Patterns may differ by track; test multiple venues

---

## Next Step

1. **Download data** (Kaggle, horseracingdatasets.com)
2. **Define hypothesis** (what are you testing?)
3. **Run analysis** (use templates above)
4. **Validate on current data** (test against 2026 results)
5. **Document findings** (track your hypothesis journal)

Good luck! 🏇
