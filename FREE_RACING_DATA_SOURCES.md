# Free & Public Horse Racing Data Sources
## For Parimutuel Opportunity Hypothesis Testing

---

## 🎯 TL;DR — Start Here

**Best sources for immediate research:**
1. **Historical data:** Kaggle datasets (free, downloadable CSVs)
2. **Live racecards:** Official track websites (NYRA, Belmont, etc.)
3. **Results + payouts:** OffTrackBetting.com, Equibase.com (free browsing)
4. **API:** The Racing API (free trial available)

---

## 📊 TIER 1: Free Historical Data (No API Key Needed)

### **Kaggle Datasets** ⭐ (Easiest to Start)
**URL:** kaggle.com  
**Data:** Historical results, odds, payouts  
**Coverage:** Multiple countries, various date ranges  
**Cost:** FREE  
**Quality:** User-uploaded, varying quality  
**Use Cases:** Hypothesis testing, ML training  

**Available datasets:**
- **"Horse Racing" by hwaitt** — General horse racing dataset
- **"Horse Racing Results 2017-2020"** — Specific year range
- **"Big Data Derby"** by jpmiller — Comprehensive race data
- **"Historic Australian Horse Racing"** — Good for non-USA comparison
- **"Predicting Horse Race Outcomes"** — Pre-labeled dataset

**How to access:**
```bash
# Download directly (after Kaggle login)
# Or use Kaggle API:
pip install kaggle
kaggle datasets download -d hwaitt/horse-racing
```

### **Horse Racing Datasets** ⭐ (Specialized)
**URL:** horseracingdatasets.com  
**Data:** Kentucky Derby, track stats, prep race data  
**Coverage:** USA (Kentucky-focused)  
**Cost:** FREE  
**Quality:** Curated, researched  

**Available data:**
- Kentucky Derby historical data (all years, payouts, odds)
- Kentucky Derby prep race statistics
- Track statistics (Churchill Downs, Keeneland, Turfway, etc.)
- Winning odds patterns by distance
- Morning line vs. final odds analysis

**Example datasets:**
- "Win, exacta, trifecta payouts on $2 wager" with race charts
- "Kentucky Derby entrants stats" (morning line, final odds, ratings, breeding)
- "Track stats 2008-2019" (average field size, winning odds, favorite win %)

---

## 🔴 TIER 2: Live Racecards & Results (Browser, No API)

### **Official Track Websites** ⭐ (Best Quality)
All free, no registration required for viewing results.

#### **NYRA Tracks**
- **Aqueduct:** https://www.nyra.com/aqueduct/racing/results/
  - Live results, entries, HD stream
  - Free admission to track
  
- **Belmont Park:** https://www.nyra.com/belmont-stakes/racing/results/
  - Entries, race conditions, payouts
  - Note: Aqueduct closing in 2026, Belmont becoming primary

#### **Other Major Tracks**
- Churchill Downs: https://www.churchilldowns.com/ (Kentucky Derby home)
- Keeneland: https://www.keeneland.com/racing/ (Major stakes)
- Santa Anita: https://www.santaanita.com/ (California racing)
- Saratoga: https://www.saratogajuly.com/ (Summer stakes)

### **Aggregator Sites** (Consolidate Results)
- **OffTrackBetting.com** — Free results, payouts, entries
  - https://www.offtrackbetting.com/results/
  - Includes: Aqueduct, Belmont, major tracks
  - Data: Win/place/show payouts, exacta, trifecta, superfecta

- **Equibase.com** — Official racing data (free viewing, limited API)
  - https://www.equibase.com/
  - Results, past performances, historical charts
  - **Note:** ToS prohibits scraping, but free to browse

- **DRF (Daily Racing Form)** — Free results section
  - https://www.drf.com/race-results
  - Daily results, payouts, charts
  - Historical (free tier limited)

---

## 💻 TIER 3: APIs with Free Trials/Tiers

### **The Racing API** ⭐ (Most Comprehensive)
**URL:** theracingapi.com  
**Data:** Racecards, results, odds, historical (10+ years)  
**Coverage:** UK, Ireland, USA, Australia, 25+ countries  
**Update frequency:** Every 3-10 minutes  
**Cost:** Free trial available (check current offer)  
**Free tier:** Limited (check website for current limitations)  

**Features:**
- 500,000+ historical results & racecards
- Bookmaker odds (20+ sources for UK/Ireland)
- Real-time updates
- RESTful API, works with any programming language

**Rate limits:** Standard API rate limits apply  
**Documentation:** Comprehensive API docs

### **Podium Sports Racing API** (Enterprise-Grade)
**URL:** podiumsports.com  
**Data:** Racecards, historical odds (all changes tracked), results  
**Coverage:** UK, Ireland, USA, France, South Africa  
**Cost:** Likely requires contact for pricing  
**Key feature:** Full odds history (every update tracked)

**Unique data:**
- All odds changes throughout pre-race period
- Historical odds tracking for analysis
- Jockey/trainer/owner statistics
- Entrant form, ratings, distance history

---

## 🌍 TIER 4: Regional & Specialty Sources

### **Australia/New Zealand**
- **Racing Victoria:** https://www.racing.vic.gov.au/ (official results)
- **Racing NSW:** https://www.racing.nsw.gov.au/ (official results)
- **Ratings2Win Axis:** Australia/Hong Kong data (some free browsing)

### **UK/Ireland** 
- **Racing Post:** https://www.racingpost.com/ (free articles, limited data)
- **Sky Sports Racing:** https://www.skysports.com/racing/ (free results)
- **At The Races:** https://www.attheraces.com/ (UK/EU racing, free to view)

### **Canada**
- **Standardbred Canada:** https://www.standardbredcanada.ca/ (harness racing results)

---

## 📈 TIER 5: For Pattern Analysis (Parimutuel Opportunity)

### **Payout Pattern Archives**
These sources track historical payouts to find anomalies/patterns:

- **horseracingdatasets.com**
  - Kentucky Derby historical payouts (all years)
  - Morning line vs. final odds
  - Favorite vs. long-shot win rates

- **Kaggle datasets**
  - Filter by win payouts, exacta payouts, superfecta payouts
  - Compare across disciplines (thoroughbred vs. harness)
  - Analyze by distance, track condition, field size

### **Odds Movement Archives**
Looking for how odds shift pre-race?

- **Podium Sports API** (tracks every odds change)
- **The Racing API** (current odds, historical if available)
- **Racing Post** (some odds history in race articles)

---

## 🔧 PRACTICAL SETUP FOR RESEARCH

### **Best Workflow: Hypothesis Testing**

```python
# Step 1: Grab historical data
import kaggle
kaggle.api.dataset_download_files('hwaitt/horse-racing', path='./data')

# Step 2: Analyze for parimutuel patterns
import pandas as pd
df = pd.read_csv('./data/races.csv')

# Filter: favorites, longshots, exacta payouts
favorites_payout = df[df['odds'] <= 3.0]['payout'].mean()
longshots_payout = df[df['odds'] >= 10.0]['payout'].mean()

# Step 3: Test hypothesis
print(f"Favorite avg payout: ${favorites_payout:.2f}")
print(f"Longshot avg payout: ${longshots_payout:.2f}")
print(f"Margin: {longshots_payout - favorites_payout:.2f}")

# Step 4: Get current data for validation
# Use The Racing API free trial or OffTrackBetting.com
```

### **To Monitor Live Opportunities**

```bash
# Daily check script
curl -s https://www.offtrackbetting.com/results/1/aqueduct.html | grep -i "payout"
# Or use Playwright for JavaScript-heavy sites

python3 -c "
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto('https://www.nyra.com/aqueduct/racing/results/')
    # Extract payouts, odds, results
"
```

---

## 📋 Data Available by Source

| Source | Racecards | Odds | Payouts | Historical | Live | Cost | Notes |
|--------|-----------|------|---------|-----------|------|------|-------|
| Kaggle | ✅ | ✅ | ✅ | ✅ Very old | ❌ | FREE | Varies by dataset |
| HRD.com | ⚠️ Derby only | ✅ | ✅ | ✅ Complete | ❌ | FREE | KY-focused |
| Official Tracks | ✅ Current | ⚠️ Limited | ✅ | ✅ Recent | ✅ | FREE | Best quality |
| OffTrackBetting | ✅ | ⚠️ Limited | ✅ | ✅ Recent | ✅ | FREE | Aggregated |
| Equibase | ✅ | ✅ | ✅ | ✅ | ✅ | FREE/Paid | Anti-bot ToS |
| Racing API | ✅ | ✅ | ✅ | ✅ (10y) | ✅ | Trial | Best API |
| Podium Sports | ✅ | ✅✅ | ✅ | ✅ (full history) | ✅ | Paid | Odds tracking |

---

## ⚠️ Important Notes

### **Copyright & ToS**
- ✅ **Safe to use:** Kaggle (open datasets), official racing commission sites, public domain data
- ⚠️ **Grey area:** Viewing results on Equibase/Racing Post for research
- ❌ **Risky:** Automated scraping against ToS (Equibase, Racing Post, ADW sites)

### **Data Quality Issues**
- Kaggle datasets vary widely — check metadata & reviews
- Historical payouts may not reflect current parimutuel rules
- Morning line odds ≠ final odds (odds shift significantly)
- Track conditions affect payouts (weather, surface changes)

### **For Parimutuel Opportunity Testing**
Consider:
- **Hypothesis:** "Favorites are undervalued in exacta pools"
- **Data needed:** Historical final odds + exacta payouts
- **Sources:** Kaggle + horseracingdatasets.com (both have this data)
- **Validation:** Use The Racing API free trial to test on current races

---

## 🚀 Quick Start: Next 2 Hours

```
1. (15 min)  Sign up for Kaggle, download "Horse Racing Results 2017-2020"
2. (15 min)  Visit horseracingdatasets.com, download Kentucky Derby dataset
3. (30 min)  Load CSVs into pandas, do basic analysis:
             - What % of favorites win?
             - What's average payout by odds bracket?
             - Any undervalued/overvalued combinations?
4. (30 min)  Set up script to monitor live races:
             - Scrape OffTrackBetting or official track sites
             - Compare current odds to historical patterns
5. (30 min)  Test hypothesis on 1 day of live racing
```

---

## 📞 If You Need More

- **Real-time odds:** The Racing API (free trial)
- **Complete odds history:** Podium Sports (contact for trial)
- **USA-specific data:** Track official websites or horseracingdatasets.com
- **International:** Kaggle has Australian, HK, international datasets

---

## Summary: Your Best Bets

**For historical hypothesis testing:**
→ **Kaggle + horseracingdatasets.com** (free, comprehensive)

**For current racecards & results:**
→ **Official track websites + OffTrackBetting.com** (free, real-time)

**For API-based analysis:**
→ **The Racing API** (free trial, most flexible)

**For odds tracking (parimutuel opportunity):**
→ **Podium Sports** (contact for pricing) OR manual tracking from track sites

Good luck with your parimutuel research! 🏇
