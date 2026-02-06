# 🐎 Fortuna - Complete Pseudocode Blueprint (Jules Edition)

**Status:** Unified Monolith Discovery & Analytics Architecture
**Version:** 3.1.0 (SQLite Integrated)
**Last Updated:** January 2026

---

## TABLE OF CONTENTS

1.  System Overview
2.  Architecture: The Monolithic Core
3.  Data Acquisition (Discovery)
4.  Persistence Layer (SQLite)
5.  Performance Auditing (Analytics)
6.  CLI & Automation

---

## 1. SYSTEM OVERVIEW

```
╔════════════════════════════════════════════════════════════════╗
║             FORTUNA - Racing Discovery Platform               ║
║      Unified Monolith Architecture for Global Racing          ║
╚════════════════════════════════════════════════════════════════╝

MISSION:
  • Acquire race data from global sources (Equibase, Racing Post, etc.).
  • Normalize and deduplicate data into a canonical Race format.
  • Identify "Goldmine" opportunities (2nd favorite >= 5.0 odds).
  • Persist predictions and outcomes in a robust SQLite database.
  • Audit historical performance to calculate ROI and Strike Rates.

CORE TENETS:
  • Monolithic Portability: `fortuna.py` contains all logic for discovery and storage.
  • SQLite Persistence: Replaced brittle JSON with thread-safe SQLite (`fortuna.db`).
  • Smart Fetching: Adaptive engine using curl_cffi, httpx, and browserforge.
  • Result Harvesting: Dedicated adapters for Equibase, RP, and ATR results.
```

---

## 2. ARCHITECTURE: THE MONOLITHIC CORE

The system consists of two primary Python scripts and a shared SQLite database.

```
+------------------------------------------------------+
| fortuna.py (Core Monolith)                           |
|                                                      |
|  +------------------+      +--------------------+    |
|  | Discovery Engine |----->| SQLite DB (FortunaDB)|    |
|  +------------------+      +---------|----------+    |
|           |                          |               |
|  +--------v----------+               |               |
|  | Analyzer Engine   |               |               |
|  +-------------------+               |               |
+--------------------------------------|---------------+
                                       |
+--------------------------------------v---------------+
| fortuna_analytics.py (Analytics Add-on)              |
|                                                      |
|  +------------------+      +--------------------+    |
|  | Results Harvester|----->| Auditor Engine     |    |
|  +------------------+      +--------------------+    |
+------------------------------------------------------+
```

---

## 3. DATA ACQUISITION (DISCOVERY)

### 3.1 Discovery Process (`fortuna.py`)

```pseudocode
PROCEDURE Run_Discovery(target_dates):
  // 1. Auto-discover all adapters inheriting from BaseAdapterV3
  adapters <- DISCOVER_ADAPTER_CLASSES()

  // 2. Fetch racecards concurrently
  raw_races <- FETCH_IN_PARALLEL(adapters, target_dates)

  // 3. Deduplicate using composite keys
  // Key = canonical_venue | race_number | date | discipline
  unique_races <- DEDUPLICATE(raw_races)

  // 4. Analyze for Goldmines
  // Goldmine = 2nd Fav >= 5.0 AND Field Size <= 8
  qualified_races <- SimplySuccessAnalyzer.qualify(unique_races)

  // 5. Persist to Database
  db <- FortunaDB()
  AWAIT db.log_tips(qualified_races)

  // 6. Generate Reports
  GENERATE summary_grid.txt
  GENERATE goldmine_report.txt
END PROCEDURE
```

### 3.2 Smart Fetching Engine

```pseudocode
CLASS SmartFetcher:
  FUNCTION fetch(url):
    // Tiered Engine Selection
    IF curl_cffi available: USE curl_cffi
    ELSE IF scrapling available: USE Playwright/Camoufox
    ELSE: USE httpx

    // Header Generation
    headers <- browserforge.generate_headers()
    headers['User-Agent'] <- fingerprint.navigator.userAgent

    RETURN execute_request(url, headers)
END CLASS
```

---

## 4. PERSISTENCE LAYER (SQLite)

### 4.1 Schema Definition

```sql
CREATE TABLE tips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id TEXT NOT NULL,
    venue TEXT NOT NULL,
    race_number INTEGER NOT NULL,
    start_time TEXT NOT NULL,
    report_date TEXT NOT NULL,
    is_goldmine INTEGER NOT NULL,
    gap12 TEXT,
    top_five TEXT,
    selection_number INTEGER,
    audit_completed INTEGER DEFAULT 0,
    verdict TEXT, -- CASHED, BURNED, VOID
    net_profit REAL,
    trifecta_payout REAL,
    ...
    UNIQUE(race_id, report_date)
);
```

### 4.2 Thread-Safe Implementation (`FortunaDB`)

```pseudocode
CLASS FortunaDB:
  INIT:
    self.executor <- ThreadPoolExecutor(max_workers=1)
    self.conn <- sqlite3.connect(db_path, check_same_thread=False)
    self.conn.execute("PRAGMA journal_mode=WAL")

  ASYNC FUNCTION log_tips(tips):
    RUN_IN_EXECUTOR:
      WITH transaction:
        FOR tip IN tips:
          IF NOT exists_within_dedup_window(tip.race_id):
            INSERT INTO tips ...
```

---

## 5. PERFORMANCE AUDITING (ANALYTICS)

### 5.1 Audit Workflow (`fortuna_analytics.py`)

```pseudocode
PROCEDURE Run_Analytics(target_dates):
  auditor <- AuditorEngine()

  // 1. Identify unverified tips from DB
  unverified <- auditor.get_unverified_tips()

  // 2. Harvest actual results
  results <- Harvest_Results(Equibase, RacingPost, SportingLife)

  // 3. Match and Evaluate
  FOR tip IN unverified:
    result <- FIND_MATCH(tip, results)
    IF found:
      outcome <- Evaluate_Outcome(tip, result)
      // Calculate Net Profit based on $2.00 unit for FTP bets
      auditor.update_audit_result(tip.id, outcome)

  // 4. Generate ROI Report
  audited <- auditor.get_all_audited_tips()
  report <- Generate_ROI_Report(audited)
  PRINT report
  SAVE TO analytics_report.txt
END PROCEDURE
```

---

## 6. CLI & AUTOMATION

### 6.1 Discovery CLI
`python fortuna.py --date 2026-01-28`

### 6.2 Analytics CLI
`python fortuna_analytics.py --days 2 --migrate`

### 6.3 Automation (GitHub Actions)
`fortuna_standalone.yml` triggers discovery and analytics hourly, uploading `goldmine_report.txt` and `analytics_report.txt` as artifacts.
