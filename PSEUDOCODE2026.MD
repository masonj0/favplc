# 🐎 Fortuna Faucet - Complete Pseudocode Blueprint

**Status:** Unified Monolith Architecture
**Version:** 3.0.0
**Last Updated:** February 20, 2026

---

## TABLE OF CONTENTS

1.  System Overview
2.  Architecture: The Unified Monolith
3.  Backend Engine (Fortuna Core)
4.  Frontend Interface (Integrated Dashboard)
5.  Data Models & Persistence
6.  Deployment & Automation (CI/CD)
7.  End-to-End Workflow

---

## 1. SYSTEM OVERVIEW

```
╔════════════════════════════════════════════════════════════════╗
║         FORTUNA FAUCET - Racing Analysis Platform             ║
║      Unified Monolith Architecture for Cross-Platform Use      ║
╚════════════════════════════════════════════════════════════════╝

MISSION:
  • Acquire race data from 20+ global sources (APIs + web scraping).
  • Normalize and deduplicate data into a canonical Race format.
  • Apply analytical filters to surface high-value betting opportunities.
  • Serve a static dashboard and a REST API from a single, self-contained executable.
  • Provide cross-platform access via launcher scripts or a standalone .exe.

CORE TENETS:
  • Single Source: `fortuna.py` contains the entire engine, server, and analyzer.
  • Zero Dependencies: The primary Windows artifact is a single .exe built with PyInstaller.
  • Robust Scoring: Grade-based (A+ to C) qualification system using composite scores.
  • Automated & Repeatable Builds: The entire application is built and tested via `build_monolith.py`.
```

---

## 2. ARCHITECTURE: THE UNIFIED MONOLITH

The application is a **Unified Monolith**. A single Python script, `fortuna.py`, contains the complete application logic. When built with PyInstaller, it runs a FastAPI web server that serves both the dashboard and the REST API, often launching in a native window via `pywebview`.

```
+------------------------------------------------------+
| FortunaFaucetPortableApp.exe (Single Executable)     |
|                                                      |
|  +-------------------------------------------------+ |
|  | Python Environment (fortuna.py)                 | |
|  |                                                 | |
|  |  +-------------------------------------------+  | |
|  |  | FastAPI / Uvicorn Server (localhost:8013) |  | |
|  |  |                                           |  | |
|  |  |  +------------------+  +----------------+ |  | |
|  |  |  | API Router       |  | HTML Dashboard | |  | |
|  |  |  | (/api/*)         |  | (Root /)       | |  | |
|  |  |  +------------------+  +----------------+ |  | |
|  |  |           |                     |         |  | |
|  |  +-----------|---------------------|---------+  | |
|  |             |                     |            | |
|  |  +----------v-----------+  +-------v--------+   | |
|  |  | DiscoveryEngine      |  | Desktop Window |   | |
|  |  | (Adapters)           |  | (pywebview)    |   | |
|  +-------------------------+-----------------------+ |
|                                                      |
+------------------------------------------------------+
```

---

## 3. BACKEND ENGINE (FORTUNA CORE)

### 3.1 Entry Point & Multi-Mode Execution (`fortuna.py`)

The engine supports various modes: CLI, Monitor, and GUI.

```pseudocode
PROCEDURE Main_Entry_Point
  PARSE command line arguments

  IF --monitor:
    RUN FavoriteToPlaceMonitor in continuous loop
  ELSE IF --gui:
    START FastAPI server in background thread
    LAUNCH pywebview window pointing to localhost:8013
  ELSE:
    RUN Discovery once and generate reports
END PROCEDURE
```

### 3.2 Integrated FastAPI Server (`start_desktop_app`)

The backend defines a FastAPI app to serve the dashboard and provide API access to the underlying SQLite database.

```pseudocode
FUNCTION start_desktop_app()
  app <- CREATE FastAPI()

  @app.get("/")
  FUNCTION get_dashboard():
    tips <- FETCH latest tips from FortunaDB
    GENERATE HTML with dynamic tips table
    RETURN HTML Response

  RUN uvicorn server on port 8013
END FUNCTION
```

### 3.3 Advanced Analytical Scoring (`SimplySuccessAnalyzer`)

The analyzer uses a multi-factor composite score to grade betting opportunities.

```pseudocode
CLASS SimplySuccessAnalyzer:
  FUNCTION qualify_races(races):
    FOR race IN races:
      // 1. Calculate Primary Metrics
      gap12 <- (2nd_Fav_Odds - Fav_Odds) / Fav_Odds
      place_prob <- calculate_implied_place_probability(Fav_Odds, Field_Size)
      market_depth <- calculate_market_depth(Whole_Field_Odds)
      modifier <- get_condition_modifier(Race_Type)

      // 2. Generate Composite Score
      score <- (gap12 * 40) + (market_depth) + (place_prob * 40) + (modifier * 20)

      // 3. Assign Grade
      grade <- MATCH_THRESHOLD(score) // A+ (>=70), A (>=60), etc.

      // 4. Determine "Best Bet" and "Goldmine" status
      IF gap12 >= 0.35:
        IF grade in ('A+', 'A'): is_best_bet <- True
        IF Field_Size <= 9 AND 2nd_Fav_Odds >= 4.5 AND NOT Maiden:
          is_goldmine <- True
```

---

## 4. FRONTEND INTERFACE (INTEGRATED DASHBOARD)

The frontend is served directly as HTML by the FastAPI server, providing a zero-latency, same-origin experience without external JS dependencies.

```pseudocode
// Simplified Dashboard Logic
TEMPLATE Dashboard:
  DISPLAY "Fortuna Intelligence Dashboard"
  DISPLAY current stats (Tips Found, Cashed, Net Profit)

  TABLE of Opportunities:
    FOR tip IN tips:
      RENDER Venue, Race Number, Selection, Predicted Odds, and Grade

  SCRIPT:
    AUTO-REFRESH page every 30 seconds
```

---

## 5. DATA MODELS & PERSISTENCE

All data is persisted in a thread-safe SQLite database (`fortuna.db`) using a Version 6 schema.

```
TABLE tips:
  - race_id (Unique Key)
  - venue, race_number, start_time
  - qualification_grade (A+, A, B, etc.)
  - composite_score, place_prob, predicted_ev
  - gap12, market_depth, is_handicap
  - audit_completed, verdict, net_profit
```

---

## 6. DEPLOYMENT & AUTOMATION (CI/CD)

The build process is managed by `build_monolith.py`, which packages `fortuna.py` and its dependencies.

```pseudocode
PROCEDURE Build_Monolith:
  1. Create build metadata (timestamp, git commit)
  2. Collect all dependencies (browserforge, playwright, curl_cffi, etc.)
  3. Execute PyInstaller with --onefile and --noupx
  4. Verify the generated EXE launches successfully
END PROCEDURE
```

---

## 7. END-TO-END WORKFLOW

### 7.1 Portable App Workflow

```
WORKFLOW user_runs_portable_app:
  STEP 1: User runs `FortunaFaucetPortableApp.exe`.
  STEP 2: The process extracts its embedded environment to a temp directory.
  STEP 3: The FastAPI server starts and the Desktop Window (pywebview) opens.
  STEP 4: The app automatically runs discovery in the background to find new races.
  STEP 5: The dashboard updates in real-time as new Goldmines are identified.
```

---
*This concludes the blueprint for the Fortuna Faucet unified monolith architecture.*
