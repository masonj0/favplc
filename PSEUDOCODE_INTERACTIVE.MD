# 🐎 Fortuna Faucet — Interactive System Blueprint

**Status:** Hybrid Three-Module Architecture (Automated + Interactive)
**Version:** 3.2.0
**Last Updated:** March 20, 2026

---

## TABLE OF CONTENTS

1. [System Overview](#1-system-overview)
2. [Architecture: Hybrid Three-Mode Design](#2-architecture-hybrid-three-mode-design)
3. [Module 1 — Discovery Engine (`fortuna.py` / `fortuna_interactive.py`)](#3-module-1--discovery-engine-fortunapy--fortunainteractivepy)
4. [Module 2 — Analytics & Audit Engine (`fortuna_analytics.py`)](#4-module-2--analytics--audit-engine-fortuna_analyticspy)
5. [Module 3 — Shared Utilities (`fortuna_utils.py`)](#5-module-3--shared-utilities-fortuna_utilspy)
6. [Data Models & Persistence](#6-data-models--persistence)
7. [Analytical Scoring System](#7-analytical-scoring-system)
8. [Adapter Inventory](#8-adapter-inventory)
9. [Deployment & Automation (CI/CD)](#9-deployment--automation-cicd)
10. [End-to-End Workflow](#10-end-to-end-workflow)

---

## 1. SYSTEM OVERVIEW

```
╔════════════════════════════════════════════════════════════════╗
║           FORTUNA FAUCET — Racing Analysis Platform           ║
║               Hybrid Architecture v3.2.0                      ║
╚════════════════════════════════════════════════════════════════╝

MISSION:
  • Acquire racecard data from 20+ global sources (APIs + web scraping + manual ingest).
  • Normalize and deduplicate into a canonical Race/Runner format.
  • Score and grade opportunities using a composite analytical model.
  • Store predictions in SQLite and audit them against actual results.
  • Report on P&L, hit rate vs breakeven, and adapter health over time.

CORE TENETS:
  • Separation of concerns: discovery, analytics, and utilities are
    distinct modules with clean import boundaries.
  • Resilient fetching: a SmartFetcher tries multiple browser/HTTP
    engines in order of health score, falling back gracefully.
  • Reproducible results: all tips carry a canonical key and are
    matched against results via multi-stage fuzzy lookup.
  • Automated operation: all workflows run on GitHub Actions schedules
    with SQLite persisted via Actions cache between runs.
```

---

## 2. ARCHITECTURE: HYBRID THREE-MODE DESIGN

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                          GITHUB ACTIONS CI/CD                                 │
│                                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  ┌──────────────┐  │
│  │  Discovery   │  │  Analytics   │  │   DB Auditor &    │  │ Manual Ingest│  │
│  │  Workflow    │  │  Workflow    │  │  Pipeline Health  │  │   Workflow   │  │
│  │ (scheduled)  │  │ (scheduled)  │  │    Workflow       │  │  (manual)    │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬──────────┘  └──────┬───────┘  │
│         │                 │                   │                    │          │
└─────────┼─────────────────┼───────────────────┼────────────────────┼──────────┘
          │                 │                   │                    │
          ▼                 ▼                   ▼                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                fortuna.py  /  fortuna_interactive.py                          │
│  ┌──────────────────────────────────────────────────────┐   ┌──────────────┐  │
│  │  Discovery Engine                                    │   │  Ingest Mode │  │
│  │  • 20+ Adapter classes (BaseAdapterV3 subclasses)   │   │  • Link Disc │  │
│  │  • SmartFetcher (Automated + Manual Ingest logic)    │   │  • File-based│  │
│  │  • Race / Runner / OddsData Pydantic models         │   │  • Interactive│ │
│  │  • SimplySuccessAnalyzer (scoring & grading)        │   └──────────────┘  │
│  │  • FortunaDB (SQLite read/write)                    │                     │
│  │  • FastAPI server + pywebview GUI (optional)        │                     │
└──────────────────────────────────────────────────────┘                     │
│                                                              │
│           ▲  imports                      writes ▼          │
└───────────┼──────────────────────────────────────┬──────────┘
            │                                      │
┌───────────┴──────────────────────────────────────▼──────────┐
│                   fortuna_utils.py                          │
│  • VENUE_MAP, DRF_VENUE_MAP, RACING_KEYWORDS               │
│  • normalize_venue_name(), get_canonical_venue()           │
│  • parse_odds_to_decimal(), SmartOddsExtractor             │
│  • get_places_paid(), ensure_eastern(), etc.               │
└─────────────────────────────────────────────────────────────┘
            ▲  imports
┌───────────┴──────────────────────────────────────────────────┐
│                  fortuna_analytics.py                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Harvest Engine                                      │   │
│  │  • 10+ ResultsAdapter classes (parallel fetch)      │   │
│  │  • ResultRace / ResultRunner models (extend fortuna) │   │
│  │  • _race_quality() deduplication heuristic          │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Audit Engine (AuditorEngine)                        │   │
│  │  • Builds canonical key map from harvested results  │   │
│  │  • Multi-stage tip-to-result matching               │   │
│  │  • Verdict computation (CASHED / BURNED / VOID)     │   │
│  │  • Batch DB persistence                             │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Report Generator                                    │   │
│  │  • Text analytics report (stdout + file)            │   │
│  │  • GitHub Actions Job Summary (Markdown)            │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘

             Both modules share:  fortuna.db  (SQLite)
```

---

## 3. MODULE 1 — Discovery Engine (`fortuna.py` / `fortuna_interactive.py`)

### 3.1 Entry Point & Execution Modes

The discovery engine now operates in two distinct tiers: **Automated** and **Interactive/Manual**.

#### Automated Discovery (`fortuna.py`)
- **Default (once):** Runs discovery for today's races, scores them, writes tips to the DB, and saves a report.
- **Monitor mode (`--monitor`):** Runs discovery in a continuous loop on a configurable interval.
- **Monitor once (`--monitor --once`):** A single monitor cycle, used by CI/CD.
- **GUI mode (`--gui`):** Starts a FastAPI/Uvicorn server on port 8013, then launches a pywebview desktop window pointed at it.
- **Fetch-only mode (`--fetch-only`):** Fetches and saves raw race data without scoring or reporting — useful for debugging adapters.
- **Region filter (`--region USA | INT | GLOBAL`):** Restricts which adapter pool is used for discovery.
- **Quality filter (`--quality solid | lousy`):** Restricts to the `SOLID_DISCOVERY_ADAPTERS` set or its complement.

#### Interactive/Manual Ingest (`fortuna_interactive.py`)
- **Interactive Mode:** When run in a terminal, prompts the user to paste HTML content for each URL. Use `EOF` to submit and `SKIP` to bypass.
- **Link Discovery Mode (`--list-links`):** Identifies all URLs the engine needs to fetch and prints them along with the expected slug-based filenames for `manual_fetch/`.
- **File-based Ingest:** Automatically detects and loads HTML files from the `manual_fetch/` directory matching the target URL slugs. This mode is used for GitHub Actions automation.
- **Sequential Execution:** Unlike the automated engine, the interactive version processes all requests sequentially to maintain a clean input buffer.

### 3.2 Region & Quality Adapter Pools

Discovery adapters are partitioned into three region pools and one quality partition:

```
USA_DISCOVERY_ADAPTERS:
  Equibase, TwinSpires, RacingPostB2B, StandardbredCanada,
  AtTheRaces, NYRABets

INT_DISCOVERY_ADAPTERS:
  TAB, BetfairDataScientist

GLOBAL_DISCOVERY_ADAPTERS:
  SkyRacingWorld, AtTheRaces, AtTheRacesGreyhound, RacingPost,
  Oddschecker, Timeform, SportingLife, SkySports, RacingAndSports

SOLID_DISCOVERY_ADAPTERS (high-reliability subset):
  TwinSpires, SkyRacingWorld, RacingPost
```

### 3.3 SmartFetcher — Multi-Engine HTTP Layer

All adapters share a common `SmartFetcher` that maintains a health score (0.0–1.0) for each available engine. On every request, engines are tried in descending health order. A successful response increases the engine's health; a failure decreases it.

**Engine priority order (default):**

| Engine | Description |
|--------|-------------|
| Camoufox (scrapling) | Stealthy headless browser — highest anti-bot score |
| curl_cffi | Browser TLS impersonation without a full browser |
| Playwright (scrapling) | Full headless Chromium via scrapling wrapper |
| Playwright Legacy | Direct Playwright without scrapling — fallback |
| httpx | Plain async HTTP — fastest, lowest stealth |

If `browserforge` is available, all requests are augmented with a generated browser fingerprint (User-Agent, headers) consistent with the chosen engine.

### 3.4 Adapter Base Class (`BaseAdapterV3`)

Every discovery adapter inherits from `BaseAdapterV3` and optionally mixes in:

- `BrowserHeadersMixin` — injects browser-realistic headers
- `JSONParsingMixin` — helpers for JSON response handling
- `DebugMixin` — saves raw HTML/JSON snapshots on parse failure
- `RacePageFetcherMixin` — shared logic for index → detail page navigation

Each adapter must implement:

- `SOURCE_NAME` — a unique string identifier
- `fetch_races(target_date)` — returns a list of `Race` objects

### 3.5 Race & Runner Data Models

**`Runner`** carries:
- `name` (cleaned: country suffixes stripped, leading numbers stripped)
- `number` (saddle cloth number)
- `win_odds` (decimal float)
- `odds` (dict of `OddsData` keyed by source name)
- `scratched`, `trainer`, `jockey`

**`Race`** carries:
- `id` (generated canonical key — see 3.6)
- `venue` (normalized through `VENUE_MAP` on construction)
- `race_number`, `start_time` (always stored in US Eastern Time)
- `runners`, `discipline`, `surface`, `distance`
- `race_type`, `is_handicap` (bool, used by scoring and place-count logic)
- `available_bets` (list of exotic types found in the page HTML)
- `qualification_score`, metadata dict (carries scoring signals)

### 3.6 Race ID Generation

A race ID is a deterministic, URL-safe string:

```
Format:  {prefix}_{venue_slug}_{YYYYMMDD}_{HHMM}_R{race_number}{discipline_suffix}

Example: sl_aqueduct_20260222_1430_R5_t

Discipline suffixes:  _t (thoroughbred)  _h (harness)  _g (greyhound)  _q (quarter horse)
```

Venue slugs are lowercase, alphanumeric-only, derived from the canonical venue name. If a venue slug exceeds 25 characters (suggesting race-title contamination), the generator logs a warning and attempts first-word recovery.

### 3.7 Deduplication

After all adapters have returned races, duplicates are merged using the canonical key:

```
Canonical key format:  {venue_slug}|{race_number}|{YYYYMMDD}|{HHMM}|{discipline_initial}

Merge strategy:
  1. Group races by canonical key
  2. For each group, keep the race with the highest trust_ratio
     (fraction of runners with valid, non-placeholder odds)
  3. Merge odds from all sources into the surviving race's runner pool
```

### 3.8 SimplySuccessAnalyzer — Scoring Pipeline

See [Section 7](#7-analytical-scoring-system) for the full scoring model.

### 3.9 FortunaDB — SQLite Persistence

`FortunaDB` wraps an async SQLite connection. Key operations:

- `save_tips(tips)` — upserts a batch of scored races; ignored if a tip with the same `race_id` already exists
- `get_unverified_tips(lookback_hours)` — returns tips with `audit_completed = 0` within the time window
- `update_audit_results_batch(outcomes)` — writes verdict, profit, and result fields to matched tips
- `get_all_audited_tips()` / `get_recent_tips(limit)` — used by report generation

Schema migrations are applied automatically via `ALTER TABLE` guards on startup. See [Section 6](#6-data-models--persistence).

### 3.10 Integrated FastAPI Dashboard (GUI mode only)

When launched with `--gui`, a FastAPI/Uvicorn server starts on `localhost:8013`. The root route `/` generates an HTML dashboard from live DB data. The `pywebview` window displays this as a native desktop application. This mode is intended for Windows local use; it is not part of the CI/CD pipeline.

---

## 4. MODULE 2 — Analytics & Audit Engine (`fortuna_analytics.py`)

### 4.1 Purpose

`fortuna_analytics.py` is the second half of the operational loop. Where `fortuna.py` predicts, `fortuna_analytics.py` verifies. It runs on a schedule after races have concluded, harvests actual results from dedicated results adapters, matches them against stored predictions, computes verdicts, and generates a performance report.

### 4.2 Results Adapter Pools

Results adapters are partitioned by region, with a separate quality partition:

```
USA_RESULTS_ADAPTERS:
  RacingPostUSAResults, SportingLifeResults, EquibaseResults,
  StandardbredCanadaResults, DRFResults,
  [NYRABetsResults marked DEPRECATED — API endpoint returns 403]

INT_RESULTS_ADAPTERS:
  RacingPostResults, RacingPostTote, AtTheRacesResults,
  AtTheRacesGreyhoundResults, SportingLifeResults, SkySportsResults,
  RacingAndSportsResults, TimeformResults

SOLID_RESULTS_ADAPTERS (high-reliability subset):
  StandardbredCanadaResults, RacingPostResults, SportingLifeResults,
  AtTheRacesGreyhoundResults, TimeformResults, SkySportsResults
```

### 4.3 ResultRace and ResultRunner Models

`ResultRunner` extends `Runner` with:
- `position` (string as reported: "1st", "2/12", "W", etc.)
- `position_numeric` (parsed integer: 1, 2, 3…)
- `final_win_odds`, `win_payout`, `place_payout`, `show_payout`

`ResultRace` extends `Race` with:
- `trifecta_payout`, `trifecta_combination`
- `exacta_payout`, `exacta_combination`
- `superfecta_payout`, `superfecta_combination`
- `is_fully_parsed` (flag set when all runners have positions and odds)
- Two computed properties:
  - `canonical_key` — full 5-part key including time
  - `relaxed_key` — 4-part key without time (for fuzzy matching)

### 4.4 Harvest Pipeline

```
For each target date:

  1. Instantiate all applicable results adapters
     (filtered by region, quality, or explicit include list)

  2. Run all adapters concurrently (up to _MAX_CONCURRENT_FETCHES = 10)

  3. For each adapter's returned races:
       Score each race with _race_quality():
         quality = number_of_runners
                 + (10 × runners_with_valid_odds)
                 + (5  × runners_with_place_payout)
                 + (3  × runners_with_numeric_position)
                 + (20 if trifecta_payout present)
                 + (20 if superfecta_payout present)

  4. Deduplicate: for each canonical key, keep the highest-quality result

  5. Log race counts and max_odds per adapter to harvest_summary dict
     and persist to harvest_logs table
```

### 4.5 Audit Pipeline (AuditorEngine)

```
Given: a list of ResultRace objects and a list of unverified tips from DB

Step 1 — Build results map:
  For each result race:
    Store under canonical_key (5-part: venue|race|date|time|disc)
    Also store under relaxed_key (4-part: venue|race|date|disc)
    On relaxed key collision, keep the canonical entry

Step 2 — For each unverified tip:
  Construct the tip's canonical key from stored venue/race/time/discipline

  Attempt match in order (stopping at first success):
    a. Exact canonical key match
    b. Same venue+race+date, time within ±90 minutes, same discipline
    c. Same venue+race+date, no time component, same discipline
    d. Same venue+race+date, any discipline (single match only)
    e. Race number ±1 shifted fallback (catches off-by-one indexing)

  If no match found: tip remains unverified, logged as warning

Step 3 — Evaluate matched tip:
  Identify selection runner by saddle cloth number, then by name
  Determine places_paid = get_places_paid(field_size, is_handicap)

  If selection finished within places_paid:
    If actual place_payout available:
      verdict = CASHED,           profit = place_payout − £2.00
    Else:
      estimated_return = £2.00 × max(1.1, 1.0 + (win_odds − 1.0) ÷ 5)
      verdict = CASHED_ESTIMATED, profit = estimated_return − £2.00

  If selection finished outside places_paid:
    verdict = BURNED,             profit = −£2.00

  If selection not found in result or position unknown:
    verdict = VOID,               profit = £0.00

Step 4 — Batch-persist all verdicts to DB
```

### 4.6 Report Generation

After each audit run, two reports are produced:

**Text analytics report** (printed to stdout and saved as `analytics_report.txt`):
- Lifetime summary: total bets, win rate, net P&L, ROI
- Rolling performance by discipline, by grade, by venue
- Recent tip ledger (last 20 results with verdict icons)

**GitHub Actions Job Summary** (Markdown, written to `$GITHUB_STEP_SUMMARY`):
- Harvest performance table (adapter → race count, success rate)
- Predictions table for the day's tips
- Proof section linking to any saved artifacts

---

## 5. MODULE 3 — Shared Utilities (`fortuna_utils.py`)

This module is a pure-logic library with no side effects. Both `fortuna.py` and `fortuna_analytics.py` import from it.

### 5.1 Venue Normalisation

**`VENUE_MAP`** — ~180 entries mapping raw venue strings (both full names and exchange codes) to canonical forms.

```
Examples:
  "AQU"           → "Aqueduct"
  "AQUEDUCT"      → "Aqueduct"
  "DUNSTALL PARK" → "Wolverhampton"
  "CATTERICK BRIDGE" → "Catterick"
```

**`DRF_VENUE_MAP`** — 24 entries mapping DRF 2–3 letter track codes to canonical names, used by the planned DRFResults adapter.

**`normalize_venue_name(raw)`** — the normalisation pipeline:

```
1. Strip parenthetical content, hyphens → spaces, collapse whitespace
2. Strip race/meeting name noise: scan for keywords from RACING_KEYWORDS
   (HANDICAP, STAKES, CHASE, HURDLE, PRIZE, etc.) and truncate there
3. Strip trailing time strings (e.g. "Ludlow 13:30" → "Ludlow")
4. Collapse repeated words (e.g. "Bahrain Bahrain" → "Bahrain")
5. Direct match against VENUE_MAP (uppercase key)
6. Longest-prefix word-boundary match against VENUE_MAP
7. Same on the original unstripped string (up to 4 words)
8. Legacy substring prefix match
9. Fallback: title-case the cleaned string as-is
```

**`get_canonical_venue(name)`** — applies `normalize_venue_name` then strips all non-alphanumeric characters for use in race IDs and canonical keys.

### 5.2 Odds Parsing

**`parse_odds_to_decimal(raw)`** — handles:
- Fractional: `"7/4"`, `"7-4"`, `"7 TO 4"` → `2.75`
- Decimal: `"5.00"` → `5.0`
- Integer: `"5"` (interpreted as 5/1) → `6.0`
- Special strings: `"EVS"`, `"EVENS"` → `2.0`
- Scratch markers: `"SCR"`, `"NR"`, `"VOID"` → `None`

**`SmartOddsExtractor`** — scans raw HTML text or selectolax nodes for odds using keyword proximity, decimal patterns, and fractional patterns in sequence.

**`is_valid_odds(value)`** — returns True if the value is a float in the range `[1.01, 1000.0)` and not in `COMMON_PLACEHOLDERS`.

### 5.3 Place Count Logic

**`get_places_paid(field_size, is_handicap)`** — UK industry standard rules:

| Field size | Non-handicap | Handicap |
|-----------|-------------|---------|
| 1–4 | 1 place (win only) | 1 place |
| 5–7 | 2 places | 2 places |
| 8–15 | 3 places | 3 places |
| 16+ | 3 places | **4 places** |

### 5.4 Timezone Utilities

All datetimes are stored and compared in US Eastern Time. Three helpers enforce this:

- `now_eastern()` — current time as an Eastern-aware datetime
- `to_eastern(dt)` — converts any timezone-aware or naive datetime to Eastern
- `ensure_eastern(dt)` — idempotent: returns Eastern if already Eastern, converts otherwise

---

## 6. DATA MODELS & PERSISTENCE

### 6.1 SQLite Schema (Current — v5/v6)

**`tips` table** — one row per prediction:

| Column | Type | Description |
|--------|------|-------------|
| `race_id` | TEXT (PK) | Generated canonical ID |
| `venue` | TEXT | Normalised venue name |
| `race_number` | INTEGER | Race number on the card |
| `start_time` | TEXT | ISO 8601, Eastern Time |
| `discipline` | TEXT | Thoroughbred / Harness / Greyhound |
| `selection_name` | TEXT | Predicted runner's name |
| `selection_number` | INTEGER | Saddle cloth number |
| `predicted_odds` | REAL | Decimal odds at discovery time |
| `gap12` | REAL | (2nd_fav − fav) ÷ fav — ratio format |
| `field_size` | INTEGER | Number of active runners |
| `market_depth` | REAL | Composite spread of mid-field odds |
| `place_prob` | REAL | Modelled place probability [0, 1] |
| `predicted_ev` | REAL | Expected value of a £2 place bet |
| `race_type` | TEXT | e.g. "Handicap", "Maiden", "Stakes" |
| `condition_modifier` | REAL | Scoring adjustment for race conditions |
| `qualification_grade` | TEXT | A+, A, B, C, D |
| `composite_score` | REAL | Raw score before grade assignment |
| `is_goldmine` | INTEGER | 0/1 boolean |
| `is_best_bet` | INTEGER | 0/1 boolean |
| `is_handicap` | INTEGER | 0/1/NULL |
| `source` | TEXT | Discovery adapter name |
| `audit_completed` | INTEGER | 0 = pending, 1 = audited |
| `verdict` | TEXT | CASHED / CASHED_ESTIMATED / BURNED / VOID |
| `net_profit` | REAL | Profit/loss in £ (stake = £2.00) |
| `audit_timestamp` | TEXT | When audit occurred |
| `match_confidence` | TEXT | exact / relaxed / shifted |
| `actual_2nd_fav_odds` | REAL | Actual 2nd favourite odds at result time |
| `selection_position` | INTEGER | Where selection actually finished |
| `top1_place_payout` | REAL | Place payout for winner |
| `top2_place_payout` | REAL | Place payout for runner-up |
| `trifecta_payout` | REAL | Trifecta dividend if available |
| `superfecta_payout` | REAL | Superfecta dividend if available |
| `predicted_2nd_fav_odds` | REAL | Predicted 2nd fav odds at discovery |
| `actual_top_5` | TEXT | Comma-separated finishing order |

**`harvest_logs` table** — one row per adapter per harvest run:

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER (PK) | Auto-increment |
| `timestamp` | TEXT | ISO 8601 |
| `adapter_name` | TEXT | Results adapter source name |
| `race_count` | INTEGER | Races returned (0 = failure) |
| `max_odds` | REAL | Highest odds seen |
| `region` | TEXT | USA / INT / GLOBAL |

**`schema_version` table** — tracks applied migrations:

| Column | Type |
|--------|------|
| `version` | INTEGER |
| `applied_at` | TEXT |

### 6.2 Cache Persistence (GitHub Actions)

The database is persisted between workflow runs using the Actions cache:

```
Cache key pattern:  fortuna-db-v3-master-{run_number}

Restore strategy:   fortuna-db-v3-master-  (prefix match — takes latest)

Each workflow restores the latest DB at start, saves an updated
copy at end. The purge workflow deletes all stale cache entries
after saving a clean version.
```

---

## 7. ANALYTICAL SCORING SYSTEM

### 7.1 Input Requirements

A race is eligible for scoring only if:
- It has at least 2 non-scratched runners with valid odds
- The favourite's odds are in the range `[1.01, 1000.0)`
- The second favourite is identifiable
- Field size is at most 11 runners (configurable via `config.toml`)

### 7.2 Primary Signal: Gap Ratio (`gap12`)

The gap ratio measures the relative odds distance between the favourite and second favourite:

```
gap12 = (second_favourite_odds − favourite_odds) ÷ favourite_odds

Interpretation:
  gap12 < 0.20  →  Tight market, 2nd fav closely matched to fav
  gap12 = 0.50  →  2nd fav is 50% longer than fav (moderate separation)
  gap12 ≥ 1.00  →  2nd fav is at least double the fav (dominant favourite)
```

A higher gap12 indicates the market believes the favourite is significantly superior to the field — the key condition for the strategy to work.

### 7.3 Secondary Signal: Place Probability (`place_prob`)

The implied probability that the favourite places, accounting for field size and handicap status:

```
places_paid = get_places_paid(field_size, is_handicap)

If field_size >= 2 and places_paid >= 2:
  win_prob   = 1 ÷ favourite_odds
  place_prob = min(win_prob + (1 − win_prob) × (places_paid − 1) ÷ (field_size − 1), 0.97)

Edge cases:
  field_size = 1:  place_prob = 1.0 (walkover)
  places_paid = 1: place_prob = win_prob (win-only market)
```

### 7.4 Secondary Signal: Market Depth (`market_depth`)

A measure of how spread the mid-field odds are, scaled to [0, 10]:

```
Collect odds for all non-scratched runners.
Compute the median of all runner odds.
market_depth = min( (median_odds ÷ favourite_odds − 1.0) × 4.0, 10.0 )

Interpretation:
  A deep market (high median relative to fav) suggests a wide quality gap —
  the fav stands out from a moderate-quality field.
```

### 7.5 Condition Modifier

A small adjustment applied based on race type:

| Race Type | Modifier |
|-----------|----------|
| Maiden | −0.15 |
| Claiming | +0.08 |
| Stakes / Graded | −0.10 |
| Default | +0.00 |

### 7.6 Composite Score and Grade Assignment

```
composite_score = (gap12    × 40.0)
                + (market_depth     )   [max 10]
                + (place_prob × 40.0)
                + (condition_modifier × 20.0)

Grade thresholds:
  composite ≥ 70  →  A+
  composite ≥ 60  →  A
  composite ≥ 50  →  B+
  composite ≥ 42  →  B
  composite ≥ 32  →  C
  composite < 32  →  D  (below qualification threshold)
```

### 7.7 Goldmine and Best Bet Flags

A tip is flagged **Best Bet** if:
- `gap12 ≥ 0.55` AND
- `qualification_grade` is A+ or A

A tip is flagged **Goldmine** if all Best Bet conditions are met, plus:
- `field_size ≤ 9` AND
- `second_favourite_odds ≥ 4.5` AND
- Race type is NOT Maiden

### 7.8 Predicted Expected Value

The expected value of a standard £2.00 place bet, computed at discovery time:

```
If place_prob > 0 and at least one valid odds value exists:
  estimated_place_payout = 2.00 × max(1.1, 1.0 + (fav_odds − 1.0) ÷ 5)
  predicted_ev = (place_prob × estimated_place_payout) − ((1 − place_prob) × 2.00)

A positive predicted_ev suggests the bet has theoretical value at these odds.
This is a discovery-time estimate; actual verdict uses real place payouts.
```

---

## 8. ADAPTER INVENTORY

### 8.1 Discovery Adapters (20 total)

| Adapter | Region | Discipline | Notes |
|---------|--------|-----------|-------|
| `NYRABetsAdapter` | USA | Thoroughbred | NYRA circuit: Aqueduct, Belmont, Saratoga |
| `EquibaseAdapter` | USA | Thoroughbred | All US TB tracks |
| `TwinSpiresAdapter` | USA | Thoroughbred | JSON API |
| `StandardbredCanadaAdapter` | USA | Harness | Mohawk, Woodbine, Rideau, etc. |
| `RacingPostB2BAdapter` | USA | Thoroughbred | B2B JSON endpoint |
| `AtTheRacesAdapter` | Global | Thoroughbred | UK/IRE/INT; also in USA pool |
| `AtTheRacesGreyhoundAdapter` | Global | Greyhound | UK greyhound tracks |
| `SportingLifeAdapter` | Global | Thoroughbred | UK/IRE JSON API |
| `SkySportsAdapter` | Global | Thoroughbred | UK/IRE JSON API |
| `RacingPostAdapter` | Global | Thoroughbred | UK/IRE/INT; in SOLID set |
| `OddscheckerAdapter` | Global | Thoroughbred | Multi-market odds aggregator |
| `TimeformAdapter` | Global | Thoroughbred | UK/IRE form-based |
| `SkyRacingWorldAdapter` | Global | Thoroughbred | AUS/INT; in SOLID set |
| `RacingAndSportsAdapter` | Global | Thoroughbred | AUS/NZ/INT |
| `TabAdapter` | INT | Thoroughbred | AUS/NZ TAB markets |
| `BetfairDataScientistAdapter` | INT | Thoroughbred | AUS/NZ model output |

### 8.2 Results Adapters (11 total)

| Adapter | Region | 30-Day Success | Notes |
|---------|--------|---------------|-------|
| `RacingPostUSAResults` | USA | ✅ 100% | Primary USA TB results source |
| `StandardbredCanadaResults` | USA | ✅ 100% | Harness; Mohawk etc. |
| `SportingLifeResults` | USA+INT | ✅ ~100% | UK/IRE primary; also picks up USA |
| `EquibaseResults` | USA | ✅ Active | Uses Camoufox to bypass blocks |
| `DRFResults` | USA | ✅ Active | Supplementary USA TB text charts |
| `NYRABetsResults` | USA | ⚠️ Deprecated | Internal API returning 403 |
| `RacingPostResults` | INT | ✅ ~100% | Primary INT TB results source |
| `SkySportsResults` | INT | ✅ ~100% | UK/IRE |
| `AtTheRacesResults` | INT | ⚠️ Intermittent | Known crash issues |
| `AtTheRacesGreyhoundResults` | INT | ✅ ~100% | UK greyhound |
| `TimeformResults` | INT | ⚠️ Intermittent | Known crash issues |
| `RacingAndSportsResults` | INT | ❌ 0% | 403 from GitHub runners |
| `RacingPostTote` | INT | ⚠️ Low | Sparse data; supplements RacingPostResults |

---

## 9. DEPLOYMENT & AUTOMATION (CI/CD)

All production workflows run on GitHub Actions using Ubuntu 22.04 runners. The SQLite database (`fortuna.db`) is persisted between runs via the Actions cache system.

### 9.1 Workflow: Discovery (`fortuna.py`)

**Trigger:** Scheduled (multiple times daily) + manual dispatch
**What it does:**
1. Restores `fortuna.db` from cache
2. Installs Python 3.11 dependencies including Playwright Chromium
3. Runs `fortuna.py --monitor --once --region GLOBAL` (or `--region USA`)
4. Tips are scored, graded, and written to the DB
5. A Job Summary is written with the day's graded opportunities
6. DB is saved back to cache under a new key

### 9.2 Workflow: Manual Ingest (`fortuna_manual_fetch.yml`)

**Trigger:** Manual dispatch via GitHub UI
**Inputs:** `daypart` (Q1-Q4), `list_links_only` (bool)
**What it does:**
1. Allows the user to select a DayPart and choose between discovering URLs or ingesting files.
2. In **List Links** mode: Executes `fortuna_interactive.py --list-links` to show which HTML files need to be uploaded to `manual_fetch/`.
3. In **Ingest** mode: Executes a sequential discovery pass, loading HTML content from the `manual_fetch/` directory, then performs scoring and grading.
4. Uploads the resulting reports and database state as artifacts.

### 9.3 Workflow: Analytics (`fortuna_analytics.py`)

**Trigger:** Scheduled (after races conclude each day) + manual dispatch
**What it does:**
1. Restores `fortuna.db` from cache
2. Runs `fortuna_analytics.py --days 2 --include-lifetime-stats`
3. Results adapters harvest actual outcomes for the past 2 days
4. Unverified tips are matched and verdicts assigned
5. Analytics report is printed and uploaded as an artifact
6. Job Summary shows harvest performance and P&L
7. DB saved back to cache

### 9.4 Workflow: DB Auditor & Pipeline Health

**Trigger:** Scheduled (3× daily) + after "List of Best Bets" workflow completes + manual dispatch
**What it does:**

Runs a self-contained diagnostic script with four phases:

- **Phase 1 — Audit execution:** Runs `fortuna_analytics.py` as a subprocess and captures its output
- **Phase 2 — Database forensics:** Queries the DB directly for 12 diagnostic sections (2A–2L), including freshness, discipline×verdict matrix, breakeven analysis, rolling 7-day P&L, goldmine vs standard performance, odds gap buckets, venue leaderboard, stuck tip age distribution, scoring column inventory, and grade performance
- **Phase 3 — Pipeline health** (triggered on stale data, audit failure, or manual force): Checks harvest log success rates and schema version
- **Phase 4 — Automated assessment:** Assigns HEALTHY / WARNING / CRITICAL status and emits specific findings with recommended actions

### 9.5 Workflow: Purge Corrupted Data

**Trigger:** Manual dispatch only (safety measure)
**Inputs:** `execute` (true/false — dry run by default), `aggressive` (also delete stale tips)
**What it does:**
1. Restores `fortuna.db`
2. Runs `scripts/purge_corrupted_data.py` with appropriate flags
3. Deletes corrupted tips (venue contamination, race number collisions)
4. Resets audit state on tips with known bad data
5. Deletes zero-race harvest log entries
6. Runs SQLite VACUUM
7. Saves the cleaned DB to cache and deletes all stale master caches

### 9.6 Local / Windows Portable Mode

For local use, `fortuna.py` can be packaged as a self-contained Windows executable using PyInstaller. The `build_monolith.py` script handles:

- Collecting all dependencies (browserforge, playwright, curl_cffi, scrapling, etc.)
- Embedding a Playwright Chromium browser
- Running PyInstaller with `--onefile`
- Verifying the generated executable launches without error

---

## 10. END-TO-END WORKFLOW

### 10.1 Daily Prediction Cycle

```
Morning (CI/CD — scheduled):

  1. Discovery workflow runs
     └── 16 adapters fetch today's races from global sources
     └── Races are normalised, deduplicated, and scored
     └── Qualified tips (grade C or better) are saved to fortuna.db
     └── Job Summary shows today's A+/A/Goldmine tips

  2. DB Auditor runs (triggered by completion of step 1)
     └── Confirms new tips landed
     └── Reports pending audit count and pipeline health
```

### 10.2 Daily Audit Cycle

```
Evening (CI/CD — scheduled after races finish):

  1. Analytics workflow runs
     └── Results adapters fetch outcomes for past 2 days
     └── Harvested results are deduplicated by quality score
     └── AuditorEngine matches tips to results via canonical key lookup
     └── Verdicts (CASHED / BURNED / VOID) written to fortuna.db
     └── Analytics report shows P&L, hit rate, breakeven margin

  2. DB Auditor runs (scheduled independently, 3× daily)
     └── Phase 2 produces full forensics: discipline matrix, rolling
         7-day chart, venue leaderboard, grade performance table
     └── Phase 4 computes overall system health status
```

### 10.3 Data Flow Summary

```
  Raw HTML / JSON                      SQLite (fortuna.db)
  from 20+ websites                         │
       │                                    │
       ▼                                    │
  SmartFetcher                             ┌▼──────────────────┐
  (multi-engine)                           │  tips table        │
       │                                   │  (predictions +   │
       ▼                                   │   verdicts +      │
  Adapter parse                            │   scoring cols)   │
       │                                   └───────────────────┘
       ▼                                            ▲
  Race/Runner                                       │
  models                                     AuditorEngine
       │                                     (match + verdict)
       ▼                                            ▲
  SimplySuccessAnalyzer                             │
  (score + grade)                            ResultRace/Runner
       │                                            ▲
       ▼                                            │
  FortunaDB.save_tips()               ResultsAdapter.fetch()
                                       (11 adapters, parallel)
```

### 10.4 Canonical Key — The Matching Backbone

Every tip and every harvested result is identified by the same key format:

```
{venue_slug} | {race_number} | {YYYYMMDD} | {HHMM} | {discipline_initial}

Example:  aqueduct|5|20260222|1430|T
```

The audit engine builds a lookup map of results keyed by this format, then matches each unverified tip against it. The multi-stage fallback (exact → time-window → no-time → no-discipline → race-number-shift) ensures that minor timing discrepancies between discovery and results sources don't cause false non-matches.

---

*This concludes the Fortuna Faucet system blueprint. All pseudocode is descriptive and non-executable.*
