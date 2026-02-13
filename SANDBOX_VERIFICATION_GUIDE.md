# Sandbox Verification Guide: Fortuna Engine Validation

This document provides a detailed walkthrough for verifying that the Fortuna
engine correctly discovers races and audits results within the Jules sandbox
environment.

---

## 🚀 Overview: The Sandbox as a Proving Ground

The Jules sandbox is a fully instrumented Linux environment capable of
executing the entire Fortuna stack — both **discovery** (finding upcoming
races) and **analytics** (auditing results against predictions).

**IMPORTANT:** The sandbox verification process is self-contained.
**NEVER** use `submit` or any Git commands to relay verification results.
Use the `message_user` tool to present final reports directly to the user.

---

## 🛠️ System Components

### 1. Discovery Engine — `fortuna.py`

The monolithic discovery pipeline. Orchestrates all discovery adapters,
applies `SimplySuccess` analyzer filters, and generates the standardized
summary grid.

| Mode | Command | Purpose |
|---|---|---|
| **Discovery** | `python3 fortuna.py` | Full adapter sweep, grid + goldmine report |
| **Monitor** | `python3 fortuna.py --monitor --once` | Single-pass live monitoring of imminent races (0–20 MTP window), then exit |

**Key outputs:**
- `summary_grid.txt` — Tiered alphabetic race grid
- `goldmine_report.txt` — High-value opportunity breakdown
- `race_data.json` — Structured race data for downstream consumption
- `fortuna.db` — SQLite database with tip history

### 2. Analytics Engine — `fortuna_analytics.py`

The results harvesting and audit pipeline. Fetches actual race results,
matches them against recorded predictions, computes profit/loss, and
generates performance reports.

| Mode | Command | Purpose |
|---|---|---|
| **Auto (today + yesterday)** | `python3 fortuna_analytics.py` | Audit last 2 days, auto-detect region |
| **Specific date** | `python3 fortuna_analytics.py --date 2026-02-07` | Audit a single date |
| **Region-locked** | `python3 fortuna_analytics.py --region USA` | Only use US results adapters |
| **With lifetime stats** | `python3 fortuna_analytics.py --include-lifetime-stats` | Append ROI/strike-rate summary |
| **Verbose** | `python3 fortuna_analytics.py -v` | Debug-level logging |
| **Migration** | `python3 fortuna_analytics.py --migrate` | Migrate legacy JSON → SQLite |

**Key outputs:**
- `analytics_report.txt` — Human-readable audit report
- `results_harvest.json` — Adapter performance metrics
- `fortuna.db` — Updated with audit verdicts + payouts

---

## ⚡ The "Immediate Gold" Filter (0–20 MTP)

The discovery pipeline implements a strict time filter to maximize signal
relevance:

| Criteria | Value |
|---|---|
| **MTP window** | 0 < Minutes to Post ≤ 20 |
| **Purpose** | Focus on races about to start, when odds are most stable |

### Goldmine Categorization

| Tier | Criteria |
|---|---|
| **Immediate Gold (Superfecta)** | MTP ≤ 20, 2nd-fav odds ≥ 5.0, Superfecta available |
| **Immediate Gold** | MTP ≤ 20, 2nd-fav odds ≥ 5.0, no Superfecta |
| **Future Goldmine** | Qualifying race, MTP > 20 |

The `1Gap2` metric (odds gap between 1st and 2nd favorite) is reported for
every goldmine candidate.

---

## 📡 Adapter Fleet

### Discovery Adapters (in `fortuna.py`)

| Discipline | Adapters |
|---|---|
| **Thoroughbred** | AtTheRaces, RacingPost, SportingLife, SkySports, Equibase, TwinSpires, Oddschecker, Timeform |
| **Harness** | StandardbredCanada, Tab, SkySports |
| **Greyhound** | AtTheRacesGreyhound, BoyleSports |

### Results Adapters (in `fortuna_analytics.py`)

| Region | Adapters |
|---|---|
| **USA** | `EquibaseResultsAdapter` |
| **International** | `RacingPostResultsAdapter`, `AtTheRacesResultsAdapter`, `SportingLifeResultsAdapter`, `SkySportsResultsAdapter` |

All results adapters inherit from `PageFetchingResultsAdapter` and share
common fetch/parse infrastructure. Region selection is automatic based on
time of day (US: 9am–11pm ET) or can be forced with `--region`.

---

## 📊 Summary Grid Format

The summary grid (`summary_grid.txt`) provides immediate visual confirmation
of discovered data.

| Feature | Description |
|---|---|
| **Tiered sections** | Primary (Superfecta available) → Secondary (all others) |
| **CATEG column** | **T** = Thoroughbred, **H** = Harness, **G** = Greyhound |
| **Race coding** | `a`=R1, `b`=R2, … Uppercase = Goldmine qualifier |
| **1Gap2** | Odds gap between 1st and 2nd favorite |

---

## 🗄️ Database Schema

The `fortuna.db` SQLite database is the single source of truth for tip
history and audit results. Key tables:

| Table | Purpose |
|---|---|
| `tips` | All recorded predictions with audit status |
| `harvests` | Adapter performance log per run |

Key fields on `tips`:
- `audit_completed` — Boolean, flipped after result matching
- `verdict` — `CASHED` / `BURNED` / `VOID`
- `net_profit` — Dollar P/L (based on $2 base unit)
- `superfecta_payout`, `trifecta_payout` — Exotic dividends when available

---

## ✅ Verification Procedure

### Step 1 — Run Discovery

```bash
python3 fortuna.py
```

Confirm outputs exist and contain data:
```bash
wc -l summary_grid.txt goldmine_report.txt
cat goldmine_report.txt | head -40
```

### Step 2 — Run Analytics (if auditable tips exist)

```bash
python3 fortuna_analytics.py -v
```

Confirm audit report:
```bash
cat analytics_report.txt | head -50
```

### Step 3 — Verify Database

```bash
sqlite3 fortuna.db "
  SELECT COUNT(*) as total,
         SUM(CASE WHEN audit_completed = 1 THEN 1 ELSE 0 END) as audited,
         SUM(CASE WHEN verdict = 'CASHED' THEN 1 ELSE 0 END) as cashed
  FROM tips;
"
```

### Step 4 — Report to User

1. Read generated reports with `read_file`
2. Format a comprehensive summary
3. Call `message_user` with the content
4. **DO NOT COMMIT** run-specific artifacts

---

## 🔧 Troubleshooting

### Missing Dependencies

```bash
# Install all requirements (prefer a requirements.txt in production)
python3 -m pip install \
  httpx \
  "structlog>=23.0" \
  "pydantic>=2.5,<3" \
  pydantic-settings \
  pandas \
  selectolax \
  tenacity \
  scrapling \
  curl_cffi \
  msgspec \
  browserforge \
  camoufox \
  redis

# Browser engines
python3 -m browserforge update
playwright install --with-deps chromium
```

### Common Failures

| Symptom | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: fortuna` | Wrong working directory | `cd` to repo root |
| All adapters return 0 races | Network/firewall blocking | Check `curl -I https://www.equibase.com` |
| `sqlite3.OperationalError: database is locked` | Concurrent access | Kill other Python processes |
| `TimeoutError` on adapter fetch | Slow network / rate-limited | Increase `TIMEOUT` on adapter or retry |
| Empty `goldmine_report.txt` | No races in MTP window | Try `--monitor` or wait for race cards |
| `playwright._impl._errors.Error` | Chromium not installed | Re-run `playwright install --with-deps chromium` |
| `ImportError: scrapling` | Optional dependency missing | `pip install scrapling` (non-critical) |

### Debug Mode

For maximum visibility into adapter behavior:
```bash
# Discovery with debug snapshots
python3 fortuna.py --debug

# Analytics with full trace
python3 fortuna_analytics.py -v --days 1
```

Debug snapshots are saved to `debug_snapshots/` and contain raw HTML
from each adapter fetch — invaluable for diagnosing selector breakage.
