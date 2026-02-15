# Fortuna Sandbox Verification — Agent Instructions

## ⚠️ CRITICAL RULES

1. **NEVER** run `submit`, `git commit`, `git push`, or any Git commands
2. **NEVER** create or modify files outside the repo working directory
3. All results are reported via `message_user` — not committed
4. If a step fails, log the error and continue to the next step
5. Report ALL outcomes — including failures — to the user

---

## Phase 0: Environment Setup

Run these commands IN ORDER. If any command fails, stop and report
the failure to the user before continuing.

```bash
cd /path/to/repo   # Adjust to actual repo root

# Verify repo structure
ls fortuna.py fortuna_analytics.py
# EXPECTED: both files listed. If not, STOP and report.

# Install dependencies
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt 2>&1 | tail -5

# If requirements.txt doesn't exist, use this fallback:
# python3 -m pip install httpx "structlog>=23.0" "pydantic>=2.5,<3" \
#   pydantic-settings pandas selectolax tenacity scrapling curl_cffi \
#   msgspec browserforge camoufox redis aiosqlite

# Browser setup
python3 -m browserforge update 2>&1 | tail -3
playwright install --with-deps chromium 2>&1 | tail -3
```

### Verify imports before proceeding

```bash
python3 -c "
import fortuna
import fortuna_analytics
print('IMPORTS: OK')
print(f'Adapters: {len(fortuna.get_adapter_classes())} discovery')
print(f'Results:  {len(fortuna_analytics.get_results_adapter_classes())} results')
"
```

**Decision:**
- If output contains `IMPORTS: OK` → proceed to Phase 1
- If `ModuleNotFoundError` → report the missing module name, attempt
  `pip install <module>`, retry once
- If any other error → report full traceback, skip to Phase 4

---

## Phase 1: Discovery

### 1a. Run discovery

```bash
export PYTHONPATH=.
timeout 600 python3 fortuna.py 2>&1 | tee discovery_output.log | tail -20
EXIT_CODE=$?
```

**Decision:**
- Exit code 0 → proceed to validation
- Exit code 124 (timeout) → report "Discovery timed out after 10 minutes"
- Any other exit code → report last 30 lines of `discovery_output.log`

### 1b. Validate outputs

```bash
python3 -c "
import json, os

results = {}

# Check each expected output
for fname, min_size in [
    ('summary_grid.txt', 10),
    ('goldmine_report.txt', 10),
    ('race_data.json', 2),
    ('fortuna.db', 1000),
]:
    exists = os.path.exists(fname)
    size = os.path.getsize(fname) if exists else 0
    ok = exists and size >= min_size
    results[fname] = {'exists': exists, 'size': size, 'ok': ok}

# Validate JSON structure
try:
    with open('race_data.json') as f:
        data = json.load(f)
    bet_now = len(data.get('bet_now_races', []))
    might_like = len(data.get('you_might_like_races', []))
    results['race_data.json']['races'] = bet_now + might_like
except Exception as e:
    results['race_data.json']['error'] = str(e)

# Print report
print('=== DISCOVERY VALIDATION ===')
all_ok = True
for fname, info in results.items():
    status = '✅' if info.get('ok') else '❌'
    extra = ''
    if 'races' in info:
        extra = f' ({info[\"races\"]} races)'
    elif 'error' in info:
        extra = f' ERROR: {info[\"error\"]}'
    print(f'{status} {fname}: {info.get(\"size\", 0)} bytes{extra}')
    if not info.get('ok'):
        all_ok = False

print(f'\nOVERALL: {\"PASS\" if all_ok else \"PARTIAL\"} ')
"
```

**Decision:**
- `OVERALL: PASS` → proceed to Phase 2
- `OVERALL: PARTIAL` → note which files failed, proceed anyway
  (discovery may find 0 races if run outside racing hours — this
  is normal, not an error)

### 1c. Capture discovery content for final report

```bash
echo "=== SUMMARY GRID (first 40 lines) ==="
head -40 summary_grid.txt 2>/dev/null || echo "(empty)"

echo ""
echo "=== GOLDMINE REPORT (first 30 lines) ==="
head -30 goldmine_report.txt 2>/dev/null || echo "(empty)"

echo ""
echo "=== RACE COUNTS ==="
python3 -c "
import json
try:
    d = json.load(open('race_data.json'))
    bn = d.get('bet_now_races', [])
    ym = d.get('you_might_like_races', [])
    gold = [r for r in bn + ym if r.get('is_goldmine')]
    print(f'Bet Now: {len(bn)}')
    print(f'You Might Like: {len(ym)}')
    print(f'Goldmine: {len(gold)}')
    if gold:
        print(f'Top Goldmine: {gold[0].get(\"track\")} R{gold[0].get(\"race_number\")} — #{gold[0].get(\"selection_number\")} @ {gold[0].get(\"second_fav_odds\", \"?\")}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null || echo "(no race data)"
```

Save this output — you'll include it in the Phase 4 report.

---

## Phase 2: Analytics Audit

### 2a. Check if audit is possible

```bash
python3 -c "
import sqlite3, os
if not os.path.exists('fortuna.db'):
    print('NO_DB')
    exit(0)
conn = sqlite3.connect('fortuna.db')
cur = conn.cursor()
try:
    cur.execute('SELECT COUNT(*) FROM tips WHERE audit_completed = 0')
    pending = cur.fetchone()[0]
    cur.execute('SELECT COUNT(*) FROM tips')
    total = cur.fetchone()[0]
    print(f'TIPS: {total} total, {pending} pending audit')
    if pending > 0:
        print('AUDIT_NEEDED')
    elif total > 0:
        print('ALL_AUDITED')
    else:
        print('EMPTY_DB')
except Exception as e:
    print(f'DB_ERROR: {e}')
finally:
    conn.close()
"
```

**Decision:**
- `NO_DB` → skip to Phase 3 (discovery didn't create any tips)
- `EMPTY_DB` → skip to Phase 3
- `AUDIT_NEEDED` → run audit (2b)
- `ALL_AUDITED` → run audit anyway (may find new results)
- `DB_ERROR` → report error, skip to Phase 3

### 2b. Run analytics

```bash
timeout 600 python3 fortuna_analytics.py -v --days 2 2>&1 \
  | tee analytics_output.log | tail -20
EXIT_CODE=$?
```

**Decision:**
- Exit code 0 → proceed to validation
- Exit code 124 → report "Analytics timed out"
- Other → report last 30 lines of `analytics_output.log`

### 2c. Validate audit outputs

```bash
python3 -c "
import os
for fname in ['analytics_report.txt', 'results_harvest.json']:
    exists = os.path.exists(fname)
    size = os.path.getsize(fname) if exists else 0
    status = '✅' if exists and size > 10 else '❌'
    print(f'{status} {fname}: {size} bytes')
"
```

### 2d. Capture audit content

```bash
echo "=== ANALYTICS REPORT (first 40 lines) ==="
head -40 analytics_report.txt 2>/dev/null || echo "(no report)"
```

---

## Phase 3: Database Verification

```bash
python3 -c "
import sqlite3, os

if not os.path.exists('fortuna.db'):
    print('No database found — first run, no data yet.')
    exit(0)

conn = sqlite3.connect('fortuna.db')
cur = conn.cursor()

print('=== DATABASE STATUS ===')
print()

# Tip statistics
try:
    cur.execute('''
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN audit_completed = 1 THEN 1 ELSE 0 END) as audited,
            SUM(CASE WHEN verdict LIKE 'CASHED%' THEN 1 ELSE 0 END) as cashed,
            SUM(CASE WHEN verdict = 'BURNED' THEN 1 ELSE 0 END) as burned,
            SUM(CASE WHEN verdict = 'VOID' THEN 1 ELSE 0 END) as voided,
            SUM(CASE WHEN audit_completed = 0 THEN 1 ELSE 0 END) as pending,
            COALESCE(SUM(net_profit), 0.0) as profit
        FROM tips
    ''')
    row = cur.fetchone()
    total, audited, cashed, burned, voided, pending, profit = row
    print(f'Total tips:  {total}')
    print(f'Audited:     {audited}')
    print(f'  Cashed:    {cashed}')
    print(f'  Burned:    {burned}')
    print(f'  Void:      {voided}')
    print(f'Pending:     {pending}')
    decided = (cashed or 0) + (burned or 0)
    if decided > 0:
        wr = (cashed or 0) / decided * 100
        print(f'Win rate:    {wr:.1f}% ({cashed}/{decided})')
    print(f'Net P/L:     \${profit:+.2f}')
except Exception as e:
    print(f'Tips query failed: {e}')

print()

# Recent results
try:
    cur.execute('''
        SELECT venue, race_number, verdict, net_profit
        FROM tips
        WHERE audit_completed = 1
        ORDER BY audit_timestamp DESC
        LIMIT 5
    ''')
    rows = cur.fetchall()
    if rows:
        print('Last 5 audited:')
        for venue, rnum, verdict, profit in rows:
            icon = '✅' if verdict and 'CASHED' in verdict else '❌' if verdict == 'BURNED' else '⚪'
            p = profit or 0.0
            print(f'  {icon} {venue} R{rnum}: {verdict} (\${p:+.2f})')
    else:
        print('No audited results yet.')
except Exception as e:
    print(f'Recent query failed: {e}')

print()

# Adapter harvest log
try:
    cur.execute('''
        SELECT COUNT(*) FROM harvests
    ''')
    harvest_count = cur.fetchone()[0]
    print(f'Harvest log entries: {harvest_count}')
except Exception:
    print('No harvest table found (normal on first run)')

conn.close()
print()
print('=== END DATABASE STATUS ===')
"
```

---

## Phase 4: Report to User

Compile ALL captured output from Phases 1–3 into a single message.

**Use `read_file` to get:**
1. `summary_grid.txt` (first 50 lines)
2. `goldmine_report.txt` (first 40 lines)
3. `analytics_report.txt` (first 40 lines, if exists)

**Format the message as:**

```
## Fortuna Sandbox Verification Report

### Environment
- Python: [version]
- Working directory: [path]
- Time of run: [timestamp]

### Discovery Results
[Phase 1c output]

### Summary Grid (first 50 lines)
[content of summary_grid.txt]

### Goldmine Report (first 40 lines)
[content of goldmine_report.txt]

### Analytics Audit
[Phase 2d output, or "Skipped — no tips to audit"]

### Database Status
[Phase 3 output]

### Issues Encountered
[list any errors from any phase, or "None"]
```

**Call `message_user` with this formatted content.**

**DO NOT:**
- Run `submit`
- Run any `git` commands
- Modify any source files
- Create branches or PRs

---

## Timing Expectations

| Phase | Typical Duration | Timeout |
|---|---|---|
| Phase 0 (setup) | 2–5 min | None (manual abort if stuck) |
| Phase 1 (discovery) | 3–10 min | 10 min |
| Phase 2 (analytics) | 2–8 min | 10 min |
| Phase 3 (DB check) | < 5 sec | None |
| Phase 4 (reporting) | < 30 sec | None |

**Total expected: 8–20 minutes.**

## Off-Hours Behavior

Racing schedules vary by region and time:

| ET Time | Expected Behavior |
|---|---|
| 6am–9am | UK/IRE cards active, some discoveries |
| 9am–6pm | Peak — US + international, most races |
| 6pm–11pm | US evening cards, some international |
| 11pm–6am | Minimal — AUS/NZ only, 0 races is NORMAL |

**0 races discovered is NOT an error outside peak hours.**
Report it as "No active racing at time of run" — not as a failure.

## Failure Decision Tree

```
START
  │
  ├─ Import fails?
  │    → pip install missing module, retry ONCE
  │    → Still fails? Report traceback, STOP
  │
  ├─ Discovery returns 0 races?
  │    → Check time of day (see Off-Hours table)
  │    → Off-hours? Report "normal — no active racing"
  │    → Peak hours? Report "⚠️ 0 races during peak — possible issue"
  │    → Continue to Phase 2 either way
  │
  ├─ Discovery crashes?
  │    → Report last 30 lines of output
  │    → Continue to Phase 2 (may have partial DB data)
  │
  ├─ Analytics returns 0 audited?
  │    → Normal if no prior tips exist
  │    → Report "no tips to audit" (not an error)
  │    → Continue to Phase 3 (DB may have useful state)
  │
  ├─ Analytics crashes?
  │    → Report error
  │    → Continue to Phase 3 (DB may have useful state)
  │
  └─ DB empty?
       → Normal on first run
       → Report "fresh database, no historical data"
```
