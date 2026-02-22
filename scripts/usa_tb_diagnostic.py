#!/usr/bin/env python3
import sqlite3
import os
import sys
import asyncio
import json
import re
import traceback
import urllib.request
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Set, Tuple, Type

# Add current directory to path so we can import fortuna and fortuna_analytics
sys.path.insert(0, '.')
import fortuna
import fortuna_analytics

# ═══════════════════════════════════════════════════════════════════
# SETUP & CONSTANTS
# ═══════════════════════════════════════════════════════════════════
EASTERN = ZoneInfo("America/New_York")
S = os.environ.get("GITHUB_STEP_SUMMARY", "/dev/stdout")
SNAP_DIR = Path("_diag_snapshots")
SNAP_DIR.mkdir(exist_ok=True)
buf = []
_evidence = {}
# Increased deadline buffer to handle setup time
_DEADLINE = datetime.now() + timedelta(minutes=32)
today = datetime.now(EASTERN)

BLOCK_SIGS = (
    "pardon our interruption", "checking your browser",
    "just a moment", "cloudflare", "access denied",
    "captcha", "incapsula", "attention required",
)

# ── Date logic ──────────────────────────────────────────────────────
results_date = os.environ.get("TEST_DATE", "").strip() or (
    today - timedelta(days=1)).strftime("%Y-%m-%d")

usa_date_env = os.environ.get("USA_DATE", "").strip()
if usa_date_env:
    usa_test_date = usa_date_env
else:
    # BUG FIX: Ensure Saturday doesn't jump back 7 days if run on Saturday
    days_since_sat = (today.weekday() - 5) % 7
    last_sat = today - timedelta(days=days_since_sat)
    usa_test_date = last_sat.strftime("%Y-%m-%d")

EQB_BASE = "https://www.equibase.com"
DATE_SHORT = datetime.strptime(usa_test_date, "%Y-%m-%d").strftime("%m%d%y")

# ── Global configuration ────────────────────────────────────────────
ADAPTER_TIMEOUT = 60
MAX_CONCURRENT = 4

# ── Helpers ─────────────────────────────────────────────────────────
def emit(line=""):
    buf.append(line)

def flush():
    with open(S, "a") as f:
        f.write("\n".join(buf) + "\n")
    buf.clear()

def deadline_exceeded():
    return datetime.now() >= _DEADLINE

def remaining_minutes():
    return max(0, (_DEADLINE - datetime.now()).total_seconds() / 60)

def save_snapshot(name, content, url=""):
    safe = re.sub(r'[^\w\-.]', '_', name)[:80]
    try:
        (SNAP_DIR / f"{safe}.html").write_text(content[:120_000], encoding="utf-8")
        (SNAP_DIR / f"{safe}.meta.txt").write_text(
            f"URL: {url}\nLength: {len(content)}\n"
            f"Saved: {datetime.now(EASTERN).isoformat()}\n", encoding="utf-8")
    except Exception:
        pass

def is_blocked(html):
    if not html or len(html) < 200: return True
    if len(html) > 15_000:
        return False
    return any(sig in html.lower() for sig in BLOCK_SIGS)

def count_tables(html):
    return len(re.findall(r'<table[\s>]', html, re.I))

def http_get(url, ranged=True):
    """Uses ranged GET if possible to bypass some HEAD blocks."""
    req = urllib.request.Request(url)
    req.add_header("User-Agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36")
    req.add_header("Accept", "text/html,*/*;q=0.8")
    if "equibase" in url:
        req.add_header("Referer", f"{EQB_BASE}/")

    if ranged:
        req.add_header("Range", "bytes=0-10000")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code == 405 and ranged: # Method not allowed for ranged GET? Try normal.
             return http_get(url, ranged=False)
        return e.code, str(e)
    except Exception as e:
        return 999, str(e)

def cffi_get(url, impersonate="chrome120"):
    try:
        from curl_cffi import requests as cffi_http
        r = cffi_http.get(url, impersonate=impersonate, timeout=20,
            headers={"Accept": "text/html,*/*;q=0.8",
                     "Referer": f"{EQB_BASE}/"})
        return r.status_code, r.text
    except ImportError:
        return None, None
    except Exception as e:
        return 998, str(e)

def get_adapters(atype):
    def _all_subs(cls):
        subs = set(cls.__subclasses__())
        return subs.union(s for c in subs for s in _all_subs(c))

    return [c for c in _all_subs(fortuna.BaseAdapterV3)
            if not getattr(c, "__abstractmethods__", None)
            and getattr(c, "ADAPTER_TYPE", "discovery") == atype
            and hasattr(c, "SOURCE_NAME")]

# ── Pre-compute adapter lists ──────────────────────────────────────
all_results_classes = fortuna_analytics.get_results_adapter_classes()
discovery_classes   = get_adapters("discovery")
usa_adapter_names   = set(getattr(fortuna, "USA_RESULTS_ADAPTERS", []))
usa_results_classes = [c for c in all_results_classes
                        if c.SOURCE_NAME in usa_adapter_names]

# ── DB setup ────────────────────────────────────────────────────────
db_path = "fortuna.db"
if not os.path.exists(db_path):
    sqlite3.connect(db_path).close()
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

# ═══════════════════════════════════════════════════════════════════
# SECTIONS
# ═══════════════════════════════════════════════════════════════════

def q1_db_reality():
    emit("## Q1: Database Reality Check\n")
    has_tips = bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tips'"
    ).fetchone())

    if not has_tips:
        emit("❌ `tips` table does not exist.\n")
        _evidence['db_has_tips'] = False
        return

    _evidence['db_has_tips'] = True

    # Discipline × audit breakdown
    emit("### Discipline breakdown (all time)")
    emit("| Discipline | Total | Audited | Stuck | Hit Rate |")
    emit("|------------|------:|--------:|------:|---------:|")
    for row in conn.execute("""
        SELECT COALESCE(discipline,'?') as disc,
               COUNT(*) as total,
               SUM(CASE WHEN audit_completed=1 THEN 1 ELSE 0 END) as ok,
               SUM(CASE WHEN audit_completed=0 THEN 1 ELSE 0 END) as stuck,
               ROUND(CAST(SUM(CASE WHEN verdict IN ('CASHED','CASHED_ESTIMATED') THEN 1 ELSE 0 END) AS REAL) /
                     NULLIF(SUM(CASE WHEN verdict IN ('CASHED','CASHED_ESTIMATED','BURNED') THEN 1 ELSE 0 END), 0) * 100, 1) as hit
        FROM tips GROUP BY discipline ORDER BY total DESC
    """):
        d = {'H':'Harness','T':'Thoroughbred','G':'Greyhound'}.get(
            row['disc'], row['disc'])
        emit(f"| {d} | {row['total']} | {row['ok']} | "
             f"{row['stuck']} | {row['hit'] or 0:.1f}% |")
    emit("")

    # USA T per-venue
    emit("### USA Thoroughbred venues")
    emit("```")
    usa_t_total = usa_t_stuck = 0
    for row in conn.execute("""
        SELECT venue, COUNT(*) as n,
               SUM(CASE WHEN audit_completed=1 THEN 1 ELSE 0 END) as ok,
               SUM(CASE WHEN audit_completed=0 THEN 1 ELSE 0 END) as stuck,
               MIN(start_time) as first_seen,
               MAX(start_time) as last_seen
        FROM tips WHERE discipline='T'
        GROUP BY venue ORDER BY n DESC
    """):
        usa_t_total += row['n']
        usa_t_stuck += row['stuck']
        flag = "❌" if row['stuck'] > 0 and row['ok'] == 0 else (
               "⚠️" if row['stuck'] > row['ok'] else "✅")
        emit(f"  {flag} {row['venue']:28s}  total={row['n']:3d}  "
             f"ok={row['ok']:3d}  stuck={row['stuck']:3d}  "
             f"{(row['first_seen'] or '')[:10]}→"
             f"{(row['last_seen'] or '')[:10]}")
    emit("```")
    _evidence['usa_t_total'] = usa_t_total
    _evidence['usa_t_stuck'] = usa_t_stuck

    # Last 7 days trend
    emit("\n### Last 7 days: new T tips per day")
    emit("```")
    week_counts = []
    for row in conn.execute("""
        SELECT DATE(start_time) as d, COUNT(*) as n
        FROM tips WHERE discipline='T'
          AND start_time >= DATE('now','-7 days')
        GROUP BY d ORDER BY d
    """):
        week_counts.append(row['n'])
        emit(f"  {row['d']}  {row['n']:3d} tips")
    if not week_counts:
        emit("  (none)")
    emit("```")
    _evidence['usa_t_last_7d'] = sum(week_counts)

    # Time-of-day histogram for stuck T tips
    emit("\n### 🇺🇸 Stuck T tips — time-of-day distribution (Eastern)")
    emit("```")
    usa_t_stuck_rows = conn.execute("""
        SELECT venue, start_time, race_id, race_number
        FROM tips
        WHERE audit_completed=0 AND discipline='T'
        ORDER BY start_time
    """).fetchall()
    hour_counts = {}
    stuck_prefixes = set()
    for row in usa_t_stuck_rows:
        prefix = row['race_id'].split('_')[0] if '_' in row['race_id'] else row['race_id']
        stuck_prefixes.add(prefix)
        try:
            st = datetime.fromisoformat(
                str(row['start_time']).replace('Z', '+00:00'))
            st_et = st.astimezone(EASTERN)
            h = st_et.hour
            hour_counts[h] = hour_counts.get(h, 0) + 1
            emit(f"  {row['venue']:28s} "
                 f"{st_et.strftime('%Y-%m-%d %H:%M ET'):<18s}  "
                 f"{row['race_id']}")
        except Exception:
            emit(f"  {row['venue']:28s} "
                 f"{str(row['start_time'])[:16]:<18s}  "
                 f"{row['race_id']}")
    if not usa_t_stuck_rows:
        emit("  (no stuck Thoroughbred tips)")
    emit("```")
    _evidence['stuck_prefixes'] = stuck_prefixes

    if hour_counts:
        emit("\n**Hour buckets (Eastern):**")
        for h in sorted(hour_counts):
            bar = "█" * hour_counts[h]
            emit(f"  {h:02d}:xx ET  {bar} ({hour_counts[h]})")
    emit("")

def q2_adapters():
    emit("## Q2: Adapter Inventory\n")
    emit(f"**Discovery adapters:** {len(discovery_classes)} · "
         f"**Results adapters:** {len(all_results_classes)}\n")

    solid = set(getattr(fortuna, 'SOLID_RESULTS_ADAPTERS', []))
    for attr in ['SOLID_RESULTS_ADAPTERS', 'USA_RESULTS_ADAPTERS',
                 'INT_RESULTS_ADAPTERS']:
        val = getattr(fortuna, attr, None)
        if val is not None:
            emit(f"- `fortuna.{attr}` = `{val}`")
    emit("")

    emit("### Results adapters")
    emit("```")
    for cls in all_results_classes:
        name = cls.SOURCE_NAME
        q = "SOLID" if name in solid else "LOUSY"
        usa = " ◀ USA" if name in usa_adapter_names else ""
        base = getattr(cls, 'BASE_URL', '?')
        engine = '?'
        try:
            engine = str(cls()._configure_fetch_strategy(
                ).primary_engine).split('.')[-1]
        except Exception:
            pass
        emit(f"  {name:40s} [{q:5s}]{usa}")
        emit(f"    {'':40s}  {base}  ({engine})")
    emit("```")

    emit("\n### Discovery adapters")
    emit("```")
    usa_disc_count = 0
    for cls in discovery_classes:
        name = cls.SOURCE_NAME
        base = getattr(cls, 'BASE_URL', '?')
        is_usa = any(k in name.lower() or k in base.lower()
                    for k in ("equibase", "usa", "twinspires", "nyra"))
        if is_usa:
            usa_disc_count += 1
        tag = " ◀ USA" if is_usa else ""
        emit(f"  {name:40s}{tag}  {base}")
    emit("```\n")
    _evidence['usa_discovery_adapters'] = usa_disc_count
    _evidence['usa_results_adapters'] = len(usa_results_classes)

def q3_network():
    emit("## Q3: Network Reachability\n")
    domains = sorted(set(
        getattr(cls, 'BASE_URL', '') for cls in all_results_classes
    ) | {
        "https://greyhounds.attheraces.com",
        "https://www.twinspires.com",
        "https://www.equibase.com",
        "https://www.nyrabets.com",
        "https://brk0201-iapi-webservice.nyrabets.com",
    })
    emit("```")
    for url in domains:
        if not url:
            continue
        # IMPROVEMENT 10: Use Ranged GET instead of HEAD
        status, info = http_get(url)
        if status < 400:
            emit(f"  ✅ {url:55s} → {status}")
        else:
            emit(f"  ❌ {url:55s} → {status} ({info[:30]})")
    emit("```\n")

def q4_equibase():
    emit("## Q4: Equibase Surgical Probe\n")
    eqb_urls = {
        "entry_index":     f"{EQB_BASE}/static/entry/index.html?SAP=TN",
        "entry_dated":     f"{EQB_BASE}/static/entry/{DATE_SHORT}USA-TB.html",
        "results_index":   f"{EQB_BASE}/static/chart/summary/index.html?SAP=TN",
        "results_dated":   f"{EQB_BASE}/static/chart/summary/{DATE_SHORT}sum.html",
    }

    emit("### 4a: urllib (baseline)")
    emit("| Key | Status | Length | Blocked? | Tables | Date Links |")
    emit("|-----|-------:|-------:|:--------:|-------:|-----------:|")
    eqb_urllib_ok = False
    for key, url in eqb_urls.items():
        status, html = http_get(url, ranged=False)
        blocked = is_blocked(html)
        tables = count_tables(html)
        dlinks = len(re.findall(
            rf'href="[^"]*{DATE_SHORT}[^"]*\.html"', html, re.I))
        save_snapshot(f"q4_urllib_{key}", html, url)
        flag = "🔴 YES" if blocked else "🟢 no"
        emit(f"| `{key}` | {status} | {len(html):,} | "
             f"{flag} | {tables} | {dlinks} |")
        if not blocked and tables > 0:
            eqb_urllib_ok = True
    emit("")
    _evidence['eqb_urllib_reachable'] = eqb_urllib_ok

    emit("### 4b: curl_cffi (adapter-realistic)")
    emit("| Key | Impersonate | Status | Length | Blocked? | Tables |")
    emit("|-----|-------------|-------:|-------:|:--------:|-------:|")
    eqb_cffi_ok = False
    cffi_targets = [
        ("entry_index",   eqb_urls["entry_index"]),
        ("results_index", eqb_urls["results_index"]),
    ]
    for label, url in cffi_targets:
        for imp in ("chrome120", "chrome110"):
            status, html = cffi_get(url, imp)
            if status is None:
                emit(f"| `{label}` | {imp} | — | — | — | curl_cffi N/A |")
                continue
            blocked = is_blocked(html)
            tables = count_tables(html)
            save_snapshot(f"q4_cffi_{label}_{imp}", html, url)
            flag = "🔴 YES" if blocked else "🟢 no"
            emit(f"| `{label}` | {imp} | {status} | "
                 f"{len(html):,} | {flag} | {tables} |")
            if not blocked and tables > 0:
                eqb_cffi_ok = True
    emit("")
    _evidence['eqb_cffi_reachable'] = eqb_cffi_ok

def q5_racing_post():
    emit("## Q5: Racing Post Index — Slug Audit\n")
    _evidence['rp_slug_gaps'] = []
    _evidence['rp_usa_links_count'] = 0
    _evidence['rp_all_links_count'] = 0

    try:
        from curl_cffi import requests as cffi_req
        rp_url = f"https://www.racingpost.com/results/{usa_test_date}"
        rp_resp = cffi_req.get(rp_url, impersonate="chrome120",
            timeout=35, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120",
                "Accept": "text/html,*/*;q=0.8",
                "Referer": "https://www.racingpost.com/",
            })
        emit(f"**HTTP {rp_resp.status_code}** — {len(rp_resp.text):,} chars\n")
        save_snapshot("q5_rp_index", rp_resp.text, rp_url)

        if rp_resp.status_code == 200 and len(rp_resp.text) > 5000:
            from selectolax.parser import HTMLParser as _SL
            _hrefs = [a.attributes.get("href", "")
                      for a in _SL(rp_resp.text).css('a[href*="/results/"]')]

            _rp_base = "https://www.racingpost.com"
            race_links = sorted(set(
                (h if h.startswith("http") else _rp_base + h)
                for h in _hrefs
                if usa_test_date in h and len(h.split("/")) >= 4
            ))
            _evidence['rp_all_links_count'] = len(race_links)
            emit(f"**Total dated links:** {len(race_links)}\n")

            def _slug(u):
                try: return u.split("/results/")[1].split("/")[0]
                except Exception: return ""

            slugs_on_page = sorted(set(_slug(u) for u in race_links if _slug(u)))

            try:
                _rp_usa_cls = next(c for c in all_results_classes if c.SOURCE_NAME == "RacingPostUSAResults")
                _known = _rp_usa_cls._USA_TRACK_SLUGS
                _known_bare = {s.replace("-", ""): s for s in _known}

                usa_links = [u for u in race_links if _rp_usa_cls._is_usa_link(u)]
                _evidence['rp_usa_links_count'] = len(usa_links)
                emit(f"**Links passing `_is_usa_link()`:** {len(usa_links)}\n")

                gaps = [s for s in slugs_on_page if s and s not in _known and s.replace("-", "") not in _known_bare]
                _evidence['rp_slug_gaps'] = gaps

                if gaps:
                    emit(f"### ⚠️ {len(gaps)} slugs NOT in `_USA_TRACK_SLUGS`")
                    emit("```python\n# Add to RacingPostUSAResultsAdapter._USA_TRACK_SLUGS:")
                    for g in gaps: emit(f'    "{g}",')
                    emit("```\n")
            except StopIteration:
                emit("⚠️ `RacingPostUSAResultsAdapter` not found.\n")

    except Exception as e:
        emit(f"❌ RP fetch failed: {e}")

def q6_stuck_tips():
    emit("## Q6: Slug Coverage & Key Anatomy — Stuck Tips\n")
    _evidence['venue_slug_misses'] = []

    if not _evidence.get('db_has_tips'):
        emit("ℹ️ No tips table — skipping.\n")
        return

    # Key anatomy
    emit("### Canonical keys for stuck T tips (first 20)")
    emit("```")
    emit(f"  {'race_id':<45s}  {'tip_key':<42s}  {'canonical'}")
    emit(f"  {'─'*45}  {'─'*42}  {'─'*22}")
    for row in conn.execute("""
        SELECT race_id, venue, race_number, start_time, discipline
        FROM tips WHERE audit_completed=0 AND discipline='T'
        ORDER BY start_time LIMIT 20
    """):
        tip = dict(row)
        key = fortuna_analytics.AuditorEngine._tip_canonical_key(tip)
        canon = fortuna.get_canonical_venue(tip['venue'])
        emit(f"  {tip['race_id']:<45s}  {str(key):<42s}  {canon}")
    emit("```\n")

    # Slug coverage audit
    try:
        _rp_usa_cls = next(c for c in all_results_classes if c.SOURCE_NAME == "RacingPostUSAResults")
        _slugs = _rp_usa_cls._USA_TRACK_SLUGS
        _slug_bare = {s.replace("-", ""): s for s in _slugs}

        stuck_venues = [row[0] for row in conn.execute(
            "SELECT DISTINCT venue FROM tips WHERE audit_completed=0 AND discipline='T' ORDER BY venue"
        ).fetchall()]

        emit("### Stuck venue → slug match")
        emit("```")
        emit(f"  {'VENUE':<30s} {'CANONICAL':<22s} {'VERDICT':<10s}  NOTE")
        emit(f"  {'─'*30} {'─'*22} {'─'*10}  {'─'*30}")
        misses = []
        for venue in stuck_venues:
            canon = fortuna.get_canonical_venue(venue)
            direct = next((s for s in _slugs if s.replace("-", "") in canon), None)
            reverse = next((orig for bare, orig in _slug_bare.items() if canon in bare), None)
            match = direct or reverse
            if match:
                emit(f"  {venue:<30s} {canon:<22s} {'✅ MATCH':<10s}  slug: '{match}'")
            else:
                suggested = re.sub(r"\s+", "-", venue.lower().strip())
                emit(f"  {venue:<30s} {canon:<22s} {'❌ MISS':<10s}  try: \"{suggested}\"")
                misses.append(venue)
        emit("```")
        _evidence['venue_slug_misses'] = misses
    except StopIteration:
        emit("⚠️ `RacingPostUSAResultsAdapter` not found.\n")

async def _test_adapter(cls, date):
    name = cls.SOURCE_NAME
    lines = []
    races = None # BUG FIX 5: Initialize races
    adapter = None
    try:
        adapter = cls()
        t0 = datetime.now()
        races = await adapter.get_races(date)
        elapsed = (datetime.now() - t0).total_seconds()

        if not races:
            lines.append(f"- ⚠️ `{name}` → **0 races** in {elapsed:.1f}s")
        else:
            venues = sorted(set(r.venue for r in races))
            lines.append(f"- ✅ `{name}` → **{len(races)} races** in {elapsed:.1f}s")
            lines.append(f"  Venues: {', '.join(venues[:8])}")
            for r in races[:3]:
                lines.append(f"  Key: `{r.canonical_key}` runners: {len(r.runners)}")
    except asyncio.TimeoutError:
        lines.append(f"- ⏱️ `{name}` → TIMED OUT")
    except Exception as e:
        lines.append(f"- ❌ `{name}` → **{type(e).__name__}:** `{str(e)[:100]}`")
        lines.append(f"  ```\n  {traceback.format_exc()[:400]}\n  ```")
    finally:
        # BUG FIX 4: Ensure adapter is closed
        if adapter:
            try: await adapter.close()
            except Exception: pass
    return name, lines, (len(races) if races else 0)

async def _run_sweep(classes, date, label):
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    total_found = 0
    tasks = []

    async def bounded(cls):
        async with sem:
            try:
                return await asyncio.wait_for(_test_adapter(cls, date), timeout=ADAPTER_TIMEOUT + 5)
            except Exception as e:
                return cls.SOURCE_NAME, [f"- ❌ `{cls.SOURCE_NAME}` crashed: {e}"], 0

    tasks = [asyncio.create_task(bounded(c)) for c in classes]
    for coro in asyncio.as_completed(tasks):
        try:
            name, lines, count = await coro
            total_found += count
        except Exception as e:
            lines = [f"- ❌ (unknown) → {e}"]
        for line in lines: emit(line)
        emit("")
        flush()

    try: await fortuna.GlobalResourceManager.cleanup()
    except Exception: pass
    return total_found

def q7_global_results():
    emit(f"## Q7: Full Results Adapter Sweep ({results_date})\n")
    emit("Control group — proves other regions work.\n")
    flush()
    # IMPROVEMENT 9: Capture Q7 return value
    _evidence['res_all_found'] = asyncio.run(_run_sweep(all_results_classes, results_date, "all"))

def q8_usa_results():
    emit(f"## Q8: USA Results Adapters — Saturday ({usa_test_date})\n")
    emit(f"Testing **{len(usa_results_classes)}** adapters: `{[c.SOURCE_NAME for c in usa_results_classes]}`\n")
    flush()
    _evidence['res_usa_found'] = asyncio.run(_run_sweep(usa_results_classes, usa_test_date, "usa"))

async def _test_discovery():
    sem = asyncio.Semaphore(3)
    tb_total = 0

    async def probe(cls):
        name = cls.SOURCE_NAME
        adapter = None
        try:
            adapter = cls()
            races = await asyncio.wait_for(adapter.get_races(results_date), timeout=ADAPTER_TIMEOUT)
            tb = [r for r in (races or []) if (getattr(r, 'discipline', '') or '').upper().startswith('T')]
            vs = sorted(set(r.venue for r in tb)) if tb else []
            return name, len(races or []), len(tb), vs
        except asyncio.TimeoutError:
            return name, -1, 0, []
        except Exception as e:
            return name, -2, 0, [str(e)[:60]]
        finally:
            # BUG FIX 4: Close discovery adapter
            if adapter:
                try: await adapter.close()
                except Exception: pass

    async def bounded(cls):
        async with sem: return await probe(cls)

    tasks = [asyncio.create_task(bounded(c)) for c in discovery_classes]
    for coro in asyncio.as_completed(tasks):
        name, total, tb, info = await coro
        tb_total += max(tb, 0)
        if total == -1: emit(f"- ⏱️ `{name}`: TIMED OUT")
        elif total == -2: emit(f"- ❌ `{name}`: {info[0] if info else '?'}")
        elif tb == 0: emit(f"- ⚠️ `{name}`: {total} total, **0 thoroughbred**")
        else: emit(f"- ✅ `{name}`: {total} total, **{tb} thoroughbred** — {', '.join(info[:6])}")
        flush()

    try: await fortuna.GlobalResourceManager.cleanup()
    except Exception: pass
    return tb_total

def q9_discovery():
    emit("## Q9: Discovery Pipeline Test\n")
    if not discovery_classes:
        emit("❌ No discovery adapters registered.\n")
        _evidence['disc_tb_found'] = 0
    else:
        _evidence['disc_tb_found'] = asyncio.run(_test_discovery())
    emit("")

def q10_playwright():
    emit("## Q10: Playwright Browser Ping\n")
    async def _ping():
        from playwright.async_api import async_playwright
        urls = [
            f"https://www.racingpost.com/results/{usa_test_date}",
            f"{EQB_BASE}/static/chart/summary/index.html",
            "https://www.nyrabets.com/results",
        ]
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
            ctx = await browser.new_context(user_agent="Mozilla/5.0 Chrome/125.0.0.0")
            for url in urls:
                try:
                    page = await ctx.new_page()
                    t0 = datetime.now()
                    resp = await page.goto(url, wait_until="domcontentloaded", timeout=12000)
                    el = (datetime.now() - t0).total_seconds()
                    html = await page.content()
                    title = await page.title()
                    status = resp.status if resp else "?"
                    blocked = any(sig in html.lower() or sig in title.lower() for sig in BLOCK_SIGS)
                    save_snapshot(f"q10_pw_{url.split('//')[1][:20]}", html, url)
                    if blocked: emit(f"- ❌ `{url}` → **BOT BLOCKED** ({title[:30]}) {status}")
                    else: emit(f"- ✅ `{url}` → {title[:30]}… | {status} ({len(html):,}c) {el:.1f}s")
                    await page.close()
                except Exception as e: emit(f"- ❌ `{url}` → `{str(e)[:60]}`")
                flush()
            await browser.close()

    try: asyncio.run(_ping())
    except Exception as e: emit(f"⚠️ Playwright failed: {e}")
    emit("")

def q11_harvest():
    emit("## Q11: Harvest History\n")
    # Log files
    for fname in ('results_harvest.json', 'discovery_harvest.json'):
        p = Path(fname)
        if p.exists():
            try:
                data = json.loads(p.read_text())
                usa = {k: v for k, v in data.items() if any(s in k.lower() for s in ('equibase','usa','nyra'))}
                if usa: emit(f"### `{fname}` — USA entries\n```json\n{json.dumps(usa, indent=2)}\n```\n")
            except Exception: pass

    # DB Logs
    has_harvest = bool(conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='harvest_logs'").fetchone())
    if has_harvest:
        emit("### Last 20 harvest entries")
        emit("```")
        for row in conn.execute("SELECT timestamp, region, adapter_name, race_count FROM harvest_logs ORDER BY id DESC LIMIT 20"):
            rc = row['race_count'] or 0
            emit(f"  {'✅' if rc>0 else '❌'} {str(row['timestamp'])[:19]}  {row['region'] or '?':6s}  {row['adapter_name']:35s} races={rc:3d}")
        emit("```\n")

        emit("### 30-day USA harvest success rates")
        emit("```")
        # BUG FIX 3: Don't overwrite rate, use a dict
        _evidence['harvest_success_rates'] = {}
        for row in conn.execute("""
            SELECT adapter_name, COUNT(*) as attempts, SUM(CASE WHEN race_count>0 THEN 1 ELSE 0 END) as ok
            FROM harvest_logs WHERE (adapter_name LIKE '%Equibase%' OR adapter_name LIKE '%USA%' OR adapter_name LIKE '%NYRA%' OR region='USA')
            AND timestamp >= DATE('now','-30 days') GROUP BY adapter_name ORDER BY attempts DESC
        """):
            rate = (row['ok']/row['attempts']*100) if row['attempts'] else 0
            emit(f"  {'✅' if rate>50 else '❌'} {row['adapter_name']:35s}  {row['ok']}/{row['attempts']} ({rate:.0f}%)")
            _evidence['harvest_success_rates'][row['adapter_name']] = rate
        emit("```\n")

def q12_data_quality():
    emit("## Q12: Scoring Signal Data Quality\n")
    if not _evidence.get('db_has_tips'):
        emit("ℹ️ No tips table — skipping.\n")
        return

    scoring_cols = [
        'gap12', 'market_depth', 'place_prob',
        'predicted_ev', 'race_type', 'condition_modifier',
        'qualification_grade', 'composite_score',
        'is_goldmine', 'is_best_bet',
        'is_superfecta_key',
    ]

    emit("| Column | Population | Signal | Status |")
    emit("|--------|-----------:|-------:|:------:|")

    quality_issues = []
    for col in scoring_cols:
        stats = conn.execute(f"""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN {col} IS NOT NULL THEN 1 ELSE 0 END) as filled,
                   SUM(CASE WHEN {col} IS NOT NULL AND {col} != 0 AND {col} != '0.0' AND {col} != 'D' THEN 1 ELSE 0 END) as signal
            FROM tips
            WHERE start_time >= DATE('now', '-7 days')
        """).fetchone()

        total = stats['total']
        filled = stats['filled']
        pct = (filled / total * 100) if total else 0

        sig_pct = (stats['signal'] / total * 100) if total else 0

        icon = "✅" if pct > 90 else ("🟡" if pct > 50 else "🔴")
        emit(f"| `{col}` | {filled}/{total} ({pct:.1f}%) | {stats['signal']} ({sig_pct:.1f}%) | {icon} |")

        if pct < 20 and total > 5:
            quality_issues.append(f"Scoring signal `{col}` is mostly NULL ({pct:.1f}%)")

        if sig_pct < 0.1 and total > 10 and col in ('gap12', 'market_depth', 'place_prob'):
            quality_issues.append(f"Scoring signal `{col}` has NO active variance (all defaults)")

    emit("")
    _evidence['quality_issues'] = quality_issues

def q13_diagnosis():
    emit("## Q13: 🎯 Automated Root-Cause Diagnosis\n")
    diagnosis = []
    severity = "INFO"

    disc_tb = _evidence.get('disc_tb_found', 0)
    t_7d = _evidence.get('usa_t_last_7d', 0)

    if disc_tb == 0 and t_7d == 0:
        severity = "CRITICAL"
        diagnosis.append("🔴 **CRITICAL: Zero USA T races from discovery AND zero new tips in 7 days.**")
    elif disc_tb > 0:
        diagnosis.append(f"🟢 Discovery found **{disc_tb}** T race(s).")

    res_usa = _evidence.get('res_usa_found', 0)
    if res_usa == 0:
        if severity != "CRITICAL": severity = "HIGH"
        diagnosis.append("🟠 **No USA results adapter returned data** on the Saturday test date.")
    else:
        diagnosis.append(f"🟢 USA results adapters returned **{res_usa}** race(s) on Saturday.")

    # Slug gaps
    gaps = _evidence.get('rp_slug_gaps', [])
    if gaps:
        if severity == "INFO": severity = "MEDIUM"
        diagnosis.append(f"🟡 **{len(gaps)} RP slug(s)** on page but NOT in `_USA_TRACK_SLUGS`.")

    # Quality issues
    quality = _evidence.get('quality_issues', [])
    if quality:
        if severity in ("INFO", "MEDIUM"): severity = "HIGH"
        for q in quality:
            diagnosis.append(f"🔴 **DATA QUALITY: {q}**")

    emit(f"### Severity: **{severity}**\n")
    for line in diagnosis: emit(line)
    emit("\n---")

# ═══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════
emit("# 🏇 Fortuna Diagnostic v4 — USA Thoroughbred Pipeline\n")
emit(f"> General date: **{results_date}** · USA date: **{usa_test_date}** · Generated: {today.strftime('%Y-%m-%d %H:%M ET')}\n")
flush()

for section in [q1_db_reality, q2_adapters, q3_network, q4_equibase, q5_racing_post,
                q6_stuck_tips, q7_global_results, q8_usa_results, q9_discovery,
                q10_playwright, q11_harvest, q12_data_quality, q13_diagnosis]:
    if deadline_exceeded():
        emit(f"\n⚠️ Deadline exceeded before `{section.__name__}`.")
        break
    try:
        section()
    except Exception as e:
        emit(f"❌ {section.__name__} crashed: {e}")
        emit(f"```\n{traceback.format_exc()[:500]}\n```")
    flush()

conn.close()
emit(f"\n*Diagnostic v4 complete — {remaining_minutes():.1f} min remaining.*")
flush()
