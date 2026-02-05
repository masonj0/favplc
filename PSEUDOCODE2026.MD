# 🐎 Fortuna Faucet - Complete Pseudocode Blueprint

**Status:** Unified Monolith Architecture
**Version:** 3.0.0
**Last Updated:** January 13, 2026

---

## TABLE OF CONTENTS

1.  System Overview
2.  Architecture: The Unified Monolith
3.  Backend Engine (Python/FastAPI)
4.  Frontend Interface (TypeScript/Next.js)
5.  Data Models & API Specification
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
  • Serve a static frontend and a REST API from a single, self-contained executable.
  • Provide cross-platform access via launcher scripts for a Podman container.

CORE TENETS:
  • Single Origin: The backend serves the frontend, eliminating all CORS and cross-origin security issues.
  • Zero Dependencies: The primary Windows artifact is a single .exe file that requires no external installation or setup.
  • Containerization: For non-Windows users, a Podman container provides a consistent, isolated environment.
  • Automated & Repeatable Builds: The entire application is built, tested, and packaged via a deterministic CI/CD pipeline (`build-monolith.yml`).
```

---

## 2. ARCHITECTURE: THE UNIFIED MONOLITH

The application is a **Unified Monolith**. A single Python executable, built with PyInstaller, contains the complete application. It runs a FastAPI web server that both serves the static Next.js frontend and provides the backend REST API.

```
+------------------------------------------------------+
| fortuna-monolith.exe (Single Executable)             |
|                                                      |
|  +-------------------------------------------------+ |
|  | Python Environment                              | |
|  |                                                 | |
|  |  +-------------------------------------------+  | |
|  |  | FastAPI / Uvicorn Server (localhost:8000) |  | |
|  |  |                                           |  | |
|  |  |  +------------------+  +----------------+ |  | |
|  |  |  | API Router       |  | Static Files   | |  | |
|  |  |  | (/api/*)         |  | (/, /_next/*)  | |  | |
|  |  |  +------------------+  +----------------+ |  | |
|  |  |           |                     |         |  | |
|  |  +-----------|---------------------|---------+  | |
|  |             |                     |            | |
|  |  +----------v-----------+  +-------v--------+   | |
|  |  | OddsEngine (engine.py)|  | Bundled Next.js|   | |
|  |  +----------------------+  | 'out' directory|   | |
|  |                            +----------------+   | |
|  +-------------------------------------------------+ |
|                                                      |
+------------------------------------------------------+
```

---

## 3. BACKEND ENGINE (PYTHON/FASTAPI)

### 3.1 Entry Point & Server Startup (`web_service/backend/main.py`)

The entry point directly runs the Uvicorn server, loading the FastAPI app instance from `api.py`.

```pseudocode
// main.py
PROCEDURE Main_Python_Entry_Point
  // Path modifications for PyInstaller compatibility
  IF running in a frozen (PyInstaller) environment:
    base_path <- sys._MEIPASS
  ELSE:
    base_path <- path to project root
  ADD base_path to sys.path

  // Programmatically launch the FastAPI application
  CALL uvicorn.run(
    app="web_service.backend.api:app",
    host="0.0.0.0",
    port=8000
  )
END PROCEDURE
```

### 3.2 Application & Frontend Serving (`web_service/backend/api.py`)

The `api.py` file is the core of the backend. It defines the FastAPI app, manages the application lifecycle, and includes the critical logic for serving the static frontend.

```pseudocode
// api.py

// --- Lifespan Management ---
ASYNC FUNCTION lifespan_manager(app: FastAPI):
  // ON STARTUP:
  CONFIGURE logging
  CREATE OddsEngine instance
  STORE engine in app.state
  YIELD
  // ON SHUTDOWN:
  AWAIT app.state.engine.close() // Gracefully close connections

// --- FastAPI App Initialization ---
app <- CREATE FastAPI(lifespan=lifespan_manager)
ADD CORS middleware
ADD Rate Limiting middleware
INCLUDE API router (for /api/* routes)

// --- CRITICAL: Unified Frontend Serving ---
FUNCTION get_ui_directory():
  IF running in a frozen (PyInstaller) environment:
    RETURN path to bundled 'ui' directory (sys._MEIPASS/ui)
  ELSE:
    RETURN path to `web_platform/frontend/out`
END FUNCTION

UI_DIR <- get_ui_directory()
VERIFY UI_DIR and index.html exist, otherwise RAISE RuntimeError

// --- SPA Middleware ---
// All requests that are not for '/api/*' or a known static file type
// will be served the `index.html` file. This allows the Next.js
// client-side router to handle all frontend navigation.
app.add_middleware(SPAMiddleware)

// --- Static File Mount ---
// This serves the actual .js, .css, and image files from the
// bundled 'ui' directory.
app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
```

### 3.3 Resilient Fetching Strategy (`web_service/backend/engine.py`)

The `OddsEngine` is designed for resilience, using a multi-tiered fallback strategy to ensure data is returned even when some sources fail.

```pseudocode
// engine.py

CLASS OddsEngine:
  INIT():
    self.adapters: Dict[str, Adapter]
    self.health_monitor: AdapterHealthMonitor
    self.stale_data_cache: StaleDataCache
    // ... other initializations

  ASYNC FUNCTION fetch_all_odds(date):
    // Tier 1: Attempt to fetch from healthy adapters
    healthy_adapters <- self.health_monitor.get_healthy_adapters()
    live_results <- FETCH_IN_PARALLEL(healthy_adapters)

    IF count(live_results) >= MINIMUM_REQUIRED_SOURCES:
      MERGE and RETURN live_results

    // Tier 2: Augment with degraded adapters if necessary
    degraded_adapters <- self.health_monitor.get_degraded_adapters()
    degraded_results <- FETCH_IN_PARALLEL(degraded_adapters)
    live_results.extend(degraded_results)

    IF count(live_results) > 0:
      MERGE, CACHE, and RETURN live_results

    // Tier 3: Fall back to stale data from the last successful run
    stale_data <- self.stale_data_cache.get(date)
    IF stale_data IS NOT NULL:
      LOG "Using stale data as a last resort."
      RETURN stale_data with a 'stale' freshness flag

    // Final fallback: Return an error response
    RETURN error_response("All live adapters failed and no stale cache was available.")

```
---

## 4. FRONTEND INTERFACE (TYPESCRIPT/NEXT.JS)

### 4.1 Configuration (`next.config.mjs`)

The frontend is a standard Next.js application configured for static export. This means it is pre-built into a set of HTML, CSS, and JS files that can be served by any static web server.

```javascript
// next.config.mjs
const nextConfig = {
  output: 'export',   // CRITICAL: Generates a static site in the 'out' directory
  distDir: 'out',
  images: { unoptimized: true }, // Required for static export
  trailingSlash: true,
};
```

### 4.2 Main Dashboard Component (`src/components/LiveRaceDashboard.tsx`)

With the unified architecture, the frontend no longer needs a complex IPC mechanism. It behaves like a standard web application, making HTTP requests to the same origin that served it.

```pseudocode
// LiveRaceDashboard.tsx (Simplified)
COMPONENT LiveRaceDashboard:
  STATE:
    races: Race[] <- []
    status: 'loading' | 'success' | 'error' <- 'loading'
    errorMessage: string <- ""

  EFFECT on mount:
    fetchQualifiedRaces() // Fetch data immediately on component load

  ASYNC FUNCTION fetchQualifiedRaces():
    TRY:
      // Make a standard, same-origin API call. No full URL needed.
      response <- AWAIT fetch("/api/races/qualified/trifecta")
      IF NOT response.ok:
        RAISE new Error(`API Error: ${response.statusText}`)

      data <- AWAIT response.json()
      setRaces(data.races)
      setStatus('success')
    CATCH e:
      setStatus('error')
      setErrorMessage(e.message)
END COMPONENT
```

---

## 5. DATA MODELS & API SPECIFICATION

The data models and API endpoints remain largely the same, with the key difference being that they are all served from the `localhost:8000` origin.

```
ENDPOINT GET /api/health
  Response (200 OK): {"status":"ok"}

ENDPOINT GET /api/races/qualified/trifecta
  Response (200 OK):
    {
      "qualified_races": List[Race],
      "analysis_metadata": { ... }
    }
```

---

## 6. DEPLOYMENT & AUTOMATION (CI/CD)

The primary build workflow is `.github/workflows/build-monolith.yml`. It creates the single Windows executable. The `.github/workflows/build-podman.yml` workflow provides a container-based alternative for cross-platform use.

### 6.1 Monolith Build (`build-monolith.yml`)

```pseudocode
// build-monolith.yml
WORKFLOW Build_Fortuna_Monolith_EXE:
  // Phase 1: Build Frontend
  SETUP Node.js
  RUN "npm ci" and "npm run build" in /web_platform/frontend
  COPY the 'out' directory to a staging area (`frontend_dist`)

  // Phase 2: Build Backend Executable
  SETUP Python
  INSTALL Python dependencies from requirements.txt
  INSTALL PyInstaller
  CREATE required data/log directories

  // Phase 3: Package with PyInstaller
  // The spec file (`fortuna-monolith.spec`) is configured to:
  // 1. Identify `web_service/backend/main.py` as the entry point.
  // 2. Bundle the `frontend_dist` directory into the .exe at the root 'ui'.
  // 3. Add application icon and version info.
  EXECUTE PyInstaller using `fortuna-monolith.spec`

  // Phase 4: Smoke Test
  START the generated `fortuna-monolith.exe` in the background
  POLL `http://127.0.0.1:8000/api/health` until it responds with 200 OK or times out
  IF timeout THEN FAIL build
  KILL the process

  // Phase 5: Upload Artifact
  UPLOAD the `fortuna-monolith.exe` as a build artifact
```

### 6.2 Race Report Generation (`unified-race-report.yml`)

This workflow runs on a schedule or manually to generate the daily race reports.

```pseudocode
// unified-race-report.yml
WORKFLOW Generate_Race_Report:
  // Phase 1: Setup Environment
  SETUP Python
  INSTALL dependencies from requirements.txt

  // Phase 2: Run Reporter Script
  // The script directly invokes the OddsEngine and AnalyzerEngine
  // without needing a live web server.
  EXECUTE `scripts/fortuna_reporter.py`

  // Phase 3: Publish Artifacts
  // The script generates a comprehensive set of artifacts for observability.
  UPLOAD the following files:
    - race-report.html (The primary user-facing report)
    - qualified_races.json (Data for the HTML report)
    - raw_race_data.json (Unfiltered data for debugging)
    - reporter_output.log (Full log of the reporter script)
    - github_summary.md (For display in the GitHub Actions UI)
```
---

## 7. END-TO-END WORKFLOW

### 7.1 Windows User Workflow

The user downloads and runs a single `.exe` file.

```
WORKFLOW user_launches_monolith_exe:
  STEP 1: User executes `fortuna-monolith.exe`.
  STEP 2: The embedded Python environment starts.
  STEP 3: The Uvicorn server starts inside the process.
  STEP 4: The FastAPI application initializes, mounts the bundled 'ui' directory, and opens the API endpoints.
  STEP 5: The user's default web browser is automatically opened to `http://127.0.0.1:8000`.
  STEP 6: The browser loads `index.html` from the FastAPI server.
  STEP 7: The Next.js application hydrates and makes same-origin API calls to `/api/*` to fetch data.
```

### 7.2 Cross-Platform (Podman) User Workflow

The user runs a launcher script that manages a Podman container.

```
WORKFLOW user_launches_podman_script:
  STEP 1: User executes `launcher.bat` or `./launcher.sh`.
  STEP 2: The script pulls the latest `ghcr.io/masonj0/fortuna-faucet` image.
  STEP 3: The script starts a Podman container, mapping port 8000 and volume mounting local `data` and `logs` directories.
  STEP 4: The container runs the same Python application, which starts the Uvicorn/FastAPI server.
  STEP 5: The user's default web browser is automatically opened to `http://127.0.0.1:8000`.
  STEP 6: The workflow proceeds identically to the Windows user workflow from Step 6 onward.
```

---
*This concludes the blueprint for the Fortuna Faucet unified monolith architecture.*
