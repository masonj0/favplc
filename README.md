# 🐎 Fortuna Faucet

Tracking global racing opportunities ideal for betting Favorite To Place.

## Mission-Critical: Continuous Global Fetching

The Fortuna engine is built for 24/7 intelligence. Our primary goal is a steady stream of fetching that spans **3 or more continents** (North America, Europe, Australia/Asia, and South Africa), ensuring that the "Next to Jump" list is never empty.

### Fetching Categories
- **Entries**: Mission-critical discovery that creates predictions using our advanced scoring model.
- **Results**: Vital validation of our strategies and performance auditing.

## Core Features

- **Multi-Continent Discovery**: Thoroughbred, Harness, and Greyhound racing data from US, UK, IRE, AUS, SA, and more.
- **Advanced Scoring**: New Grade-based system (A+, A, B+, B, C) using composite scores that factor in gap12, market depth, place probability, and race-type modifiers.
- **Goldmine Identification**: Automatic detection of high-value opportunities where the 2nd favorite has strong odds and a significant gap from the favorite.
- **Odds Hygiene**: Real-time filtering of placeholder and default odds to ensure high-signal predictions.
- **Performance Auditing**: Automated verification of predictions against official results via `fortuna_analytics.py`.
- **Portable App & GUI**: Standalone Windows executable with an integrated desktop dashboard (`--gui`).

## Usage

### Discovery & Reporting
To run a single discovery pass and generate reports:
```bash
python fortuna.py
```

### Live Monitor
To run in monitor mode with a live-updating terminal dashboard:
```bash
python fortuna.py --monitor
```

### Desktop GUI
To launch the integrated desktop dashboard in a native window:
```bash
python fortuna.py --gui
```

### Performance Analytics
To audit recent predictions and see ROI reports:
```bash
python fortuna_analytics.py --days 2
```

## Building Windows EXE

To build the standalone Windows executable (`FortunaFaucetPortableApp.exe`):

```bash
python build_monolith.py
```

**First run requirement:**
The app requires Playwright for some discovery adapters. On first run, you may need to install the browser binaries:
```bash
python -m playwright install chromium
```

Alternatively, run Fortuna with the `--install-browsers` flag:
```bash
python fortuna.py --install-browsers
```

You can also set the environment variable `FORTUNA_AUTO_INSTALL_BROWSERS=1` to enable automatic installation when dependencies are missing.
