# Fortuna Faucet

Tracking all the races that are ideal for betting Favorite To Place.

## Mission-Critical: Continuous Global Fetching

The Fortuna engine is built for 24/7 intelligence. Our primary goal is a steady stream of fetching that spans **3 or more continents** (North America, Europe, Australia/Asia, and South Africa), ensuring that the "Next to Jump" list is never empty.

### Fetching Categories
- **Entries**: Mission-critical discovery that creates predictions.
- **Results**: Vital validation of our strategies and performance.

## Core Features

- **Multi-Continent Discovery**: Thoroughbred, Harness, and Greyhound racing data from US, UK, IRE, AUS, SA, and more.
- **Odds Hygiene**: Real-time filtering of placeholder and default odds to ensure high-signal predictions.
- **Performance Auditing**: Automated verification of predictions against official results.
- **Portable App**: Fully functional as a standalone tool.

## Building Windows EXE

To build the standalone Windows executable:

```bash
python build_monolith.py
```

**First run requirement:**
The app requires Playwright for some discovery adapters. On first run, you may need to install the browser binaries:
```bash
python -m playwright install chromium
```
