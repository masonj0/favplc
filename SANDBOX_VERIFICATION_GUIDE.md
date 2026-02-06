# Sandbox Verification Guide: Race Discovery Validation

This document provides a detailed walkthrough for verifying that the Fortuna engine is correctly discovering races within the Jules sandbox environment. This process validates the core logic changes, new adapters, and tracking features.

## 🚀 The Paradigm Shift: The Sandbox as a Proving Ground

The Jules sandbox is a fully instrumented Linux environment capable of executing the entire Fortuna stack. We can prove success locally using the following tools.

**IMPORTANT:** The sandbox verification process is a self-contained activity designed to generate real-time reports within this VM. **NEVER** use `submit` or any Git commands to relay verification results. Instead, use the `message_user` tool to present your final Race Discovery Grid, Goldmine Report, and Monitor results directly to the user.

---

## 🛠️ Primary Tools for Verification

### 1. The Monolith: `fortuna.py`
This is the high-stability, monolithic discovery pipeline (the "Anthology"). It orchestrates the entire fleet of functional adapters, applies the `SimplySuccess` analyzer filters, and generates the standardized alphabetic summary grid.

**Execution (Discovery Mode):**
```bash
python3 fortuna.py
```

**Execution (Monitor Mode):**
```bash
python3 fortuna.py --monitor --once
```

---

## ⚡ The "Immediate Gold" Filter (0-20 MTP Shift)

To maximize signal relevance, the discovery pipeline now implements a strict **Time Filter** known as "Immediate Gold." This filter ensures that only races starting within a specific window are prioritized for analysis.

### The Golden Zone:
- **Criteria:** 0 < Minutes to Post (MTP) ≤ 20.
- **Why?** Focuses processing power on races about to start, ensuring the "Goldmine" opportunities are identified exactly when they are most actionable.

---

## 🏆 Goldmine Race Intelligence Report

The Goldmine report has been reworked to provide deeper insights into high-value opportunities, specifically focusing on the "Immediate Gold" window and the `1Gap2` (odds gap) metric.

### New Categorization:
1.  **Immediate Gold (superfecta):**
    - Races starting within **20 minutes**.
    - **2nd favorite** has decimal odds of **5.0 or higher**.
    - **Superfecta** bet is available (explicitly or via T-track field size rules).
2.  **Immediate Gold:**
    - Same as above, but **without** Superfecta availability.
3.  **All Remaining Goldmine Races:**
    - Qualifying races starting more than 20 minutes in the future.

---

## 📡 Expanded Adapter Fleet

The engine now utilizes an expanded fleet of validated adapters to ensure maximum coverage across Thoroughbred, Harness, and Greyhound racing, including results and dividends.

### Core Validated Adapters:
- **Thoroughbred:** AtTheRaces, RacingPostB2B, SportingLife, SkySports, Equibase, TwinSpires, Oddschecker, Timeform, RacingPost.
- **Harness:** StandardbredCanada, Tab, SkySports.
- **Greyhound:** AtTheRacesGreyhound, BoyleSports.
- **Results/Dividends:** RacingPostTote.

---

## 📊 The Tiered Summary Grid

The summary grid (`summary_grid.txt`) provides immediate visual confirmation of discovered data and is now partitioned for better transparency.

### Key Features:
1.  **Tiered Reporting:**
    - **Primary Section:** Displays qualified races that have **Superfectas** available.
    - **Secondary Section ("All Remaining Races"):** Displays all other qualified races.
2.  **CATEG Column:** Identifies tracks as **T** (Thoroughbred), **H** (Harness), or **G** (Greyhound).
3.  **Alphabetic Race Coding:** Race numbers are mapped to letters (`a`=1, `b`=2). Goldmine races are Uppercase.
4.  **1Gap2 Metric:** The report now includes the `1Gap2` value for each goldmine candidate.

---

## 🏁 The Final Step: Reporting Results

Once you have executed the discovery tools and confirmed their output (e.g., `summary_grid.txt`, `goldmine_report.txt`, `hot_tips_db.json`), the verification process concludes by presenting these artifacts to the user.

**Procedure:**
1.  Read the generated text reports using `read_file`.
2.  Format a comprehensive summary.
3.  Call `message_user` with the content of the reports.
4.  **DO NOT COMMIT** these run-specific artifacts.

---

## 🔧 Troubleshooting the Sandbox

If scripts fail due to missing dependencies, ensure the following are installed:
```bash
python3 -m pip install httpx structlog pydantic==2.5.0 pandas selectolax tenacity scrapling pydantic-settings redis curl_cffi msgspec browserforge camoufox
python3 -m browserforge update
playwright install --with-deps chromium
```
