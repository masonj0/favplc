# 🐎 JB's Guide to Interactive Processing & Scoring (v3.3.1)

Hello JB! This guide explains the best way to manually fetch racecards and perform interactive scoring to find the best bets with the highest 2nd-favorite odds.

## Step 1: Discover the Required Links
Run the link discovery mode to see which URLs you need to fetch.

```bash
python fortuna_interactive.py --list-links
```

This will generate a **`manual_links.html`** dashboard.

## Step 2: Fetch and Save Racecard Data
1.  Open **`manual_links.html`** in your browser.
2.  Use the **JavaScript Date Picker** at the top to select the race date (it defaults to today).
3.  **Tier 1 (JSON/API):** Right-click each link and choose "Save Link As..." (save as `.json`).
4.  **Tier 2 (HTML):** Click to open the link, then use "Save Page As..." (choose "Webpage, Complete").
5.  **Placement:** Save all files directly into the **`manual_fetch/`** directory.
    *   *Tip:* Use the target filenames shown in the dashboard to ensure the engine finds them.

## Step 3: Ingest the Saved Data
Perform a structural discovery sweep of your manually fetched files.

```bash
python fortuna_interactive.py --quarter-fetch
```

This reads the files from `manual_fetch/`, parses the races, and saves a structural snapshot in the `snapshots/` directory.

## Step 4: Interactive Processing & Scoring
Score the races and find the best betting opportunities.

```bash
python fortuna_interactive.py --score-now
```

### What happens now?
-   **JB's Preferred Sort:** The output is automatically sorted to put **Goldmines with the best 2nd-favorite odds first**.
-   **2nd-ODDS Column:** Look for the new `2nd-ODDS` column in the summary grid to quickly identify the highest value.
-   **Reports:** Detailed analysis is saved to `summary_grid.txt` and `goldmine_report.txt`.

---
*Powered by Fortuna Faucet Intelligence*
