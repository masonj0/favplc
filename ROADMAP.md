# Fortuna Faucet Roadmap

## The Auditor: Real-Time Race Verification

The Auditor is a background process designed to provide real-time verification of race predictions against official results, calculating profitability and tracking performance.

The core logic is located in the [`python_service/auditor.py`](python_service/auditor.py) script.

### Frontend Integration

The Auditor is designed to have its data displayed on the frontend. The `AuditorEngine` class in the script exposes two key methods for this purpose:

1.  `get_rolling_metrics()`: This function is intended to power a "Last Hour" overlay on the UI, providing key performance indicators such as strike rate, net profit, and betting volume.
2.  `get_recent_activity()`: This function provides a list of the most recent bet outcomes (e.g., "CASHED", "BURNED", "PENDING") for display in a real-time activity feed or history log.

The intended architecture is for the backend to create API endpoints (e.g., `/api/auditor/metrics`, `/api/auditor/activity`) that expose these functions. The frontend would then call these endpoints to fetch and display the data.

**Note:** As of the last update, the web scraping component of the Auditor is non-functional. The description above outlines the intended design and capabilities, which are not yet fully operational.

## Mission Critical: 24/7 Global Fetching

Our primary goal is to ensure a continuous stream of predictions by maintaining active fetchers across **3 or more continents**, including crucial contributions from North America, Europe, Australia/Asia, and **South Africa**.

### Ultimate Targets
- **High-Fidelity Integration**: The extreme ultimate targets for the engine are **TwinSpires**, **TVG**, and **FanDuel Racing**. These are the primary platforms for user betting, and ensuring perfect synchronization with their live odds is the project's highest ambition.

### Categories of Success
- **Entries**: Mission-critical discovery data that generates predictions.
- **Results**: Vital validation data that proves our performance and refines strategies.

### Impossible Challenge
- **Extra Credit**: Successfully fetching from `https://www.racingandsports.com.au/form-guide` to further solidify our Australian data stream.

This dual approach ensures the "Faucet" never runs dry and the quality of predictions remains high.
