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
