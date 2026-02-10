import re

with open('fortuna.py', 'r') as f:
    content = f.read()

# Fix 18: Share FortunaDB instance in run_discovery
# Replace the second 'db = FortunaDB()' with empty string if 'db' is already in scope
# or just reuse the existing one.
content = content.replace(
    '                    if harvest_summary:\n                        db = FortunaDB()\n                        await db.log_harvest(harvest_summary, region=region)',
    '                    if harvest_summary:\n                        await db.log_harvest(harvest_summary, region=region)'
)

# JB's Fix: run_discovery filtering
content = content.replace(
    '''        unique_races = list(next_races_map.values())
        if not unique_races:
            logger.warning("🔭 No 'Immediate Gold' races found (0-20 mins).")
            # We continue instead of returning to allow the rest of the discovery process (saving reports, etc)
            # but no tips will be logged or processed.

        logger.info("Filtered to Next Race per track in Golden Zone", count=len(unique_races))

        # Save raw fetched/merged races if requested
        if save_path:
            try:
                with open(save_path, "w") as f:
                    json.dump([r.model_dump(mode='json') for r in unique_races], f, indent=4)
                logger.info("Saved races to file", path=save_path)
            except Exception as e:
                logger.error("Failed to save races", error=str(e))

        if fetch_only:
            logger.info("Fetch-only mode active. Skipping analysis and reporting.")
            return

        # Analyze
        analyzer = SimplySuccessAnalyzer()
        result = analyzer.qualify_races(unique_races)
        qualified = result.get("races", [])

        # Generate Grid & Goldmine
        grid = generate_summary_grid(qualified, all_races=unique_races)''',
    '''        golden_zone_races = list(next_races_map.values())
        if not golden_zone_races:
            logger.warning("🔭 No 'Immediate Gold' races found (0-20 mins).")

        logger.info("Filtered to Next Race per track in Golden Zone", count=len(golden_zone_races))
        logger.info("Total unique races available for summary", count=len(unique_races))

        # Save raw fetched/merged races if requested (Save EVERYTHING unique)
        if save_path:
            try:
                with open(save_path, "w") as f:
                    json.dump([r.model_dump(mode='json') for r in unique_races], f, indent=4)
                logger.info("Saved all unique races to file", path=save_path)
            except Exception as e:
                logger.error("Failed to save races", error=str(e))

        if fetch_only:
            logger.info("Fetch-only mode active. Skipping analysis and reporting.")
            return

        # Analyze ONLY Golden Zone races for immediate tips
        analyzer = SimplySuccessAnalyzer()
        result = analyzer.qualify_races(golden_zone_races)
        qualified = result.get("races", [])

        # Generate Grid & Goldmine (Grid uses unique_races for the broader context)
        grid = generate_summary_grid(qualified, all_races=unique_races)'''
)

with open('fortuna.py', 'w') as f:
    f.write(content)
