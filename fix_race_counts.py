import re

with open('fortuna.py', 'r') as f:
    content = f.read()

# 1. Fix FavoriteToPlaceMonitor.build_race_summaries aggressive filtering
content = content.replace(
    '''                if cutoff and (st < now - timedelta(minutes=30) or st > cutoff):
                    continue''',
    '                # Time window filtering removed to ensure all unique races are counted'
)

# 2. Fix run_discovery aggressive filtering
# We want to keep the logger info but not the actual filtering of all_races_raw
content = re.sub(
    r'        if cutoff:\s*\n\s*original_count = len\(all_races_raw\)\s*\n\s*filtered_races = \[\]\s*\n\s*for r in all_races_raw:\s*\n\s*st = r\.start_time\s*\n\s*if isinstance\(st, str\):\s*\n\s*try:\s*\n\s*st = datetime\.fromisoformat\(st\.replace\(\'Z\', \'\+00:00\'\)\)\s*\n\s*except \(ValueError, TypeError\):\s*\n\s*continue\s*\n\s*if st\.tzinfo is None:\s*\n\s*st = st\.replace\(tzinfo=EASTERN\)\s*\n\s*if now <= st <= cutoff:\s*\n\s*filtered_races\.append\(r\)\s*\n\s*all_races_raw = filtered_races\s*\n\s*logger\.info\(\s*\"Filtered races by time window\",\s*window_hours=window_hours,\s*before=original_count,\s*after=len\(all_races_raw\)\s*\)',
    '        # Initial time window filtering removed to ensure all unique races are tracked for reporting',
    content
)

# 3. Fix generate_summary_grid to show all unique races
content = content.replace(
    '    for race in races:',
    '    for race in (all_races or races):'
)

with open('fortuna.py', 'w') as f:
    f.write(content)
