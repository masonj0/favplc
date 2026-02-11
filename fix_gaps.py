import re

with open('fortuna.py', 'r') as f:
    content = f.read()

# 1. Update SimplySuccessAnalyzer.qualify_races
content = content.replace(
    '''                r_with_odds = sorted(valid_r_with_odds, key=lambda x: x[1])
                race.top_five_numbers = ", ".join([str(r[0].number or '?') for r in r_with_odds[:5]])

            # Stability Check: Only if 1st and 2nd fav are established''',
    '''                r_with_odds = sorted(valid_r_with_odds, key=lambda x: x[1])
                race.top_five_numbers = ", ".join([str(r[0].number or '?') for r in r_with_odds[:5]])

                if len(r_with_odds) >= 2:
                    sec_fav = r_with_odds[1][0]
                    race.metadata['selection_number'] = sec_fav.number
                    race.metadata['selection_name'] = sec_fav.name

            # Stability Check: Only if 1st and 2nd fav are established'''
)

# 2. Update run_discovery Golden Zone
content = content.replace(
    'if -5 < mtp <= 20:',
    'if -10 < mtp <= 45:'
)
content = content.replace(
    'logger.warning("🔭 No \'Immediate Gold\' races found (0-20 mins).")',
    'logger.warning("🔭 No \'Immediate Gold\' races found (-10 to 45 mins).")'
)

# 3. Update FavoriteToPlaceMonitor Golden Zone
content = content.replace(
    '# THE GOLDEN ZONE: -5 to 20 mins\n            if -5 < mtp <= 20:',
    '# THE GOLDEN ZONE: -10 to 45 mins\n            if -10 < mtp <= 45:'
)

# 4. Update get_bet_now_races and get_you_might_like_races criteria
content = content.replace(
    'if r.mtp is not None and 0 < r.mtp <= 20',
    'if r.mtp is not None and -10 < r.mtp <= 45'
)
content = content.replace(
    'but 0 < MTP <= 20 and 2nd Fav Odds >= 4.0',
    'but -10 < MTP <= 45 and 2nd Fav Odds >= 4.0'
)
content = content.replace(
    'Criteria: MTP <= 20 minutes AND 2nd Favorite Odds >= 5.0',
    'Criteria: -10 < MTP <= 45 minutes AND 2nd Favorite Odds >= 5.0'
)

# 5. Update HotTipsTracker
content = content.replace(
    'future_limit = now + timedelta(minutes=20)',
    'future_limit = now + timedelta(minutes=45)'
)
content = content.replace(
    'if st > future_limit:',
    'if st > future_limit or st < now - timedelta(minutes=10):'
)

# Log selection_number
content = content.replace(
    '''                "top_five": r.top_five_numbers,
                "predicted_2nd_fav_odds": r.metadata.get('predicted_2nd_fav_odds')
            }''',
    '''                "top_five": r.top_five_numbers,
                "selection_number": r.metadata.get('selection_number'),
                "selection_name": r.metadata.get('selection_name'),
                "predicted_2nd_fav_odds": r.metadata.get('predicted_2nd_fav_odds')
            }'''
)

with open('fortuna.py', 'w') as f:
    f.write(content)
