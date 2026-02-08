import json
import sys
import os

def generate_summary():
    if not os.path.exists('race_data.json'):
        print('⚠️ race_data.json not found')
        return

    try:
        with open('race_data.json') as f:
            d = json.load(f)

        total = d.get('total_races', 0)
        bet_now = d.get('bet_now_count', 0)
        might_like = d.get('you_might_like_count', 0)

        print(f'**Total Races:** {total}')
        print(f'**BET NOW:** {bet_now}')
        print(f'**You Might Like:** {might_like}')
        print('')

        if d.get('bet_now_races'):
            print('#### 🎯 BET NOW OPPORTUNITIES')
            print('| SUP | MTP | DISC | TRACK | R# | FIELD | ODDS | TOP 5 |')
            print('|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|')
            for r in d['bet_now_races']:
                sup = '✅' if r.get('superfecta_offered') else '❌'
                mtp = r.get('mtp', 'N/A')
                disc = r.get('discipline', 'N/A')
                track = r.get('track', 'N/A')
                race_num = r.get('race_number', 'N/A')
                field = r.get('field_size', 'N/A')
                fav = f"{r['favorite_odds']:.2f}" if r.get('favorite_odds') else 'N/A'
                sec = f"{r['second_fav_odds']:.2f}" if r.get('second_fav_odds') else 'N/A'
                odds = f"{fav}, {sec}"
                top5 = f"`{r['top_five_numbers']}`" if r.get('top_five_numbers') else 'N/A'
                print(f'| {sup} | {mtp} | {disc} | {track} | {race_num} | {field} | {odds} | {top5} |')
            print('')

        if d.get('you_might_like_races'):
            print('#### 🌟 YOU MIGHT LIKE')
            print('| SUP | MTP | DISC | TRACK | R# | FIELD | ODDS | TOP 5 |')
            print('|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|')
            for r in d['you_might_like_races']:
                sup = '✅' if r.get('superfecta_offered') else '❌'
                mtp = r.get('mtp', 'N/A')
                disc = r.get('discipline', 'N/A')
                track = r.get('track', 'N/A')
                race_num = r.get('race_number', 'N/A')
                field = r.get('field_size', 'N/A')
                fav = f"{r['favorite_odds']:.2f}" if r.get('favorite_odds') else 'N/A'
                sec = f"{r['second_fav_odds']:.2f}" if r.get('second_fav_odds') else 'N/A'
                odds = f"{fav}, {sec}"
                top5 = f"`{r['top_five_numbers']}`" if r.get('top_five_numbers') else 'N/A'
                print(f'| {sup} | {mtp} | {disc} | {track} | {race_num} | {field} | {odds} | {top5} |')

    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    generate_summary()
