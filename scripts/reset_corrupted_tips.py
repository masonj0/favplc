import sqlite3
import os

DB_PATH = os.environ.get("FORTUNA_DB_PATH", "fortuna.db")

def reset_corrupted_tips():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Reset ATR (Thoroughbred) and ATRG (Greyhound) tips
        # They were corrupted by malformed venues/race numbers (ATR) or bad odds parsing (ATRG)
        query = """
        UPDATE tips
        SET audit_completed = 0,
            verdict = NULL,
            net_profit = NULL,
            actual_2nd_fav_odds = NULL,
            selection_position = NULL,
            actual_top_5 = NULL,
            audit_timestamp = NULL,
            trifecta_payout = NULL,
            trifecta_combination = NULL,
            superfecta_payout = NULL,
            superfecta_combination = NULL,
            top1_place_payout = NULL,
            top2_place_payout = NULL
        WHERE (race_id LIKE 'atr_%' OR race_id LIKE 'atrg_%')
          AND audit_completed = 1;
        """
        cursor.execute(query)
        affected = cursor.rowcount
        conn.commit()
        print(f"Successfully reset {affected} ATR/ATRG tips for re-auditing.")
    except Exception as e:
        print(f"Error resetting tips: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    reset_corrupted_tips()
