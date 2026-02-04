import sqlite3
import json
from pathlib import Path

def dump_db():
    db_path = Path("backend/tracking_data.db")
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM tracking_objects ORDER BY last_seen DESC LIMIT 20")
        rows = cursor.fetchall()

        if not rows:
            print("No data found in tracking_objects table.")
            return

        print(f"{'ID':<4} | {'TrackID':<8} | {'Class':<10} | {'Status':<8} | {'Last Seen'}")
        print("-" * 60)

        for row in rows:
            data = dict(row)
            # Truncate JSON for display
            ai_info = data.get('ai_info_json')
            ai_status = "Available" if ai_info and ai_info != "null" else "None"
            
            print(f"{data['id']:<4} | {data['track_id']:<8} | {data['class_name']:<10} | {data['status']:<8} | {data['last_seen']}")
            if ai_status == "Available":
                try:
                    info = json.loads(ai_info)
                    print(f"  AI Info: {info.get('commonName', 'N/A')} ({info.get('scientificName', 'N/A')})")
                except:
                    print("  AI Info: [Error parsing JSON]")

    except sqlite3.OperationalError as e:
        print(f"Error reading database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    dump_db()
