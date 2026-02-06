"""
Database module for tracking object persistence using SQLite.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import threading
import os


class DatabaseManager:
    """Manages SQLite database for tracking objects."""
    
    def __init__(self, db_path: str = "tracking_data.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Ensure directory exists
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        self.lock = threading.Lock()
        self._initialize_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    def _initialize_database(self):
        """Create database tables if they don't exist."""
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                # Create tracking_objects table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS tracking_objects (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        track_id INTEGER UNIQUE NOT NULL,
                        class_name TEXT NOT NULL,
                        first_seen TIMESTAMP NOT NULL,
                        last_seen TIMESTAMP NOT NULL,
                        ai_info_json TEXT,
                        status TEXT DEFAULT 'active',
                        frame_snapshot TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for faster queries
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_track_id 
                    ON tracking_objects(track_id)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_status 
                    ON tracking_objects(status)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_class_name 
                    ON tracking_objects(class_name)
                """)
                
                conn.commit()
                print("✓ Database initialized successfully")
            except Exception as e:
                print(f"✗ Error initializing database: {e}")
                raise
            finally:
                conn.close()
    
    def create_tracking_object(
        self, 
        track_id: int, 
        class_name: str, 
        first_seen: datetime,
        ai_info: Optional[Dict] = None,
        frame_snapshot: Optional[str] = None
    ) -> Optional[int]:
        """
        Create a new tracking object entry.
        
        Args:
            track_id: Unique tracking ID
            class_name: Detected class name
            first_seen: Timestamp when first detected
            ai_info: Wildlife information dictionary
            frame_snapshot: Base64 encoded frame snapshot
            
        Returns:
            Database row ID if successful, None otherwise
        """
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                ai_info_json = json.dumps(ai_info) if ai_info else None
                
                cursor.execute("""
                    INSERT INTO tracking_objects 
                    (track_id, class_name, first_seen, last_seen, ai_info_json, status, frame_snapshot)
                    VALUES (?, ?, ?, ?, ?, 'active', ?)
                """, (track_id, class_name, first_seen, first_seen, ai_info_json, frame_snapshot))
                
                conn.commit()
                row_id = cursor.lastrowid
                print(f"✓ Created tracking object: track_id={track_id}, class={class_name}")
                return row_id
            except sqlite3.IntegrityError:
                # Track already exists, just make sure it's active
                try:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE tracking_objects 
                        SET status = 'active', last_seen = ?
                        WHERE track_id = ?
                    """, (first_seen, track_id))
                    conn.commit()
                    print(f"↺ Reactivated existing track ID {track_id}")
                    return track_id
                except Exception as e:
                    print(f"✗ Error reactivating track {track_id}: {e}")
                    return None
            except Exception as e:
                print(f"✗ Error creating tracking object: {e}")
                return None
            finally:
                conn.close()
    
    def update_last_seen(self, track_id: int, last_seen: datetime) -> bool:
        """
        Update the last_seen timestamp for a tracking object.
        
        Args:
            track_id: Tracking ID to update
            last_seen: New timestamp
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE tracking_objects 
                    SET last_seen = ?
                    WHERE track_id = ?
                """, (last_seen, track_id))
                
                conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                print(f"✗ Error updating last_seen for track {track_id}: {e}")
                return False
            finally:
                conn.close()
    
    def update_ai_info(self, track_id: int, ai_info: Dict) -> bool:
        """
        Update AI information for a tracking object.
        
        Args:
            track_id: Tracking ID to update
            ai_info: Wildlife information dictionary
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                ai_info_json = json.dumps(ai_info)
                
                cursor.execute("""
                    UPDATE tracking_objects 
                    SET ai_info_json = ?
                    WHERE track_id = ?
                """, (ai_info_json, track_id))
                
                conn.commit()
                print(f"✓ Updated AI info for track_id={track_id}")
                return cursor.rowcount > 0
            except Exception as e:
                print(f"✗ Error updating AI info for track {track_id}: {e}")
                return False
            finally:
                conn.close()
    
    def get_tracking_object(self, track_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a tracking object by track_id.
        
        Args:
            track_id: Tracking ID to retrieve
            
        Returns:
            Dictionary with tracking object data or None
        """
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM tracking_objects WHERE track_id = ?
                """, (track_id,))
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_dict(row)
                return None
            except Exception as e:
                print(f"✗ Error getting tracking object {track_id}: {e}")
                return None
            finally:
                conn.close()
    
    def get_all_active_tracks(self) -> List[Dict[str, Any]]:
        """
        Get all active tracking objects.
        
        Returns:
            List of tracking object dictionaries
        """
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM tracking_objects 
                    WHERE status = 'active'
                    ORDER BY last_seen DESC
                """)
                
                rows = cursor.fetchall()
                return [self._row_to_dict(row) for row in rows]
            except Exception as e:
                print(f"✗ Error getting active tracks: {e}")
                return []
            finally:
                conn.close()
    
    def get_tracking_history(
        self, 
        track_id: Optional[int] = None,
        class_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get tracking history with optional filters.
        
        Args:
            track_id: Filter by specific track ID
            class_name: Filter by class name
            start_date: Filter by start date
            end_date: Filter by end date
            status: Filter by status ('active' or 'inactive')
            limit: Maximum number of results
            
        Returns:
            List of tracking object dictionaries
        """
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                query = "SELECT * FROM tracking_objects WHERE 1=1"
                params = []
                
                if track_id is not None:
                    query += " AND track_id = ?"
                    params.append(track_id)
                
                if class_name:
                    query += " AND class_name = ?"
                    params.append(class_name)
                
                if start_date:
                    query += " AND first_seen >= ?"
                    params.append(start_date)
                
                if end_date:
                    query += " AND last_seen <= ?"
                    params.append(end_date)
                
                if status:
                    query += " AND status = ?"
                    params.append(status)
                
                query += " ORDER BY first_seen DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [self._row_to_dict(row) for row in rows]
            except Exception as e:
                print(f"✗ Error getting tracking history: {e}")
                return []
            finally:
                conn.close()
    
    def deactivate_track(self, track_id: int) -> bool:
        """
        Mark a tracking object as inactive.
        
        Args:
            track_id: Tracking ID to deactivate
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE tracking_objects 
                    SET status = 'inactive'
                    WHERE track_id = ?
                """, (track_id,))
                
                conn.commit()
                if cursor.rowcount > 0:
                    print(f"✓ Deactivated track_id={track_id}")
                    return True
                return False
            except Exception as e:
                print(f"✗ Error deactivating track {track_id}: {e}")
                return False
            finally:
                conn.close()
    
    def delete_track(self, track_id: int) -> bool:
        """
        Hard delete a tracking object from the database.
        
        Args:
            track_id: Tracking ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM tracking_objects 
                    WHERE track_id = ?
                """, (track_id,))
                
                conn.commit()
                if cursor.rowcount > 0:
                    print(f"🗑️ Deleted track_id={track_id}")
                    return True
                return False
            except Exception as e:
                print(f"✗ Error deleting track {track_id}: {e}")
                return False
            finally:
                conn.close()

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """
        Convert a database row to a dictionary.
        
        Args:
            row: SQLite row object
            
        Returns:
            Dictionary representation
        """
        data = dict(row)
        
        # Parse JSON fields
        if data.get('ai_info_json'):
            try:
                data['ai_info'] = json.loads(data['ai_info_json'])
            except json.JSONDecodeError:
                data['ai_info'] = None
        else:
            data['ai_info'] = None
        
        # Remove the JSON string field
        data.pop('ai_info_json', None)
        
        return data
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                # Total tracks
                cursor.execute("SELECT COUNT(*) FROM tracking_objects")
                total = cursor.fetchone()[0]
                
                # Active tracks
                cursor.execute("SELECT COUNT(*) FROM tracking_objects WHERE status = 'active'")
                active = cursor.fetchone()[0]
                
                # Inactive tracks
                cursor.execute("SELECT COUNT(*) FROM tracking_objects WHERE status = 'inactive'")
                inactive = cursor.fetchone()[0]
                
                # Tracks by class
                cursor.execute("""
                    SELECT class_name, COUNT(*) as count 
                    FROM tracking_objects 
                    GROUP BY class_name
                """)
                by_class = {row[0]: row[1] for row in cursor.fetchall()}
                
                return {
                    "total_tracks": total,
                    "active_tracks": active,
                    "inactive_tracks": inactive,
                    "tracks_by_class": by_class
                }
            except Exception as e:
                print(f"✗ Error getting stats: {e}")
                return {}
            finally:
                conn.close()
    def get_recent_animal_history(self, limit: int = 2) -> List[Dict[str, Any]]:
        """
        Get the most recent confirmed animal detections.
        
        Args:
            limit: Maximum number of recent detections to fetch
            
        Returns:
            List of dictionaries with animal info
        """
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                # Query for tracking objects that have AI info and are animals
                cursor.execute("""
                    SELECT class_name, ai_info_json 
                    FROM tracking_objects 
                    WHERE ai_info_json IS NOT NULL 
                    AND status = 'inactive'
                    ORDER BY last_seen DESC 
                    LIMIT ?
                """, (limit,))
                
                results = []
                for row in cursor.fetchall():
                    try:
                        ai_info = json.loads(row['ai_info_json'])
                        # Only include if confirmed as an animal
                        if ai_info.get('is_animal'):
                            results.append({
                                "class_name": row['class_name'],
                                "common_name": ai_info.get('commonName'),
                                "scientific_name": ai_info.get('scientificName')
                            })
                    except Exception as e:
                        print(f"✗ Error parsing ai_info_json: {e}")
                        continue
                
                return results
            except Exception as e:
                print(f"✗ Error getting recent animal history: {e}")
                return []
            finally:
                conn.close()
    def get_unique_species_history(self, minutes: int = 10) -> List[Dict[str, Any]]:
        """
        Get unique species detected in the last X minutes.
        Deduplication is based on scientificName from AI info.
        
        Args:
            minutes: Time window in minutes
            
        Returns:
            List of unique animal sightings
        """
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                # Fetch animal sightings from the last 10 minutes
                # Use subquery to get latest for each scientific name if needed, 
                # but simplest is to fetch all and deduplicate in Python for clarity with JSON parsing.
                cursor.execute("""
                    SELECT class_name, ai_info_json, frame_snapshot, last_seen
                    FROM tracking_objects 
                    WHERE ai_info_json IS NOT NULL 
                    AND last_seen >= datetime('now', '-' || ? || ' minutes')
                    ORDER BY last_seen DESC
                """, (minutes,))
                
                seen_species = set()
                results = []
                
                for row in cursor.fetchall():
                    try:
                        ai_info = json.loads(row['ai_info_json'])
                        # Only confirmed animals
                        if not ai_info.get('is_animal'):
                            continue
                            
                        sci_name = ai_info.get('scientificName')
                        if sci_name and sci_name not in seen_species:
                            seen_species.add(sci_name)
                            results.append({
                                "class_name": row['class_name'],
                                "ai_info": ai_info,
                                "frame_snapshot": row['frame_snapshot'],
                                "last_seen": row['last_seen']
                            })
                    except Exception:
                        continue
                
                return results
            except Exception as e:
                print(f"✗ Error getting unique species history: {e}")
                return []
            finally:
                conn.close()
