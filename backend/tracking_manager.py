"""
Tracking manager module for detecting new tracking IDs and managing lifecycle.
"""

import asyncio
import cv2
import base64
import time
from datetime import datetime
from typing import Dict, List, Set, Optional, Any
from pathlib import Path
import sys
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from backend.database import DatabaseManager
from backend.ai_module import get_wildlife_info


class TrackingManager:
    """Manages tracking lifecycle and AI information collection."""
    
    def __init__(
        self, 
        db_manager: DatabaseManager,
        enable_ai: bool = True,
        ai_timeout: float = 30.0
    ):
        """
        Initialize tracking manager.
        
        Args:
            db_manager: Database manager instance
            enable_ai: Whether to enable AI information collection
            ai_timeout: Timeout for AI calls in seconds
        """
        self.db_manager = db_manager
        self.enable_ai = enable_ai
        self.ai_timeout = ai_timeout
        
        # Track currently active IDs
        self.active_track_ids: Set[int] = set()
        
        # Track IDs pending AI processing
        self.pending_ai_processing: Set[int] = set()
        
        # WebSocket connections for broadcasting
        self.ws_connections: Set[Any] = set()
        
        # Track last seen timestamps for persistence
        self.track_last_seen: Dict[int, datetime] = {}
        self.TRACK_PERSISTENCE_TIMEOUT = 10.0  # Seconds
        
        # Load existing active tracks from database
        self._load_active_tracks()
        
        print(f"✓ TrackingManager initialized (AI: {enable_ai})")
    
    def _load_active_tracks(self):
        """Load active tracks from database on startup."""
        active_tracks = self.db_manager.get_all_active_tracks()
        self.active_track_ids = {track['track_id'] for track in active_tracks}
        print(f"✓ Loaded {len(self.active_track_ids)} active tracks from database")
    
    def register_websocket(self, websocket):
        """Register a WebSocket connection for broadcasting."""
        self.ws_connections.add(websocket)
    
    def unregister_websocket(self, websocket):
        """Unregister a WebSocket connection."""
        self.ws_connections.discard(websocket)
    
    async def broadcast_message(self, message: Dict):
        """
        Broadcast a message to all connected WebSocket clients.
        
        Args:
            message: Message dictionary to broadcast
        """
        disconnected = set()
        for ws in self.ws_connections:
            try:
                await ws.send_json(message)
            except Exception as e:
                print(f"✗ Error broadcasting to WebSocket: {e}")
                disconnected.add(ws)
        
        # Remove disconnected clients
        for ws in disconnected:
            self.ws_connections.discard(ws)
    
    async def process_detections(
        self, 
        frame: np.ndarray, 
        detections: List[Dict[str, Any]]
    ):
        """
        Process detections and manage tracking lifecycle.
        
        Args:
            frame: Current video frame
            detections: List of detection dictionaries from tracker
        """
        current_time = datetime.now()
        current_track_ids = set()
        
        for detection in detections:
            track_id = detection.get('track_id')
            if track_id is None:
                continue
            
            current_track_ids.add(track_id)
            self.track_last_seen[track_id] = current_time # Update persistence timer
            class_name = detection.get('class_name', 'unknown')
            
            # Check if this is a new tracking ID
            if track_id not in self.active_track_ids:
                await self._handle_new_track(
                    track_id, 
                    class_name, 
                    frame, 
                    detection, 
                    current_time
                )
            else:
                # Update last_seen for existing track
                self.db_manager.update_last_seen(track_id, current_time)
        
        # Handle disappeared tracks with persistence grace period
        disappeared_from_frame = self.active_track_ids - current_track_ids
        for track_id in disappeared_from_frame:
            last_seen = self.track_last_seen.get(track_id)
            
            # If we don't have a record or it's past the grace period
            if not last_seen or (current_time - last_seen).total_seconds() > self.TRACK_PERSISTENCE_TIMEOUT:
                await self._handle_disappeared_track(track_id)
                if track_id in self.track_last_seen:
                    del self.track_last_seen[track_id]
    
    async def _handle_new_track(
        self, 
        track_id: int, 
        class_name: str, 
        frame: np.ndarray,
        detection: Dict[str, Any],
        timestamp: datetime
    ):
        """
        Handle a newly detected tracking ID.
        
        Args:
            track_id: New tracking ID
            class_name: Detected class name
            frame: Current video frame
            detection: Detection dictionary
            timestamp: Current timestamp
        """
        print(f"🆕 New track detected: ID={track_id}, class={class_name}")
        
        # Extract frame crop for AI processing and history thumbnail
        frame_snapshot = self._extract_frame_crop(frame, detection)
        
        # Create database entry
        self.db_manager.create_tracking_object(
            track_id=track_id,
            class_name=class_name,
            first_seen=timestamp,
            ai_info=None,  # Will be updated after AI processing
            frame_snapshot=frame_snapshot
        )
        
        # Add to active tracks
        self.active_track_ids.add(track_id)
        
        # Broadcast new track event
        await self.broadcast_message({
            "type": "track_new",
            "data": {
                "track_id": track_id,
                "class_name": class_name,
                "first_seen": timestamp.isoformat(),
                "ai_info": None
            }
        })
        
        # Start AI processing in background
        if self.enable_ai and track_id not in self.pending_ai_processing:
            self.pending_ai_processing.add(track_id)
            asyncio.create_task(
                self._process_ai_info(track_id, class_name, frame_snapshot)
            )
    
    async def _handle_disappeared_track(self, track_id: int):
        """
        Handle a tracking ID that has disappeared.
        
        Args:
            track_id: Disappeared tracking ID
        """
        print(f"👋 Track disappeared: ID={track_id}")
        
        # Remove from active set
        self.active_track_ids.discard(track_id)
        
        # Deactivate in database
        self.db_manager.deactivate_track(track_id)
        
        # Broadcast track removed event
        await self.broadcast_message({
            "type": "track_removed",
            "data": {
                "track_id": track_id
            }
        })
    
    def _extract_frame_crop(
        self, 
        frame: np.ndarray, 
        detection: Dict[str, Any]
    ) -> Optional[str]:
        """
        Extract and encode a cropped region of the frame.
        
        Args:
            frame: Full video frame
            detection: Detection dictionary with bbox
            
        Returns:
            Base64 encoded JPEG string or None
        """
        try:
            bbox = detection.get('bbox')
            if not bbox or len(bbox) != 4:
                return None
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Add padding (15% around the object)
            w_obj = x2 - x1
            h_obj = y2 - y1
            padding_w = int(w_obj * 0.15)
            padding_h = int(h_obj * 0.15)
            
            x1 = x1 - padding_w
            y1 = y1 - padding_h
            x2 = x2 + padding_w
            y2 = y2 + padding_h
            
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            # Crop frame
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                return None
            
            # Encode to JPEG
            success, buffer = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                return base64.b64encode(buffer).decode('utf-8')
            
            return None
        except Exception as e:
            print(f"✗ Error extracting frame crop: {e}")
            return None
    
    async def _process_ai_info(
        self, 
        track_id: int, 
        class_name: str, 
        frame_snapshot: Optional[str]
    ):
        """
        Process AI information for a tracking object (runs in background).
        
        Args:
            track_id: Tracking ID
            class_name: Detected class name
            frame_snapshot: Base64 encoded frame snapshot
        """
        try:
            snapshot_size = len(frame_snapshot) if frame_snapshot else 0
            print(f"🤖 Starting AI processing for track_id={track_id}, class={class_name}, snapshot_size={snapshot_size} bytes")
            
            # Fetch recent animal history for context
            history_str = None
            try:
                recent_animals = self.db_manager.get_recent_animal_history(limit=2)
                if recent_animals:
                    history_items = [f"{a['common_name']} ({a['scientific_name']})" for a in recent_animals]
                    history_str = ", ".join(history_items)
                    print(f"📜 Including history context for ID={track_id}: {history_str}")
            except Exception as e:
                print(f"⚠️ Error fetching animal history for context: {e}")

            # Run AI processing in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            wildlife_info = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    get_wildlife_info,
                    class_name,
                    frame_snapshot,
                    history_str,
                    "image/jpeg"
                ),
                timeout=self.ai_timeout
            )
            
            # Convert Pydantic model to dict
            ai_info_dict = wildlife_info.model_dump()
            
            # Update database
            self.db_manager.update_ai_info(track_id, ai_info_dict)
            
            print(f"✓ AI processing complete for track_id={track_id}")
            
            # Broadcast update
            await self.broadcast_message({
                "type": "track_updated",
                "data": {
                    "track_id": track_id,
                    "ai_info": ai_info_dict
                }
            })
            
        except asyncio.TimeoutError:
            print(f"⏱ AI processing timeout for track_id={track_id}")
        except Exception as e:
            print(f"✗ Error processing AI info for track_id={track_id}: {e}")
        finally:
            self.pending_ai_processing.discard(track_id)
    
    def get_active_tracks_data(self) -> List[Dict[str, Any]]:
        """
        Get data for all currently active tracks.
        
        Returns:
            List of tracking object dictionaries
        """
        return self.db_manager.get_all_active_tracks()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get tracking manager statistics.
        
        Returns:
            Dictionary with statistics
        """
        db_stats = self.db_manager.get_stats()
        return {
            **db_stats,
            "active_track_ids": list(self.active_track_ids),
            "pending_ai_processing": len(self.pending_ai_processing),
            "ws_connections": len(self.ws_connections)
        }
