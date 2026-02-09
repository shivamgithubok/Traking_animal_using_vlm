import asyncio
import threading
import queue
import time
import cv2
import numpy as np
import base64
import json
import logging
from datetime import datetime
from typing import Dict, List, Set, Optional, Any
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import Config
from backend.database import DatabaseManager
import backend.ai_broker as ai_broker

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrackingManager")

class TrackingManager:
    """
    Industrial-Grade Tracking Manager.
    Uses a Producer-Consumer pattern (Queue) to decouple fast video tracking 
    from slow CPU-based AI inference.
    """
    
    def __init__(
        self, 
        db_manager: DatabaseManager,
        recording_manager: Any = None,
        enable_ai: bool = True,
        ai_timeout: float = 30.0
    ):
        self.db_manager = db_manager
        self.recording_manager = recording_manager
        self.enable_ai = enable_ai
        self.ai_timeout = ai_timeout
        
        # --- STATE MANAGEMENT ---
        self.active_track_ids: Set[int] = set()
        self.track_last_seen: Dict[int, datetime] = {}
        self.ws_connections: Set[Any] = set()
        
        # --- INDUSTRIAL TUNING PARAMS ---
        self.TRACK_PERSISTENCE_TIMEOUT = 5.0   # How long to keep ID after it leaves frame
        self.STABILITY_THRESHOLD = 1.0         # Wait 1s before sending to AI (Reduces false positives)
        
        # --- QUEUE ARCHITECTURE ---
        # Limit queue size to 20. If AI is too slow, we drop new requests 
        # rather than crashing the server with OOM (Out of Memory).
        self.ai_queue = queue.PriorityQueue(maxsize=20)
        
        # Tracks waiting to be stable enough for AI
        # Format: {track_id: {'start_time': datetime, 'snapshot': b64, 'queued': bool, 'class_name': str}}
        self.pending_tracks_stability = {} 
        
        # Capture the main event loop to allow the thread to talk back to async functions
        self.main_loop = asyncio.get_event_loop()
        
        self.running = True
        
        # 1. Load data from DB (Fixes your previous error)
        self._load_active_tracks()
        
        # 2. Start the Background Worker Thread
        self.worker_thread = threading.Thread(target=self._ai_worker_loop, daemon=True)
        self.worker_thread.start()
        
        logger.info(f"✓ TrackingManager initialized. AI Worker Running.")

    def _load_active_tracks(self):
        """Load active tracks from database on startup."""
        try:
            active_tracks = self.db_manager.get_all_active_tracks()
            self.active_track_ids = {track['track_id'] for track in active_tracks}
            logger.info(f"✓ Loaded {len(self.active_track_ids)} active tracks from DB")
        except Exception as e:
            logger.error(f"Error loading tracks: {e}")

    # --- WEBSOCKET HANDLING ---
    def register_websocket(self, websocket):
        self.ws_connections.add(websocket)
    
    def unregister_websocket(self, websocket):
        self.ws_connections.discard(websocket)
    
    async def broadcast_message(self, message: Dict):
        """Async broadcast to all clients"""
        disconnected = set()
        for ws in self.ws_connections:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.add(ws)
        for ws in disconnected:
            self.ws_connections.discard(ws)

    def _broadcast_threadsafe(self, message: Dict):
        """Helper to broadcast from the sync worker thread"""
        if self.main_loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast_message(message), self.main_loop)

    # --- MAIN VIDEO LOOP (30 FPS) ---
    async def process_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]):
        """
        Runs every frame. MUST BE FAST. 
        Only updates coordinates and checks stability.
        Does NOT run AI.
        """
        current_time = datetime.now()
        current_track_ids = set()
        
        for detection in detections:
            track_id = detection.get('track_id')
            if track_id is None: continue
            
            # 1. Filter Humans (Privacy / Irrelevance)
            class_name = detection.get('class_name', 'unknown')
            if class_name.lower() == "person":
                continue

            current_track_ids.add(track_id)
            self.track_last_seen[track_id] = current_time

            # 2. Handle New Tracks
            if track_id not in self.active_track_ids:
                await self._handle_new_track(track_id, class_name, frame, detection, current_time)
            else:
                self.db_manager.update_last_seen(track_id, current_time)

            # 3. Check Stability for AI Queue
            # This ensures we don't spam AI with objects that flicker in/out
            if self.enable_ai and track_id in self.pending_tracks_stability:
                track_data = self.pending_tracks_stability[track_id]
                
                # Check if it has existed long enough
                duration = (current_time - track_data['start_time']).total_seconds()
                
                # Update snapshot with a newer one if needed? (Optional)
                
                if duration > self.STABILITY_THRESHOLD and not track_data['queued']:
                    # STABLE! Queue it.
                    logger.info(f"🚀 Track {track_id} is stable ({duration:.1f}s). Queuing for AI.")
                    track_data['queued'] = True # Mark as queued so we don't add it twice
                    
                    try:
                        # Priority 1 (High). Could use 0 for Bears, 2 for Deer, etc.
                        job = (1, track_id, track_data['class_name'], track_data['snapshot'])
                        self.ai_queue.put_nowait(job)
                    except queue.Full:
                        logger.warning(f"⚠️ AI Queue Full! Dropping AI request for Track {track_id}")

        # 4. Handle Disappeared Tracks
        disappeared = self.active_track_ids - current_track_ids
        for track_id in disappeared:
            last_seen = self.track_last_seen.get(track_id)
            if not last_seen or (current_time - last_seen).total_seconds() > self.TRACK_PERSISTENCE_TIMEOUT:
                await self._handle_disappeared_track(track_id)

    async def _handle_new_track(self, track_id, class_name, frame, detection, timestamp):
        """Registers track in DB and preps for stability check"""
        logger.info(f"🆕 New track: {track_id} ({class_name})")
        
        # Extract Crop
        snapshot = self._extract_frame_crop(frame, detection)
        
        # DB Entry
        self.db_manager.create_tracking_object(
            track_id, class_name, timestamp, None, snapshot
        )
        self.active_track_ids.add(track_id)
        
        # Notify UI
        await self.broadcast_message({
            "type": "track_new", 
            "data": {"track_id": track_id, "class_name": class_name, "frame_snapshot": snapshot}
        })
        
        # Add to Stability Monitor (Not AI Queue yet!)
        self.pending_tracks_stability[track_id] = {
            'start_time': timestamp,
            'snapshot': snapshot, # Keep the first clear shot
            'class_name': class_name,
            'queued': False
        }
        
        # Start Recording
        if self.recording_manager:
            # Pass full frame shape or just start
            self.recording_manager.start_recording(track_id, class_name, frame.shape[:2])

    async def _handle_disappeared_track(self, track_id):
        logger.info(f"👋 Track {track_id} disappeared.")
        self.active_track_ids.discard(track_id)
        self.track_last_seen.pop(track_id, None)
        self.pending_tracks_stability.pop(track_id, None)
        
        self.db_manager.deactivate_track(track_id)
        
        if self.recording_manager:
            self.recording_manager.stop_recording(track_id)

        await self.broadcast_message({
            "type": "track_removed", "data": {"track_id": track_id}
        })

    # --- BACKGROUND WORKER THREAD (The "Industrial" Part) ---
    def _ai_worker_loop(self):
        """
        Runs in a separate thread. Consumes AI requests one by one.
        This prevents the 'Thundering Herd' effect on the CPU.
        """
        logger.info("👷 AI Worker Thread Started")
        
        while self.running:
            try:
                # 1. Get Job (Blocks for 1 sec)
                # Structure: (priority, track_id, class_name, snapshot)
                _, track_id, class_name, frame_snapshot = self.ai_queue.get(timeout=1.0)
                
                # 2. Check Validity (Did it leave while in queue?)
                if track_id not in self.active_track_ids:
                    logger.info(f"⏩ Skipping stale track {track_id}")
                    self.ai_queue.task_done()
                    continue

                # 3. EXECUTE INFERENCE (Synchronous/Blocking is OK here)
                self._execute_ai_analysis(track_id, class_name, frame_snapshot)
                
                self.ai_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker Loop Error: {e}")

    def _execute_ai_analysis(self, track_id, class_name, frame_snapshot):
        """
        The heavy lifting logic.
        """
        try:
            # Context
            history_str = None
            try:
                recent = self.db_manager.get_recent_animal_history(limit=2)
                if recent:
                    history_str = ", ".join([r['common_name'] for r in recent])
            except: pass

            logger.info(f"🤖 AI Processing {track_id}...")
            
            # CALL BROKER (Blocks this thread, not the video)
            wildlife_info = ai_broker.get_wildlife_info(
                class_name, frame_snapshot, history_str
            )

            # Normalize data
            if hasattr(wildlife_info, "model_dump"):
                ai_info_dict = wildlife_info.model_dump()
            else:
                ai_info_dict = wildlife_info

            # Handle Non-Animals (False Positives)
            if not ai_info_dict.get("is_animal", True):
                logger.info(f"🚫 Track {track_id} rejected by AI.")
                self.db_manager.delete_track(track_id)
                if self.recording_manager:
                    self.recording_manager.cancel_recording(track_id)
                
                # We need to update state, but state is shared. 
                # Sets are thread-safe for add/remove in Python usually, but let's be safe.
                self.active_track_ids.discard(track_id)
                
                self._broadcast_threadsafe({
                    "type": "track_removed", "data": {"track_id": track_id}
                })
                return

            # Success - Update DB
            self.db_manager.update_ai_info(track_id, ai_info_dict)
            
            # Rename Recording
            if self.recording_manager and ai_info_dict.get("commonName"):
                safe_name = "".join([c if c.isalnum() else "_" for c in ai_info_dict["commonName"]])
                self.recording_manager.rename_recording(track_id, safe_name)

            logger.info(f"✓ AI Result {track_id}: {ai_info_dict.get('commonName')}")

            # Notify UI (Thread-safe)
            self._broadcast_threadsafe({
                "type": "track_updated",
                "data": {
                    "track_id": track_id,
                    "ai_info": ai_info_dict,
                    "frame_snapshot": frame_snapshot
                }
            })

        except Exception as e:
            logger.error(f"AI Analysis Failed for {track_id}: {e}")

    # --- HELPERS ---
    def _extract_frame_crop(self, frame, detection):
        try:
            bbox = detection.get('bbox')
            if not bbox: return None
            x1, y1, x2, y2 = map(int, bbox)
            
            # Add 10% Padding
            w_obj = x2 - x1
            h_obj = y2 - y1
            px = int(w_obj * 0.1)
            py = int(h_obj * 0.1)
            
            h, w = frame.shape[:2]
            x1 = max(0, x1 - px)
            y1 = max(0, y1 - py)
            x2 = min(w, x2 + px)
            y2 = min(h, y2 + py)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: return None
            
            success, buffer = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if success:
                return base64.b64encode(buffer).decode('utf-8')
        except Exception:
            return None
        return None