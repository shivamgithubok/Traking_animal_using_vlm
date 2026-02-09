from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
import cv2
import base64
import asyncio
import json
import time
import threading
import os
from collections import deque
from pathlib import Path
from typing import Set, List, Dict, Optional, Tuple, Any
import sys
from datetime import datetime, timedelta
import uuid

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import Config
from backend.tracker import ObjectTracker, CameraCapture
from backend.database import DatabaseManager
from backend.tracking_manager import TrackingManager
import backend.ai_broker as ai_broker

# ---------------- VIDEO RECORDING MODULES ---------------- #

class VideoRecorder:
    """Manages individual video recording session for a tracked object."""
    
    def __init__(self, track_id: int, class_name: str, frame_shape: Tuple[int, int], 
                 fps: int, codec: str, output_dir: str, pre_buffer_frames: List = None):
        self.track_id = track_id
        self.class_name = class_name
        self.fps = fps
        self.output_dir = output_dir
        
        # Create class-specific directory
        self.class_dir = os.path.join(output_dir, class_name)
        os.makedirs(self.class_dir, exist_ok=True)
        
        # Calculate start time (accounting for buffer)
        buffer_duration = len(pre_buffer_frames) / fps if pre_buffer_frames else 0
        self.start_time = datetime.now() - timedelta(seconds=buffer_duration)
        
        # Generate filename: YYYYMMDD_HHMMSS_track_ID.mp4
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.filename_base = f"{timestamp}_track_{track_id}"
        self.video_path = os.path.join(self.class_dir, f"{self.filename_base}.mp4")
        self.metadata_path = os.path.join(self.class_dir, f"{self.filename_base}.json")
        
        # Initialize video writer
        height, width = frame_shape
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(self.video_path, fourcc, fps, (width, height))
        
        self.end_time = None
        self.frame_count = 0
        self.is_active = True
        
        # 1. Flush pre-buffer frames immediately
        if pre_buffer_frames:
            for old_frame in pre_buffer_frames:
                self.write_frame(old_frame)
        
        print(f"📹 Started recording: {self.video_path}")
    
    def write_frame(self, frame):
        """Write a frame to the video file."""
        if self.is_active and self.writer is not None:
            self.writer.write(frame)
            self.frame_count += 1
    
    def stop(self):
        """Stop recording and save metadata."""
        if not self.is_active:
            return
        
        self.is_active = False
        self.end_time = datetime.now()
        
        # Release video writer
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        
        # If the class was renamed, move the file now
        self._finalize_move_if_needed()
        
        # Calculate duration
        duration = (self.end_time - self.start_time).total_seconds()
        
        final_video_path = self.video_path
        final_metadata_path = self.metadata_path
        
        # Save metadata
        metadata = {
            "event_id": self.filename_base, # Use filename as ID for API
            "track_id": self.track_id,
            "class_name": self.class_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": round(duration, 2),
            "frame_count": self.frame_count,
            "fps": self.fps,
            "video_path": final_video_path,
            "detected_classes": [self.class_name] # For API compatibility
        }
        
        # Ensure directory exists (might have changed during rename)
        os.makedirs(os.path.dirname(final_video_path), exist_ok=True)
        
        with open(final_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Stopped recording: {final_video_path} ({duration:.1f}s)")
        return metadata

    def rename(self, new_class_name: str, codec_str: str):
        """Update the target class name and paths for when the recording stops."""
        if not self.is_active or self.class_name == new_class_name:
            return

        print(f"🔄 Queuing rename for track {self.track_id}: {self.class_name} -> {new_class_name}")

        # 1. Prepare new directory info
        new_class_dir = os.path.join(self.output_dir, new_class_name)
        
        # 2. Update existing video file path (we'll move it on stop())
        old_video_path = self.video_path
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.filename_base = f"{timestamp}_track_{self.track_id}"
        
        # We don't move the file yet because it's still open by the writer!
        # We just update the paths where we WANT it to be eventually.
        self.target_video_path = os.path.join(new_class_dir, f"{self.filename_base}.mp4")
        self.target_metadata_path = os.path.join(new_class_dir, f"{self.filename_base}.json")
        self.new_class_name = new_class_name

    def _finalize_move_if_needed(self):
        """Move the recorded file to its final destination after stopping."""
        if hasattr(self, 'new_class_name') and self.new_class_name != self.class_name:
            try:
                os.makedirs(os.path.dirname(self.target_video_path), exist_ok=True)
                if os.path.exists(self.video_path):
                    os.rename(self.video_path, self.target_video_path)
                
                # Update internal state for metadata accuracy
                self.video_path = self.target_video_path
                self.metadata_path = self.target_metadata_path
                self.class_name = self.new_class_name
                print(f"📂 Moved recording to final destination: {self.video_path}")
            except Exception as e:
                print(f"✗ Error moving file to final destination: {e}")

    def cancel(self):
        """Stop recording and delete files."""
        if not self.is_active:
            return
        
        self.is_active = False
        if self.writer is not None:
            self.writer.release()
        
        try:
            if os.path.exists(self.video_path):
                os.remove(self.video_path)
            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)
            print(f"🗑️ Cancelled and deleted recording: {self.video_path}")
        except Exception as e:
            print(f"✗ Error deleting cancelled recording: {e}")


class RecordingManager:
    """Manages multiple concurrent video recordings for tracked objects."""
    
    def __init__(self, output_dir: str = "events", fps: int = 20, 
                 codec: str = "mp4v", 
                 buffer_seconds: int = 5,   # Pre-record duration
                 timeout_seconds: int = 5,  # Post-record duration
                 enabled: bool = True):
        
        self.output_dir = output_dir
        self.fps = fps
        self.codec = codec
        self.disappear_timeout = timeout_seconds
        self.enabled = enabled
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Active recordings: {track_id: VideoRecorder}
        self.active_recordings: Dict[int, VideoRecorder] = {}
        # Track last seen time: {track_id: timestamp}
        self.last_seen: Dict[int, float] = {}
        
        # Ring Buffer
        self.buffer_maxlen = int(fps * buffer_seconds)
        self.frame_buffer = deque(maxlen=self.buffer_maxlen)
        
        self.lock = threading.Lock()
        
        print(f"🎬 RecordingManager initialized: {buffer_seconds}s pre-buffer, {timeout_seconds}s timeout")
    
    def update_tracks(self, frame, detections: List[Dict]):
        """
        Update recordings based on current frame detections.
        Expected detections format: [{'track_id': int, 'class': str, ...}]
        """
        if not self.enabled:
            return
        
        current_time = time.time()
        current_track_ids = set()
        
        with self.lock:
            # Always update buffer with the provided frame (ANNOTATED now)
            self.frame_buffer.append(frame)

            for det in detections:
                track_id = det.get('track_id')
                if track_id is None:
                    continue
                
                current_track_ids.add(track_id)
                # Handle key naming differences (tracker might use class_name or class)
                class_name = det.get('class_name') or det.get('class') or 'unknown'
                
                # CRITICAL: IGNORE HUMANS
                if class_name.lower() == "person":
                    continue

                self.last_seen[track_id] = current_time
                
                if track_id not in self.active_recordings:
                    self.start_recording(track_id, class_name, frame.shape[:2])
                else:
                    self.active_recordings[track_id].write_frame(frame)
            
            self.cleanup_disappeared_tracks(current_track_ids, current_time)
    
    def start_recording(self, track_id: int, class_name: str, frame_shape: Tuple[int, int]):
        try:
            # Capture snapshot of buffer history
            pre_event_frames = list(self.frame_buffer)
            
            recorder = VideoRecorder(
                track_id=track_id,
                class_name=class_name,
                frame_shape=frame_shape,
                fps=self.fps,
                codec=self.codec,
                output_dir=self.output_dir,
                pre_buffer_frames=pre_event_frames
            )
            self.active_recordings[track_id] = recorder
        except Exception as e:
            print(f"❌ Error starting recording for track {track_id}: {e}")
    
    def stop_recording(self, track_id: int):
        if track_id in self.active_recordings:
            try:
                self.active_recordings[track_id].stop()
                del self.active_recordings[track_id]
                if track_id in self.last_seen:
                    del self.last_seen[track_id]
            except Exception as e:
                print(f"❌ Error stopping recording for track {track_id}: {e}")
    
    def cancel_recording(self, track_id: int):
        """Cancel and delete recording for track."""
        if track_id in self.active_recordings:
            try:
                self.active_recordings[track_id].cancel()
                del self.active_recordings[track_id]
                if track_id in self.last_seen:
                    del self.last_seen[track_id]
            except Exception as e:
                print(f"❌ Error cancelling recording for track {track_id}: {e}")
    
    def rename_recording(self, track_id: int, new_class_name: str):
        """Update the class name and target path for a recording."""
        with self.lock:
            if track_id in self.active_recordings:
                try:
                    self.active_recordings[track_id].rename(new_class_name, self.codec)
                except Exception as e:
                    print(f"❌ Error renaming recording for track {track_id}: {e}")
    
    def cleanup_disappeared_tracks(self, current_track_ids: set, current_time: float):
        disappeared_tracks = []
        
        for track_id, recorder in self.active_recordings.items():
            if track_id not in current_track_ids:
                last_seen_time = self.last_seen.get(track_id, current_time)
                time_since_seen = current_time - last_seen_time
                
                if time_since_seen < self.disappear_timeout:
                    # Keep recording using most recent frame to prevent video gaps
                    if self.frame_buffer:
                        recorder.write_frame(self.frame_buffer[-1])
                else:
                    disappeared_tracks.append(track_id)
        
        for track_id in disappeared_tracks:
            self.stop_recording(track_id)
            
    def cleanup(self):
        with self.lock:
            for track_id in list(self.active_recordings.keys()):
                self.stop_recording(track_id)
    
    def get_stats(self):
        return {
            "active_recordings_count": len(self.active_recordings),
            "recording_ids": list(self.active_recordings.keys())
        }

# ---------------- FASTAPI SETUP ---------------- #

app = FastAPI(title="Object Tracking Stream", version="2.0.0")

# Global instances
tracker: ObjectTracker = None
camera: CameraCapture = None
recording_manager: RecordingManager = None
db_manager: DatabaseManager = None
tracking_manager: TrackingManager = None
active_connections: Set[WebSocket] = set()
last_vlm_only_time = 0
vlm_only_description = ""

frontend_path = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

@app.on_event("startup")
async def startup_event():
    global tracker, camera, recording_manager, db_manager, tracking_manager
    print("\n" + "="*60)
    print("🚀 Starting Server with Annotated Recording & Tracking Memory")
    print("="*60)
    
    try:
        tracker = ObjectTracker()
        camera = CameraCapture()
        
        # Initialize the new RecordingManager
        recording_manager = RecordingManager(
            output_dir="events",
            fps=Config.TARGET_FPS, 
            codec="avc1",
            buffer_seconds=5,      
            timeout_seconds=5      
        )
        
        # Initialize database manager
        backend_dir = Path(__file__).parent
        db_path = str(backend_dir / "tracking_data.db")
        db_manager = DatabaseManager(db_path=db_path)
        
        # Initialize tracking manager
        tracking_manager = TrackingManager(
            db_manager=db_manager,
            recording_manager=recording_manager,
            enable_ai=True,
            ai_timeout=30.0
        )
        
        print("\n✓ Server initialization complete")
        print(f"✓ Events will be saved to: events/")
        print(f"✓ Tracking database: backend/tracking_data.db")
        print(f"\n📡 Server running at http://{Config.HOST}:{Config.PORT}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Initialization failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    global camera, recording_manager, active_connections
    print("\n🛑 Shutting down server...")
    
    for connection in active_connections.copy():
        try: await connection.close()
        except: pass
    active_connections.clear()
    
    if recording_manager:
        recording_manager.cleanup()
    
    if camera:
        camera.release()
    print("✓ Cleanup complete\n")

@app.get("/")
async def read_root():
    html_file = frontend_path / "index.html"
    return FileResponse(html_file)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "camera_opened": camera.is_opened() if camera else False,
        "active_connections": len(active_connections),
        "recording_stats": recording_manager.get_stats() if recording_manager else {}
    }

# ---------------- REVISED API ENDPOINTS ---------------- #

@app.get("/api/events")
async def list_events() -> List[Dict]:
    """
    List all recorded events.
    """
    if not recording_manager:
        raise HTTPException(status_code=503, detail="Recording manager not initialized")
    
    base_dir = Path(recording_manager.output_dir)
    if not base_dir.exists():
        return []
    
    events = []
    
    for class_dir in base_dir.iterdir():
        if class_dir.is_dir():
            for metadata_file in class_dir.glob("*.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        data = json.load(f)
                    
                    video_filename = metadata_file.with_suffix('.mp4').name
                    video_path = class_dir / video_filename
                    
                    # Try to fetch AI info from database if available
                    ai_info = None
                    if db_manager and "track_id" in data:
                        obj = db_manager.get_tracking_object(data["track_id"])
                        if obj and obj.get("ai_info"):
                            ai_info = obj["ai_info"]

                    events.append({
                        "event_id": data.get("event_id", metadata_file.stem), 
                        "track_id": data.get("track_id"),
                        "class_name": data.get("class_name", class_dir.name),
                        "start_time": data.get("start_time"),
                        "end_time": data.get("end_time"),
                        "duration_seconds": data.get("duration_seconds"),
                        "detected_classes": [data.get("class_name", "unknown")],
                        "has_video": video_path.exists(),
                        "path_ref": f"{class_dir.name}/{video_filename}",
                        "ai_info": ai_info,
                        "frame_snapshot": obj.get("frame_snapshot") if obj else None
                    })
                except Exception as e:
                    print(f"Error reading metadata {metadata_file}: {e}")
                    continue

    events.sort(key=lambda x: x.get("start_time", ""), reverse=True)
    return events

@app.get("/api/events/{event_id}/video")
async def stream_event_video(event_id: str):
    base_dir = Path(recording_manager.output_dir)
    target_file = None
    
    # Find the video file
    for class_dir in base_dir.iterdir():
        if class_dir.is_dir():
            potential_path = class_dir / f"{event_id}.mp4"
            if potential_path.exists():
                target_file = potential_path
                break
    
    if not target_file:
        raise HTTPException(status_code=404, detail="Event video not found")
    
    # --- CHANGE 2: Use FileResponse instead of StreamingResponse ---
    # This allows the browser to seek, pause, and see duration
    return FileResponse(path=target_file, media_type="video/mp4")
    

@app.get("/api/events/{event_id}/metadata")
async def get_event_metadata(event_id: str):
    base_dir = Path(recording_manager.output_dir)
    target_file = None
    
    for class_dir in base_dir.iterdir():
        if class_dir.is_dir():
            potential_path = class_dir / f"{event_id}.json"
            if potential_path.exists():
                target_file = potential_path
                break

    if not target_file:
        raise HTTPException(status_code=404, detail="Metadata not found")
        
    with open(target_file, 'r') as f:
        data = json.load(f)
        
        # Enrich with database AI info
        if db_manager and "track_id" in data:
            obj = db_manager.get_tracking_object(data["track_id"])
            if obj:
                if obj.get("ai_info"):
                    data["ai_info"] = obj["ai_info"]
                data["frame_snapshot"] = obj.get("frame_snapshot")
                
        return data

@app.get("/api/history")
async def get_history() -> List[Dict]:
    """
    Get unique species history from the last 10 minutes.
    """
    if not db_manager:
        return []
    return db_manager.get_unique_species_history(minutes=5)

@app.get("/api/config/vlm_mode")
async def get_vlm_mode():
    return {"mode": ai_broker.get_vlm_mode()}

@app.post("/api/config/vlm_mode")
async def set_vlm_mode(data: Dict[str, str]):
    mode = data.get("mode")
    if not mode:
        raise HTTPException(status_code=400, detail="Mode is required")
    
    if ai_broker.set_vlm_mode(mode):
        # Broadcast to all clients
        for websocket in active_connections:
            try:
                await websocket.send_json({
                    "type": "vlm_mode_updated",
                    "data": {"mode": mode}
                })
            except:
                pass
        return {"status": "success", "mode": mode}
    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Use 'local' or 'cloud'")

@app.get("/api/config/vlm_only")
async def get_vlm_only():
    return {"enabled": Config.VLM_ONLY_ENABLED, "interval": Config.VLM_ONLY_INTERVAL}

@app.post("/api/config/vlm_only")
async def set_vlm_only(data: Dict[str, Any]):
    enabled = data.get("enabled")
    interval = data.get("interval")
    
    if enabled is not None:
        Config.VLM_ONLY_ENABLED = bool(enabled)
    if interval is not None:
        Config.VLM_ONLY_INTERVAL = int(interval)
        
    print(f"🔄 [CONFIG] VLM-Only Mode: {'ENABLED' if Config.VLM_ONLY_ENABLED else 'DISABLED'} (Interval: {Config.VLM_ONLY_INTERVAL}s)")
    
    # Broadcast to all
    for websocket in active_connections:
        try:
            await websocket.send_json({
                "type": "vlm_only_updated",
                "data": {"enabled": Config.VLM_ONLY_ENABLED, "interval": Config.VLM_ONLY_INTERVAL}
            })
        except: pass
        
    return {"status": "success", "enabled": Config.VLM_ONLY_ENABLED, "interval": Config.VLM_ONLY_INTERVAL}

# ---------------- WEBSOCKET STREAM ---------------- #

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global active_connections
    await websocket.accept()
    active_connections.add(websocket)
    
    # Register with tracking manager for lifecycle events
    if tracking_manager:
        tracking_manager.register_websocket(websocket)
        
    print(f"✓ New WebSocket connection (Total: {len(active_connections)})")
    
    try:
        await websocket.send_json({"type": "config", "data": Config.get_info()})
        
        while True:
            if not camera or not camera.is_opened():
                await asyncio.sleep(1)
                continue
            
            success, frame = camera.read()
            if not success:
                await asyncio.sleep(0.1)
                continue
            
            # Track objects - Returns 'annotated_frame' (with boxes) and metadata
            annotated_frame, metadata = tracker.process_frame(frame, track=True)
            
            # --- UPDATE: Send ANNOTATED frame to recorder ---
            detections_list = []
            if 'detections' in metadata:
                detections_list = metadata['detections']
            
            if recording_manager:
                # CHANGED: frame -> annotated_frame
                recording_manager.update_tracks(annotated_frame, detections_list)
            
            # Process tracking lifecycle and AI info collection
            if tracking_manager:
                await tracking_manager.process_detections(frame, detections_list)
            # ------------------------------------------------
            
            # Encode for live stream
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), Config.JPEG_QUALITY]
            success, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
            
            if success:
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                
                rec_stats = recording_manager.get_stats()
                metadata['is_recording'] = rec_stats['active_recordings_count'] > 0
                metadata['recording_info'] = rec_stats
                
                # VLM-ONLY Processing
                global last_vlm_only_time, vlm_only_description
                current_time = time.time()
                if Config.VLM_ONLY_ENABLED and (current_time - last_vlm_only_time >= Config.VLM_ONLY_INTERVAL):
                    # NEW: Only trigger scene analysis if an animal is currently detected
                    if detections_list:
                        last_vlm_only_time = current_time
                        # Run in thread to avoid blocking the stream
                        def run_vlm_only(img_b64):
                            global vlm_only_description
                            try:
                                vlm_only_description = ai_broker.analyze_scene(img_b64)
                                print(f"🤖 [VLM-ONLY] {vlm_only_description}")
                            except Exception as e:
                                print(f"✗ [VLM-ONLY] Error: {e}")
                        
                        threading.Thread(target=run_vlm_only, args=(jpg_as_text,)).start()
                    else:
                        # Reset message when no animal is present to clarify why it's not updating
                        vlm_only_description = "Scene analysis paused (no animal detected)"

                metadata['vlm_only_description'] = vlm_only_description or "Analyzing scene..."

                await websocket.send_json({
                    "type": "frame",
                    "image": jpg_as_text,
                    "metadata": metadata
                })
            
            await asyncio.sleep(1 / Config.TARGET_FPS)
            
    except WebSocketDisconnect:
        print(f"✗ WebSocket disconnected")
    except Exception as e:
        print(f"✗ WebSocket error: {e}")
    finally:
        active_connections.discard(websocket)
        if tracking_manager:
            tracking_manager.unregister_websocket(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        ws_ping_interval=Config.WEBSOCKET_PING_INTERVAL,
        ws_ping_timeout=Config.WEBSOCKET_PING_TIMEOUT
    )