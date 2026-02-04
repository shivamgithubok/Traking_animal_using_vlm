The Video Codec: You are using mp4v (MPEG-4 Part 2). Modern browsers (Chrome, Edge, Firefox) do not support this codec, even if the file extension is .mp4. They require H.264 (avc1).
The API Response: You are using StreamingResponse. While this works for live streams, it breaks playback for static files because it doesn't support "Range Requests" (which allow the browser to seek, rewind, and determine the file duration).





# 🎯 Real-Time Object Tracking Stream

A modern web-based real-time object tracking system using **YOLO**, **FastAPI**, and **WebSocket** technology. Stream video from your webcam to a beautiful web interface with live object detection and tracking.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![YOLO](https://img.shields.io/badge/YOLO-v8-orange.svg)

## ✨ Features

- 🔴 **Real-time video streaming** from webcam to browser via WebSocket
- 🎯 **YOLO object detection & tracking** with ByteTrack
- 📊 **Live statistics** (FPS, frame count, latency, object count)
- 🎨 **Modern dark UI** with glassmorphism effects
- 📱 **Responsive design** works on desktop and mobile
- 🔄 **Auto-reconnect** with exponential backoff
- 🎮 **Interactive controls** (connect/disconnect)
- 📈 **Detection metadata** with confidence scores and track IDs

## 📋 Requirements

- Python 3.8 or higher
- Webcam
- Modern web browser (Chrome, Firefox, Edge, Safari)

## 🚀 Installation

1. **Clone or navigate to the directory:**
   ```bash
   cd streaming-server
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLO model (if not already present):**
   The `yolov8n.pt` model will be automatically downloaded on first run.

## 🎮 Usage

1. **Start the server:**
   ```bash
   python backend/main.py
   ```

   Or with custom configuration:
   ```bash
   # Custom port
   PORT=8080 python backend/main.py
   
   # Custom camera
   CAMERA_INDEX=1 python backend/main.py
   
   # Custom YOLO model
   YOLO_MODEL=yolov8s.pt python backend/main.py
   ```

2. **Open your browser:**
   Navigate to `http://localhost:8000`

3. **Start streaming:**
   The stream will start automatically. Use the controls to connect/disconnect.

## ⚙️ Configuration

Configure the server using environment variables or by editing `config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `CAMERA_INDEX` | `0` | Camera device index |
| `CAMERA_WIDTH` | `1280` | Camera resolution width |
| `CAMERA_HEIGHT` | `720` | Camera resolution height |
| `TARGET_FPS` | `30` | Target frames per second |
| `YOLO_MODEL` | `yolov8n.pt` | YOLO model file (n/s/m/l/x) |
| `TRACKER_CONFIG` | `bytetrack.yaml` | Tracker configuration |
| `JPEG_QUALITY` | `80` | JPEG compression quality (1-100) |
| `CONFIDENCE_THRESHOLD` | `0.25` | Detection confidence threshold |

### Example with environment variables:
```bash
# Windows (PowerShell)
$env:PORT=8080; $env:YOLO_MODEL="yolov8s.pt"; python backend/main.py

# macOS/Linux
PORT=8080 YOLO_MODEL=yolov8s.pt python backend/main.py
```

## 📁 Project Structure

```
streaming-server/
├── backend/
│   ├── main.py          # FastAPI server with WebSocket
│   └── tracker.py       # YOLO object tracking logic
├── frontend/
│   ├── index.html       # Web interface
│   ├── style.css        # Styling
│   └── app.js           # WebSocket client
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore patterns
└── README.md           # This file
```

## 🔌 API Endpoints

### HTTP Endpoints

- `GET /` - Serve the web interface
- `GET /health` - Health check endpoint
  ```json
  {
    "status": "healthy",
    "camera_opened": true,
    "active_connections": 1,
    "config": {...}
  }
  ```

### WebSocket Endpoint

- `WS /ws` - Real-time video stream

**Message Types:**

1. **Config Message** (Server → Client):
   ```json
   {
     "type": "config",
     "data": {
       "host": "0.0.0.0",
       "port": 8000,
       "camera_resolution": "1280x720",
       "target_fps": 30,
       "yolo_model": "yolov8n.pt",
       "jpeg_quality": 80
     }
   }
   ```

2. **Frame Message** (Server → Client):
   ```json
   {
     "type": "frame",
     "image": "base64_encoded_jpeg",
     "metadata": {
       "num_detections": 2,
       "detections": [
         {
           "track_id": 1,
           "class_id": 0,
           "class_name": "person",
           "confidence": 0.95,
           "bbox": [100, 150, 300, 450]
         }
       ],
       "detected_classes": ["person"],
       "fps": 28.5,
       "frame_count": 1234
     }
   }
   ```

3. **Error Message** (Server → Client):
   ```json
   {
     "type": "error",
     "message": "Error description"
   }
   ```

## 🎨 Features Walkthrough

### Real-time Detection
- Objects are detected and tracked in real-time
- Bounding boxes and labels displayed on video
- Track IDs maintained across frames

### Statistics Panel
- **FPS**: Actual processing frame rate
- **Frames**: Total frames processed
- **Latency**: Client-side frame rendering latency
- **Objects**: Number of objects currently detected

### Detections Panel
- Shows all detected object classes
- Displays count per class
- Shows confidence scores
- Track IDs when available

### Auto-reconnect
- Automatically reconnects on connection loss
- Exponential backoff retry strategy
- Visual feedback during reconnection

## 🛠️ Troubleshooting

### Camera not opening
```bash
# List available cameras (Linux/macOS)
ls /dev/video*

# Try different camera index
CAMERA_INDEX=1 python backend/main.py
```

### Low FPS
- Use a smaller YOLO model (`yolov8n.pt`)
- Reduce camera resolution
- Lower JPEG quality
- Check CPU/GPU usage

### WebSocket connection fails
- Check firewall settings
- Verify port is not in use
- Try a different port
- Check browser console for errors

### YOLO model download issues
```bash
# Manually download model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## 📝 Development

### Running in development mode with auto-reload:
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Testing WebSocket connection:
```bash
# Using wscat
wscat -c ws://localhost:8000/ws
```

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Object detection
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [OpenCV](https://opencv.org/) - Computer vision

---

**Made with ❤️ using YOLO + FastAPI + WebSocket**
"# Traking_animal_using_vlm" 
