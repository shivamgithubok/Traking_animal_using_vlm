import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import Config
import backend.ai_module as cloud_ai
import backend.local_ai_local as local_ai

def get_wildlife_info(detected_class: str, base64_image: Optional[str] = None, history: Optional[str] = None, mime_type: str = "image/jpeg") -> Any:
    """
    Broker function that routes identification requests to either Local or Cloud VLM.
    """
    mode = getattr(Config, "VLM_MODE", "cloud").lower()
    
    if mode == "local":
        print(f"🤖 [BROKER] Routing to LOCAL VLM (Ollama: {Config.LOCAL_AI_MODEL})")
        # Ensure we use the model from config
        return local_ai.get_wildlife_info(detected_class, base64_image, history)
    else:
        print(f"☁️ [BROKER] Routing to CLOUD VLM (OpenRouter)")
        return cloud_ai.get_wildlife_info(detected_class, base64_image, history, mime_type)

def set_vlm_mode(mode: str):
    """Update the VLM mode in runtime."""
    if mode.lower() in ["local", "cloud"]:
        Config.VLM_MODE = mode.lower()
        print(f"🔄 [BROKER] VLM Mode switched to: {Config.VLM_MODE.upper()}")
        return True
    return False

def get_vlm_mode():
    """Get the current VLM mode."""
    return getattr(Config, "VLM_MODE", "cloud")
