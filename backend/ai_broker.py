import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import Config
import backend.ai_module as cloud_ai
import backend.local_ai_local as local_ai
import backend.florence_ai as florence_ai

def get_wildlife_info(detected_class: str, base64_image: Optional[str] = None, history: Optional[str] = None, mime_type: str = "image/jpeg") -> Any:
    """
    Broker function that routes identification requests to either Local, Cloud, or Florence VLM.
    """
    mode = getattr(Config, "VLM_MODE", "cloud").lower()
    
    if mode == "local":
        print(f"🤖 [BROKER] Routing to LOCAL VLM (Ollama: {Config.LOCAL_AI_MODEL})")
        # Ensure we use the model from config
        return local_ai.get_wildlife_info(detected_class, base64_image, history)
    elif mode == "florence":
        print(f"🖼️ [BROKER] Routing to FLORENCE-2 ({Config.FLORENCE_MODEL})")
        # Florence-2 returns a dict like {'<CAPTION>': '...'}
        result = florence_ai.analyze_base64(base64_image)
        # Wrap it in a compatible format
        caption = result.get("<CAPTION>", "No description available.")
        return {
            "is_animal": True, # Assume true for VLM-only mode or specific detection
            "detected_class": detected_class,
            "commonName": "Florence-2 Analysis",
            "scientificName": None,
            "description": caption,
            "habitat": None,
            "behavior": None,
            "safetyInfo": None,
            "conservationStatus": "Unknown",
            "isDangerous": False
        }
    else:
        print(f"☁️ [BROKER] Routing to CLOUD VLM (OpenRouter)")
        return cloud_ai.get_wildlife_info(detected_class, base64_image, history, mime_type)

def analyze_scene(base64_image: str) -> str:
    """Specifically for VLM-Only mode: describe the whole frame."""
    print(f"🖼️ [BROKER] Analyzing scene with Florence-2...")
    result = florence_ai.analyze_base64(base64_image, prompt="<DETAILED_CAPTION>")
    return result.get("<DETAILED_CAPTION>", "No scene description available.")

def set_vlm_mode(mode: str):
    """Update the VLM mode in runtime."""
    valid_modes = ["local", "cloud", "florence"]
    if mode.lower() in valid_modes:
        Config.VLM_MODE = mode.lower()
        print(f"🔄 [BROKER] VLM Mode switched to: {Config.VLM_MODE.upper()}")
        return True
    return False

def get_vlm_mode():
    """Get the current VLM mode."""
    return getattr(Config, "VLM_MODE", "cloud")
