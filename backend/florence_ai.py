# --- START OF FILE florence_ai.py ---

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os
import io
import base64
from typing import Optional, Dict, Any

# Global model and processor instances
_model = None
_processor = None
_device = "cpu"

def init_model(model_id: str = 'microsoft/Florence-2-base'):
    global _model, _processor, _device
    if _model is not None:
        return

    # Check for CUDA availability
    if torch.cuda.is_available():
        _device = "cuda"
    else:
        _device = "cpu"

    print(f"🖼️ Loading Florence-2 model {model_id} on {_device}...")
    
    # --- HELPER: Load Model with Offline Fallback ---
    def load_with_fallback(local_only=False):
        try:
            kwargs = {
                "trust_remote_code": True,
                "local_files_only": local_only
            }
            
            # Device mapping logic
            if _device == "cpu":
                kwargs["torch_dtype"] = torch.float32
                kwargs["low_cpu_mem_usage"] = False
            else:
                kwargs["torch_dtype"] = torch.float16 if _device == "cuda" else torch.float32
                kwargs["device_map"] = "auto"

            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            
            # Handle CPU specific loading if not using device_map
            if _device == "cpu":
                model = model.to(_device)
            
            return model
        except Exception as e:
            if not local_only:
                # If online load failed, raise to trigger fallback
                raise e 
            else:
                # If offline load failed, it means the model isn't downloaded yet
                print(f"❌ Could not load model locally. You must have internet to download it first.")
                raise e

    try:
        # 1. Try loading normally (checks internet for updates)
        try:
            print("Trying to connect to HuggingFace Hub...")
            _model = load_with_fallback(local_only=False)
        except Exception as e:
            # 2. If internet fails, try loading purely from local cache
            print(f"⚠️ Network error ({str(e)}). Switching to OFFLINE mode...")
            _model = load_with_fallback(local_only=True)
            print("✓ Successfully loaded from local cache.")

        # Ensure weights are tied
        _model.tie_weights()
        _model.eval()

        # Load Processor (Apply same offline logic)
        try:
            _processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        except Exception:
            print("⚠️ Loading processor from local cache...")
            _processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
        
        # Robust fix for 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'
        for cfg in [_model.config, _model.generation_config]:
            if cfg is not None and not hasattr(cfg, "forced_bos_token_id"):
                setattr(cfg, "forced_bos_token_id", None)
                
        print(f"✓ Florence-2 model loaded and patched on {_device}")
    
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load Florence-2.")
        print(f"   If this is your first run, you MUST have an internet connection to download the model.")
        print(f"   Error details: {e}")
        _model = None
        _processor = None
        # Don't raise here, allow the app to start without VLM if it fails
        return

def analyze_image(image: Image.Image, prompt: str = "<CAPTION>") -> Dict[str, Any]:
    global _model, _processor
    
    if _model is None:
        try:
            init_model()
        except:
            return {prompt: "Error: Model failed to load."}
            
    if _model is None:
        return {prompt: "Error: Model not loaded."}

    # Florence-2 tasks: <CAPTION>, <DETAILED_CAPTION>, <MORE_DETAILED_CAPTION>, <OD>
    try:
        inputs = _processor(text=prompt, images=image, return_tensors="pt").to(_device, _model.dtype) # Match model dtype

        with torch.no_grad():
            generated_ids = _model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3
            )
        
        generated_text = _processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = _processor.post_process_generation(
            generated_text, 
            task=prompt, 
            image_size=(image.width, image.height)
        )
        
        return parsed_answer
    except Exception as e:
        print(f"❌ Florence inference error: {e}")
        return {prompt: f"Error during inference: {e}"}

def analyze_base64(base64_str: Optional[str], prompt: str = "<CAPTION>") -> Dict[str, Any]:
    # Handle None or empty string
    if not base64_str:
        return {prompt: "Error: No image data provided"}

    try:
        # Decode base64 to PIL Image
        img_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        return analyze_image(image, prompt)
    except Exception as e:
        print(f"❌ Florence base64 decode error: {e}")
        return {prompt: f"Error decoding image: {e}"}

if __name__ == "__main__":
    # Quick test if run directly
    import time
    # Check for a dummy image for testing
    if os.path.exists("test.jpg"):
        init_model()
        if _model:
            image = Image.open("test.jpg").convert("RGB")
            print("Testing Florence-2...")
            start = time.time()
            result = analyze_image(image)
            print(f"Result: {result}")
            print(f"Time: {time.time() - start:.2f}s")
    else:
        print("Test image 'test.jpg' not found.")