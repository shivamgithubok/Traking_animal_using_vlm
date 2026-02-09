import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(os.getcwd())

try:
    from backend import florence_ai
    print("Testing Florence-2 initialization...")
    florence_ai.init_model()
    print("✅ Florence-2 initialized successfully!")
    
    # Optional: test with dummy image
    from PIL import Image
    import numpy as np
    dummy_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    print("Testing Florence-2 inference with dummy image...")
    result = florence_ai.analyze_image(dummy_image)
    print(f"✅ Result: {result}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
