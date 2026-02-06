import json
import base64
import requests
from typing import Optional, Dict, Any
import ollama
from pydantic import BaseModel, Field
import sys
from pathlib import Path

# Add parent directory to path to import config if needed
sys.path.append(str(Path(__file__).parent.parent))
from config import Config

class Wildlife(BaseModel):
    """Wildlife information model mirroring ai_module.py for compatibility."""
    id: Optional[str] = Field(default=None, description="Unique identifier")
    detected_class: str = Field(description="YOLO detection class name (e.g., buffalo, elephant)")
    is_animal: bool = Field(description="Whether the detected object is an animal")
    commonName: Optional[str] = None
    scientificName: Optional[str] = None
    description: Optional[str] = None
    habitat: Optional[str] = None
    behavior: Optional[str] = None
    safetyInfo: Optional[str] = None
    conservationStatus: Optional[str] = Field(default=None, description="LC, NT, VU, EN, or CR")
    isDangerous: Optional[bool] = Field(default=None)
    imageUrl: Optional[str] = None

def identify_wildlife(detected_class: str, base64_image: Optional[str] = None, history: Optional[str] = None, model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Identify wildlife locally using Ollama and return detailed information.
    
    Args:
        detected_class: YOLO detection class name
        base64_image: Optional base64-encoded image string
        history: Optional text describing previous sightings for context
        model_name: Ollama model to use (defaults to Config.LOCAL_AI_MODEL)
        
    Returns:
        Dictionary containing wildlife information matching Wildlife model structure
    """
    if model_name is None:
        model_name = getattr(Config, "LOCAL_AI_MODEL", "llava:7b")

    # Build prompt
    prompt = f"Identify the object: {detected_class}. "
    if history:
        prompt += f"Recent sighting history for context: {history}. "
    
    prompt += """
    Based on the image (if provided) and the detected class, provide detailed information about the animal. 
    If the image does not contain a clear animal, set 'is_animal' to false.
    
    Output strictly valid JSON with the following keys:
    - 'is_animal': boolean
    - 'commonName': string
    - 'scientificName': string
    - 'description': string
    - 'habitat': string
    - 'behavior': string
    - 'safetyInfo': string
    - 'conservationStatus': string (one of: LC, NT, VU, EN, CR, Unknown)
    - 'isDangerous': boolean
    """

    try:
        images = []
        if base64_image:
            # Ollama accepts base64 strings or bytes in the 'images' list
            images = [base64_image]
            print(f"📸 Calling local Ollama ({model_name}) for {detected_class}...")

        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            images=images,
            format='json'
        )

        content = response['response']
        print(f"✓ Ollama responded for {detected_class}")
        
        wildlife_data = json.loads(content)
        
        # Add detected_class back to the data
        wildlife_data["detected_class"] = detected_class
        
        return wildlife_data

    except Exception as e:
        print(f"✗ Ollama Error: {e}")
        # Return a fallback response
        return {
            "is_animal": True, # Assume true if detected by YOLO
            "detected_class": detected_class,
            "commonName": detected_class.capitalize(),
            "scientificName": "Unknown",
            "description": f"Local AI processing failed: {str(e)}",
            "habitat": "Unknown",
            "behavior": "Unknown",
            "safetyInfo": "Use caution.",
            "conservationStatus": "Unknown",
            "isDangerous": False
        }

def get_wildlife_info(detected_class: str, base64_image: Optional[str] = None, history: Optional[str] = None, model_name: Optional[str] = None) -> Wildlife:
    """
    Get wildlife information and return as Wildlife model instance.
    """
    data = identify_wildlife(detected_class, base64_image, history, model_name)
    return Wildlife(**data)

if __name__ == "__main__":
    # Test configuration
    test_image_url = "http://cdn.britannica.com/16/234216-050-C66F8665/beagle-hound-dog.jpg"
    detected_class = "dog"
    
    print("=" * 70)
    print("Local Wildlife Identification System (Ollama) - Test")
    print("=" * 70)
    print(f"Detected Class: {detected_class}")
    print(f"Image URL: {test_image_url}\n")
    
    try:
        # Download and encode image to base64
        print("Downloading image...")
        response = requests.get(test_image_url, timeout=30)
        response.raise_for_status()
        base64_image = base64.b64encode(response.content).decode('utf-8')
        print(f"✓ Image encoded ({len(base64_image)} characters)\n")
        
        # Get wildlife information
        print("Querying local LLM for wildlife information...")
        wildlife = get_wildlife_info(
            detected_class=detected_class,
            base64_image=base64_image
        )
        wildlife.imageUrl = test_image_url
        print("✓ Information retrieved\n")
        
        # Display results
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Detected Class: {wildlife.detected_class}")
        print(f"Common Name: {wildlife.commonName}")
        print(f"Scientific Name: {wildlife.scientificName}")
        print(f"\nDescription:\n  {wildlife.description}")
        print(f"\nHabitat:\n  {wildlife.habitat}")
        print(f"\nBehavior:\n  {wildlife.behavior}")
        print(f"\nSafety Information:\n  {wildlife.safetyInfo}")
        print(f"\nConservation Status: {wildlife.conservationStatus}")
        print(f"Is Dangerous: {wildlife.isDangerous}")
        print("=" * 70)
        
        # JSON output
        print("\nJSON Output:")
        print(json.dumps(wildlife.model_dump(), indent=2))
        
    except requests.RequestException as e:
        print(f"✗ Network error: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()