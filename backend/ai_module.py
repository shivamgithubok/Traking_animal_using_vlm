import json
from typing import Optional, Dict, Any
from openai import OpenAI
from pydantic import BaseModel, Field
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import Config

# Initialize cliAent with OpenRouter's base URL
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=Config.OPENROUTER_API_KEY,
)


class Wildlife(BaseModel):
    """Wildlife information model."""
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


def identify_wildlife(detected_class: str, base64_image: Optional[str] = None, history: Optional[str] = None, mime_type: str = "image/jpeg") -> Dict[str, Any]:
    """
    Identify wildlife from YOLO detection class name and return detailed information.
    
    Args:
        detected_class: YOLO detection class name (e.g., "buffalo", "elephant", "bear")
        base64_image: Optional base64-encoded image string
        history: Optional text describing previous sightings for context
        mime_type: MIME type of the image (default: "image/jpeg")
        
    Returns:
        Dictionary containing wildlife information matching Wildlife model structure
    """
    # Build context message
    context_msg = "Identify if the detected object is an animal. "
    if history:
        context_msg += f"Recent sighting history for context: {history}. "
    
    context_msg += "If it IS an animal, provide detailed information including common name, scientific name, description, habitat, behavior, safety information, conservation status (LC, NT, VU, EN, or CR), and whether it is dangerous to humans. If it IS NOT an animal, set 'is_animal' to false."

    # Build user message content
    user_content = [
        {
            "type": "text",
            "text": context_msg
        }
    ]
    
    # Add image if provided
    if base64_image:
        print(f"📸 Including image in AI request for {detected_class} ({len(base64_image)} bytes)")
        data_url = f"data:{mime_type};base64,{base64_image}"
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": data_url
            }
        })
    else:
        print(f"⚠️ No image provided for AI identification of {detected_class}")
    
    response = client.chat.completions.create(
        model="qwen/qwen3-vl-30b-a3b-instruct:nitro",  # Use a model that supports JSON schema
        messages=[
            {
                "role": "system",
                "content": "You are a professional wildlife expert and biologist. Identify the animal in the image and provide scientific data. If the image does not contain a clear animal, set 'is_animal' to false. For 'conservationStatus', choose from [LC, NT, VU, EN, CR] or use 'Unknown' if the status is not readily available for that species. Output strictly valid JSON."
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "wildlife_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "is_animal": {
                            "type": "boolean",
                            "description": "Whether the detected object is an animal"
                        },
                        "commonName": {
                            "type": "string",
                            "description": "Common name of the animal"
                        },
                        "scientificName": {
                            "type": "string",
                            "description": "Scientific name (binomial nomenclature)"
                        },
                        "description": {
                            "type": "string",
                            "description": "Detailed description of the animal"
                        },
                        "habitat": {
                            "type": "string",
                            "description": "Natural habitat and distribution"
                        },
                        "behavior": {
                            "type": "string",
                            "description": "Typical behavior patterns"
                        },
                        "safetyInfo": {
                            "type": "string",
                            "description": "Safety information for human encounters"
                        },
                        "conservationStatus": {
                            "type": "string",
                            "enum": ["LC", "NT", "VU", "EN", "CR", "Unknown"],
                            "description": "IUCN conservation status (use 'Unknown' if not sure)"
                        },
                        "isDangerous": {
                            "type": "boolean",
                            "description": "Whether the animal is dangerous to humans"
                        }
                    },
                    "required": [
                        "is_animal",
                        "commonName",
                        "scientificName",
                        "description",
                        "habitat",
                        "behavior",
                        "safetyInfo",
                        "conservationStatus",
                        "isDangerous"
                    ],
                    "additionalProperties": False
                }
            }
        }
    )

    # Extract and parse the JSON string from the response
    content = response.choices[0].message.content
    wildlife_data = json.loads(content)
    
    # Add detected_class back to the data (it's an input parameter, not from VLM)
    wildlife_data["detected_class"] = detected_class
    
    return wildlife_data


def get_wildlife_info(detected_class: str, base64_image: Optional[str] = None, history: Optional[str] = None, mime_type: str = "image/jpeg") -> Wildlife:
    """
    Get wildlife information and return as Wildlife model instance.
    
    Args:
        detected_class: YOLO detection class name
        base64_image: Optional base64-encoded image string
        history: Optional text describing previous sightings for context
        mime_type: MIME type of the image (default: "image/jpeg")
        
    Returns:
        Wildlife model instance with all information populated
    """
    data = identify_wildlife(detected_class, base64_image, history, mime_type)
    return Wildlife(**data)


if __name__ == "__main__":
    import base64
    import requests
    
    # Test configuration
    test_image_url = "http://cdn.britannica.com/16/234216-050-C66F8665/beagle-hound-dog.jpg"
    detected_class = "dog"
    
    print("=" * 70)
    print("Wildlife Identification System - Test")
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
        print("Querying LLM for wildlife information...")
        wildlife = get_wildlife_info(
            detected_class=detected_class,
            base64_image=base64_image,
            mime_type="image/jpeg"
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
        print(f"Image URL: {wildlife.imageUrl}")
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

