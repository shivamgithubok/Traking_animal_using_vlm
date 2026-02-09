import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os
import time
# Set device to CPU explicitly
device = "cpu"
# Using base model for faster CPU testing as requested/implied by user edit
model_id = 'microsoft/Florence-2-base'

print(f"Loading model {model_id} on {device}...")

# Load model and processor
# Explicitly use float32 for CPU and trust_remote_code for Florence-2
# We use torch.float32 because many CPU kernels are optimized for it
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True,
    torch_dtype=torch.float32
).to(device).eval()

processor = AutoProcessor.from_pretrained(
    model_id, 
    trust_remote_code=True
)

def test_vlm(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print(f"Processing image: {image_path}")
    start_time = time.time()
    # Florence-2 works best with RGB images
    image = Image.open(image_path).convert("RGB")
    
    # Florence-2 tasks: <CAPTION>, <DETAILED_CAPTION>, <MORE_DETAILED_CAPTION>, <OD>
    prompt = "<CAPTION>"
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float32)

    print("Generating response (this may take a minute on CPU)...")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3
        )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))

    print("\n--- Model Output ---")
    print(parsed_answer)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print("--------------------")

if __name__ == "__main__":
    # Using the local animal.jpg found in the workspace
    test_image = r"c:\Users\Asus\python\LZ\The_VLM\african-wildlife-sample.jpg"
    test_vlm(test_image)
