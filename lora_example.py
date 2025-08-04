#!/usr/bin/env python3
"""
Simple example of how to use the LoRA training and inference API

This example shows:
1. How to prepare training images
2. How to train a LoRA
3. How to use the trained LoRA for inference
"""

import requests
import json
import time
import base64
import zipfile
import io
from PIL import Image

# Configuration
API_KEY = "rpa_C55TBQG7H6FM7G3Q7A6JM7ZJCDKA3I2J3EO0TAH8fxyddo"  # Your API key
ENDPOINT_ID = "qgihilkw9mdlsk"  # Your endpoint ID
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def images_to_zip_base64(image_paths):
    """
    Convert a list of image file paths to a base64-encoded zip file
    
    Args:
        image_paths: List of paths to image files
        
    Returns:
        str: Base64 encoded zip file
    """
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, image_path in enumerate(image_paths):
            # Add each image to the zip with a simple filename
            filename = f"image_{i+1}.{image_path.split('.')[-1]}"
            zip_file.write(image_path, filename)
    
    # Convert to base64
    zip_data = zip_buffer.getvalue()
    return base64.b64encode(zip_data).decode('utf-8')

def train_lora(training_images_zip, output_name, instance_prompt="a photo of TOK person"):
    """
    Train a LoRA model
    
    Args:
        training_images_zip: Base64 encoded zip file containing training images
        output_name: Name for the trained LoRA model
        instance_prompt: Prompt template for training (use TOK as placeholder)
        
    Returns:
        dict: Training result
    """
    payload = {
        "input": {
            "endpoint": "train_lora",
            "training_images_zip": training_images_zip,
            "output_name": output_name,
            "instance_prompt": instance_prompt,
            "resolution": 512,
            "max_train_steps": 500,
            "learning_rate": 1e-4,
            "lora_rank": 4,
            "lora_alpha": 32
        }
    }
    
    # Submit async training job
    response = requests.post(f"{BASE_URL}/run", headers=headers, json=payload)
    response.raise_for_status()
    
    job_data = response.json()
    job_id = job_data["id"]
    
    print(f"Training started! Job ID: {job_id}")
    print("This may take several minutes...")
    
    # Poll for completion
    while True:
        status_response = requests.get(f"{BASE_URL}/status/{job_id}", headers=headers)
        status_data = status_response.json()
        
        status = status_data.get("status", "UNKNOWN")
        print(f"Status: {status}")
        
        if status == "COMPLETED":
            return status_data["output"]
        elif status == "FAILED":
            raise Exception(f"Training failed: {status_data}")
        
        time.sleep(10)  # Wait 10 seconds before checking again

def generate_with_lora(prompt, lora_path, **kwargs):
    """
    Generate an image using a trained LoRA
    
    Args:
        prompt: Text prompt for generation
        lora_path: Path to the trained LoRA file
        **kwargs: Additional generation parameters
        
    Returns:
        dict: Generation result
    """
    payload = {
        "input": {
            "endpoint": "generate",
            "prompt": prompt,
            "lora_path": lora_path,
            "height": kwargs.get("height", 1024),
            "width": kwargs.get("width", 1024),
            "guidance_scale": kwargs.get("guidance_scale", 3.5),
            "num_inference_steps": kwargs.get("num_inference_steps", 28),
            "seed": kwargs.get("seed", None),
            "return_base64": True
        }
    }
    
    # Submit async generation job
    response = requests.post(f"{BASE_URL}/run", headers=headers, json=payload)
    response.raise_for_status()
    
    job_data = response.json()
    job_id = job_data["id"]
    
    print(f"Generation started! Job ID: {job_id}")
    
    # Poll for completion
    while True:
        status_response = requests.get(f"{BASE_URL}/status/{job_id}", headers=headers)
        status_data = status_response.json()
        
        status = status_data.get("status", "UNKNOWN")
        print(f"Status: {status}")
        
        if status == "COMPLETED":
            return status_data["output"]
        elif status == "FAILED":
            raise Exception(f"Generation failed: {status_data}")
        
        time.sleep(5)  # Wait 5 seconds before checking again

def save_image_from_base64(base64_string, filename):
    """Save a base64 encoded image to file"""
    image_data = base64.b64decode(base64_string)
    with open(filename, 'wb') as f:
        f.write(image_data)
    print(f"Image saved as: {filename}")

def example_workflow():
    """
    Example workflow showing how to:
    1. Train a LoRA on your images
    2. Generate new images with the trained LoRA
    """
    
    print("🚀 LoRA Training and Inference Example")
    print("=" * 50)
    
    # Step 1: Prepare your training images
    # In this example, you would replace these with your actual image paths
    training_image_paths = [
        # "path/to/your/image1.jpg",
        # "path/to/your/image2.jpg", 
        # "path/to/your/image3.jpg",
        # etc...
    ]
    
    if not training_image_paths:
        print("❌ Please add your training image paths to the training_image_paths list")
        print("   You need at least 3-5 images of the person/object you want to train on")
        return
    
    print(f"📸 Preparing {len(training_image_paths)} training images...")
    
    # Convert images to zip
    training_zip = images_to_zip_base64(training_image_paths)
    
    # Step 2: Train the LoRA
    print("🧠 Starting LoRA training...")
    
    try:
        training_result = train_lora(
            training_images_zip=training_zip,
            output_name=f"my_person_lora_{int(time.time())}",
            instance_prompt="a photo of TOK person"  # TOK will be replaced with your subject
        )
        
        print("✅ Training completed!")
        print(f"📁 LoRA saved at: {training_result['lora_path']}")
        print(f"⏱️  Training took: {training_result['training_time']:.2f} seconds")
        
        lora_path = training_result['lora_path']
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Step 3: Generate images with the trained LoRA
    print("\n🎨 Generating images with trained LoRA...")
    
    prompts = [
        "a professional headshot of TOK person in a business suit",
        "TOK person as a superhero, comic book style",
        "a portrait of TOK person in the style of a Renaissance painting"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\n📝 Generating: {prompt}")
        
        try:
            result = generate_with_lora(
                prompt=prompt,
                lora_path=lora_path,
                seed=42 + i  # Different seed for each image
            )
            
            if result.get("success"):
                filename = f"lora_generated_{i+1}.png"
                save_image_from_base64(result["image_base64"], filename)
            else:
                print(f"❌ Generation failed: {result}")
                
        except Exception as e:
            print(f"❌ Generation failed: {e}")
    
    print("\n🎉 Workflow completed!")
    print("Check your directory for the generated images.")

if __name__ == "__main__":
    example_workflow()