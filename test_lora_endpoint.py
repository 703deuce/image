#!/usr/bin/env python3
"""
Test the LoRA training and inference capabilities of the FLUX.1-dev RunPod endpoint
"""

import requests
import json
import time
import base64
import zipfile
import io
from PIL import Image, ImageDraw
import os

# Your endpoint details
API_KEY = "rpa_C55TBQG7H6FM7G3Q7A6JM7ZJCDKA3I2J3EO0TAH8fxyddo"
ENDPOINT_ID = "qgihilkw9mdlsk"
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def create_sample_training_images():
    """Create sample training images for testing LoRA training"""
    print("🎨 Creating sample training images...")
    
    # Create a temp directory for images
    os.makedirs("temp_training_images", exist_ok=True)
    
    # Create 5 simple colored squares as training images
    colors = ['red', 'blue', 'green', 'yellow', 'purple']
    
    for i, color in enumerate(colors):
        # Create a 512x512 image with a colored square
        img = Image.new('RGB', (512, 512), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw a colored square in the center
        square_size = 200
        x1 = (512 - square_size) // 2
        y1 = (512 - square_size) // 2
        x2 = x1 + square_size
        y2 = y1 + square_size
        
        draw.rectangle([x1, y1, x2, y2], fill=color)
        
        # Add some text
        draw.text((10, 10), f"Training image {i+1} - {color} square", fill='black')
        
        # Save the image
        img.save(f"temp_training_images/training_{i+1}_{color}.png")
    
    print(f"✅ Created {len(colors)} training images")
    return "temp_training_images"

def create_training_zip(image_dir):
    """Create a zip file from training images and encode as base64"""
    print("📦 Creating training data zip...")
    
    # Create zip file in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename in os.listdir(image_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(image_dir, filename)
                zip_file.write(file_path, filename)
    
    # Get zip data and encode as base64
    zip_data = zip_buffer.getvalue()
    zip_base64 = base64.b64encode(zip_data).decode('utf-8')
    
    print(f"✅ Created zip file with {len(os.listdir(image_dir))} images")
    print(f"📏 Zip size: {len(zip_data)} bytes, Base64 size: {len(zip_base64)} characters")
    
    return zip_base64

def test_lora_training():
    """Test LoRA training endpoint"""
    print("\n🚀 Testing LoRA training...")
    
    # Create sample training data
    image_dir = create_sample_training_images()
    training_zip = create_training_zip(image_dir)
    
    # Prepare training request
    payload = {
        "input": {
            "endpoint": "train_lora",
            "training_images_zip": training_zip,
            "output_name": f"test_lora_{int(time.time())}",
            "instance_prompt": "a photo of TOK colored square",
            "resolution": 512,
            "max_train_steps": 100,  # Reduced for faster testing
            "learning_rate": 1e-4,
            "lora_rank": 4,
            "user_id": "test_user"
        }
    }
    
    try:
        print("📤 Submitting LoRA training request...")
        response = requests.post(f"{BASE_URL}/run", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if "id" in result:
            job_id = result["id"]
            print(f"✅ LoRA training job submitted! Job ID: {job_id}")
            return job_id
        else:
            print(f"❌ No job ID in response: {result}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to submit LoRA training job: {e}")
        return None
    finally:
        # Clean up temp files
        import shutil
        if os.path.exists(image_dir):
            shutil.rmtree(image_dir)

def poll_for_training_result(job_id):
    """Poll for LoRA training completion"""
    print(f"\n⏳ Polling for LoRA training job {job_id} completion...")
    
    max_attempts = 120  # 10 minutes max (5 second intervals)
    attempt = 0
    
    while attempt < max_attempts:
        try:
            response = requests.get(f"{BASE_URL}/status/{job_id}", headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            status = result.get("status", "UNKNOWN")
            print(f"   Attempt {attempt + 1}: Status = {status}")
            
            if status == "COMPLETED":
                print("✅ LoRA training completed successfully!")
                return result
            elif status == "FAILED":
                print(f"❌ LoRA training failed: {result}")
                return None
            elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                print(f"   Training is {status.lower()}... waiting 5 seconds")
                time.sleep(5)
                attempt += 1
            else:
                print(f"   Unknown status: {status}, continuing to poll...")
                time.sleep(5)
                attempt += 1
                
        except Exception as e:
            print(f"   ❌ Error polling: {e}")
            time.sleep(5)
            attempt += 1
    
    print("❌ Timeout waiting for LoRA training completion")
    return None

def test_lora_inference(lora_path):
    """Test image generation with trained LoRA"""
    print(f"\n🎨 Testing inference with LoRA: {lora_path}")
    
    payload = {
        "input": {
            "endpoint": "generate",
            "prompt": "a beautiful TOK colored square in a fantasy landscape, highly detailed, 4k",
            "lora_path": lora_path,
            "height": 1024,
            "width": 1024,
            "guidance_scale": 3.5,
            "num_inference_steps": 28,
            "seed": 42,
            "return_base64": True
        }
    }
    
    try:
        print("📤 Submitting LoRA inference request...")
        response = requests.post(f"{BASE_URL}/run", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if "id" in result:
            job_id = result["id"]
            print(f"✅ LoRA inference job submitted! Job ID: {job_id}")
            return job_id
        else:
            print(f"❌ No job ID in response: {result}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to submit LoRA inference job: {e}")
        return None

def save_lora_result(result, prefix="lora_test"):
    """Save the LoRA inference result"""
    try:
        output = result.get("output", {})
        
        if not output.get("success", False):
            print(f"❌ Generation was not successful: {output}")
            return False
            
        if "image_base64" not in output:
            print("❌ No image_base64 in result")
            return False
            
        # Decode and save image
        image_data = base64.b64decode(output["image_base64"])
        image = Image.open(io.BytesIO(image_data))
        
        filename = f"{prefix}_{int(time.time())}.png"
        image.save(filename)
        print(f"💾 LoRA inference result saved as: {filename}")
        
        # Print generation details
        if "parameters_used" in output:
            print(f"📋 Parameters: {json.dumps(output['parameters_used'], indent=2)}")
            
        if "generation_time" in output:
            print(f"⏱️  Generation time: {output['generation_time']:.2f} seconds")
            
        return True
        
    except Exception as e:
        print(f"❌ Error saving LoRA result: {e}")
        return False

def test_health():
    """Test health endpoint before running LoRA tests"""
    print("🏥 Testing health endpoint...")
    
    payload = {
        "input": {
            "endpoint": "health"
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/runsync", headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        print(f"✅ Health check: {json.dumps(result, indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def main():
    """Run the complete LoRA test"""
    print("🚀 Testing FLUX.1-dev LoRA Training and Inference")
    print("=" * 60)
    print(f"🎯 Endpoint: {ENDPOINT_ID}")
    print(f"🔗 URL: {BASE_URL}")
    print("=" * 60)
    
    # Test health first
    if not test_health():
        print("❌ Health check failed, aborting LoRA tests")
        return
    
    # Test LoRA training
    training_job_id = test_lora_training()
    if not training_job_id:
        print("❌ Failed to submit LoRA training job, aborting")
        return
    
    # Wait for training to complete
    training_result = poll_for_training_result(training_job_id)
    if not training_result:
        print("❌ LoRA training failed or timed out")
        return
    
    # Extract LoRA path from training result
    training_output = training_result.get("output", {})
    if not training_output.get("success", False):
        print(f"❌ LoRA training was not successful: {training_output}")
        return
    
    lora_path = training_output.get("lora_path")
    if not lora_path:
        print("❌ No LoRA path in training result")
        return
    
    print(f"✅ LoRA trained successfully!")
    print(f"📁 LoRA path: {lora_path}")
    print(f"⏱️  Training time: {training_output.get('training_time', 'unknown')} seconds")
    
    # Test inference with the trained LoRA
    inference_job_id = test_lora_inference(lora_path)
    if not inference_job_id:
        print("❌ Failed to submit LoRA inference job")
        return
    
    # Wait for inference to complete
    inference_result = poll_for_training_result(inference_job_id)  # Reuse the same polling function
    if not inference_result:
        print("❌ LoRA inference failed or timed out")
        return
    
    # Save the result
    if save_lora_result(inference_result, "lora_inference"):
        print("\n🎉 LoRA test completed successfully!")
        print("✅ Health check passed")
        print("✅ LoRA training completed")
        print("✅ LoRA inference completed")
        print("✅ Image saved locally")
        print(f"🧬 LoRA model saved at: {lora_path}")
    else:
        print("\n⚠️  LoRA test partially successful - training completed but inference save failed")

if __name__ == "__main__":
    main()