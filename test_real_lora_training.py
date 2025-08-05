#!/usr/bin/env python3
"""
Test real LoRA training with test.zip - no more dummy data!
"""

import requests
import json
import time
import base64
from PIL import Image
import io
import zipfile
import os

# Your endpoint details
API_KEY = "rpa_C55TBQG7H6FM7G3Q7A6JM7ZJCDKA3I2J3EO0TAH8fxyddo"
ENDPOINT_ID = "qgihilkw9mdlsk"
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def zip_to_base64(zip_path):
    """Convert zip file to base64"""
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    
    with open(zip_path, 'rb') as f:
        zip_data = f.read()
    
    return base64.b64encode(zip_data).decode('utf-8')

def train_real_lora():
    """Train real LoRA with test.zip images"""
    print("🚀 REAL LoRA Training Test")
    print("=" * 60)
    print("📁 Using: test.zip (your actual training images)")
    print("🎯 Training: REAL FLUX LoRA (no more dummy data!)")
    print("⚙️  Config: Optimized for 12 images, 1000 steps")
    print("=" * 60)
    
    # Check if test.zip exists
    zip_path = "test.zip"
    if not os.path.exists(zip_path):
        print(f"❌ test.zip not found in current directory")
        print("📋 Please ensure test.zip is in the same folder as this script")
        return None
    
    print(f"📦 Found test.zip ({os.path.getsize(zip_path)} bytes)")
    
    # Convert zip to base64
    print("🔄 Converting test.zip to base64...")
    try:
        training_images_zip = zip_to_base64(zip_path)
        print(f"✅ Converted to base64 ({len(training_images_zip)} chars)")
    except Exception as e:
        print(f"❌ Failed to convert zip: {e}")
        return None
    
    # Training configuration - optimized for single person with 12 images
    timestamp = int(time.time())
    output_name = f"real_flux_lora_{timestamp}"
    
    payload = {
        "input": {
            "endpoint": "train_lora",
            "training_images_zip": training_images_zip,
            "output_name": output_name,
            "instance_prompt": "a photo of TOK person",  # Your trigger token
            
            # Optimal settings for 12 images (from your specs)
            "resolution": 768,  # Good for face consistency
            "max_train_steps": 1000,  # Perfect for 12 images
            "learning_rate": 5e-5,  # Sweet spot for LoRA
            "lora_rank": 16,  # Good for facial identity
            "lora_alpha": 16,  # Usually same as rank
            "lora_dropout": 0.1,
            "train_batch_size": 1,  # Limited by VRAM
            "mixed_precision": "fp16"  # Save VRAM
        }
    }
    
    try:
        print("📤 Submitting REAL LoRA training job...")
        print(f"📝 Output name: {output_name}")
        print(f"🎛️  Steps: {payload['input']['max_train_steps']}")
        print(f"🎛️  Learning rate: {payload['input']['learning_rate']}")
        print(f"🎛️  Resolution: {payload['input']['resolution']}")
        print(f"🎛️  LoRA rank: {payload['input']['lora_rank']}")
        
        response = requests.post(f"{BASE_URL}/run", headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        if "id" not in result:
            print(f"❌ No job ID: {result}")
            return None
            
        job_id = result["id"]
        print(f"✅ Training job submitted! ID: {job_id}")
        
        # Wait for completion (this will take a while for real training!)
        print("⏳ Training REAL LoRA (this will take several minutes)...")
        print("📊 Progress will be logged during training...")
        
        final_result = wait_for_completion(job_id)
        
        if final_result:
            output = final_result.get("output", {})
            
            if output.get("success"):
                lora_path = output.get("lora_path")
                training_time = output.get("training_time", 0)
                
                print(f"\n🎉 REAL LoRA TRAINING COMPLETE!")
                print("=" * 60)
                print(f"✅ LoRA saved to: {lora_path}")
                print(f"⏱️  Training time: {training_time:.2f} seconds")
                print(f"📁 Location: /runpod-volume/cache/lora/")
                print("🎯 Ready for inference testing!")
                print("=" * 60)
                
                return {
                    "success": True,
                    "lora_path": lora_path,
                    "output_name": output_name,
                    "training_time": training_time
                }
            else:
                print(f"❌ Training failed: {output}")
                return None
        else:
            print("❌ Training job failed")
            return None
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def test_trained_lora(lora_path):
    """Test the newly trained LoRA with the working prompts"""
    print(f"\n🧪 Testing Trained LoRA: {lora_path}")
    print("=" * 50)
    
    # Test with both prompts that work with the current LoRA
    test_prompts = [
        "Professional headshot of TOK 4k",
        "TOK her in a trendy fashion event outfit"
    ]
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🎨 Test {i}: {prompt}")
        print("-" * 40)
        
        payload = {
            "input": {
                "endpoint": "generate",
                "prompt": prompt,
                "lora_path": lora_path,
                "height": 1024,
                "width": 1024,
                "guidance_scale": 3.0,
                "num_inference_steps": 28,
                "seed": 12345 + i,
                "return_base64": True
            }
        }
        
        try:
            response = requests.post(f"{BASE_URL}/run", headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            if "id" not in result:
                print(f"❌ No job ID: {result}")
                continue
                
            job_id = result["id"]
            print(f"✅ Generation submitted: {job_id}")
            
            print("⏳ Generating with new LoRA...", end="")
            final_result = wait_for_completion(job_id, verbose=False)
            
            if final_result:
                filename = save_image(final_result, f"new_lora_test_{i}")
                if filename:
                    print(f" ✅ Saved: {filename}")
                    results.append(filename)
                else:
                    print(" ❌ Save failed")
            else:
                print(" ❌ Generation failed")
                
        except Exception as e:
            print(f"❌ Test {i} failed: {e}")
    
    if results:
        print(f"\n🏆 TESTING COMPLETE!")
        print("=" * 40)
        print("📁 Generated images:")
        for filename in results:
            print(f"   📸 {filename}")
        print("\n👀 IMPORTANT: Compare these with previous results!")
        print("✅ If they show YOUR person, real training worked!")
        print("❌ If generic people, training needs adjustment")
    
    return results

def wait_for_completion(job_id, verbose=True):
    """Wait for job completion"""
    if verbose:
        print("⏳ Processing", end="")
    
    while True:
        try:
            response = requests.get(f"{BASE_URL}/status/{job_id}", headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            status = result.get("status", "UNKNOWN")
            
            if status == "COMPLETED":
                if verbose:
                    print(" ✅ COMPLETED!")
                return result
            elif status == "FAILED":
                error = result.get('error', 'Unknown error')
                if verbose:
                    print(f" ❌ FAILED: {error}")
                return None
            else:
                if verbose:
                    print(".", end="", flush=True)
                time.sleep(5)  # Longer interval for training
                
        except Exception as e:
            if verbose:
                print(f" ❌ Error: {e}")
            return None

def save_image(result, prefix):
    """Save the generated image"""
    try:
        output = result.get("output", {})
        
        if output.get("success") and "image_base64" in output:
            image_data = base64.b64decode(output["image_base64"])
            image = Image.open(io.BytesIO(image_data))
            
            timestamp = int(time.time())
            filename = f"{prefix}_{timestamp}.png"
            image.save(filename)
            
            return filename
        else:
            return None
            
    except Exception as e:
        print(f"❌ Save error: {e}")
        return None

if __name__ == "__main__":
    print("🔥 REAL FLUX LoRA TRAINING TEST")
    print("🎯 Training with your actual images from test.zip")
    print("⚙️  Using optimized settings for 12 images")
    print()
    
    # Step 1: Train real LoRA
    training_result = train_real_lora()
    
    if training_result and training_result.get("success"):
        lora_path = training_result["lora_path"]
        
        # Step 2: Test the trained LoRA
        test_results = test_trained_lora(lora_path)
        
        print(f"\n🚀 FULL WORKFLOW COMPLETE!")
        print("=" * 60)
        print(f"✅ Trained LoRA: {lora_path}")
        if test_results:
            print(f"✅ Generated {len(test_results)} test images")
            print("🎯 Check if they show YOUR trained person!")
        print("=" * 60)
    else:
        print("\n❌ Training failed - check logs for details")