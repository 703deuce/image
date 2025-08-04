#!/usr/bin/env python3
"""
Test LoRA training with Replicate-style parameters (1000 steps, higher quality)
"""

import requests
import json
import time
import base64
from PIL import Image
import io

# Your endpoint details
API_KEY = "rpa_C55TBQG7H6FM7G3Q7A6JM7ZJCDKA3I2J3EO0TAH8fxyddo"
ENDPOINT_ID = "qgihilkw9mdlsk"
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def encode_zip_to_base64(zip_path):
    """Convert zip file to base64"""
    print(f"📦 Reading zip file: {zip_path}")
    
    try:
        with open(zip_path, 'rb') as f:
            zip_data = f.read()
        
        zip_base64 = base64.b64encode(zip_data).decode('utf-8')
        
        print(f"✅ Zip encoded successfully")
        print(f"📏 Zip size: {len(zip_data)} bytes, Base64 size: {len(zip_base64)} characters")
        
        return zip_base64
        
    except Exception as e:
        print(f"❌ Error reading zip file: {e}")
        return None

def train_lora_replicate_style():
    """Train LoRA with Replicate's proven parameters"""
    print("🧬 Training LoRA with Replicate-style parameters")
    print("=" * 60)
    
    # Encode the user's zip file
    training_zip = encode_zip_to_base64("test.zip")
    if not training_zip:
        return None
    
    # Replicate's proven parameters - all configurable via API
    payload = {
        "input": {
            "endpoint": "train_lora",
            "training_images_zip": training_zip,
            "output_name": f"replicate_style_lora_{int(time.time())}",
            
            # Core training parameters
            "instance_prompt": "a photo of TOK person",
            "max_train_steps": 1000,  # Replicate's proven default
            "resolution": 768,  # Higher resolution for better quality
            
            # LoRA architecture parameters  
            "lora_rank": 16,  # Higher rank = more capacity (Replicate uses 16)
            "lora_alpha": 32,  # Standard alpha value
            "lora_dropout": 0.1,  # Standard dropout
            
            # Training hyperparameters
            "learning_rate": 1e-4,  # Standard learning rate
            "train_batch_size": 1,  # Batch size
            "gradient_accumulation_steps": 4,  # Gradient accumulation
            "lr_scheduler": "constant",  # Learning rate scheduler
            "lr_warmup_steps": 100,  # Warmup steps
            
            # Metadata
            "user_id": "replicate_test_user"
        }
    }
    
    print("📝 Training Configuration:")
    config = payload["input"]
    print(f"   🎯 Output name: {config['output_name']}")
    print(f"   📝 Instance prompt: {config['instance_prompt']}")
    print(f"   🔢 Training steps: {config['max_train_steps']} (Replicate standard)")
    print(f"   📐 Resolution: {config['resolution']}x{config['resolution']}")
    print(f"   🧬 LoRA rank: {config['lora_rank']} (higher = more capacity)")
    print(f"   📊 Learning rate: {config['learning_rate']}")
    print(f"   🔄 Batch size: {config['train_batch_size']}")
    print(f"   📈 Gradient accumulation: {config['gradient_accumulation_steps']}")
    
    try:
        print("\n📤 Submitting training request...")
        response = requests.post(f"{BASE_URL}/run", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if "id" in result:
            job_id = result["id"]
            print(f"✅ LoRA training job submitted successfully!")
            print(f"🆔 Job ID: {job_id}")
            print(f"⏳ This will take longer (~3-5 minutes) due to higher quality settings")
            return job_id
        else:
            print(f"❌ No job ID in response: {result}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to submit LoRA training job: {e}")
        if hasattr(e, 'response') and e.response:
            try:
                error_detail = e.response.json()
                print(f"📝 Error details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"📝 Raw error response: {e.response.text}")
        return None

def poll_for_training_result(job_id):
    """Poll for LoRA training completion with longer timeout for quality training"""
    print(f"\n⏳ Polling for LoRA training job {job_id} completion...")
    print("   📈 Quality training takes longer - please be patient...")
    
    max_attempts = 300  # 25 minutes max (5 second intervals) - longer for quality training
    attempt = 0
    
    while attempt < max_attempts:
        try:
            response = requests.get(f"{BASE_URL}/status/{job_id}", headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            status = result.get("status", "UNKNOWN")
            
            if attempt % 12 == 0 or status in ["COMPLETED", "FAILED"]:  # Print every minute
                elapsed_minutes = (attempt * 5) / 60
                print(f"   [{time.strftime('%H:%M:%S')}] Attempt {attempt + 1}: Status = {status} (elapsed: {elapsed_minutes:.1f}m)")
            
            if status == "COMPLETED":
                print("✅ LoRA training completed successfully!")
                return result
            elif status == "FAILED":
                print(f"❌ LoRA training failed!")
                print(f"📝 Error details: {json.dumps(result, indent=2)}")
                return None
            elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                if attempt % 24 == 0 and attempt > 0:  # Print progress every 2 minutes
                    elapsed_minutes = (attempt * 5) / 60
                    print(f"   📈 Training progress: {status.lower()}, elapsed time: {elapsed_minutes:.1f} minutes")
                time.sleep(5)
                attempt += 1
            else:
                print(f"   🤔 Unknown status: {status}, continuing to poll...")
                time.sleep(5)
                attempt += 1
                
        except Exception as e:
            print(f"   ❌ Error polling: {e}")
            time.sleep(5)
            attempt += 1
    
    print("❌ Timeout waiting for LoRA training completion")
    return None

def generate_test_image(lora_path):
    """Generate a test image with the trained LoRA"""
    print(f"\n🎨 Testing the trained LoRA...")
    
    payload = {
        "input": {
            "endpoint": "generate",
            "prompt": "a professional portrait of TOK person with natural lighting, high quality, detailed",
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
        print("📤 Submitting test generation...")
        response = requests.post(f"{BASE_URL}/run", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if "id" in result:
            job_id = result["id"]
            print(f"✅ Generation job submitted! Job ID: {job_id}")
            return job_id
        else:
            print(f"❌ No job ID in response: {result}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to submit generation job: {e}")
        return None

def save_generated_image(result, filename_prefix="replicate_style"):
    """Save the generated image"""
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
        filename = f"{filename_prefix}_{int(time.time())}.png"
        
        with open(filename, 'wb') as f:
            f.write(image_data)
            
        print(f"💾 Generated image saved as: {filename}")
        
        # Print generation details
        if "generation_time" in output:
            print(f"⏱️  Generation time: {output['generation_time']:.2f} seconds")
            
        return True
        
    except Exception as e:
        print(f"❌ Error saving image: {e}")
        return False

def main():
    """Run Replicate-style LoRA training test"""
    print("🚀 Replicate-Style LoRA Training Test")
    print("=" * 60)
    print("🎯 Using 1000 steps, 768 resolution, rank 16 for quality results")
    print("📦 Training data: test.zip")
    print("=" * 60)
    
    # Start LoRA training with Replicate parameters
    training_job_id = train_lora_replicate_style()
    if not training_job_id:
        print("❌ Failed to submit training, aborting test")
        return
    
    # Wait for training to complete (longer timeout for quality)
    training_result = poll_for_training_result(training_job_id)
    if not training_result:
        print("❌ LoRA training failed or timed out")
        return
    
    # Extract LoRA information
    training_output = training_result.get("output", {})
    if not training_output.get("success", False):
        print(f"❌ LoRA training was not successful: {training_output}")
        return
    
    lora_path = training_output.get("lora_path")
    if not lora_path:
        print("❌ No LoRA path in training result")
        return
    
    print(f"\n🎉 Replicate-Style LoRA Training Success!")
    print("=" * 60)
    print(f"📁 LoRA saved at: {lora_path}")
    print(f"⏱️  Training time: {training_output.get('training_time', 'unknown')} seconds")
    print(f"🔧 Final config: {json.dumps(training_output.get('config_used', {}), indent=2)}")
    
    # Test generation with the new high-quality LoRA
    print(f"\n🎨 Testing generation with the new LoRA...")
    generation_job_id = generate_test_image(lora_path)
    if not generation_job_id:
        print("❌ Failed to submit generation job")
        return
    
    # Wait for generation to complete
    generation_result = poll_for_training_result(generation_job_id)
    if not generation_result:
        print("❌ Generation failed or timed out")
        return
    
    # Save the result
    if save_generated_image(generation_result, "replicate_quality"):
        print(f"\n🏆 Replicate-Style LoRA Test Complete!")
        print("=" * 60)
        print("✅ High-quality training (1000 steps) completed")
        print("✅ LoRA saved to persistent storage")
        print("✅ Test generation completed")
        print("✅ Image saved locally")
        print(f"🧬 Your high-quality LoRA: {lora_path}")
        print("\n💡 This LoRA should show much better resemblance to your training images!")
    else:
        print("\n⚠️  Training successful, but image save failed")

if __name__ == "__main__":
    main()