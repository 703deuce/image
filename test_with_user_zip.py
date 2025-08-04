#!/usr/bin/env python3
"""
Test LoRA training with the user's test.zip file
"""

import requests
import json
import time
import base64

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

def test_lora_training_with_user_zip():
    """Test LoRA training with user's test.zip"""
    print("🚀 Testing LoRA training with test.zip")
    print("=" * 50)
    
    # Encode the user's zip file
    training_zip = encode_zip_to_base64("test.zip")
    if not training_zip:
        return None
    
    # Prepare training request
    payload = {
        "input": {
            "endpoint": "train_lora",
            "training_images_zip": training_zip,
            "output_name": f"user_test_lora_{int(time.time())}",
            "instance_prompt": "a photo of TOK person",
            "resolution": 512,
            "max_train_steps": 300,  # Reduced for faster testing
            "learning_rate": 1e-4,
            "lora_rank": 4,
            "user_id": "test_user_real_data"
        }
    }
    
    try:
        print("📤 Submitting LoRA training request...")
        print(f"📝 Training config:")
        print(f"   • Output name: {payload['input']['output_name']}")
        print(f"   • Instance prompt: {payload['input']['instance_prompt']}")
        print(f"   • Training steps: {payload['input']['max_train_steps']}")
        print(f"   • Resolution: {payload['input']['resolution']}")
        
        response = requests.post(f"{BASE_URL}/run", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if "id" in result:
            job_id = result["id"]
            print(f"✅ LoRA training job submitted successfully!")
            print(f"🆔 Job ID: {job_id}")
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
    """Poll for LoRA training completion"""
    print(f"\n⏳ Polling for LoRA training job {job_id} completion...")
    print("   This may take 5-15 minutes depending on your data...")
    
    max_attempts = 180  # 15 minutes max (5 second intervals)
    attempt = 0
    
    while attempt < max_attempts:
        try:
            response = requests.get(f"{BASE_URL}/status/{job_id}", headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            status = result.get("status", "UNKNOWN")
            
            if attempt % 10 == 0 or status in ["COMPLETED", "FAILED"]:  # Print every 50 seconds or on completion
                print(f"   [{time.strftime('%H:%M:%S')}] Attempt {attempt + 1}: Status = {status}")
            
            if status == "COMPLETED":
                print("✅ LoRA training completed successfully!")
                return result
            elif status == "FAILED":
                print(f"❌ LoRA training failed!")
                print(f"📝 Error details: {json.dumps(result, indent=2)}")
                return None
            elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                if attempt % 20 == 0:  # Print progress every 100 seconds
                    print(f"   📈 Training progress: {status.lower()}, elapsed time: {attempt * 5} seconds")
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

def test_inference_with_trained_lora(lora_path):
    """Test generating an image with the newly trained LoRA"""
    print(f"\n🎨 Testing inference with trained LoRA...")
    print(f"📁 LoRA path: {lora_path}")
    
    payload = {
        "input": {
            "endpoint": "generate",
            "prompt": "a professional headshot of TOK person in business attire, high quality, studio lighting",
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
            print(f"✅ Inference job submitted! Job ID: {job_id}")
            return job_id
        else:
            print(f"❌ No job ID in response: {result}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to submit inference job: {e}")
        return None

def save_generated_image(result, filename_prefix="lora_generated"):
    """Save the generated image from LoRA inference"""
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
        if "parameters_used" in output:
            print(f"📋 Generation parameters:")
            params = output['parameters_used']
            print(f"   • Prompt: {params.get('prompt', 'N/A')[:60]}...")
            print(f"   • Size: {params.get('width', 'N/A')}x{params.get('height', 'N/A')}")
            print(f"   • Steps: {params.get('num_inference_steps', 'N/A')}")
            print(f"   • Guidance: {params.get('guidance_scale', 'N/A')}")
            
        if "generation_time" in output:
            print(f"⏱️  Generation time: {output['generation_time']:.2f} seconds")
            
        return True
        
    except Exception as e:
        print(f"❌ Error saving image: {e}")
        return False

def main():
    """Run the complete test with user's zip file"""
    print("🧬 LoRA Training Test with User Data")
    print("=" * 60)
    print(f"🎯 Endpoint: {ENDPOINT_ID}")
    print(f"📦 Training data: test.zip")
    print("=" * 60)
    
    # Start LoRA training
    training_job_id = test_lora_training_with_user_zip()
    if not training_job_id:
        print("❌ Failed to submit LoRA training, aborting test")
        return
    
    # Wait for training to complete
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
    
    print(f"\n🎉 LoRA Training Successful!")
    print(f"📁 LoRA saved at: {lora_path}")
    print(f"⏱️  Training time: {training_output.get('training_time', 'unknown')} seconds")
    print(f"🔧 Config used: {json.dumps(training_output.get('config_used', {}), indent=2)}")
    
    # Test inference with the trained LoRA
    inference_job_id = test_inference_with_trained_lora(lora_path)
    if not inference_job_id:
        print("❌ Failed to submit inference job")
        return
    
    # Wait for inference to complete
    inference_result = poll_for_training_result(inference_job_id)  # Reuse same polling function
    if not inference_result:
        print("❌ LoRA inference failed or timed out")
        return
    
    # Save the generated image
    if save_generated_image(inference_result, "lora_test_result"):
        print("\n🏆 Complete LoRA Test Success!")
        print("=" * 40)
        print("✅ Training completed successfully")
        print("✅ LoRA model saved to persistent storage")
        print("✅ Inference completed successfully")
        print("✅ Generated image saved locally")
        print(f"🧬 Your LoRA: {lora_path}")
        print("\n🎯 You can now use this LoRA for future generations!")
    else:
        print("\n⚠️  Training successful, but image save failed")

if __name__ == "__main__":
    main()