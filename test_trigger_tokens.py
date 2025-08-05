#!/usr/bin/env python3
"""
Test different trigger tokens that might be used in the LoRA
Common FLUX LoRA trigger tokens: TOK, PERSON, subject_name, ohwx, etc.
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

def test_trigger_tokens():
    """Test different possible trigger tokens"""
    print("🔑 Testing Different Trigger Tokens")
    print("=" * 60)
    print("🎯 Testing various trigger tokens that might activate LoRA")
    print("=" * 60)
    
    # Different trigger token patterns commonly used
    trigger_tests = [
        {
            "name": "TOK person (standard)",
            "prompt": "professional headshot of TOK person, studio lighting, detailed face",
            "token": "TOK person"
        },
        {
            "name": "TOK woman (gender specific)",
            "prompt": "professional headshot of TOK woman, studio lighting, detailed face", 
            "token": "TOK woman"
        },
        {
            "name": "TOK (just the token)",
            "prompt": "professional headshot of TOK, studio lighting, detailed face",
            "token": "TOK"
        },
        {
            "name": "ohwx person (another common token)",
            "prompt": "professional headshot of ohwx person, studio lighting, detailed face",
            "token": "ohwx person"
        },
        {
            "name": "subject token",
            "prompt": "professional headshot of subject, studio lighting, detailed face",
            "token": "subject"
        },
        {
            "name": "No trigger (baseline)",
            "prompt": "professional headshot of a person, studio lighting, detailed face",
            "token": "none"
        }
    ]
    
    results = []
    
    for i, test in enumerate(trigger_tests, 1):
        print(f"\n🧪 TEST {i}: {test['name']}")
        print(f"📝 Prompt: {test['prompt']}")
        print(f"🔑 Token: {test['token']}")
        print("-" * 50)
        
        payload = {
            "input": {
                "endpoint": "generate",
                "prompt": test["prompt"],
                "lora_path": "/runpod-volume/cache/lora/flux-lora.safetensors",
                "height": 768,  # Smaller for faster testing
                "width": 768,
                "guidance_scale": 3.0,
                "num_inference_steps": 20,  # Faster for testing
                "seed": 100 + i,  # Different seed for each test
                "return_base64": True
            }
        }
        
        try:
            print("📤 Submitting generation...")
            
            response = requests.post(f"{BASE_URL}/run", headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            if "id" not in result:
                print(f"❌ No job ID: {result}")
                continue
                
            job_id = result["id"]
            print(f"✅ Job submitted: {job_id}")
            
            # Wait for completion
            print("⏳ Generating...", end="")
            final_result = wait_for_completion(job_id)
            
            if final_result:
                filename = save_image(final_result, f"trigger_test_{i}_{test['token'].replace(' ', '_')}")
                if filename:
                    print(f"💾 Saved: {filename}")
                    results.append({
                        "test": test['name'],
                        "token": test['token'],
                        "filename": filename,
                        "success": True
                    })
                else:
                    print("❌ Save failed")
                    results.append({
                        "test": test['name'],
                        "token": test['token'],
                        "success": False
                    })
            else:
                print("❌ Generation failed")
                results.append({
                    "test": test['name'],
                    "token": test['token'],
                    "success": False
                })
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append({
                "test": test['name'],
                "token": test['token'],
                "error": str(e),
                "success": False
            })
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 TRIGGER TOKEN TEST SUMMARY")
    print("="*60)
    
    for result in results:
        status = "✅" if result.get("success") else "❌"
        print(f"{status} {result['token']:<15} | {result['test']}")
        if result.get("filename"):
            print(f"   📁 {result['filename']}")
    
    print("\n💡 Compare the images to see which trigger token works!")
    print("🔍 Look for the one that shows your trained person vs generic faces")
    
    return results

def wait_for_completion(job_id):
    """Wait for job completion"""
    while True:
        try:
            response = requests.get(f"{BASE_URL}/status/{job_id}", headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            status = result.get("status", "UNKNOWN")
            
            if status == "COMPLETED":
                print(" ✅")
                return result
            elif status == "FAILED":
                error = result.get('error', 'Unknown error')
                print(f" ❌ FAILED: {error}")
                return None
            else:
                print(".", end="", flush=True)
                time.sleep(2)
                
        except Exception as e:
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
            print(f"❌ No image in output: {output}")
            return None
            
    except Exception as e:
        print(f"❌ Save error: {e}")
        return None

if __name__ == "__main__":
    print("🔑 TRIGGER TOKEN DISCOVERY TEST")
    print("🎯 Finding the correct trigger token for your LoRA")
    print()
    
    results = test_trigger_tokens()
    
    print(f"\n🏆 TEST COMPLETE!")
    print("📋 Check all generated images to find which trigger works!")