#!/usr/bin/env python3
"""
Comprehensive LoRA debugging based on the provided checklist
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

def test_comprehensive_lora_debug():
    """Test all the potential LoRA issues systematically"""
    print("🔍 Comprehensive LoRA Debugging")
    print("=" * 60)
    print("🎯 Testing all potential causes of 'LoRA loads but no effect'")
    print("=" * 60)
    
    tests = [
        {
            "name": "1. HIGH GUIDANCE (7.0) + Correct Token",
            "prompt": "professional headshot photo of TOK woman, studio lighting, detailed face",
            "guidance_scale": 7.0,
            "reason": "Higher guidance makes LoRA effects more visible"
        },
        {
            "name": "2. VERY HIGH GUIDANCE (10.0) + Female Token", 
            "prompt": "portrait of TOK woman, beautiful face, high quality",
            "guidance_scale": 10.0,
            "reason": "Maximum guidance to force LoRA activation"
        },
        {
            "name": "3. LOWER STEPS + High Guidance",
            "prompt": "TOK woman headshot, professional lighting",
            "guidance_scale": 8.0,
            "num_inference_steps": 15,
            "reason": "Fewer steps with strong guidance"
        },
        {
            "name": "4. Original Training Prompt Style",
            "prompt": "a photo of TOK person",
            "guidance_scale": 6.0,
            "reason": "Match the original training prompt format"
        }
    ]
    
    results = []
    
    for i, test in enumerate(tests, 1):
        print(f"\n🧪 TEST {i}: {test['name']}")
        print(f"📝 Prompt: {test['prompt']}")
        print(f"🎛️  Guidance: {test['guidance_scale']}")
        print(f"💡 Reason: {test['reason']}")
        print("-" * 50)
        
        payload = {
            "input": {
                "endpoint": "generate",
                "prompt": test["prompt"],
                "lora_path": "/runpod-volume/cache/lora/flux-lora.safetensors",
                "height": 768,  # Smaller for faster testing
                "width": 768,
                "guidance_scale": test["guidance_scale"],
                "num_inference_steps": test.get("num_inference_steps", 20),
                "seed": 111 + i,  # Different seed for each test
                "return_base64": True
            }
        }
        
        try:
            response = requests.post(f"{BASE_URL}/run", headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            result = response.json()
            
            if "id" not in result:
                print(f"❌ No job ID: {result}")
                continue
                
            job_id = result["id"]
            print(f"✅ Job submitted: {job_id}")
            
            # Wait for completion
            final_result = wait_for_completion(job_id)
            
            if final_result:
                filename = save_test_image(final_result, f"debug_test_{i}")
                if filename:
                    print(f"💾 Saved: {filename}")
                    results.append({
                        "test": test["name"],
                        "filename": filename,
                        "success": True
                    })
                else:
                    results.append({
                        "test": test["name"], 
                        "success": False,
                        "error": "Failed to save image"
                    })
            else:
                results.append({
                    "test": test["name"],
                    "success": False,
                    "error": "Generation failed"
                })
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append({
                "test": test["name"],
                "success": False,
                "error": str(e)
            })
    
    # Results summary
    print("\n🏁 DEBUGGING RESULTS SUMMARY")
    print("=" * 60)
    for i, result in enumerate(results, 1):
        status = "✅" if result["success"] else "❌"
        print(f"{status} Test {i}: {result['test']}")
        if result["success"]:
            print(f"   📁 File: {result['filename']}")
        else:
            print(f"   ❌ Error: {result.get('error', 'Unknown')}")
    
    print("\n💡 ANALYSIS INSTRUCTIONS:")
    print("1. Open all generated images")
    print("2. Compare them to your training data (woman in sparkly dress)")
    print("3. Look for images that show the correct person")
    print("4. Note which guidance scale worked best")
    print("5. If none work, we have a deeper adapter/compatibility issue")
    
    return results

def wait_for_completion(job_id):
    """Wait for job completion"""
    print("⏳ Waiting...", end="")
    
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

def save_test_image(result, prefix):
    """Save test image"""
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

def main():
    print("🚨 COMPREHENSIVE LORA DEBUGGING")
    print("🎯 Based on community best practices for FLUX.1-dev")
    print("📋 Testing: Guidance scale, prompts, and LoRA activation")
    print()
    
    results = test_comprehensive_lora_debug()
    
    print(f"\n🏆 Generated {len([r for r in results if r['success']])} test images")
    print("🔍 Check which ones actually show the woman from your training data!")

if __name__ == "__main__":
    main()