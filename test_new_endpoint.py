#!/usr/bin/env python3
"""
Test the new FLUX.1-dev RunPod endpoint
"""

import requests
import json
import time
import base64
from PIL import Image
import io

# Your new endpoint details
API_KEY = "rpa_C55TBQG7H6FM7G3Q7A6JM7ZJCDKA3I2J3EO0TAH8fxyddo"
ENDPOINT_ID = "qgihilkw9mdlsk"
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def test_health():
    """Test health endpoint"""
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

def submit_image_generation():
    """Submit async image generation request"""
    print("\n🎨 Submitting image generation request...")
    
    payload = {
        "input": {
            "endpoint": "generate",
            "prompt": "A majestic dragon flying over mountains at sunset, fantasy art style, highly detailed, 4k",
            "height": 1024,
            "width": 1024,
            "guidance_scale": 3.5,
            "num_inference_steps": 28,
            "seed": 42,
            "return_base64": True
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/run", headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        if "id" in result:
            job_id = result["id"]
            print(f"✅ Job submitted successfully! Job ID: {job_id}")
            return job_id
        else:
            print(f"❌ No job ID in response: {result}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to submit job: {e}")
        return None

def poll_for_result(job_id):
    """Poll for job completion"""
    print(f"\n⏳ Polling for job {job_id} completion...")
    
    max_attempts = 60  # 5 minutes max (5 second intervals)
    attempt = 0
    
    while attempt < max_attempts:
        try:
            response = requests.get(f"{BASE_URL}/status/{job_id}", headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            status = result.get("status", "UNKNOWN")
            print(f"   Attempt {attempt + 1}: Status = {status}")
            
            if status == "COMPLETED":
                print("✅ Job completed successfully!")
                return result
            elif status == "FAILED":
                print(f"❌ Job failed: {result}")
                return None
            elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                print(f"   Job is {status.lower()}... waiting 5 seconds")
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
    
    print("❌ Timeout waiting for job completion")
    return None

def save_image_from_result(result):
    """Save image from the result"""
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
        
        filename = f"flux_test_{int(time.time())}.png"
        image.save(filename)
        print(f"💾 Image saved as: {filename}")
        
        # Print generation details
        if "parameters_used" in output:
            print(f"📋 Parameters: {json.dumps(output['parameters_used'], indent=2)}")
            
        if "generation_time" in output:
            print(f"⏱️  Generation time: {output['generation_time']:.2f} seconds")
            
        return True
        
    except Exception as e:
        print(f"❌ Error saving image: {e}")
        return False

def main():
    """Run the complete test"""
    print("🚀 Testing FLUX.1-dev RunPod Endpoint")
    print("=" * 50)
    print(f"🎯 Endpoint: {ENDPOINT_ID}")
    print(f"🔗 URL: {BASE_URL}")
    print("=" * 50)
    
    # Test health first
    if not test_health():
        print("❌ Health check failed, aborting test")
        return
    
    # Submit image generation
    job_id = submit_image_generation()
    if not job_id:
        print("❌ Failed to submit job, aborting test")
        return
    
    # Poll for result
    result = poll_for_result(job_id)
    if not result:
        print("❌ Failed to get result, test failed")
        return
    
    # Save the image
    if save_image_from_result(result):
        print("\n🎉 Test completed successfully!")
        print("✅ Health check passed")
        print("✅ Image generation completed")
        print("✅ Image saved locally")
    else:
        print("\n⚠️  Test partially successful - job completed but image save failed")

if __name__ == "__main__":
    main()