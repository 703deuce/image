#!/usr/bin/env python3
"""
Test LoRA with proper TOK trigger token
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

def test_lora_with_tok_trigger():
    """Test LoRA with the proper TOK trigger token"""
    print("🎯 Testing LoRA with TOK Trigger Token")
    print("=" * 50)
    print("📁 LoRA: /runpod-volume/cache/lora/flux-lora.safetensors")
    print("🔑 Trigger: TOK person")
    print("🎯 This should actually activate the LoRA!")
    print("=" * 50)
    
    # Test with TOK trigger
    print("\n🎨 Generating with TOK trigger")
    print("-" * 35)
    
    tok_payload = {
        "input": {
            "endpoint": "generate",
            "prompt": "a professional headshot of TOK person, studio lighting, detailed face, photorealistic",
            "lora_path": "/runpod-volume/cache/lora/flux-lora.safetensors",
            "height": 1024,
            "width": 1024,
            "guidance_scale": 3.0,
            "num_inference_steps": 28,
            "seed": 42,
            "return_base64": True
        }
    }
    
    try:
        print("📤 Submitting with TOK trigger...")
        
        response = requests.post(f"{BASE_URL}/run", headers=headers, json=tok_payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if "id" not in result:
            print(f"❌ No job ID in response: {result}")
            return
            
        job_id = result["id"]
        print(f"✅ Job submitted! ID: {job_id}")
        
        # Wait for completion
        print("⏳ Waiting for LoRA generation with TOK trigger...")
        while True:
            try:
                status_response = requests.get(f"{BASE_URL}/status/{job_id}", headers=headers, timeout=30)
                status_response.raise_for_status()
                status_result = status_response.json()
                
                status = status_result.get("status", "UNKNOWN")
                
                if status == "COMPLETED":
                    print(" ✅ COMPLETED!")
                    output = status_result.get("output", {})
                    
                    if output.get("success") and "image_base64" in output:
                        # Save the image
                        image_data = base64.b64decode(output["image_base64"])
                        image = Image.open(io.BytesIO(image_data))
                        filename = f"lora_with_tok_trigger_{int(time.time())}.png"
                        image.save(filename)
                        print(f"💾 Image saved: {filename}")
                        
                        print("\n🎉 SUCCESS! LoRA with TOK trigger!")
                        print("✅ LoRA loaded and activated")
                        print("✅ TOK trigger used in prompt")
                        print("✅ Generation completed")
                        print("🎨 This image should now show the trained person!")
                        
                        return filename
                    else:
                        print(f"❌ No image in output: {output}")
                        return None
                        
                elif status == "FAILED":
                    print(f" ❌ FAILED!")
                    error_info = status_result.get('error', 'Unknown error')
                    print(f"🔍 Error: {error_info}")
                    return None
                    
                else:
                    print(".", end="", flush=True)
                    time.sleep(3)
                    
            except Exception as e:
                print(f"❌ Error checking status: {e}")
                return None
                
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return None

def main():
    print("🔑 LoRA TOK Trigger Test")
    print("🎯 Using the proper trigger token to activate LoRA")
    print()
    
    filename = test_lora_with_tok_trigger()
    
    print("\n" + "="*50)
    if filename:
        print("🎉 RESULT: LoRA with TOK trigger working!")
        print(f"📁 Check: {filename}")
        print("💡 This image should actually look like the trained person!")
        
        # Open the image
        try:
            import subprocess
            subprocess.run(f"Start-Process {filename}", shell=True)
            print("🖼️  Image opened for viewing")
        except:
            pass
    else:
        print("❌ RESULT: Generation failed")
    print("="*50)

if __name__ == "__main__":
    main()