#!/usr/bin/env python3
"""
Test LoRA with "her" to generate the correct female subject
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

def generate_with_her_pronoun():
    """Generate with 'her' pronoun to get the female subject"""
    print("👩 Testing LoRA with 'her' for Female Subject")
    print("=" * 55)
    print("📁 LoRA: /runpod-volume/cache/lora/flux-lora.safetensors")
    print("🔑 Subject: TOK her (female from training)")
    print("🎯 Should generate the WOMAN from training images!")
    print("=" * 55)
    
    # Generate with "her" to specify female
    payload = {
        "input": {
            "endpoint": "generate",
            "prompt": "professional headshot photo of TOK, her beautiful face, studio lighting, high quality, detailed",
            "lora_path": "/runpod-volume/cache/lora/flux-lora.safetensors",
            "height": 1024,
            "width": 1024,
            "guidance_scale": 3.0,
            "num_inference_steps": 28,
            "seed": 789012,  # Different seed
            "return_base64": True
        }
    }
    
    try:
        print("📤 Submitting generation with female-specific prompt...")
        print(f"📝 Prompt: {payload['input']['prompt']}")
        
        response = requests.post(f"{BASE_URL}/run", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if "id" not in result:
            print(f"❌ No job ID in response: {result}")
            return None
            
        job_id = result["id"]
        print(f"✅ Job submitted! ID: {job_id}")
        
        # Wait for completion
        print("⏳ Generating female subject from LoRA...")
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
                        
                        timestamp = int(time.time())
                        filename = f"tok_female_subject_{timestamp}.png"
                        image.save(filename)
                        
                        print(f"💾 Image saved: {filename}")
                        print("\n👩 Female Subject Generated!")
                        print("✅ LoRA loaded and applied")
                        print("✅ Female-specific prompt used")
                        print("✅ Should show the WOMAN from training!")
                        
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
    print("👩 LoRA Female Subject Test")
    print("🎯 Generate the WOMAN from your training images")
    print()
    
    filename = generate_with_her_pronoun()
    
    print("\n" + "="*60)
    if filename:
        print("🎉 SUCCESS! Female subject generated!")
        print(f"📁 File: {filename}")
        print("💡 This should look like the WOMAN in your training images!")
        
        # Open the image
        try:
            import subprocess
            subprocess.run(f"Start-Process {filename}", shell=True)
            print("🖼️  Image opened for viewing")
        except:
            pass
    else:
        print("❌ Generation failed")
    print("="*60)

if __name__ == "__main__":
    main()