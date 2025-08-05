#!/usr/bin/env python3
"""
Test the new FLUX.1-Krea-dev endpoint with custom VAE
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

def test_krea_vs_regular():
    """Compare FLUX.1-Krea-dev vs regular FLUX.1-dev with same prompt"""
    print("🎯 FLUX.1-Krea-dev vs FLUX.1-dev Comparison")
    print("=" * 60)
    print("🔄 Testing both endpoints with identical settings")
    print("🎨 Should see quality differences due to Krea model + custom VAE")
    print("=" * 60)
    
    test_prompt = "Professional headshot of TOK 4k"
    
    # Test regular FLUX.1-dev
    print("\n1️⃣ REGULAR FLUX.1-dev Generation")
    print("-" * 40)
    
    regular_payload = {
        "input": {
            "endpoint": "generate",  # Regular endpoint
            "prompt": test_prompt,
            "lora_path": "/runpod-volume/cache/lora/flux-lora.safetensors",
            "height": 1024,
            "width": 1024,
            "guidance_scale": 3.0,
            "num_inference_steps": 28,
            "seed": 54321,  # Fixed seed for comparison
            "return_base64": True
        }
    }
    
    regular_result = run_generation(regular_payload, "regular_flux")
    
    # Test FLUX.1-Krea-dev
    print("\n2️⃣ FLUX.1-Krea-dev Generation")
    print("-" * 40)
    
    krea_payload = {
        "input": {
            "endpoint": "generate_krea",  # New Krea endpoint
            "prompt": test_prompt,
            "lora_path": "/runpod-volume/cache/lora/flux-lora.safetensors",
            "height": 1024,
            "width": 1024,
            "guidance_scale": 3.0,
            "num_inference_steps": 28,
            "seed": 54321,  # Same seed for comparison
            "return_base64": True
        }
    }
    
    krea_result = run_generation(krea_payload, "krea_flux")
    
    # Summary
    print("\n🏁 COMPARISON COMPLETE!")
    print("=" * 60)
    if regular_result and krea_result:
        print("✅ Both models generated successfully!")
        print(f"📁 Regular FLUX: {regular_result}")
        print(f"📁 Krea FLUX: {krea_result}")
        print("\n🔍 Compare the images:")
        print("   • Color quality and saturation")
        print("   • Fine details and textures") 
        print("   • Overall image fidelity")
        print("   • Krea should show enhanced quality!")
    else:
        if not regular_result:
            print("❌ Regular FLUX generation failed")
        if not krea_result:
            print("❌ Krea FLUX generation failed")
    
    return regular_result, krea_result

def test_krea_fashion_outfit():
    """Test Krea with the fashion outfit prompt"""
    print("\n👗 FLUX.1-Krea-dev Fashion Test")
    print("=" * 50)
    print("🎯 Prompt: 'TOK her in a trendy fashion event outfit'")
    print("🎨 Testing Krea model quality with fashion generation")
    print("=" * 50)
    
    payload = {
        "input": {
            "endpoint": "generate_krea",
            "prompt": "TOK her in a trendy fashion event outfit",
            "lora_path": "/runpod-volume/cache/lora/flux-lora.safetensors",
            "height": 1024,
            "width": 1024,
            "guidance_scale": 3.0,
            "num_inference_steps": 28,
            "seed": 98765,
            "return_base64": True
        }
    }
    
    result = run_generation(payload, "krea_fashion")
    
    if result:
        print(f"\n🎉 Krea Fashion Generation Complete!")
        print(f"📁 Saved: {result}")
        print("👗 Should show enhanced fashion detail quality!")
    
    return result

def run_generation(payload, prefix):
    """Run a generation and save the result"""
    try:
        print(f"📤 Submitting to {payload['input']['endpoint']}...")
        
        response = requests.post(f"{BASE_URL}/run", headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        if "id" not in result:
            print(f"❌ No job ID: {result}")
            return None
            
        job_id = result["id"]
        print(f"✅ Job submitted: {job_id}")
        
        # Wait for completion
        print("⏳ Generating...", end="")
        final_result = wait_for_completion(job_id)
        
        if final_result:
            filename = save_image(final_result, prefix)
            if filename:
                print(f" ✅ Saved: {filename}")
                return filename
            else:
                print(" ❌ Save failed")
        else:
            print(" ❌ Generation failed")
        
        return None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def wait_for_completion(job_id):
    """Wait for job completion"""
    while True:
        try:
            response = requests.get(f"{BASE_URL}/status/{job_id}", headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            status = result.get("status", "UNKNOWN")
            
            if status == "COMPLETED":
                return result
            elif status == "FAILED":
                error = result.get('error', 'Unknown error')
                print(f" ❌ FAILED: {error}")
                return None
            else:
                print(".", end="", flush=True)
                time.sleep(3)
                
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
            
            # Log model info if available
            model_used = output.get("model_used", "unknown")
            print(f" [{model_used}]", end="")
            
            return filename
        else:
            print(f"❌ No image in output: {output}")
            return None
            
    except Exception as e:
        print(f"❌ Save error: {e}")
        return None

def test_health_check():
    """Test the health endpoint to see both models"""
    print("\n🏥 Health Check - Both Models")
    print("=" * 40)
    
    health_payload = {
        "input": {
            "endpoint": "health"
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/run", headers=headers, json=health_payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if "id" in result:
            job_id = result["id"]
            status_response = requests.get(f"{BASE_URL}/status/{job_id}", headers=headers, timeout=30)
            status_response.raise_for_status()
            status_result = status_response.json()
            
            if status_result.get("status") == "COMPLETED":
                health = status_result.get("output", {})
                print(f"✅ API Status: {health.get('status', 'unknown')}")
                print(f"📊 Regular FLUX loaded: {health.get('model_loaded', False)}")
                print(f"🎨 Krea FLUX loaded: {health.get('krea_model_loaded', False)}")
                return health
        
        print("❌ Health check failed")
        return None
        
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return None

if __name__ == "__main__":
    print("🚀 FLUX.1-Krea-dev Endpoint Test")
    print("🎯 Testing new Krea model with custom VAE")
    print("🔄 Comparing against regular FLUX.1-dev")
    print()
    
    # Test health first
    health = test_health_check()
    
    # Test both models
    regular_file, krea_file = test_krea_vs_regular()
    
    # Test Krea with fashion
    fashion_file = test_krea_fashion_outfit()
    
    print("\n🎊 ALL TESTS COMPLETE!")
    print("=" * 60)
    print("📁 Generated Files:")
    if regular_file:
        print(f"   🔸 Regular FLUX: {regular_file}")
    if krea_file:
        print(f"   🔸 Krea FLUX: {krea_file}")
    if fashion_file:
        print(f"   🔸 Krea Fashion: {fashion_file}")
    
    print("\n🎯 What to Look For:")
    print("   • Krea should have better color saturation")
    print("   • Enhanced detail and texture quality")
    print("   • Improved overall image fidelity")
    print("   • Same LoRA person, but higher quality!")
    print("=" * 60)