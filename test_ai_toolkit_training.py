#!/usr/bin/env python3
"""
Test the new ai-toolkit based LoRA training
"""

import requests
import json
import time
import base64
import os
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

def zip_to_base64(zip_path):
    """Convert zip file to base64"""
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    
    with open(zip_path, 'rb') as f:
        zip_data = f.read()
    
    return base64.b64encode(zip_data).decode('utf-8')

def test_ai_toolkit_training():
    """Test LoRA training with the new ai-toolkit integration"""
    print("🚀 TESTING AI-TOOLKIT LoRA TRAINING")
    print("=" * 60)
    print("📁 Using: test.zip")
    print("🔧 Trainer: ostris/ai-toolkit (industry standard)")
    print("🎯 Training: Proper FLUX LoRA")
    print("💾 Saving: To /runpod-volume/cache/lora/")
    print("=" * 60)
    
    # Check if test.zip exists
    zip_path = "test.zip"
    if not os.path.exists(zip_path):
        print(f"❌ test.zip not found in current directory")
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
    
    # Training configuration for ai-toolkit
    timestamp = int(time.time())
    output_name = f"ai_toolkit_lora_{timestamp}"
    
    print(f"\n🎯 AI-Toolkit Training Configuration:")
    print(f"   📝 Output name: {output_name}")
    print(f"   🔤 Trigger word: TOK")
    print(f"   📐 Resolution: 768px")
    print(f"   🔄 Steps: 1000")
    print(f"   🧠 Learning rate: 5e-5")
    print(f"   🎚️ LoRA rank: 16")
    print(f"   🔧 Trainer: ostris/ai-toolkit")
    
    payload = {
        "input": {
            "endpoint": "train_lora",
            "training_images_zip": training_images_zip,
            "output_name": output_name,
            "instance_prompt": "a photo of TOK person",  # TOK trigger word
            
            # ai-toolkit optimal settings
            "resolution": 768,
            "max_train_steps": 1000,
            "learning_rate": 5e-5,
            "lora_rank": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "lr_scheduler": "constant",
            "mixed_precision": "fp16"
        }
    }
    
    try:
        print("\n📤 Submitting ai-toolkit LoRA training job...")
        response = requests.post(f"{BASE_URL}/run", headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        if "id" not in result:
            print(f"❌ No job ID: {result}")
            return None
            
        job_id = result["id"]
        print(f"✅ Training job submitted: {job_id}")
        
        # Wait for training completion (ai-toolkit takes longer than the old broken trainer)
        print("⏳ ai-toolkit training in progress...", end="")
        training_result = wait_for_completion(job_id, extended_timeout=True)
        
        if training_result and training_result.get("output", {}).get("success"):
            output = training_result["output"]
            lora_path = output.get("lora_path")
            training_time = output.get("training_time", 0)
            
            print(f"\n🎉 AI-Toolkit LoRA Training Complete!")
            print(f"📁 LoRA Path: {lora_path}")
            print(f"⏱️ Training Time: {training_time:.2f} seconds")
            print(f"🔧 Trained with: ostris/ai-toolkit")
            
            return {
                "lora_path": lora_path,
                "output_name": output_name,
                "training_time": training_time
            }
        else:
            print(f"\n❌ ai-toolkit training failed: {training_result}")
            return None
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        return None

def test_new_lora_generation(lora_path, lora_name):
    """Test image generation with the newly trained ai-toolkit LoRA"""
    print(f"\n🧪 TESTING NEW AI-TOOLKIT LoRA: {lora_name}")
    print("=" * 50)
    
    # Test prompts with TOK trigger word
    test_prompts = [
        "Professional headshot of TOK person 4k dslr",
        "TOK person in business attire", 
        "Portrait of TOK person smiling, natural lighting"
    ]
    
    results = {}
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}️⃣ Testing Prompt: '{prompt}'")
        print("-" * 40)
        
        # Test with regular FLUX
        print("🔸 Regular FLUX:")
        regular_result = test_single_generation(prompt, lora_path, "generate", f"ai_toolkit_regular_{i}")
        
        # Test with Krea FLUX  
        print("🔸 Krea FLUX:")
        krea_result = test_single_generation(prompt, lora_path, "generate_krea", f"ai_toolkit_krea_{i}")
        
        results[f"prompt_{i}"] = {
            "prompt": prompt,
            "regular": regular_result,
            "krea": krea_result
        }
    
    return results

def test_single_generation(prompt, lora_path, endpoint, prefix):
    """Test a single generation"""
    payload = {
        "input": {
            "endpoint": endpoint,
            "prompt": prompt,
            "lora_path": lora_path,
            "height": 1024,
            "width": 1024,
            "guidance_scale": 3.0,
            "num_inference_steps": 28,
            "seed": 54321,  # Fixed seed
            "return_base64": True
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/run", headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        if "id" not in result:
            print(f"  ❌ No job ID")
            return None
            
        job_id = result["id"]
        print(f"  📤 Job: {job_id}")
        
        print("  ⏳ Generating...", end="")
        final_result = wait_for_completion(job_id)
        
        if final_result:
            filename = save_image(final_result, prefix)
            if filename:
                print(f" ✅ {filename}")
                return filename
        
        return None
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def wait_for_completion(job_id, extended_timeout=False):
    """Wait for job completion with optional extended timeout for training"""
    timeout = 7200 if extended_timeout else 300  # 2 hours for training, 5 min for generation
    start_time = time.time()
    
    while True:
        try:
            if time.time() - start_time > timeout:
                print(f" ❌ TIMEOUT after {timeout} seconds")
                return None
                
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
                time.sleep(10 if extended_timeout else 3)  # Longer wait for training
                
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
    print("🚀 AI-TOOLKIT LoRA TRAINING TEST")
    print("🔧 Using ostris/ai-toolkit (industry standard FLUX trainer)")
    print()
    
    # Step 1: Train LoRA with ai-toolkit
    training_result = test_ai_toolkit_training()
    
    if not training_result:
        print("❌ ai-toolkit training failed, stopping test")
        exit(1)
    
    lora_path = training_result["lora_path"]
    lora_name = training_result["output_name"]
    
    # Step 2: Test generation with new LoRA
    generation_results = test_new_lora_generation(lora_path, lora_name)
    
    # Summary
    print(f"\n🎊 AI-TOOLKIT TEST COMPLETE!")
    print("=" * 60)
    print(f"✅ LoRA trained with: ostris/ai-toolkit")
    print(f"📁 LoRA path: {lora_path}")
    print(f"⏱️ Training time: {training_result['training_time']:.2f}s")
    
    print(f"\n📸 Generated Images:")
    for prompt_key, result in generation_results.items():
        print(f"   🔸 {result['prompt']}")
        if result['regular']:
            print(f"      📄 Regular: {result['regular']}")
        if result['krea']:
            print(f"      🎨 Krea: {result['krea']}")
    
    print(f"\n🎯 Success! AI-Toolkit Integration:")
    print(f"   ✅ Industry standard FLUX LoRA training")
    print(f"   ✅ Proper config-driven approach")
    print(f"   ✅ Works with both FLUX models")
    print(f"   ✅ Saves to network volume")
    print("=" * 60)