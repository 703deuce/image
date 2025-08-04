#!/usr/bin/env python3
"""
Test the complete LoRA workflow: train, download, and generate
"""

import requests
import json
import time
import base64
import os

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
        print(f"✅ Zip encoded successfully ({len(zip_data)} bytes)")
        return zip_base64
        
    except Exception as e:
        print(f"❌ Error reading zip file: {e}")
        return None

def list_available_loras():
    """List all available LoRAs"""
    print("\n📋 Listing available LoRAs...")
    
    payload = {"input": {"endpoint": "list_loras"}}
    
    try:
        response = requests.post(f"{BASE_URL}/runsync", headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        output = result.get("output", {})
        if output.get("success", False):
            loras = output.get("loras", [])
            print(f"✅ Found {len(loras)} LoRA models:")
            
            for i, lora in enumerate(loras, 1):
                size_mb = lora["size"] / (1024 * 1024)
                print(f"   {i}. {lora['name']} ({size_mb:.2f} MB)")
                print(f"      Created: {lora['created']}")
            
            return loras
        else:
            print(f"❌ Failed to list LoRAs: {output}")
            return []
            
    except Exception as e:
        print(f"❌ Error listing LoRAs: {e}")
        return []

def train_lora_with_zip(zip_path):
    """Train a LoRA with the provided zip file"""
    print("\n🧠 Training LoRA...")
    
    training_zip = encode_zip_to_base64(zip_path)
    if not training_zip:
        return None
    
    lora_name = f"my_lora_{int(time.time())}"
    
    payload = {
        "input": {
            "endpoint": "train_lora",
            "training_images_zip": training_zip,
            "output_name": lora_name,
            "instance_prompt": "a photo of TOK person",
            "resolution": 512,
            "max_train_steps": 200,
            "user_id": "workflow_test"
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/run", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if "id" in result:
            job_id = result["id"]
            print(f"✅ Training job submitted! Job ID: {job_id}")
            
            # Poll for completion
            while True:
                status_response = requests.get(f"{BASE_URL}/status/{job_id}", headers=headers, timeout=30)
                status_data = status_response.json()
                
                status = status_data.get("status", "UNKNOWN")
                print(f"   Status: {status}")
                
                if status == "COMPLETED":
                    output = status_data.get("output", {})
                    if output.get("success", False):
                        lora_path = output.get("lora_path")
                        training_time = output.get("training_time", 0)
                        print(f"✅ Training completed in {training_time:.2f} seconds!")
                        print(f"📁 LoRA saved at: {lora_path}")
                        return lora_name + ".safetensors"
                    else:
                        print(f"❌ Training failed: {output}")
                        return None
                elif status == "FAILED":
                    print(f"❌ Training failed: {status_data}")
                    return None
                
                time.sleep(5)
                
        else:
            print(f"❌ No job ID in response: {result}")
            return None
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def download_lora(lora_name):
    """Download a LoRA file"""
    print(f"\n💾 Downloading LoRA: {lora_name}")
    
    payload = {
        "input": {
            "endpoint": "download_lora",
            "lora_name": lora_name
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/runsync", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        output = result.get("output", {})
        if output.get("success", False):
            lora_base64 = output.get("lora_base64")
            file_size = output.get("file_size", 0)
            
            # Decode and save the file
            lora_data = base64.b64decode(lora_base64)
            local_filename = f"downloaded_{lora_name}"
            
            with open(local_filename, 'wb') as f:
                f.write(lora_data)
            
            print(f"✅ LoRA downloaded successfully!")
            print(f"📁 Saved as: {local_filename}")
            print(f"📏 File size: {file_size} bytes")
            
            return local_filename
        else:
            print(f"❌ Download failed: {output}")
            return None
            
    except Exception as e:
        print(f"❌ Download error: {e}")
        return None

def generate_with_lora(lora_name):
    """Generate an image using the LoRA"""
    print(f"\n🎨 Generating image with LoRA: {lora_name}")
    
    lora_path = f"/runpod-volume/loras/{lora_name}"
    
    payload = {
        "input": {
            "endpoint": "generate",
            "prompt": "a professional headshot of TOK person in business attire, studio lighting, high quality",
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
        response = requests.post(f"{BASE_URL}/run", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if "id" in result:
            job_id = result["id"]
            print(f"✅ Generation job submitted! Job ID: {job_id}")
            
            # Poll for completion
            while True:
                status_response = requests.get(f"{BASE_URL}/status/{job_id}", headers=headers, timeout=30)
                status_data = status_response.json()
                
                status = status_data.get("status", "UNKNOWN")
                print(f"   Status: {status}")
                
                if status == "COMPLETED":
                    output = status_data.get("output", {})
                    if output.get("success", False) and "image_base64" in output:
                        # Save the generated image
                        image_data = base64.b64decode(output["image_base64"])
                        filename = f"lora_generated_{int(time.time())}.png"
                        
                        with open(filename, 'wb') as f:
                            f.write(image_data)
                        
                        generation_time = output.get("generation_time", 0)
                        print(f"✅ Image generated in {generation_time:.2f} seconds!")
                        print(f"💾 Saved as: {filename}")
                        return filename
                    else:
                        print(f"❌ Generation failed: {output}")
                        return None
                elif status == "FAILED":
                    print(f"❌ Generation failed: {status_data}")
                    return None
                
                time.sleep(5)
                
        else:
            print(f"❌ No job ID in response: {result}")
            return None
            
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return None

def main():
    """Test the complete LoRA workflow"""
    print("🚀 Complete LoRA Workflow Test")
    print("=" * 60)
    print(f"🎯 Endpoint: {ENDPOINT_ID}")
    print(f"📦 Training data: test.zip")
    print("=" * 60)
    
    # Step 1: List existing LoRAs
    list_available_loras()
    
    # Step 2: Train a new LoRA
    trained_lora = train_lora_with_zip("test.zip")
    if not trained_lora:
        print("❌ Training failed, aborting workflow")
        return
    
    # Step 3: List LoRAs again to see the new one
    list_available_loras()
    
    # Step 4: Download the trained LoRA
    downloaded_file = download_lora(trained_lora)
    if not downloaded_file:
        print("❌ Download failed, but continuing with generation")
    
    # Step 5: Generate an image with the LoRA
    generated_image = generate_with_lora(trained_lora)
    
    # Summary
    print("\n" + "=" * 60)
    print("🏆 Workflow Summary:")
    print(f"   🧠 Training: {'✅ Success' if trained_lora else '❌ Failed'}")
    print(f"   💾 Download: {'✅ Success' if downloaded_file else '❌ Failed'}")
    print(f"   🎨 Generation: {'✅ Success' if generated_image else '❌ Failed'}")
    
    if trained_lora and downloaded_file and generated_image:
        print("\n🎉 Complete workflow successful!")
        print(f"   📁 LoRA file: {downloaded_file}")
        print(f"   🖼️  Generated image: {generated_image}")
        print("\n💡 You can now reuse this LoRA for future generations!")
    else:
        print("\n⚠️  Workflow partially completed. Check errors above.")

if __name__ == "__main__":
    main()