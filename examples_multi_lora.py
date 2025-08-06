#!/usr/bin/env python3
"""
FLUX.1-dev Multi-LoRA API Examples
Shows how to use subject LoRAs with automatic realistic skin enhancement
"""

import requests
import base64
import json
import time
from typing import Dict, Any

# API Configuration
ENDPOINT_URL = "https://api.runpod.ai/v2/your-endpoint-id/run"  # Replace with your endpoint
API_KEY = "your-api-key"  # Replace with your actual API key

def call_api(endpoint: str, **params) -> Dict[str, Any]:
    """Call the FLUX API with specified endpoint and parameters"""
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "endpoint": endpoint,
            **params
        }
    }
    
    response = requests.post(ENDPOINT_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"API call failed: {response.status_code} - {response.text}"}

def save_image_from_base64(base64_string: str, filename: str):
    """Save base64 image to file"""
    try:
        image_data = base64.b64decode(base64_string)
        with open(filename, 'wb') as f:
            f.write(image_data)
        print(f"Image saved as: {filename}")
    except Exception as e:
        print(f"Error saving image: {e}")

def example_1_basic_generation_with_auto_skin():
    """Basic generation with automatic skin enhancement (default behavior)"""
    print("\n=== Example 1: Basic Generation with Auto Skin Enhancement ===")
    
    result = call_api(
        endpoint="generate",
        prompt="portrait of a beautiful woman, professional photography, studio lighting",
        width=1024,
        height=1024,
        guidance_scale=3.0,
        num_inference_steps=30,
        seed=12345
        # use_skin_enhancement=True is the default, so skin LoRA will be auto-applied
        # skin_lora_weight=0.8 is the default strength
    )
    
    if result.get("success"):
        print("✅ Image generated successfully with automatic skin enhancement!")
        print(f"Prompt used: {result['parameters_used']['prompt']}")  # Will show aidmarealisticskin was added
        
        if result.get("image_base64"):
            save_image_from_base64(result["image_base64"], "output_auto_skin.png")
    else:
        print(f"❌ Error: {result.get('error')}")

def example_2_subject_lora_with_skin():
    """Use your trained subject LoRA + automatic skin enhancement"""
    print("\n=== Example 2: Subject LoRA + Automatic Skin Enhancement ===")
    
    # Replace with path to your actual trained LoRA
    subject_lora_path = "/runpod-volume/cache/lora/my_person_lora.safetensors"
    
    result = call_api(
        endpoint="generate",
        prompt="portrait of TOK person, smiling, professional headshot",
        lora_path=subject_lora_path,  # Your subject LoRA
        lora_weight=1.0,  # Subject LoRA strength
        # Skin enhancement is automatic:
        use_skin_enhancement=True,  # default, can be omitted
        skin_lora_weight=0.8,  # Skin LoRA strength
        width=1024,
        height=1024,
        guidance_scale=3.5,
        num_inference_steps=40,
        seed=67890
    )
    
    if result.get("success"):
        print("✅ Subject + Skin LoRAs applied successfully!")
        print(f"Prompt used: {result['parameters_used']['prompt']}")
        
        if result.get("image_base64"):
            save_image_from_base64(result["image_base64"], "output_subject_plus_skin.png")
    else:
        print(f"❌ Error: {result.get('error')}")

def example_3_disable_skin_enhancement():
    """Generate without skin enhancement (just subject LoRA)"""
    print("\n=== Example 3: Subject LoRA Only (No Skin Enhancement) ===")
    
    subject_lora_path = "/runpod-volume/cache/lora/my_person_lora.safetensors"
    
    result = call_api(
        endpoint="generate",
        prompt="portrait of TOK person, artistic style, vintage lighting",
        lora_path=subject_lora_path,
        lora_weight=1.2,
        use_skin_enhancement=False,  # Disable automatic skin LoRA
        width=1024,
        height=1024,
        guidance_scale=3.0,
        num_inference_steps=35,
        seed=11111
    )
    
    if result.get("success"):
        print("✅ Subject LoRA only (no skin enhancement)!")
        print(f"Prompt used: {result['parameters_used']['prompt']}")  # No aidmarealisticskin trigger
        
        if result.get("image_base64"):
            save_image_from_base64(result["image_base64"], "output_subject_only.png")
    else:
        print(f"❌ Error: {result.get('error')}")

def example_4_advanced_multi_lora():
    """Advanced example: Subject + Skin + Additional style LoRAs"""
    print("\n=== Example 4: Advanced Multi-LoRA Stacking ===")
    
    result = call_api(
        endpoint="generate",
        prompt="portrait of TOK person, cinematic lighting, film grain",
        # Primary subject LoRA
        lora_path="/runpod-volume/cache/lora/my_person_lora.safetensors",
        lora_weight=1.0,
        # Automatic skin enhancement
        use_skin_enhancement=True,
        skin_lora_weight=0.7,  # Slightly lower for balance
        # Additional LoRAs via advanced config
        lora_configs=[
            {
                "path": "/runpod-volume/cache/lora/cinematic_style_lora.safetensors",
                "weight": 0.6,
                "adapter_name": "cinematic_style"
            },
            {
                "path": "/runpod-volume/cache/lora/film_grain_lora.safetensors", 
                "weight": 0.4,
                "adapter_name": "film_grain"
            }
        ],
        width=1024,
        height=1344,  # Portrait aspect ratio
        guidance_scale=4.0,
        num_inference_steps=50,
        seed=99999
    )
    
    if result.get("success"):
        print("✅ Multi-LoRA stacking successful!")
        print(f"Prompt used: {result['parameters_used']['prompt']}")
        
        if result.get("image_base64"):
            save_image_from_base64(result["image_base64"], "output_multi_lora.png")
    else:
        print(f"❌ Error: {result.get('error')}")

def example_5_krea_with_skin():
    """Use FLUX.1-Krea-dev with skin enhancement"""
    print("\n=== Example 5: FLUX.1-Krea-dev + Skin Enhancement ===")
    
    result = call_api(
        endpoint="generate_krea",  # Use Krea model
        prompt="portrait of a model, high fashion photography, dramatic lighting",
        use_skin_enhancement=True,
        skin_lora_weight=0.9,  # Higher weight for fashion photography
        width=768,
        height=1024,
        guidance_scale=3.5,
        num_inference_steps=45,
        seed=55555
    )
    
    if result.get("success"):
        print("✅ Krea model with skin enhancement!")
        print(f"Model used: {result.get('model_used', 'FLUX.1-Krea-dev')}")
        
        if result.get("image_base64"):
            save_image_from_base64(result["image_base64"], "output_krea_skin.png")
    else:
        print(f"❌ Error: {result.get('error')}")

def example_6_check_api_info():
    """Check API info to see LoRA stacking capabilities"""
    print("\n=== Example 6: API Info (LoRA Stacking Details) ===")
    
    result = call_api(endpoint="info")
    
    if result.get("success"):
        print("✅ API Info Retrieved!")
        
        # Print LoRA stacking information
        lora_info = result.get("lora_stacking_info", {})
        print("\n📚 LoRA Stacking Capabilities:")
        for key, value in lora_info.items():
            print(f"  {key}: {value}")
        
        # Print LoRA parameters
        lora_params = result.get("generate_lora_parameters", {})
        print("\n🔧 LoRA Parameters:")
        for param, details in lora_params.items():
            print(f"  {param}: {details}")
    else:
        print(f"❌ Error: {result.get('error')}")

def main():
    """Run all examples"""
    print("🎨 FLUX.1-dev Multi-LoRA API Examples")
    print("=" * 50)
    
    # Check API info first
    example_6_check_api_info()
    
    # Run generation examples
    example_1_basic_generation_with_auto_skin()
    example_2_subject_lora_with_skin()
    example_3_disable_skin_enhancement()
    # example_4_advanced_multi_lora()  # Uncomment if you have additional LoRAs
    example_5_krea_with_skin()
    
    print("\n✅ All examples completed!")
    print("\nKey Features Demonstrated:")
    print("- Automatic realistic skin enhancement")
    print("- Subject LoRA + skin LoRA stacking")
    print("- Manual skin enhancement control")
    print("- FLUX.1-Krea-dev compatibility")
    print("- Advanced multi-LoRA configurations")

if __name__ == "__main__":
    main()