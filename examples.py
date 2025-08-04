#!/usr/bin/env python3
"""
Example usage patterns for FLUX.1-dev RunPod API
Demonstrates various use cases and integration patterns
"""

import os
import time
import json
from runpod_api import FluxRunPodAPI
from typing import List, Dict, Any

# Configuration - Replace with your actual values
ENDPOINT_ID = "your-endpoint-id"  # Get from RunPod dashboard
API_KEY = "your-api-key"          # Get from RunPod dashboard

def example_basic_generation():
    """
    Example 1: Basic image generation
    """
    print("🎨 Example 1: Basic Image Generation")
    print("-" * 50)
    
    api = FluxRunPodAPI(ENDPOINT_ID, API_KEY)
    
    try:
        # Simple generation with default parameters
        result = api.generate_image(
            prompt="A majestic mountain landscape at sunset with a lake reflection"
        )
        
        if result.get("status") == "COMPLETED" and result.get("output", {}).get("success"):
            # Save the image
            api.save_image_from_response(result, "basic_landscape.png")
            print("✅ Image saved as 'basic_landscape.png'")
            
            # Print generation info
            params = result["output"]["parameters_used"]
            print(f"📋 Used parameters: {json.dumps(params, indent=2)}")
        else:
            print(f"❌ Generation failed: {result}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def example_custom_parameters():
    """
    Example 2: Custom parameters for specific use cases
    """
    print("\n🎛️  Example 2: Custom Parameters")
    print("-" * 50)
    
    api = FluxRunPodAPI(ENDPOINT_ID, API_KEY)
    
    # High-quality portrait with specific settings
    portraits = [
        {
            "prompt": "Professional headshot of a confident businesswoman, studio lighting",
            "filename": "portrait_businesswoman.png",
            "settings": {
                "height": 1024,
                "width": 768,  # Portrait aspect ratio
                "guidance_scale": 5.0,  # Higher for better prompt following
                "num_inference_steps": 60,  # More steps for higher quality
                "seed": 12345
            }
        },
        {
            "prompt": "Artistic portrait of an elderly man with weathered hands, black and white",
            "filename": "portrait_elderly.png", 
            "settings": {
                "height": 1024,
                "width": 768,
                "guidance_scale": 4.5,
                "num_inference_steps": 50,
                "seed": 67890
            }
        }
    ]
    
    for i, portrait in enumerate(portraits):
        try:
            print(f"🎭 Generating portrait {i+1}: {portrait['prompt'][:50]}...")
            
            result = api.generate_image(
                prompt=portrait["prompt"],
                **portrait["settings"]
            )
            
            if result.get("status") == "COMPLETED" and result.get("output", {}).get("success"):
                api.save_image_from_response(result, portrait["filename"])
                print(f"✅ Saved: {portrait['filename']}")
            else:
                print(f"❌ Failed: {portrait['filename']}")
                
        except Exception as e:
            print(f"❌ Error generating {portrait['filename']}: {e}")

def example_batch_processing():
    """
    Example 3: Batch processing multiple prompts
    """
    print("\n📦 Example 3: Batch Processing")
    print("-" * 50)
    
    api = FluxRunPodAPI(ENDPOINT_ID, API_KEY)
    
    # Batch of nature scenes
    nature_prompts = [
        "A serene forest path with dappled sunlight",
        "Ocean waves crashing against rocky cliffs",
        "A field of wildflowers under a cloudy sky",
        "Snow-capped mountains reflected in a crystal lake",
        "A desert landscape with sand dunes and cacti"
    ]
    
    print(f"🌿 Processing {len(nature_prompts)} nature scenes...")
    
    results = []
    for i, prompt in enumerate(nature_prompts):
        try:
            print(f"   🎨 Generating {i+1}/{len(nature_prompts)}: {prompt[:40]}...")
            
            result = api.generate_image(
                prompt=prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=40,  # Faster for batch processing
                seed=i * 1000  # Consistent but different seeds
            )
            
            if result.get("status") == "COMPLETED" and result.get("output", {}).get("success"):
                filename = f"nature_batch_{i+1:02d}.png"
                api.save_image_from_response(result, filename)
                results.append({"prompt": prompt, "filename": filename, "success": True})
                print(f"   ✅ Saved: {filename}")
            else:
                results.append({"prompt": prompt, "success": False, "error": result})
                print(f"   ❌ Failed")
                
        except Exception as e:
            results.append({"prompt": prompt, "success": False, "error": str(e)})
            print(f"   ❌ Error: {e}")
    
    # Summary
    successful = sum(1 for r in results if r["success"])
    print(f"\n📊 Batch Summary: {successful}/{len(nature_prompts)} successful")

def example_async_processing():
    """
    Example 4: Asynchronous processing for better efficiency
    """
    print("\n⚡ Example 4: Async Processing")
    print("-" * 50)
    
    api = FluxRunPodAPI(ENDPOINT_ID, API_KEY)
    
    # Start multiple jobs asynchronously
    jobs = []
    prompts = [
        "A futuristic city with flying cars",
        "A medieval castle on a hilltop",
        "A space station orbiting Earth"
    ]
    
    print("🚀 Starting async jobs...")
    for i, prompt in enumerate(prompts):
        try:
            job = api.generate_image(
                prompt=prompt,
                height=1024,
                width=1024,
                sync=False  # Don't wait for completion
            )
            
            jobs.append({
                "id": job["id"],
                "prompt": prompt,
                "filename": f"async_{i+1}.png"
            })
            print(f"   📤 Started job {job['id']}: {prompt[:40]}...")
            
        except Exception as e:
            print(f"   ❌ Failed to start job: {e}")
    
    # Wait for all jobs to complete
    print("\n⏳ Waiting for jobs to complete...")
    for job in jobs:
        try:
            result = api.wait_for_completion(job["id"], timeout=300)
            
            if result.get("status") == "COMPLETED" and result.get("output", {}).get("success"):
                api.save_image_from_response(result, job["filename"])
                print(f"   ✅ Completed: {job['filename']}")
            else:
                print(f"   ❌ Failed: {job['filename']}")
                
        except Exception as e:
            print(f"   ❌ Error waiting for {job['id']}: {e}")

def example_style_variations():
    """
    Example 5: Generate variations of the same subject in different styles
    """
    print("\n🎨 Example 5: Style Variations")
    print("-" * 50)
    
    api = FluxRunPodAPI(ENDPOINT_ID, API_KEY)
    
    base_subject = "a wise old owl sitting on a tree branch"
    styles = [
        "photorealistic",
        "oil painting",
        "watercolor",
        "digital art",
        "pencil sketch",
        "anime style"
    ]
    
    print(f"🦉 Creating {len(styles)} style variations of: {base_subject}")
    
    for i, style in enumerate(styles):
        try:
            full_prompt = f"{base_subject}, {style}"
            print(f"   🎭 Style {i+1}: {style}")
            
            result = api.generate_image(
                prompt=full_prompt,
                height=1024,
                width=1024,
                guidance_scale=4.0,
                num_inference_steps=50,
                seed=999  # Same seed for all to see style differences
            )
            
            if result.get("status") == "COMPLETED" and result.get("output", {}).get("success"):
                filename = f"owl_style_{style.replace(' ', '_')}.png"
                api.save_image_from_response(result, filename)
                print(f"   ✅ Saved: {filename}")
            else:
                print(f"   ❌ Failed: {style}")
                
        except Exception as e:
            print(f"   ❌ Error with {style}: {e}")

def example_resolution_comparison():
    """
    Example 6: Compare different resolutions and their trade-offs
    """
    print("\n📐 Example 6: Resolution Comparison")
    print("-" * 50)
    
    api = FluxRunPodAPI(ENDPOINT_ID, API_KEY)
    
    prompt = "A detailed architectural drawing of a modern skyscraper"
    resolutions = [
        {"height": 512, "width": 512, "name": "small"},
        {"height": 768, "width": 768, "name": "medium"},
        {"height": 1024, "width": 1024, "name": "large"},
        {"height": 1536, "width": 1024, "name": "portrait"}
    ]
    
    print(f"🏢 Comparing {len(resolutions)} resolutions for: {prompt[:50]}...")
    
    for res in resolutions:
        try:
            print(f"   📏 {res['name']}: {res['width']}x{res['height']}")
            start_time = time.time()
            
            result = api.generate_image(
                prompt=prompt,
                height=res["height"],
                width=res["width"],
                guidance_scale=3.5,
                num_inference_steps=40,
                seed=777  # Same seed for comparison
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            if result.get("status") == "COMPLETED" and result.get("output", {}).get("success"):
                filename = f"resolution_{res['name']}_{res['width']}x{res['height']}.png"
                api.save_image_from_response(result, filename)
                print(f"   ✅ Saved: {filename} (took {generation_time:.1f}s)")
            else:
                print(f"   ❌ Failed: {res['name']}")
                
        except Exception as e:
            print(f"   ❌ Error with {res['name']}: {e}")

def example_api_monitoring():
    """
    Example 7: API health monitoring and error handling
    """
    print("\n🔍 Example 7: API Monitoring")
    print("-" * 50)
    
    api = FluxRunPodAPI(ENDPOINT_ID, API_KEY)
    
    try:
        # Check API health
        print("🏥 Checking API health...")
        health = api.health_check()
        print(f"   Status: {health.get('output', {}).get('status', 'unknown')}")
        print(f"   Model loaded: {health.get('output', {}).get('model_loaded', False)}")
        
        # Get API info
        print("\n📋 Getting API information...")
        info = api.get_api_info()
        
        if info.get("status") == "COMPLETED":
            model_info = info.get("output", {}).get("model_info", {})
            print(f"   Model: {model_info.get('name', 'unknown')}")
            print(f"   License: {model_info.get('license', 'unknown')}")
            
            # Show supported parameters
            params = info.get("output", {}).get("supported_parameters", {})
            print(f"   Supported parameters: {len(params)}")
            for param_name, param_info in list(params.items())[:3]:  # Show first 3
                print(f"     - {param_name}: {param_info.get('description', 'no description')}")
        
        # Test error handling
        print("\n🚨 Testing error handling...")
        try:
            # This should fail due to invalid parameters
            bad_result = api.generate_image(
                prompt="Test",
                height=50,  # Too small
                width=50    # Too small
            )
            print(f"   Error handling test: {bad_result.get('output', {}).get('error', 'No error?')}")
        except Exception as e:
            print(f"   ✅ Error correctly caught: {e}")
            
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")

def main():
    """
    Run all examples
    """
    print("🚀 FLUX.1-dev RunPod API Examples")
    print("=" * 60)
    
    # Check if credentials are set
    if ENDPOINT_ID == "your-endpoint-id" or API_KEY == "your-api-key":
        print("⚠️  Please update ENDPOINT_ID and API_KEY at the top of this file")
        print("   Get these values from your RunPod dashboard")
        return
    
    # Create output directory
    os.makedirs("examples_output", exist_ok=True)
    os.chdir("examples_output")
    
    try:
        # Run examples in order
        example_basic_generation()
        example_custom_parameters()
        example_batch_processing()
        example_async_processing()
        example_style_variations()
        example_resolution_comparison()
        example_api_monitoring()
        
        print("\n🎉 All examples completed!")
        print(f"📁 Check the 'examples_output' directory for generated images")
        
    except KeyboardInterrupt:
        print("\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Examples failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()