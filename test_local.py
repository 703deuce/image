#!/usr/bin/env python3
"""
Local testing script for FLUX.1-dev RunPod API
Test the handler function locally before deployment
"""

import os
import sys
import json
import time
from handler import handler

def test_health_endpoint():
    """Test the health check endpoint"""
    print("🔍 Testing health endpoint...")
    
    job = {
        "input": {
            "endpoint": "health"
        }
    }
    
    result = handler(job)
    print(f"✅ Health check result: {json.dumps(result, indent=2)}")
    return result.get("success", False)

def test_info_endpoint():
    """Test the info endpoint"""
    print("\n🔍 Testing info endpoint...")
    
    job = {
        "input": {
            "endpoint": "info"
        }
    }
    
    result = handler(job)
    print(f"✅ Info result: {json.dumps(result, indent=2)}")
    return result.get("success", False)

def test_generate_endpoint():
    """Test the image generation endpoint"""
    print("\n🔍 Testing image generation...")
    
    job = {
        "input": {
            "endpoint": "generate",
            "prompt": "A cute cat wearing a hat, digital art style",
            "height": 512,  # Smaller for faster testing
            "width": 512,
            "guidance_scale": 3.5,
            "num_inference_steps": 20,  # Fewer steps for faster testing
            "seed": 42,
            "return_base64": True
        }
    }
    
    print("⏳ Generating image (this may take a few minutes)...")
    start_time = time.time()
    
    result = handler(job)
    
    end_time = time.time()
    generation_time = end_time - start_time
    
    print(f"⏱️  Generation completed in {generation_time:.2f} seconds")
    
    if result.get("success"):
        print("✅ Image generation successful!")
        
        # Save the image if base64 is returned
        if "image_base64" in result:
            import base64
            from PIL import Image
            import io
            
            # Decode and save image
            image_data = base64.b64decode(result["image_base64"])
            image = Image.open(io.BytesIO(image_data))
            image.save("test_output.png")
            print("💾 Test image saved as 'test_output.png'")
        
        # Print parameters used
        if "parameters_used" in result:
            print(f"📋 Parameters used: {json.dumps(result['parameters_used'], indent=2)}")
            
    else:
        print(f"❌ Image generation failed: {result.get('error', 'Unknown error')}")
    
    return result.get("success", False)

def test_parameter_validation():
    """Test parameter validation"""
    print("\n🔍 Testing parameter validation...")
    
    # Test invalid height
    job = {
        "input": {
            "endpoint": "generate",
            "prompt": "Test prompt",
            "height": 100,  # Too small
            "width": 512
        }
    }
    
    result = handler(job)
    if not result.get("success"):
        print("✅ Parameter validation working - rejected invalid height")
    else:
        print("❌ Parameter validation failed - accepted invalid height")
    
    # Test missing prompt
    job = {
        "input": {
            "endpoint": "generate",
            "height": 512,
            "width": 512
        }
    }
    
    result = handler(job)
    if not result.get("success"):
        print("✅ Parameter validation working - rejected missing prompt")
    else:
        print("❌ Parameter validation failed - accepted missing prompt")

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "torch",
        "diffusers", 
        "transformers",
        "Pillow",
        "runpod"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - installed")
        except ImportError:
            print(f"❌ {package} - missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies are installed!")
        return True

def check_gpu():
    """Check GPU availability"""
    print("\n🔍 Checking GPU availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"✅ GPU available: {gpu_name}")
            print(f"📊 GPU count: {gpu_count}")
            print(f"💾 GPU memory: {gpu_memory:.1f} GB")
            
            if gpu_memory < 8:
                print("⚠️  Warning: GPU has less than 8GB memory. May not be sufficient for FLUX.1-dev")
            
            return True
        else:
            print("❌ No GPU available - will use CPU (very slow)")
            return False
            
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting FLUX.1-dev RunPod API Tests\n")
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Tests aborted due to missing dependencies")
        sys.exit(1)
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Run endpoint tests
    try:
        # Test basic endpoints
        health_ok = test_health_endpoint()
        info_ok = test_info_endpoint()
        
        # Test parameter validation
        test_parameter_validation()
        
        # Test image generation (optional if no GPU)
        if gpu_available:
            generate_ok = test_generate_endpoint()
        else:
            print("\n⚠️  Skipping image generation test (no GPU available)")
            print("   Image generation will be very slow on CPU")
            generate_ok = True
        
        # Summary
        print(f"\n📊 Test Summary:")
        print(f"   Health endpoint: {'✅' if health_ok else '❌'}")
        print(f"   Info endpoint: {'✅' if info_ok else '❌'}")
        print(f"   Image generation: {'✅' if generate_ok else '❌'}")
        
        if health_ok and info_ok and generate_ok:
            print("\n🎉 All tests passed! Ready for deployment to RunPod.")
        else:
            print("\n⚠️  Some tests failed. Check the output above for details.")
            
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()