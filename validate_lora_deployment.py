#!/usr/bin/env python3
"""
Validate that the LoRA deployment is working correctly
"""

import requests
import json

# Your endpoint details
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
    
    payload = {"input": {"endpoint": "health"}}
    
    try:
        response = requests.post(f"{BASE_URL}/runsync", headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        print(f"✅ Health check: {json.dumps(result, indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_info():
    """Test info endpoint to see available features"""
    print("\n📋 Testing info endpoint...")
    
    payload = {"input": {"endpoint": "info"}}
    
    try:
        response = requests.post(f"{BASE_URL}/runsync", headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        output = result.get("output", {})
        
        # Check if LoRA endpoints are available
        endpoints = output.get("endpoints", {})
        lora_params = output.get("lora_parameters", {})
        
        print("✅ Available endpoints:")
        for endpoint, description in endpoints.items():
            emoji = "🧬" if "lora" in endpoint.lower() else "🎨" if endpoint == "generate" else "🏥" if endpoint == "health" else "ℹ️"
            print(f"   {emoji} {endpoint}: {description}")
        
        if "train_lora" in endpoints:
            print("\n🎉 LoRA training is available!")
            
            if lora_params:
                print("\n📝 LoRA training parameters:")
                for param, info in lora_params.items():
                    required = "required" if info.get("required", False) else "optional"
                    default = f" (default: {info.get('default', 'none')})" if info.get("default") else ""
                    print(f"   • {param} ({required}){default}: {info.get('description', 'No description')}")
        else:
            print("\n❌ LoRA training endpoint not found!")
            print("   The deployment might not have completed yet.")
        
        return "train_lora" in endpoints
        
    except Exception as e:
        print(f"❌ Info check failed: {e}")
        return False

def test_lora_endpoint_exists():
    """Test if LoRA endpoint exists by trying to call it (should fail with validation error)"""
    print("\n🧪 Testing LoRA endpoint existence...")
    
    payload = {"input": {"endpoint": "train_lora"}}  # Missing required params
    
    try:
        response = requests.post(f"{BASE_URL}/runsync", headers=headers, json=payload, timeout=30)
        result = response.json()
        
        # We expect this to fail with a validation error, not an "unknown endpoint" error
        error_msg = result.get("output", {}).get("error", "")
        
        if "Unknown endpoint" in error_msg and "train_lora" in error_msg:
            print("❌ LoRA endpoint not deployed yet")
            return False
        elif "Missing required parameter" in error_msg:
            print("✅ LoRA endpoint exists (validation error as expected)")
            return True
        else:
            print(f"🤔 Unexpected response: {error_msg}")
            return False
            
    except Exception as e:
        print(f"❌ LoRA endpoint test failed: {e}")
        return False

def main():
    """Validate the deployment"""
    print("🔍 Validating LoRA Deployment")
    print("=" * 40)
    print(f"🎯 Endpoint: {ENDPOINT_ID}")
    print(f"🔗 URL: {BASE_URL}")
    print("=" * 40)
    
    # Test health
    health_ok = test_health()
    
    # Test info endpoint
    info_ok = test_info()
    
    # Test LoRA endpoint existence
    lora_ok = test_lora_endpoint_exists()
    
    print("\n" + "=" * 40)
    print("📊 Validation Summary:")
    print(f"   🏥 Health: {'✅ OK' if health_ok else '❌ Failed'}")
    print(f"   📋 Info: {'✅ OK' if info_ok else '❌ Failed'}")
    print(f"   🧬 LoRA: {'✅ Deployed' if lora_ok else '❌ Not deployed'}")
    
    if health_ok and info_ok and lora_ok:
        print("\n🎉 LoRA deployment is successful!")
        print("   You can now train LoRA models using the train_lora endpoint.")
        print("   See LORA_README.md for usage examples.")
    elif health_ok and info_ok:
        print("\n⚠️  Endpoint is healthy but LoRA not yet deployed.")
        print("   Make sure you've updated the endpoint image and redeployed.")
    else:
        print("\n❌ Deployment validation failed.")
        print("   Check your endpoint configuration and try again.")

if __name__ == "__main__":
    main()