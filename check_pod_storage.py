#!/usr/bin/env python3
"""
Check RunPod storage directly using management API
"""

import requests
import json

# Your RunPod API key
API_KEY = "rpa_C55TBQG7H6FM7G3Q7A6JM7ZJCDKA3I2J3EO0TAH8fxyddo"

def check_endpoint_info():
    """Check specific endpoint details"""
    
    print("🔍 Checking your specific endpoint...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    endpoint_id = "qgihilkw9mdlsk"
    
    try:
        # Try to get endpoint status/info
        response = requests.get(f"https://api.runpod.ai/v2/{endpoint_id}/status", headers=headers)
        print(f"📊 Endpoint status check: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
        
        # Try to get worker information
        print("\n🔍 Checking workers...")
        response = requests.get(f"https://api.runpod.ai/v2/{endpoint_id}/workers", headers=headers)
        print(f"📊 Workers check: {response.status_code}")
        if response.status_code == 200:
            workers = response.json()
            print(f"   Workers: {json.dumps(workers, indent=2)}")
        
        # Try to get endpoint configuration
        print("\n🔍 Checking endpoint config...")
        response = requests.get(f"https://api.runpod.ai/v2/{endpoint_id}", headers=headers)
        print(f"📊 Config check: {response.status_code}")
        if response.status_code == 200:
            config = response.json()
            print(f"   Config: {json.dumps(config, indent=2)}")
            
    except Exception as e:
        print(f"❌ Error checking endpoint: {e}")

def check_account_info():
    """Check account and general info"""
    
    print("\n🔍 Checking account information...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        # Try different API endpoints
        endpoints_to_try = [
            ("Account", "https://api.runpod.ai/v1/user"),
            ("Pods", "https://api.runpod.ai/v1/pods"),
            ("Volumes", "https://api.runpod.ai/v1/volumes"),
            ("Templates", "https://api.runpod.ai/v1/templates"),
        ]
        
        for name, url in endpoints_to_try:
            try:
                response = requests.get(url, headers=headers, timeout=10)
                print(f"📊 {name} ({url}): {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    if name == "Volumes" and data:
                        print(f"   📁 Found volumes: {json.dumps(data, indent=2)}")
                    elif name == "Pods" and data:
                        print(f"   🖥️  Found pods: {len(data) if isinstance(data, list) else 'Unknown'}")
                        if isinstance(data, list):
                            for pod in data[:3]:  # Show first 3 pods
                                print(f"      Pod: {pod.get('id', 'Unknown')} - {pod.get('name', 'Unknown')}")
                    else:
                        print(f"   Data: {str(data)[:200]}...")
                elif response.status_code == 404:
                    print(f"   ❌ Endpoint not found")
                else:
                    print(f"   ❌ Error: {response.text[:100]}...")
                    
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                
    except Exception as e:
        print(f"❌ Error checking account: {e}")

def estimate_storage_needs():
    """Estimate storage requirements"""
    
    print("\n📊 Storage Requirements Analysis:")
    print("=" * 50)
    
    print("🤖 FLUX.1-dev Model Components:")
    print("   • Transformer: ~12GB")
    print("   • VAE: ~1GB") 
    print("   • Text Encoders: ~1GB each (x2)")
    print("   • Safety Checker: ~1GB")
    print("   • Temporary files: ~5GB")
    print("   📊 Total needed: ~24GB minimum")
    
    print("\n💾 Common RunPod Storage Configurations:")
    print("   • Container disk: Usually 10-50GB")
    print("   • Network volume: Can be 50GB-1TB+")
    print("   • Temp storage: Usually limited")
    
    print("\n🔧 Solutions for 'No space left' error:")
    print("   1. ✅ Increase container disk to 50GB+")
    print("   2. ✅ Add network volume (50GB+)")
    print("   3. ✅ Use larger GPU instance (more storage)")
    print("   4. ✅ Model streaming (download on demand)")
    
    print("\n⚠️  Current Issue:")
    print("   • Model download failed due to insufficient disk space")
    print("   • Need to either increase storage or optimize loading")

def main():
    """Main function"""
    print("🔍 RunPod Direct Storage Check")
    print("=" * 50)
    
    # Check specific endpoint
    check_endpoint_info()
    
    # Check account info
    check_account_info()
    
    # Storage analysis
    estimate_storage_needs()
    
if __name__ == "__main__":
    main()