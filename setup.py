#!/usr/bin/env python3
"""
Setup and deployment helper for FLUX.1-dev RunPod Serverless API
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_docker():
    """Check if Docker is installed and running"""
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker found: {result.stdout.strip()}")
            return True
        else:
            print("❌ Docker not found")
            return False
    except FileNotFoundError:
        print("❌ Docker not installed")
        return False

def build_docker_image(image_name="flux-runpod-api", tag="latest"):
    """Build the Docker image"""
    print(f"🔨 Building Docker image: {image_name}:{tag}")
    
    try:
        cmd = ["docker", "build", "-t", f"{image_name}:{tag}", "."]
        result = subprocess.run(cmd, check=True)
        print(f"✅ Docker image built successfully: {image_name}:{tag}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker build failed: {e}")
        return False

def check_files():
    """Check if all required files exist"""
    required_files = [
        "handler.py",
        "requirements.txt", 
        "Dockerfile",
        "runpod_api.py",
        "README.md"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
        else:
            print(f"✅ {file}")
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files present")
    return True

def validate_requirements():
    """Validate requirements.txt"""
    try:
        with open("requirements.txt", "r") as f:
            requirements = f.read()
        
        required_packages = ["runpod", "torch", "diffusers", "transformers", "Pillow"]
        missing_packages = []
        
        for package in required_packages:
            if package not in requirements:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing packages in requirements.txt: {', '.join(missing_packages)}")
            return False
        
        print("✅ Requirements.txt validated")
        return True
        
    except Exception as e:
        print(f"❌ Error validating requirements.txt: {e}")
        return False

def create_env_template():
    """Create environment variables template"""
    env_template = """# RunPod Serverless Environment Variables
# Copy these to your RunPod endpoint configuration

PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
HF_HOME=/app/cache
TRANSFORMERS_CACHE=/app/cache
DIFFUSERS_CACHE=/app/cache
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
CUDA_LAUNCH_BLOCKING=0

# Optional: Hugging Face token for private models
# HF_TOKEN=your_hf_token_here
"""
    
    with open(".env.template", "w") as f:
        f.write(env_template)
    
    print("✅ Created .env.template with environment variables")

def create_deployment_checklist():
    """Create a deployment checklist"""
    checklist = """# FLUX.1-dev RunPod Deployment Checklist

## Pre-deployment
- [ ] Docker image built successfully
- [ ] All files present and validated
- [ ] Local testing completed (run: python test_local.py)
- [ ] Docker registry set up (DockerHub, GitHub Container Registry, etc.)

## Docker Registry
- [ ] Tag image: docker tag flux-runpod-api your-registry/flux-runpod-api:latest
- [ ] Push image: docker push your-registry/flux-runpod-api:latest
- [ ] Verify image is accessible

## RunPod Configuration
- [ ] Create new Serverless Endpoint
- [ ] Set container image: your-registry/flux-runpod-api:latest
- [ ] Configure hardware (RTX 4090+ recommended)
- [ ] Set container disk: 50GB+
- [ ] Set temp disk: 20GB+
- [ ] Add environment variables from .env.template

## Hardware Requirements
- [ ] GPU: RTX 4090 (24GB VRAM) minimum
- [ ] CPU: 8+ cores
- [ ] RAM: 32GB+ system memory
- [ ] Storage: 50GB+ container disk

## Post-deployment
- [ ] Test health endpoint
- [ ] Test info endpoint  
- [ ] Test image generation
- [ ] Update API credentials in examples.py
- [ ] Run example scripts
- [ ] Monitor performance and costs

## API Testing Commands
```bash
# Test locally first
python test_local.py

# Test deployed endpoint
python -c "
from runpod_api import FluxRunPodAPI
api = FluxRunPodAPI('your-endpoint-id', 'your-api-key')
print(api.health_check())
"

# Run examples
python examples.py
```

## Troubleshooting
- Check RunPod logs for errors
- Verify model downloads completed
- Monitor GPU memory usage
- Check endpoint scaling settings
"""
    
    with open("DEPLOYMENT_CHECKLIST.md", "w") as f:
        f.write(checklist)
    
    print("✅ Created DEPLOYMENT_CHECKLIST.md")

def generate_dockerfile_optimized():
    """Generate an optimized Dockerfile for production"""
    optimized_dockerfile = """# Optimized Dockerfile for FLUX.1-dev RunPod API
FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache  
ENV DIFFUSERS_CACHE=/app/cache
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create cache directories
RUN mkdir -p /app/cache /tmp/outputs

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    wget \\
    curl \\
    unzip \\
    && rm -rf /var/lib/apt/lists/* \\
    && apt-get clean

# Copy and install Python dependencies first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Install performance optimizations
RUN pip install --no-cache-dir ninja flash-attn --no-build-isolation || echo "Optional packages failed"

# Copy application code
COPY handler.py runpod_api.py ./

# Pre-download model (optional - increases image size but reduces cold start)
# RUN python -c "from diffusers import FluxPipeline; FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-dev', torch_dtype=torch.bfloat16)"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD python -c "import torch; print('CUDA:', torch.cuda.is_available())" || exit 1

# Set default command
CMD ["python", "handler.py"]
"""
    
    with open("Dockerfile.optimized", "w") as f:
        f.write(optimized_dockerfile)
    
    print("✅ Created Dockerfile.optimized for production use")

def main():
    """Main setup function"""
    print("🚀 FLUX.1-dev RunPod Serverless API Setup")
    print("=" * 60)
    
    # Check environment
    print("\n📋 Checking environment...")
    docker_ok = check_docker()
    files_ok = check_files()
    requirements_ok = validate_requirements()
    
    if not all([docker_ok, files_ok, requirements_ok]):
        print("\n❌ Environment check failed. Please fix the issues above.")
        return False
    
    # Create helper files
    print("\n📄 Creating helper files...")
    create_env_template()
    create_deployment_checklist()
    generate_dockerfile_optimized()
    
    # Offer to build Docker image
    print(f"\n🔨 Build Docker image?")
    build_choice = input("Build Docker image now? (y/n): ").lower().strip()
    
    if build_choice in ['y', 'yes']:
        if build_docker_image():
            print("\n✅ Setup completed successfully!")
            print("\nNext steps:")
            print("1. Review DEPLOYMENT_CHECKLIST.md")
            print("2. Test locally: python test_local.py")
            print("3. Push image to registry")
            print("4. Deploy to RunPod")
            print("5. Update credentials in examples.py")
            print("6. Test deployed API: python examples.py")
        else:
            print("\n❌ Docker build failed")
            return False
    else:
        print("\n✅ Setup files created successfully!")
        print("\nNext steps:")
        print("1. Review DEPLOYMENT_CHECKLIST.md")
        print("2. Build Docker image when ready")
        print("3. Test locally: python test_local.py")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()