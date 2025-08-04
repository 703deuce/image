# FLUX.1-dev RunPod Deployment Checklist

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
