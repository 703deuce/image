# FLUX.1-dev RunPod Serverless API

A complete RunPod Serverless API implementation for the [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) text-to-image model by Black Forest Labs.

## Overview

This API provides a serverless endpoint for generating high-quality images from text prompts using the FLUX.1-dev model. The implementation includes:

- ✅ Complete parameter support (all FLUX.1-dev parameters)
- ✅ Multiple endpoints (generate, health, info)
- ✅ Robust error handling and validation
- ✅ Memory optimization for GPU efficiency
- ✅ Base64 image encoding for easy integration
- ✅ Comprehensive documentation and examples

## Features

### Model Capabilities
- **12 billion parameter** rectified flow transformer
- **High-quality image generation** from text descriptions
- **Competitive prompt following** performance
- **Efficient guidance distillation** training
- **Commercial usage allowed** (with license restrictions)

### API Features
- **Flexible parameters** with validation and defaults
- **Multiple output formats** (PNG, JPEG)
- **Seed-based reproducibility** for consistent results
- **Health monitoring** and status endpoints
- **Comprehensive error handling**
- **Memory-efficient processing** with GPU optimization

## Quick Start

### 1. Deploy to RunPod Serverless via GitHub

This API is designed to deploy directly from GitHub to RunPod Serverless with automatic container builds.

#### **Option A: Deploy from this Repository (Recommended)**

1. **Fork or clone this repository**:
   ```bash
   git clone https://github.com/703deuce/image.git
   cd image
   ```

2. **Create a RunPod account** at [runpod.io](https://runpod.io)

3. **Create a new Serverless Endpoint**:
   - Go to Serverless → Create Endpoint
   - Choose "GitHub" as source
   - Connect your GitHub account
   - Select repository: `703deuce/image`
   - Branch: `main`
   - Dockerfile: `Dockerfile.simple`

4. **Configure the endpoint**:
   - GPU: RTX 4090, A100 40GB, or A100 80GB
   - Container Disk: 50GB+ (for model storage)
   - Volume: 20GB+ (for model caching)
   - Environment variables: Copy from `env.template`

#### **Option B: Manual Docker Deployment**

1. **Build and deploy the container**:
   ```bash
   # Build the Docker image
   docker build -f Dockerfile.simple -t flux-runpod-api .
   
   # Tag for your registry (replace with your details)
   docker tag flux-runpod-api your-registry/flux-runpod-api:latest
   
   # Push to registry
   docker push your-registry/flux-runpod-api:latest
   ```

2. **Configure the endpoint**:
   - Container Image: `your-registry/flux-runpod-api:latest`
   - Container Disk: 50GB+ (for model storage)
   - GPU: RTX 4090, A100, or similar (16GB+ VRAM recommended)

### 2. Test the API

```python
from runpod_api import FluxRunPodAPI

# Initialize with your endpoint details
api = FluxRunPodAPI(
    endpoint_id="your-endpoint-id",
    api_key="your-api-key"
)

# Generate an image
result = api.generate_image(
    prompt="A serene landscape with mountains and a lake at sunset",
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    seed=42
)

# Save the result
api.save_image_from_response(result, "output.png")
```

## API Endpoints

### 1. Generate Image (`/generate`)

Generate images from text prompts with full parameter control.

**Parameters:**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `prompt` | string | *required* | - | Text description for image generation |
| `height` | integer | 1024 | 256-2048 | Image height in pixels (multiples of 8) |
| `width` | integer | 1024 | 256-2048 | Image width in pixels (multiples of 8) |
| `guidance_scale` | float | 3.5 | 0-20 | Guidance scale for generation quality |
| `num_inference_steps` | integer | 50 | 1-100 | Number of denoising steps |
| `max_sequence_length` | integer | 512 | 1-1024 | Maximum sequence length for text encoding |
| `seed` | integer | null | - | Random seed for reproducible generation |
| `output_format` | string | "PNG" | PNG/JPEG | Output image format |
| `return_base64` | boolean | true | - | Return image as base64 string |

**Example Request:**
```json
{
  "input": {
    "endpoint": "generate",
    "prompt": "A futuristic city with flying cars and neon lights",
    "height": 1024,
    "width": 1024,
    "guidance_scale": 4.0,
    "num_inference_steps": 40,
    "seed": 123456
  }
}
```

**Example Response:**
```json
{
  "status": "COMPLETED",
  "output": {
    "success": true,
    "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
    "parameters_used": {
      "prompt": "A futuristic city with flying cars and neon lights",
      "height": 1024,
      "width": 1024,
      "guidance_scale": 4.0,
      "num_inference_steps": 40,
      "seed": 123456
    },
    "image_info": {
      "width": 1024,
      "height": 1024,
      "format": "PNG"
    }
  }
}
```

### 2. Health Check (`/health`)

Check API health and model loading status.

**Example Request:**
```json
{
  "input": {
    "endpoint": "health"
  }
}
```

**Example Response:**
```json
{
  "output": {
    "success": true,
    "status": "healthy",
    "model_loaded": true
  }
}
```

### 3. API Information (`/info`)

Get detailed information about the API and supported parameters.

**Example Request:**
```json
{
  "input": {
    "endpoint": "info"
  }
}
```

## Client SDK Usage

### Basic Image Generation

```python
from runpod_api import FluxRunPodAPI

# Initialize client
api = FluxRunPodAPI("your-endpoint-id", "your-api-key")

# Simple generation
result = api.generate_image("A beautiful sunset over the ocean")

# Save image
api.save_image_from_response(result, "sunset.png")
```

### Advanced Parameters

```python
# High-quality generation with custom parameters
result = api.generate_image(
    prompt="A detailed portrait of a wise old wizard with a long beard",
    height=1536,
    width=1024,
    guidance_scale=5.0,
    num_inference_steps=75,
    max_sequence_length=256,
    seed=98765,
    output_format="PNG"
)
```

### Async Generation

```python
# Start async generation
job = api.generate_image(
    prompt="A complex fantasy landscape",
    sync=False  # Don't wait for completion
)

job_id = job["id"]

# Check status later
result = api.wait_for_completion(job_id, timeout=300)
```

### Batch Processing

```python
prompts = [
    "A red sports car",
    "A blue mountain landscape",
    "A green forest scene"
]

results = []
for i, prompt in enumerate(prompts):
    result = api.generate_image(
        prompt=prompt,
        seed=i * 1000,  # Different seed for each
        height=1024,
        width=1024
    )
    
    filename = f"batch_image_{i}.png"
    api.save_image_from_response(result, filename)
    results.append(result)
```

## Configuration and Optimization

### GPU Requirements

**Minimum:**
- GPU: 16GB VRAM (RTX 4090, A100 40GB)
- RAM: 32GB system RAM
- Disk: 20GB for model storage

**Recommended:**
- GPU: A100 40GB or A100 80GB
- RAM: 64GB+ system RAM
- Disk: 50GB+ SSD storage

### Environment Variables

Set these in your RunPod endpoint configuration:

```bash
# Hugging Face cache settings
HF_HOME=/app/cache
TRANSFORMERS_CACHE=/app/cache
DIFFUSERS_CACHE=/app/cache

# Performance optimizations
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
CUDA_LAUNCH_BLOCKING=0
```

### Memory Optimization

The API includes several memory optimization techniques:

1. **Model CPU Offloading**: Automatically enabled to reduce VRAM usage
2. **Attention Optimization**: Uses xformers when available
3. **Garbage Collection**: Automatic cleanup after each generation
4. **CUDA Memory Management**: Efficient GPU memory handling

## Error Handling

The API provides comprehensive error handling with detailed messages:

```python
try:
    result = api.generate_image("Invalid prompt with very long text" * 1000)
except Exception as e:
    print(f"Generation failed: {e}")
    # Handle the error appropriately
```

**Common Error Types:**
- **Parameter Validation**: Invalid ranges or types
- **Memory Errors**: Insufficient GPU memory
- **Model Loading**: Model download or loading issues
- **Network Errors**: RunPod API connectivity issues

## Performance Tips

### 1. Optimal Parameters
- **Steps**: 30-50 for most cases (higher = better quality, slower)
- **Guidance Scale**: 3.5-7.0 for good prompt following
- **Resolution**: 1024x1024 for best quality/speed balance

### 2. Batch Processing
- Use consistent seeds for reproducible results
- Process multiple images in sequence to avoid model reloading
- Consider async processing for large batches

### 3. Memory Management
- Use lower resolutions for faster generation
- Enable CPU offloading (automatically done)
- Process images sequentially rather than in parallel

## License and Usage

This API implementation is provided under the MIT License. However, the FLUX.1-dev model itself is subject to the **FLUX.1 [dev] Non-Commercial License**.

### Key License Points:
- ✅ **Personal use** allowed
- ✅ **Scientific research** allowed  
- ✅ **Commercial use** allowed with restrictions
- ❌ **Prohibited uses**: See [model card](https://huggingface.co/black-forest-labs/FLUX.1-dev) for details

## Troubleshooting

### Common Issues

**1. Out of Memory Errors**
```python
# Reduce image resolution
result = api.generate_image(prompt, height=768, width=768)

# Reduce inference steps
result = api.generate_image(prompt, num_inference_steps=30)
```

**2. Model Loading Timeout**
- Ensure sufficient disk space (20GB+)
- Check internet connectivity for model download
- Verify RunPod endpoint has adequate resources

**3. Generation Quality Issues**
```python
# Increase guidance scale for better prompt following
result = api.generate_image(prompt, guidance_scale=7.0)

# Increase inference steps for higher quality
result = api.generate_image(prompt, num_inference_steps=75)
```

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your API calls will now show detailed logs
```

## Support and Contributing

For issues and feature requests:
1. Check existing documentation
2. Review error messages and logs
3. Test with minimal examples
4. Report issues with full context

## Changelog

### v1.0.0
- Initial release with full FLUX.1-dev support
- Complete parameter coverage
- Robust error handling
- Memory optimization
- Comprehensive documentation

---

**Happy generating with FLUX.1-dev! 🎨**