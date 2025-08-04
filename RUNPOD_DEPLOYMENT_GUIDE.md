# RunPod Serverless Deployment Guide

## 🎉 Repository Successfully Deployed to GitHub!

Your FLUX.1-dev API is now available at: **https://github.com/703deuce/image.git**

## 🚀 Deploy to RunPod Serverless

### Step 1: Create RunPod Account
1. Go to [runpod.io](https://runpod.io)
2. Sign up or log in to your account
3. Navigate to **Serverless** section

### Step 2: Create New Endpoint

1. **Click "Create Endpoint"**

2. **Choose "GitHub" as Source**:
   - Connect your GitHub account if not already connected
   - Grant RunPod access to your repositories

3. **Repository Configuration**:
   - **Repository**: `703deuce/image`
   - **Branch**: `master` (or `main`)
   - **Dockerfile**: `Dockerfile.simple` (recommended for stability)

4. **Hardware Configuration**:
   ```
   GPU Type: RTX 4090 (minimum) or A100 40GB/80GB (recommended)
   Container Disk: 50GB
   Volume Disk: 20GB (for model caching)
   ```

5. **Environment Variables** (copy from `env.template`):
   ```
   PYTHONUNBUFFERED=1
   PYTHONDONTWRITEBYTECODE=1
   HF_HOME=/runpod-volume/cache
   TRANSFORMERS_CACHE=/runpod-volume/cache
   DIFFUSERS_CACHE=/runpod-volume/cache
   PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   CUDA_LAUNCH_BLOCKING=0
   TOKENIZERS_PARALLELISM=false
   ```

6. **Scaling Configuration**:
   ```
   Min Workers: 0 (cost-effective)
   Max Workers: 10 (adjust based on needs)
   Idle Timeout: 300 seconds
   Max Job Timeout: 600 seconds
   ```

### Step 3: Deploy and Test

1. **Click "Create Endpoint"**
2. **Wait for Build** (first build takes 10-15 minutes)
3. **Copy Endpoint ID and API Key** from the dashboard

### Step 4: Test Your API

```python
from runpod_api import FluxRunPodAPI

# Replace with your actual credentials
api = FluxRunPodAPI("your-endpoint-id", "your-api-key")

# Test health check
print(api.health_check())

# Generate an image
result = api.generate_image("A beautiful sunset over mountains")
api.save_image_from_response(result, "test.png")
```

## 📊 Available Docker Configurations

| Dockerfile | Purpose | Recommended For |
|------------|---------|-----------------|
| `Dockerfile.simple` | Clean, minimal, guaranteed to work | **First deployment** |
| `Dockerfile` | Production-ready with optimizations | **Production use** |
| `Dockerfile.optimized` | Maximum performance optimizations | **High-performance needs** |

## 🔧 Configuration Options

### GPU Requirements

| GPU Type | VRAM | Performance | Cost/Hour |
|----------|------|-------------|-----------|
| RTX 4090 | 24GB | Good | ~$0.50 |
| A100 40GB | 40GB | Excellent | ~$1.50 |
| A100 80GB | 80GB | Best | ~$2.50 |

### Environment Variables

| Variable | Purpose | Recommended Value |
|----------|---------|-------------------|
| `HF_HOME` | Model cache location | `/runpod-volume/cache` |
| `PYTORCH_CUDA_ALLOC_CONF` | GPU memory optimization | `max_split_size_mb:512` |
| `CUDA_LAUNCH_BLOCKING` | CUDA performance | `0` |

## 🚨 Troubleshooting

### Common Issues

1. **Build Fails**:
   - Check Dockerfile syntax
   - Verify all files are committed to GitHub
   - Use `Dockerfile.simple` for guaranteed success

2. **Out of Memory**:
   - Increase GPU type (A100 40GB/80GB)
   - Add `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`
   - Reduce image resolution in requests

3. **Slow Cold Starts**:
   - Set min workers > 0
   - Use volume disk for model caching
   - Consider pre-downloading models in container

4. **Generation Timeouts**:
   - Increase max job timeout
   - Reduce `num_inference_steps`
   - Use faster GPU

### Performance Tips

1. **Enable Volume Caching**:
   - Use 20GB+ volume disk
   - Set cache paths to `/runpod-volume/cache`

2. **Optimize Parameters**:
   - Use 30-50 inference steps
   - Keep guidance scale 3.5-7.0
   - Use 1024x1024 for best quality/speed

3. **Monitor Costs**:
   - Use idle timeout to reduce costs
   - Scale workers based on demand
   - Monitor usage in RunPod dashboard

## 🎯 Next Steps

1. ✅ **Repository deployed to GitHub**
2. 🔄 **Create RunPod endpoint** (follow steps above)
3. 🧪 **Test API functionality**
4. 📈 **Monitor performance and costs**
5. 🚀 **Scale based on usage**

## 📞 Support

- **RunPod Documentation**: [docs.runpod.io](https://docs.runpod.io)
- **GitHub Issues**: [Create issue](https://github.com/703deuce/image/issues)
- **API Documentation**: See `README.md` in repository

---

**🎉 Your FLUX.1-dev API is ready for production deployment!**