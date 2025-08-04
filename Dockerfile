# Production-ready Dockerfile for FLUX.1-dev RunPod API
FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables for optimization and caching
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache
ENV DIFFUSERS_CACHE=/app/cache
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_LAUNCH_BLOCKING=0
ENV TOKENIZERS_PARALLELISM=false

# Create necessary directories
RUN mkdir -p /app/cache /tmp/outputs

# Install system dependencies with cleanup
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Install performance optimizations with error handling
RUN pip install --no-cache-dir ninja || echo "Warning: ninja install failed"
RUN pip install --no-cache-dir flash-attn --no-build-isolation || echo "Warning: flash-attn install failed (optional optimization)"

# Install additional helpful packages
RUN pip install --no-cache-dir \
    psutil \
    GPUtil || echo "Warning: monitoring packages failed"

# Copy application files
COPY handler.py runpod_api.py ./

# Create health check script  
RUN echo '#!/bin/bash' > health_check.sh && \
    echo 'python -c "import torch, diffusers, transformers; print(\"CUDA:\", torch.cuda.is_available()); print(\"Devices:\", torch.cuda.device_count()); print(\"Ready\")"' >> health_check.sh && \
    chmod +x health_check.sh

# Add health check for container monitoring
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# Set proper permissions
RUN chmod 755 /app

# Default command
CMD ["python", "handler.py"]