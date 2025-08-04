# LoRA Training and Inference with FLUX.1-dev

This endpoint now supports both **LoRA training** and **inference with LoRA models** in addition to standard FLUX.1-dev image generation.

## 🚀 Quick Start

### 1. Train a LoRA Model

```python
import requests
import base64
import zipfile
import io

# Prepare your training images as a base64-encoded zip
def create_training_zip(image_paths):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        for i, path in enumerate(image_paths):
            zf.write(path, f"image_{i}.jpg")
    return base64.b64encode(zip_buffer.getvalue()).decode()

# Train the LoRA
payload = {
    "input": {
        "endpoint": "train_lora",
        "training_images_zip": create_training_zip(["img1.jpg", "img2.jpg", "img3.jpg"]),
        "output_name": "my_person_lora",
        "instance_prompt": "a photo of TOK person",
        "max_train_steps": 500,
        "resolution": 512
    }
}

response = requests.post(f"{BASE_URL}/run", headers=headers, json=payload)
job_id = response.json()["id"]
# Poll for completion...
```

### 2. Generate with LoRA

```python
# Generate images using your trained LoRA
payload = {
    "input": {
        "endpoint": "generate",
        "prompt": "a professional headshot of TOK person in a business suit",
        "lora_path": "/runpod-volume/loras/my_person_lora.safetensors",
        "height": 1024,
        "width": 1024
    }
}

response = requests.post(f"{BASE_URL}/run", headers=headers, json=payload)
```

## 📚 Available Endpoints

### 1. `train_lora` - Train a LoRA Model

Train a custom LoRA model on your images.

**Required Parameters:**
- `training_images_zip` (string): Base64-encoded zip file containing 3+ training images
- `output_name` (string): Name for your LoRA model

**Optional Parameters:**
- `instance_prompt` (string): Training prompt template. Use `TOK` as placeholder. Default: `"a photo of TOK person"`
- `resolution` (int): Training image resolution. Default: `512`
- `max_train_steps` (int): Number of training steps. Default: `500`
- `learning_rate` (float): Learning rate. Default: `1e-4`
- `lora_rank` (int): LoRA rank (higher = more capacity). Default: `4`
- `lora_alpha` (int): LoRA alpha parameter. Default: `32`
- `user_id` (string): User identifier for organizing models

**Response:**
```json
{
    "success": true,
    "lora_path": "/runpod-volume/loras/my_person_lora.safetensors",
    "training_time": 145.6,
    "config_used": {...},
    "message": "LoRA 'my_person_lora' trained successfully"
}
```

### 2. `generate` - Generate Images (Enhanced)

Generate images with optional LoRA support.

**New Parameter:**
- `lora_path` (string, optional): Path to trained LoRA file

**Example with LoRA:**
```json
{
    "endpoint": "generate",
    "prompt": "TOK person as a superhero, comic book style",
    "lora_path": "/runpod-volume/loras/my_person_lora.safetensors",
    "height": 1024,
    "width": 1024
}
```

## 💡 Best Practices

### Training Data
- **Quantity**: 5-20 high-quality images work best
- **Quality**: Use clear, well-lit photos with good resolution
- **Variety**: Include different angles, expressions, and lighting
- **Format**: JPEG or PNG, ideally 512x512 or larger

### Training Parameters
- **For faces**: Use `resolution: 512` and `max_train_steps: 500-1000`
- **For objects/styles**: May need higher resolution and more steps
- **Memory**: Higher `lora_rank` uses more VRAM but gives more capacity

### Prompting with LoRA
- Use `TOK` as the placeholder for your trained subject
- Be descriptive: `"a professional photo of TOK person in business attire"`
- Experiment with different styles: `"TOK person in the style of a Renaissance painting"`

## 📁 File Storage

All LoRA models are stored persistently in `/runpod-volume/loras/` and can be reused across sessions.

### LoRA File Naming
- Format: `{output_name}.safetensors`
- Example: `my_person_lora.safetensors`

### Accessing Your LoRAs
List all trained LoRAs by checking the loras directory:
```bash
ls /runpod-volume/loras/
```

## 🔧 Example Scripts

### Complete Example (`lora_example.py`)
```python
# See lora_example.py for a complete workflow example
python lora_example.py
```

### Test Suite (`test_lora_endpoint.py`)
```python
# Test all LoRA functionality with sample data
python test_lora_endpoint.py
```

## ⚡ Performance Notes

- **Training Time**: 2-10 minutes depending on steps and data size
- **Memory Usage**: ~12GB VRAM for training, ~8GB for inference
- **Concurrent Jobs**: Training and inference jobs are queued separately
- **Storage**: Each LoRA is typically 50-200MB

## 🐛 Troubleshooting

### Common Issues

**"Not enough training images"**
- Ensure your zip contains at least 3 valid image files
- Check image formats (JPG, PNG supported)

**"Training failed"**
- Reduce `max_train_steps` or `resolution` for faster training
- Ensure images are not corrupted

**"LoRA not found"**
- Verify the exact path returned from training
- Check if training completed successfully

**"Out of memory"**
- Reduce `resolution` to 512 or lower
- Reduce `lora_rank` to 4 or lower

### Debug Information
Check the job output for detailed error messages and training progress.

## 🎯 Use Cases

### Personal Avatars
```python
# Train on selfies, generate professional headshots
prompt = "a professional LinkedIn photo of TOK person in a suit"
```

### Product Photography
```python
# Train on product images, generate marketing content  
prompt = "TOK product on a marble background, luxury lighting"
```

### Artistic Styles
```python
# Train on artwork samples, apply style to new subjects
prompt = "a portrait in the TOK art style, vibrant colors"
```

## 📊 API Endpoints Summary

| Endpoint | Method | Purpose | Async |
|----------|--------|---------|-------|
| `/run` with `endpoint: "train_lora"` | POST | Train LoRA model | Yes |
| `/run` with `endpoint: "generate"` | POST | Generate with optional LoRA | Yes |
| `/status/{job_id}` | GET | Check job status | - |
| `/runsync` with `endpoint: "health"` | POST | Health check | No |

---

For more examples and advanced usage, see the included example scripts!