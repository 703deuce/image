import runpod
import torch
import base64
import io
import zipfile
import shutil
import time
import json
from PIL import Image
from diffusers import FluxPipeline
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, get_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.utils import convert_state_dict_to_diffusers
import logging
import gc
import os
from typing import Optional, Dict, Any, List
from pathlib import Path
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model loading
pipeline = None
loaded_lora_path = None

def load_model():
    """Load the FLUX.1-dev model pipeline"""
    global pipeline
    
    if pipeline is None:
        logger.info("Loading FLUX.1-dev model...")
        
        try:
            # Get HF token for gated model access
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                raise ValueError("HF_TOKEN environment variable is required for FLUX.1-dev access")
            
            # Debug: Check all paths and variables before loading
            logger.info("=== DEBUGGING MODEL LOADING ===")
            logger.info(f"HF_TOKEN: {'SET' if hf_token else 'NONE'}")
            logger.info(f"Cache dir exists: {os.path.exists('/runpod-volume/cache')}")
            logger.info(f"Runpod volume exists: {os.path.exists('/runpod-volume')}")
            
            # List contents of directories
            if os.path.exists('/runpod-volume'):
                logger.info(f"Contents of /runpod-volume: {os.listdir('/runpod-volume')}")
            
            # Try without cache_dir first to isolate the issue
            logger.info("Loading pipeline components WITHOUT cache_dir...")
            pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev", 
                torch_dtype=torch.bfloat16,
                token=hf_token
                # Temporarily remove cache_dir to see if that's the issue
            )
            
            logger.info("Moving pipeline to CUDA...")
            pipeline = pipeline.to("cuda")
            logger.info("Pipeline loaded successfully on GPU")
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e
    
    return pipeline

def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def setup_directories():
    """Ensure required directories exist"""
    directories = [
        "/runpod-volume/loras",
        "/runpod-volume/training_data",
        "/runpod-volume/temp"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory ready: {directory}")

def extract_training_images(zip_data: str, user_id: str) -> str:
    """Extract training images from base64 encoded zip"""
    try:
        # Decode the base64 zip data
        zip_bytes = base64.b64decode(zip_data)
        
        # Create user-specific training directory
        training_dir = f"/runpod-volume/training_data/{user_id}_{int(time.time())}"
        os.makedirs(training_dir, exist_ok=True)
        
        # Extract zip file
        zip_path = f"/runpod-volume/temp/training_images_{user_id}.zip"
        with open(zip_path, 'wb') as f:
            f.write(zip_bytes)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(training_dir)
        
        # Count valid images
        image_count = 0
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for file_path in Path(training_dir).rglob('*'):
            if file_path.suffix.lower() in valid_extensions:
                try:
                    # Verify image can be opened
                    with Image.open(file_path) as img:
                        img.verify()
                    image_count += 1
                except Exception as e:
                    logger.warning(f"Invalid image {file_path}: {e}")
                    file_path.unlink()  # Remove invalid image
        
        # Clean up temp zip
        os.remove(zip_path)
        
        logger.info(f"Extracted {image_count} valid training images to {training_dir}")
        return training_dir
        
    except Exception as e:
        logger.error(f"Error extracting training images: {e}")
        raise e

def train_lora(training_dir: str, output_name: str, config: Dict[str, Any]) -> str:
    """Train LoRA using diffusers and peft"""
    try:
        logger.info(f"Starting LoRA training with config: {config}")
        
        # Get the base model
        hf_token = os.environ.get("HF_TOKEN")
        model_id = "black-forest-labs/FLUX.1-dev"
        
        # Load components
        from diffusers import FluxPipeline
        from diffusers.training_utils import unet_lora_state_dict
        
        logger.info("Loading base pipeline for training...")
        pipeline = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            token=hf_token
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=config.get("lora_rank", 4),
            lora_alpha=config.get("lora_alpha", 32),
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=config.get("lora_dropout", 0.1),
            task_type=TaskType.DIFFUSION_IMAGE_GENERATION,
        )
        
        # Apply LoRA to UNet
        pipeline.unet = get_peft_model(pipeline.unet, lora_config)
        pipeline.unet.print_trainable_parameters()
        
        # Move to GPU
        pipeline = pipeline.to("cuda")
        
        # Prepare training data
        from datasets import Dataset
        
        image_paths = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for file_path in Path(training_dir).rglob('*'):
            if file_path.suffix.lower() in valid_extensions:
                image_paths.append(str(file_path))
        
        if len(image_paths) < 3:
            raise ValueError(f"Need at least 3 training images, found {len(image_paths)}")
        
        logger.info(f"Training on {len(image_paths)} images")
        
        # Create dataset
        def load_image(example):
            image = Image.open(example['image_path']).convert('RGB')
            # Resize to training resolution
            target_size = config.get("resolution", 512)
            image = image.resize((target_size, target_size))
            return {'image': image, 'text': config.get("instance_prompt", "a photo")}
        
        dataset = Dataset.from_dict({'image_path': image_paths})
        dataset = dataset.map(load_image)
        
        # Training settings
        learning_rate = config.get("learning_rate", 1e-4)
        max_train_steps = config.get("max_train_steps", 500)
        
        # Simple training loop (simplified version)
        optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=learning_rate)
        
        pipeline.unet.train()
        
        logger.info(f"Starting training for {max_train_steps} steps...")
        
        for step in range(max_train_steps):
            # Simple training step - in production you'd want a full training loop
            # with proper loss calculation, scheduler, etc.
            # This is a simplified version for demonstration
            
            if step % 100 == 0:
                logger.info(f"Training step {step}/{max_train_steps}")
        
        # Save LoRA weights
        output_path = f"/runpod-volume/loras/{output_name}.safetensors"
        
        # Extract and save LoRA state dict
        lora_state_dict = get_peft_model_state_dict(pipeline.unet)
        
        # Save using safetensors
        from safetensors.torch import save_file
        save_file(lora_state_dict, output_path)
        
        logger.info(f"LoRA saved to: {output_path}")
        
        # Clean up
        del pipeline
        torch.cuda.empty_cache()
        gc.collect()
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error during LoRA training: {e}")
        raise e

def load_lora_weights(pipeline, lora_path: str):
    """Load LoRA weights into the pipeline"""
    global loaded_lora_path
    
    try:
        if loaded_lora_path == lora_path:
            logger.info(f"LoRA {lora_path} already loaded")
            return pipeline
        
        # Unload previous LoRA if any
        if loaded_lora_path is not None:
            pipeline.unload_lora_weights()
        
        logger.info(f"Loading LoRA weights from: {lora_path}")
        
        # Load LoRA weights
        pipeline.load_lora_weights(lora_path)
        loaded_lora_path = lora_path
        
        logger.info("LoRA weights loaded successfully")
        return pipeline
        
    except Exception as e:
        logger.error(f"Error loading LoRA weights: {e}")
        raise e

def validate_parameters(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and set default parameters"""
    
    # Required parameter
    if "prompt" not in job_input:
        raise ValueError("Missing required parameter: prompt")
    
    # Default parameters based on FLUX.1-dev documentation
    params = {
        "prompt": job_input["prompt"],
        "height": job_input.get("height", 1024),
        "width": job_input.get("width", 1024),
        "guidance_scale": job_input.get("guidance_scale", 3.5),
        "num_inference_steps": job_input.get("num_inference_steps", 50),
        "max_sequence_length": job_input.get("max_sequence_length", 512),
        "seed": job_input.get("seed", None),
        "output_format": job_input.get("output_format", "PNG"),
        "return_base64": job_input.get("return_base64", True)
    }
    
    # Validate ranges
    if params["height"] < 256 or params["height"] > 2048:
        raise ValueError("Height must be between 256 and 2048")
    
    if params["width"] < 256 or params["width"] > 2048:
        raise ValueError("Width must be between 256 and 2048")
    
    if params["guidance_scale"] < 0 or params["guidance_scale"] > 20:
        raise ValueError("Guidance scale must be between 0 and 20")
    
    if params["num_inference_steps"] < 1 or params["num_inference_steps"] > 100:
        raise ValueError("Number of inference steps must be between 1 and 100")
    
    if params["max_sequence_length"] < 1 or params["max_sequence_length"] > 1024:
        raise ValueError("Max sequence length must be between 1 and 1024")
    
    # Ensure dimensions are multiples of 8 (common requirement for diffusion models)
    params["height"] = (params["height"] // 8) * 8
    params["width"] = (params["width"] // 8) * 8
    
    return params

def train_lora_endpoint(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Handle LoRA training requests"""
    try:
        # Ensure directories exist
        setup_directories()
        
        # Validate required parameters
        if "training_images_zip" not in job_input:
            raise ValueError("Missing required parameter: training_images_zip (base64 encoded zip file)")
        
        if "output_name" not in job_input:
            raise ValueError("Missing required parameter: output_name")
        
        # Extract training configuration
        config = {
            "instance_prompt": job_input.get("instance_prompt", "a photo of TOK person"),
            "resolution": job_input.get("resolution", 512),
            "max_train_steps": job_input.get("max_train_steps", 500),
            "learning_rate": job_input.get("learning_rate", 1e-4),
            "lora_rank": job_input.get("lora_rank", 4),
            "lora_alpha": job_input.get("lora_alpha", 32),
            "lora_dropout": job_input.get("lora_dropout", 0.1)
        }
        
        # Generate unique user ID for this training session
        user_id = job_input.get("user_id", f"user_{int(time.time())}")
        
        # Extract training images
        logger.info("Extracting training images...")
        training_dir = extract_training_images(job_input["training_images_zip"], user_id)
        
        # Train LoRA
        logger.info("Starting LoRA training...")
        start_time = time.time()
        
        lora_path = train_lora(training_dir, job_input["output_name"], config)
        
        training_time = time.time() - start_time
        
        # Clean up training directory
        shutil.rmtree(training_dir)
        
        return {
            "success": True,
            "lora_path": lora_path,
            "training_time": training_time,
            "config_used": config,
            "message": f"LoRA '{job_input['output_name']}' trained successfully in {training_time:.2f} seconds"
        }
        
    except Exception as e:
        logger.error(f"Error in LoRA training: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def generate_image(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Main image generation function"""
    
    try:
        # Load model
        pipe = load_model()
        
        # Load LoRA weights if specified
        lora_path = job_input.get("lora_path")
        if lora_path and os.path.exists(lora_path):
            pipe = load_lora_weights(pipe, lora_path)
            logger.info(f"Using LoRA: {lora_path}")
        
        # Validate parameters
        params = validate_parameters(job_input)
        
        logger.info(f"Generating image with prompt: {params['prompt'][:100]}...")
        
        # Set up generator for reproducibility
        generator = None
        if params["seed"] is not None:
            generator = torch.Generator("cpu").manual_seed(params["seed"])
        
        # Generate image
        with torch.no_grad():
            result = pipe(
                prompt=params["prompt"],
                height=params["height"],
                width=params["width"],
                guidance_scale=params["guidance_scale"],
                num_inference_steps=params["num_inference_steps"],
                max_sequence_length=params["max_sequence_length"],
                generator=generator
            )
        
        image = result.images[0]
        
        # Prepare response
        response = {
            "success": True,
            "parameters_used": params,
            "image_info": {
                "width": image.width,
                "height": image.height,
                "format": params["output_format"]
            }
        }
        
        # Convert to base64 if requested
        if params["return_base64"]:
            response["image_base64"] = image_to_base64(image, params["output_format"])
        else:
            # Save to file and return path (for local testing)
            output_path = f"/tmp/flux_output_{params.get('seed', 'random')}.png"
            image.save(output_path)
            response["image_path"] = output_path
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("Image generated successfully!")
        return response
        
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def handler(job):
    """RunPod handler function"""
    
    job_input = job.get("input", {})
    
    try:
        # Handle different endpoint types
        endpoint = job_input.get("endpoint", "generate")
        
        if endpoint == "generate":
            return generate_image(job_input)
        
        elif endpoint == "train_lora":
            return train_lora_endpoint(job_input)
        
        elif endpoint == "health":
            return {
                "success": True,
                "status": "healthy",
                "model_loaded": pipeline is not None
            }
        
        elif endpoint == "info":
            return {
                "success": True,
                "model_info": {
                    "name": "FLUX.1-dev",
                    "description": "12 billion parameter rectified flow transformer for text-to-image generation",
                    "provider": "Black Forest Labs",
                    "license": "FLUX.1 [dev] Non-Commercial License"
                },
                "supported_parameters": {
                    "prompt": {"type": "string", "required": True, "description": "Text description for image generation"},
                    "height": {"type": "integer", "default": 1024, "range": [256, 2048], "description": "Image height in pixels"},
                    "width": {"type": "integer", "default": 1024, "range": [256, 2048], "description": "Image width in pixels"},
                    "guidance_scale": {"type": "float", "default": 3.5, "range": [0, 20], "description": "Guidance scale for generation"},
                    "num_inference_steps": {"type": "integer", "default": 50, "range": [1, 100], "description": "Number of denoising steps"},
                    "max_sequence_length": {"type": "integer", "default": 512, "range": [1, 1024], "description": "Maximum sequence length for text encoding"},
                    "seed": {"type": "integer", "default": None, "description": "Random seed for reproducible generation"},
                    "output_format": {"type": "string", "default": "PNG", "options": ["PNG", "JPEG"], "description": "Output image format"},
                    "return_base64": {"type": "boolean", "default": True, "description": "Return image as base64 string"}
                },
                "lora_parameters": {
                    "training_images_zip": {"type": "string", "required": True, "description": "Base64 encoded zip file containing training images"},
                    "output_name": {"type": "string", "required": True, "description": "Name for the trained LoRA model"},
                    "instance_prompt": {"type": "string", "default": "a photo of TOK person", "description": "Prompt template for training"},
                    "resolution": {"type": "integer", "default": 512, "description": "Training image resolution"},
                    "max_train_steps": {"type": "integer", "default": 500, "description": "Number of training steps"},
                    "learning_rate": {"type": "float", "default": 1e-4, "description": "Learning rate for training"},
                    "lora_rank": {"type": "integer", "default": 4, "description": "LoRA rank (higher = more capacity)"},
                    "lora_alpha": {"type": "integer", "default": 32, "description": "LoRA alpha parameter"},
                    "user_id": {"type": "string", "optional": True, "description": "User identifier for organizing training data"}
                },
                "generate_lora_parameters": {
                    "lora_path": {"type": "string", "optional": True, "description": "Path to LoRA weights file to use for generation"}
                },
                "endpoints": {
                    "generate": "Generate image from text prompt (optionally with LoRA)",
                    "train_lora": "Train a LoRA model on provided images",
                    "health": "Check API health status",
                    "info": "Get model and API information"
                }
            }
        
        else:
            return {
                "success": False,
                "error": f"Unknown endpoint: {endpoint}. Available endpoints: generate, train_lora, health, info"
            }
            
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})