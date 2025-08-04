import runpod
import torch
import base64
import io
from PIL import Image
from diffusers import FluxPipeline
import logging
import gc
import os
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model loading
pipeline = None

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
            
            # Load the pipeline with authentication from pre-downloaded cache
            pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev", 
                torch_dtype=torch.bfloat16,
                token=hf_token,
                cache_dir="/runpod-volume/cache"
            )
            
            # Enable CPU offload to save VRAM
            pipeline.enable_model_cpu_offload()
            
            # Enable memory efficient attention if available
            if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("Enabled xformers memory efficient attention")
                except Exception as e:
                    logger.warning(f"Could not enable xformers: {e}")
            
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

def generate_image(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Main image generation function"""
    
    try:
        # Load model
        pipe = load_model()
        
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
                "endpoints": {
                    "generate": "Generate image from text prompt",
                    "health": "Check API health status",
                    "info": "Get model and API information"
                }
            }
        
        else:
            return {
                "success": False,
                "error": f"Unknown endpoint: {endpoint}. Available endpoints: generate, health, info"
            }
            
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})