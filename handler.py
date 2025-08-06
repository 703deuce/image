import runpod
import torch
import base64
import io
import zipfile
import shutil
import time
import json
from PIL import Image
from diffusers import FluxPipeline, AutoencoderKL
# Simplified imports for LoRA functionality
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
krea_pipeline = None
loaded_lora_path = None
krea_loaded_lora_path = None

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

def load_krea_model():
    """Load the FLUX.1-Krea-dev model pipeline with custom VAE"""
    global krea_pipeline
    
    if krea_pipeline is None:
        logger.info("Loading FLUX.1-Krea-dev model...")
        
        try:
            # Get HF token for gated model access
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                raise ValueError("HF_TOKEN environment variable is required for FLUX.1-Krea-dev access")
            
            # Debug: Check cache and directories
            logger.info("=== DEBUGGING KREA MODEL LOADING ===")
            logger.info(f"HF_TOKEN: {'SET' if hf_token else 'NONE'}")
            logger.info(f"Cache dir exists: {os.path.exists('/runpod-volume/cache')}")
            logger.info(f"Loading from: black-forest-labs/FLUX.1-Krea-dev")
            
            # Load FLUX.1-Krea-dev pipeline from HuggingFace
            logger.info("Loading FLUX.1-Krea-dev pipeline components...")
            krea_pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Krea-dev",  # Correct Krea repository
                torch_dtype=torch.bfloat16,
                token=hf_token,
                cache_dir="/runpod-volume/cache"  # Same cache location as regular model
            )
            
            logger.info("✅ FLUX.1-Krea-dev pipeline loaded successfully!")
            
            # Load custom VAE (ae.safetensors) for enhanced quality
            logger.info("Loading custom VAE from Krea model...")
            try:
                # For FLUX models, the VAE might be directly in the main repo, not subfolder
                # Try multiple approaches to load the custom VAE
                custom_vae = None
                
                # Method 1: Try loading from vae subfolder
                try:
                    custom_vae = AutoencoderKL.from_pretrained(
                        "black-forest-labs/FLUX.1-Krea-dev",
                        subfolder="vae",
                        torch_dtype=torch.bfloat16,
                        token=hf_token,
                        cache_dir="/runpod-volume/cache"
                    )
                    logger.info("✅ Custom VAE loaded from vae subfolder")
                except Exception as subfolder_error:
                    logger.info(f"VAE subfolder method failed: {subfolder_error}")
                    
                    # Method 2: Try loading VAE directly from main repo
                    try:
                        # The ae.safetensors might be the main VAE file
                        custom_vae = AutoencoderKL.from_pretrained(
                            "black-forest-labs/FLUX.1-Krea-dev",
                            torch_dtype=torch.bfloat16,
                            token=hf_token,
                            cache_dir="/runpod-volume/cache",
                            # Look specifically for ae.safetensors
                            use_safetensors=True
                        )
                        logger.info("✅ Custom VAE loaded from main repository")
                    except Exception as main_error:
                        logger.warning(f"Main repo VAE loading failed: {main_error}")
                        raise Exception("Both VAE loading methods failed")
                
                if custom_vae:
                    # Replace the default VAE with the custom one (ae.safetensors)
                    krea_pipeline.vae = custom_vae
                    logger.info("✅ Custom VAE (ae.safetensors) loaded and applied to Krea pipeline")
                
            except Exception as vae_error:
                logger.warning(f"⚠️ Could not load custom VAE: {vae_error}")
                logger.info("Continuing with default VAE (reduced quality)...")
                logger.info("The Krea model will still work, but without enhanced VAE quality")
            
            # Move to GPU
            logger.info("Moving Krea pipeline to CUDA...")
            krea_pipeline = krea_pipeline.to("cuda")
            logger.info("Krea pipeline loaded successfully on GPU")
            
            logger.info("✅ FLUX.1-Krea-dev model with custom VAE loaded successfully!")
            logger.info(f"📁 Cached to: /runpod-volume/cache")
            
        except Exception as e:
            logger.error(f"❌ Error loading Krea model: {e}")
            raise e
    
    return krea_pipeline

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

def create_ai_toolkit_config(training_dir: str, output_name: str, config: Dict[str, Any]) -> str:
    """Generate ai-toolkit YAML config for FLUX LoRA training"""
    
    # Get HF token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required")
    
    # Training parameters
    instance_prompt = config.get("instance_prompt", "a photo of TOK person")
    max_train_steps = config.get("max_train_steps", 1000)
    learning_rate = config.get("learning_rate", 5e-5)
    lora_rank = config.get("lora_rank", 16)
    lora_alpha = config.get("lora_alpha", 16)
    resolution = config.get("resolution", 768)
    train_batch_size = config.get("train_batch_size", 1)
    
    # Output directory
    output_dir = f"/runpod-volume/cache/lora/{output_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create ai-toolkit config
    ai_config = f"""job: extension
config:
  name: "{output_name}"
  process:
    - type: sd_trainer
      training_folder: "{output_dir}"
      device: cuda:0
      trigger_word: "TOK"
      network:
        type: lora
        linear: {lora_rank}
        linear_alpha: {lora_alpha}
      save:
        dtype: float16
        save_every: {max_train_steps}
        max_step_saves_to_keep: 1
      datasets:
        - folder_path: "{training_dir}"
          caption_ext: "txt"
          caption_dropout_rate: 0.1
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: {resolution}
      train:
        batch_size: {train_batch_size}
        steps: {max_train_steps}
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        content_or_style: balanced
        gradient_checkpointing: true
        noise_scheduler: flowmatch
        optimizer: adamw8bit
        lr: {learning_rate}
        ema_config:
          use_ema: true
          ema_decay: 0.99
        dtype: bf16
      model:
        name_or_path: "black-forest-labs/FLUX.1-dev"
        is_flux: true
        quantize: true
        low_vram: true
      sample:
        sampler: flowmatch
        sample_every: {max_train_steps}
        width: {resolution}
        height: {resolution}
        prompts:
          - "a photo of TOK person"
          - "portrait of TOK person"
          - "TOK person smiling"
        neg: ""
        seed: 42
        walk_seed: true
        guidance_scale: 4.0
        sample_steps: 20
meta:
  name: "[time] {output_name}"
  version: "1.0"
"""
    
    # Save config file
    config_path = f"/tmp/{output_name}_config.yaml"
    with open(config_path, 'w') as f:
        f.write(ai_config)
    
    logger.info(f"Generated ai-toolkit config: {config_path}")
    return config_path

def setup_ai_toolkit():
    """Install ai-toolkit at runtime to persistent volume (like model weights)"""
    try:
        logger.info("🔍 Checking for ai-toolkit installation...")
        
        # Check if already installed in persistent volume
        if os.path.exists("/runpod-volume/ai-toolkit/run.py"):
            logger.info("✅ ai-toolkit found in persistent volume")
            return "manual"
        
        # Runtime installation to persistent volume (like model weights)
        logger.info("📦 Installing ai-toolkit at runtime to persistent volume...")
        
        import subprocess
        import sys
        
        # Clone ai-toolkit to persistent volume
        logger.info("🔄 Cloning ai-toolkit repository...")
        subprocess.run([
            "git", "clone", "--recursive",
            "https://github.com/ostris/ai-toolkit.git", 
            "/runpod-volume/ai-toolkit"
        ], check=True, cwd="/runpod-volume")
        
        # Check torch installation (RunPod may already have it)
        try:
            import torch
            logger.info(f"✅ Using existing torch {torch.__version__}")
        except ImportError:
            logger.info("🔄 Installing torch for CUDA 12.1...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "--no-cache-dir",
                "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
                "--index-url", "https://download.pytorch.org/whl/cu121"
            ], check=True)
        
        # Create python packages directory
        os.makedirs("/runpod-volume/python-packages", exist_ok=True)
        
        # Install ai-toolkit requirements to persistent volume (avoid disk space issues)
        logger.info("🔄 Installing ai-toolkit requirements to persistent volume...")
        
        # Force ALL packages to install to network storage
        pip_env = os.environ.copy()
        pip_env["PYTHONUSERBASE"] = "/runpod-volume/python-packages"
        pip_env["PIP_TARGET"] = "/runpod-volume/python-packages"
        
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "--no-cache-dir",
            "--target", "/runpod-volume/python-packages",
            "--no-deps",  # Install without dependencies first
            "-r", "/runpod-volume/ai-toolkit/requirements.txt"
        ], capture_output=True, text=True, env=pip_env)
        
        # Then install dependencies to same location
        if result.returncode == 0:
            logger.info("🔄 Installing dependencies to persistent volume...")
            result2 = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "--no-cache-dir",
                "--target", "/runpod-volume/python-packages",
                "--upgrade",  # Allow upgrades of existing packages
                "-r", "/runpod-volume/ai-toolkit/requirements.txt"
            ], capture_output=True, text=True, env=pip_env)
            
            if result2.returncode != 0:
                logger.error(f"❌ Failed to install dependencies: {result2.stderr}")
                result = result2  # Use dependency install result for error handling
        
        if result.returncode != 0:
            logger.error(f"❌ Failed to install requirements: {result.stderr}")
            raise RuntimeError(f"Requirements installation failed: {result.stderr}")
        
        logger.info("✅ ai-toolkit requirements installed successfully")
        
        # Add to Python path so packages can be found
        import sys
        if "/runpod-volume/python-packages" not in sys.path:
            sys.path.insert(0, "/runpod-volume/python-packages")
            
        # Verify critical packages can be imported
        try:
            import oyaml
            logger.info("✅ oyaml package verified")
        except ImportError as e:
            logger.error(f"❌ Critical package missing: {e}")
            raise
        
        logger.info("✅ ai-toolkit installed successfully at runtime")
        return "manual"
                
    except Exception as e:
        logger.error(f"❌ Error setting up ai-toolkit: {e}")
        raise e

def train_lora(training_dir: str, output_name: str, config: Dict[str, Any]) -> str:
    """Train FLUX LoRA using ostris/ai-toolkit"""
    try:
        logger.info(f"Starting FLUX LoRA training with ai-toolkit: {config}")
        
        # Validate training images
        image_paths = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for file_path in Path(training_dir).rglob('*'):
            if file_path.suffix.lower() in valid_extensions:
                try:
                    # Verify image can be opened
                    with Image.open(file_path) as img:
                        img.verify()
                    image_paths.append(str(file_path))
                except Exception as e:
                    logger.warning(f"Skipping invalid image {file_path}: {e}")
        
        if len(image_paths) < 3:
            raise ValueError(f"Need at least 3 valid training images, found {len(image_paths)}")
        
        logger.info(f"Training on {len(image_paths)} valid images")
        
        # Create caption files for each image
        instance_prompt = config.get("instance_prompt", "a photo of TOK person")
        for image_path in image_paths:
            caption_path = image_path.rsplit('.', 1)[0] + '.txt'
            if not os.path.exists(caption_path):
                with open(caption_path, 'w') as f:
                    f.write(instance_prompt)
        
        logger.info(f"Created {len(image_paths)} caption files")
        
        # Ensure ai-toolkit is available
        toolkit_type = setup_ai_toolkit()
        
        # Generate ai-toolkit config
        config_path = create_ai_toolkit_config(training_dir, output_name, config)
        
        # Run ai-toolkit training
        logger.info("Starting ai-toolkit training...")
        
        # Set environment variables for ai-toolkit
        env = os.environ.copy()
        env["HF_TOKEN"] = os.environ.get("HF_TOKEN")
        env["CUDA_VISIBLE_DEVICES"] = "0"
        
        # Ensure Python can find packages in persistent volume
        current_pythonpath = env.get("PYTHONPATH", "")
        python_paths = ["/runpod-volume/python-packages"]
        if current_pythonpath:
            python_paths.append(current_pythonpath)
        env["PYTHONPATH"] = ":".join(python_paths)
        
        logger.info(f"🐍 PYTHONPATH set to: {env['PYTHONPATH']}")
        
        # Also add to current Python path for safety
        import sys
        if "/runpod-volume/python-packages" not in sys.path:
            sys.path.insert(0, "/runpod-volume/python-packages")
        
        # Run ai-toolkit as script (it's not a pip package)
        import subprocess
        import sys
        
        # ai-toolkit is always used as a script, never as a package
        cmd = [
            sys.executable, "/runpod-volume/ai-toolkit/run.py", 
            config_path
        ]
        cwd = "/runpod-volume/ai-toolkit"
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            env=env,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            logger.error(f"ai-toolkit training failed:")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            raise RuntimeError(f"ai-toolkit training failed: {result.stderr}")
        
        logger.info("ai-toolkit training completed successfully!")
        logger.info(f"STDOUT: {result.stdout}")
        
        # Find the generated LoRA file
        output_dir = f"/runpod-volume/cache/lora/{output_name}"
        lora_files = list(Path(output_dir).glob("*.safetensors"))
        
        if not lora_files:
            raise FileNotFoundError(f"No LoRA files found in {output_dir}")
        
        # Use the most recent LoRA file
        lora_path = str(sorted(lora_files, key=lambda x: x.stat().st_mtime)[-1])
        
        logger.info(f"✅ LoRA training completed: {lora_path}")
        
        # Clean up config file
        if os.path.exists(config_path):
            os.remove(config_path)
        
        return lora_path
        
    except Exception as e:
        logger.error(f"❌ Error during ai-toolkit LoRA training: {e}")
        raise e

def load_multiple_lora_weights(pipeline, lora_configs: List[Dict[str, Any]]):
    """Load multiple LoRA weights into the pipeline for stacking effects
    
    Args:
        pipeline: The diffusion pipeline
        lora_configs: List of LoRA configurations, each containing:
            - path: str - Path to the LoRA file
            - weight: float - Weight/strength of the LoRA (default 1.0)
            - adapter_name: str - Unique name for the adapter (optional)
    """
    
    try:
        logger.info(f"Loading {len(lora_configs)} LoRA models for stacking...")
        
        # Unload any existing LoRAs first
        try:
            pipeline.unload_lora_weights()
            logger.info("Previous LoRAs unloaded")
        except Exception as e:
            logger.warning(f"Could not unload previous LoRAs: {e}")
        
        adapter_names = []
        adapter_weights = []
        
        for i, lora_config in enumerate(lora_configs):
            lora_path = lora_config["path"]
            weight = lora_config.get("weight", 1.0)
            adapter_name = lora_config.get("adapter_name", f"lora_{i}")
            
            # Verify LoRA file exists
            if not os.path.exists(lora_path):
                logger.warning(f"LoRA file not found, skipping: {lora_path}")
                continue
                
            logger.info(f"Loading LoRA {i+1}/{len(lora_configs)}: {os.path.basename(lora_path)} (weight={weight})")
            
            # Load LoRA weights with unique adapter name
            pipeline.load_lora_weights(lora_path, adapter_name=adapter_name)
            
            adapter_names.append(adapter_name)
            adapter_weights.append(weight)
            
            logger.info(f"✅ LoRA '{adapter_name}' loaded successfully")
        
        # Activate all adapters with their respective weights
        if adapter_names:
            pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
            logger.info(f"✅ {len(adapter_names)} LoRA adapters activated with weights: {adapter_weights}")
            
            # Log the combination being used
            for name, weight in zip(adapter_names, adapter_weights):
                logger.info(f"  - {name}: weight {weight}")
        else:
            logger.warning("No LoRAs were successfully loaded")
        
        return pipeline
        
    except Exception as e:
        logger.error(f"❌ Error loading multiple LoRA weights: {e}")
        logger.warning("Continuing generation without LoRAs")
        return pipeline

def load_lora_weights(pipeline, lora_path: str):
    """Load single LoRA weights into the pipeline (backward compatibility)"""
    lora_configs = [{"path": lora_path, "weight": 1.0, "adapter_name": "default"}]
    return load_multiple_lora_weights(pipeline, lora_configs)

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
        "guidance_scale": job_input.get("guidance_scale", 3.0),
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

def download_lora_endpoint(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Handle LoRA download requests"""
    try:
        # Validate required parameters
        if "lora_name" not in job_input:
            raise ValueError("Missing required parameter: lora_name")
        
        lora_name = job_input["lora_name"]
        lora_path = f"/runpod-volume/loras/{lora_name}"
        
        # Add .safetensors extension if not present
        if not lora_path.endswith('.safetensors'):
            lora_path += '.safetensors'
        
        # Check if LoRA file exists
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA file not found: {lora_name}")
        
        # Read and encode the LoRA file
        with open(lora_path, 'rb') as f:
            lora_data = f.read()
        
        lora_base64 = base64.b64encode(lora_data).decode('utf-8')
        file_size = len(lora_data)
        
        logger.info(f"LoRA file '{lora_name}' prepared for download ({file_size} bytes)")
        
        return {
            "success": True,
            "lora_name": lora_name,
            "lora_base64": lora_base64,
            "file_size": file_size,
            "message": f"LoRA '{lora_name}' ready for download"
        }
        
    except Exception as e:
        logger.error(f"Error downloading LoRA: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def list_loras_endpoint(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """List all available LoRAs"""
    try:
        loras_dir = "/runpod-volume/loras"
        
        if not os.path.exists(loras_dir):
            return {
                "success": True,
                "loras": [],
                "message": "No LoRAs directory found"
            }
        
        # Get all .safetensors files
        lora_files = []
        for filename in os.listdir(loras_dir):
            if filename.endswith('.safetensors'):
                filepath = os.path.join(loras_dir, filename)
                file_stats = os.stat(filepath)
                
                lora_files.append({
                    "name": filename,
                    "size": file_stats.st_size,
                    "created": time.ctime(file_stats.st_ctime),
                    "modified": time.ctime(file_stats.st_mtime)
                })
        
        # Sort by creation time (newest first)
        lora_files.sort(key=lambda x: x["created"], reverse=True)
        
        logger.info(f"Found {len(lora_files)} LoRA files")
        
        return {
            "success": True,
            "loras": lora_files,
            "count": len(lora_files),
            "message": f"Found {len(lora_files)} LoRA models"
        }
        
    except Exception as e:
        logger.error(f"Error listing LoRAs: {e}")
        return {
            "success": False,
            "error": str(e)
        }

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
        
        # Extract training configuration (optimized for single person, 12 images)
        config = {
            "instance_prompt": job_input.get("instance_prompt", "a photo of TOK person"),
            "resolution": job_input.get("resolution", 768),  # Optimal for face consistency  
            "max_train_steps": job_input.get("max_train_steps", 1000),  # Perfect for 12 images
            "learning_rate": job_input.get("learning_rate", 5e-5),  # Sweet spot for LoRA
            "lora_rank": job_input.get("lora_rank", 16),  # Good for facial identity
            "lora_alpha": job_input.get("lora_alpha", 16),  # Usually same as rank
            "lora_dropout": job_input.get("lora_dropout", 0.1),
            "train_batch_size": job_input.get("train_batch_size", 1),  # Limited by VRAM
            "gradient_accumulation_steps": job_input.get("gradient_accumulation_steps", 1),
            "lr_scheduler": job_input.get("lr_scheduler", "constant"),
            "lr_warmup_steps": job_input.get("lr_warmup_steps", 0),
            "mixed_precision": job_input.get("mixed_precision", "fp16")  # Save VRAM
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
        
        # Prepare LoRA configurations
        lora_configs = []
        
        # Add skin enhancement LoRA automatically if enabled (default: True)
        use_skin_enhancement = job_input.get("use_skin_enhancement", True)
        skin_lora_path = "/runpod-volume/cache/lora/aidmaRealisticSkin-FLUX-v0.1.safetensors"
        
        if use_skin_enhancement and os.path.exists(skin_lora_path):
            skin_weight = job_input.get("skin_lora_weight", 0.8)  # Moderate strength by default
            lora_configs.append({
                "path": skin_lora_path,
                "weight": skin_weight,
                "adapter_name": "realistic_skin"
            })
            logger.info(f"Adding realistic skin LoRA with weight {skin_weight}")
        elif use_skin_enhancement:
            logger.warning(f"Skin LoRA not found at {skin_lora_path}, continuing without it")
        
        # Add user-specified LoRA (e.g., subject LoRA)
        lora_path = job_input.get("lora_path")
        if lora_path:
            lora_weight = job_input.get("lora_weight", 1.0)
            lora_configs.append({
                "path": lora_path,
                "weight": lora_weight,
                "adapter_name": "subject_lora"
            })
            logger.info(f"Adding subject LoRA: {lora_path} with weight {lora_weight}")
        
        # Add any additional LoRAs specified in lora_configs
        additional_loras = job_input.get("lora_configs", [])
        for i, additional_lora in enumerate(additional_loras):
            lora_configs.append({
                "path": additional_lora.get("path"),
                "weight": additional_lora.get("weight", 1.0),
                "adapter_name": additional_lora.get("adapter_name", f"additional_{i}")
            })
        
        # Load all LoRAs if any are specified
        if lora_configs:
            pipe = load_multiple_lora_weights(pipe, lora_configs)
            logger.info(f"Loaded {len(lora_configs)} LoRA models")
            
            # Add trigger words if using realistic skin LoRA
            if use_skin_enhancement and os.path.exists(skin_lora_path):
                # Add trigger word to prompt if not already present
                prompt = job_input.get("prompt", "")
                if "aidmarealisticskin" not in prompt.lower():
                    # Insert trigger word naturally into the prompt
                    job_input["prompt"] = f"{prompt}, aidmarealisticskin"
                    logger.info("Added realistic skin trigger word to prompt")
        
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
            torch.cuda.synchronize()
        gc.collect()
        
        logger.info("Image generated successfully!")
        return response
        
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def generate_krea_image(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """FLUX.1-Krea-dev image generation function"""
    
    try:
        # Load Krea model
        pipe = load_krea_model()
        
        # Prepare LoRA configurations (same as regular FLUX)
        lora_configs = []
        
        # Add skin enhancement LoRA automatically if enabled (default: True)
        use_skin_enhancement = job_input.get("use_skin_enhancement", True)
        skin_lora_path = "/runpod-volume/cache/lora/aidmaRealisticSkin-FLUX-v0.1.safetensors"
        
        if use_skin_enhancement and os.path.exists(skin_lora_path):
            skin_weight = job_input.get("skin_lora_weight", 0.8)  # Moderate strength by default
            lora_configs.append({
                "path": skin_lora_path,
                "weight": skin_weight,
                "adapter_name": "realistic_skin"
            })
            logger.info(f"Adding realistic skin LoRA to Krea with weight {skin_weight}")
        elif use_skin_enhancement:
            logger.warning(f"Skin LoRA not found at {skin_lora_path}, continuing without it")
        
        # Add user-specified LoRA (e.g., subject LoRA)
        lora_path = job_input.get("lora_path")
        if lora_path:
            lora_weight = job_input.get("lora_weight", 1.0)
            lora_configs.append({
                "path": lora_path,
                "weight": lora_weight,
                "adapter_name": "subject_lora"
            })
            logger.info(f"Adding subject LoRA to Krea: {lora_path} with weight {lora_weight}")
        
        # Add any additional LoRAs specified in lora_configs
        additional_loras = job_input.get("lora_configs", [])
        for i, additional_lora in enumerate(additional_loras):
            lora_configs.append({
                "path": additional_lora.get("path"),
                "weight": additional_lora.get("weight", 1.0),
                "adapter_name": additional_lora.get("adapter_name", f"additional_{i}")
            })
        
        # Load all LoRAs if any are specified
        if lora_configs:
            pipe = load_multiple_lora_weights_krea(pipe, lora_configs)
            logger.info(f"Loaded {len(lora_configs)} LoRA models for Krea")
            
            # Add trigger words if using realistic skin LoRA
            if use_skin_enhancement and os.path.exists(skin_lora_path):
                # Add trigger word to prompt if not already present
                prompt = job_input.get("prompt", "")
                if "aidmarealisticskin" not in prompt.lower():
                    # Insert trigger word naturally into the prompt
                    job_input["prompt"] = f"{prompt}, aidmarealisticskin"
                    logger.info("Added realistic skin trigger word to Krea prompt")
        
        # Validate parameters (same validation as regular model)
        params = validate_parameters(job_input)
        
        logger.info(f"Generating Krea image with prompt: {params['prompt'][:100]}...")
        
        # Set up generator for reproducibility
        generator = None
        if params["seed"] is not None:
            generator = torch.Generator("cpu").manual_seed(params["seed"])
        
        # Generate image with Krea model
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
        
        # Prepare response (same format as regular model)
        response = {
            "success": True,
            "model_used": "FLUX.1-Krea-dev",
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
            output_path = f"/tmp/flux_krea_output_{params.get('seed', 'random')}.png"
            image.save(output_path)
            response["image_path"] = output_path
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        logger.info("Krea image generated successfully!")
        return response
        
    except Exception as e:
        logger.error(f"Error generating Krea image: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def load_multiple_lora_weights_krea(pipeline, lora_configs: List[Dict[str, Any]]):
    """Load multiple LoRA weights into the Krea pipeline for stacking effects"""
    
    try:
        logger.info(f"Loading {len(lora_configs)} LoRA models for Krea stacking...")
        
        # Unload any existing LoRAs first
        try:
            pipeline.unload_lora_weights()
            logger.info("Previous LoRAs unloaded from Krea")
        except Exception as e:
            logger.warning(f"Could not unload previous LoRAs from Krea: {e}")
        
        adapter_names = []
        adapter_weights = []
        
        for i, lora_config in enumerate(lora_configs):
            lora_path = lora_config["path"]
            weight = lora_config.get("weight", 1.0)
            adapter_name = lora_config.get("adapter_name", f"lora_{i}")
            
            # Verify LoRA file exists
            if not os.path.exists(lora_path):
                logger.warning(f"LoRA file not found, skipping: {lora_path}")
                continue
                
            logger.info(f"Loading LoRA {i+1}/{len(lora_configs)} into Krea: {os.path.basename(lora_path)} (weight={weight})")
            
            # Load LoRA weights with unique adapter name
            pipeline.load_lora_weights(lora_path, adapter_name=adapter_name)
            
            adapter_names.append(adapter_name)
            adapter_weights.append(weight)
            
            logger.info(f"✅ LoRA '{adapter_name}' loaded into Krea successfully")
        
        # Activate all adapters with their respective weights
        if adapter_names:
            pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
            logger.info(f"✅ {len(adapter_names)} LoRA adapters activated on Krea with weights: {adapter_weights}")
            
            # Log the combination being used
            for name, weight in zip(adapter_names, adapter_weights):
                logger.info(f"  - Krea {name}: weight {weight}")
        else:
            logger.warning("No LoRAs were successfully loaded into Krea")
        
        return pipeline
        
    except Exception as e:
        logger.error(f"❌ Error loading multiple LoRA weights into Krea: {e}")
        logger.warning("Continuing Krea generation without LoRAs")
        return pipeline

def load_lora_weights_krea(pipeline, lora_path: str):
    """Load single LoRA weights into the Krea pipeline (backward compatibility)"""
    lora_configs = [{"path": lora_path, "weight": 1.0, "adapter_name": "default"}]
    return load_multiple_lora_weights_krea(pipeline, lora_configs)

def handler(job):
    """RunPod handler function"""
    
    job_input = job.get("input", {})
    
    try:
        # Handle different endpoint types
        endpoint = job_input.get("endpoint", "generate")
        
        if endpoint == "generate":
            return generate_image(job_input)
        
        elif endpoint == "generate_krea":
            return generate_krea_image(job_input)
        
        elif endpoint == "train_lora":
            return train_lora_endpoint(job_input)
        
        elif endpoint == "download_lora":
            return download_lora_endpoint(job_input)
        
        elif endpoint == "list_loras":
            return list_loras_endpoint(job_input)
        
        elif endpoint == "health":
            return {
                "success": True,
                "status": "healthy",
                "model_loaded": pipeline is not None,
                "krea_model_loaded": krea_pipeline is not None
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
                    "guidance_scale": {"type": "float", "default": 3.0, "range": [0, 20], "description": "Guidance scale for generation"},
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
                    "resolution": {"type": "integer", "default": 768, "description": "Training image resolution (Replicate standard)"},
                    "max_train_steps": {"type": "integer", "default": 1000, "description": "Number of training steps (Replicate standard)"},
                    "learning_rate": {"type": "float", "default": 1e-4, "description": "Learning rate for training"},
                    "lora_rank": {"type": "integer", "default": 16, "description": "LoRA rank (higher = more capacity, Replicate standard)"},
                    "lora_alpha": {"type": "integer", "default": 32, "description": "LoRA alpha parameter"},
                    "train_batch_size": {"type": "integer", "default": 1, "description": "Training batch size"},
                    "gradient_accumulation_steps": {"type": "integer", "default": 4, "description": "Gradient accumulation steps"},
                    "lr_scheduler": {"type": "string", "default": "constant", "description": "Learning rate scheduler"},
                    "lr_warmup_steps": {"type": "integer", "default": 100, "description": "Learning rate warmup steps"},
                    "user_id": {"type": "string", "optional": True, "description": "User identifier for organizing training data"}
                },
                "generate_lora_parameters": {
                    "lora_path": {"type": "string", "optional": True, "description": "Path to LoRA weights file to use for generation (subject/style LoRA)"},
                    "lora_weight": {"type": "float", "default": 1.0, "description": "Weight/strength of the main LoRA (0.0 to 2.0)"},
                    "use_skin_enhancement": {"type": "boolean", "default": True, "description": "Automatically apply realistic skin LoRA if available"},
                    "skin_lora_weight": {"type": "float", "default": 0.8, "description": "Weight/strength of the skin enhancement LoRA (0.0 to 2.0)"},
                    "lora_configs": {"type": "array", "optional": True, "description": "Advanced: Array of additional LoRA configurations with path, weight, and adapter_name"}
                },
                "download_lora_parameters": {
                    "lora_name": {"type": "string", "required": True, "description": "Name of the LoRA file to download (with or without .safetensors extension)"}
                },
                "endpoints": {
                    "generate": "Generate image from text prompt using FLUX.1-dev with automatic skin enhancement + optional subject LoRA stacking",
                    "generate_krea": "Generate image from text prompt using FLUX.1-Krea-dev with custom VAE + automatic skin enhancement + optional subject LoRA stacking",
                    "train_lora": "Train a LoRA model on provided images",
                    "download_lora": "Download a trained LoRA model as base64",
                    "list_loras": "List all available LoRA models",
                    "health": "Check API health status",
                    "info": "Get model and API information"
                },
                "lora_stacking_info": {
                    "description": "This API supports automatic LoRA stacking for enhanced realism",
                    "automatic_skin_enhancement": "The realistic skin LoRA (aidmaRealisticSkin-FLUX-v0.1.safetensors) is automatically applied by default",
                    "trigger_word": "The 'aidmarealisticskin' trigger word is automatically added to prompts when using skin enhancement",
                    "subject_lora_support": "Your trained subject LoRAs work seamlessly with skin enhancement - specify via 'lora_path'",
                    "advanced_stacking": "Use 'lora_configs' array for complex multi-LoRA combinations beyond skin + subject",
                    "weight_control": "Fine-tune each LoRA's influence with individual weight parameters (0.0 to 2.0 range)"
                }
            }
        
        else:
            return {
                "success": False,
                "error": f"Unknown endpoint: {endpoint}. Available endpoints: generate, generate_krea, train_lora, download_lora, list_loras, health, info"
            }
            
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})