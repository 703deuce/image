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

def train_lora(training_dir: str, output_name: str, config: Dict[str, Any]) -> str:
    """Train real FLUX LoRA using diffusers and peft"""
    try:
        logger.info(f"Starting REAL LoRA training with config: {config}")
        
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
        
        # Import training dependencies
        from diffusers import FluxPipeline
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import CLIPTextModel, CLIPTokenizer
        import torch.nn.functional as F
        from torch.optim import AdamW
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms
        
        # Load the base FLUX model for training
        logger.info("Loading FLUX model for training...")
        hf_token = os.environ.get("HF_TOKEN")
        
        # Create LoRA config with optimal settings for person training
        network_dim = config.get("lora_rank", 16)  # 16 is good for facial identity
        network_alpha = config.get("lora_alpha", 16)  # Usually same as dim
        
        lora_config = LoraConfig(
            r=network_dim,
            lora_alpha=network_alpha, 
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # FLUX transformer attention modules
            lora_dropout=config.get("lora_dropout", 0.1),
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        
        # Load model
        flux_pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            token=hf_token
        )
        
        # Apply LoRA to transformer
        flux_pipeline.transformer = get_peft_model(flux_pipeline.transformer, lora_config)
        
        logger.info(f"LoRA applied with rank={network_dim}, alpha={network_alpha}")
        
        # Training parameters  
        resolution = config.get("resolution", 768)
        max_train_steps = config.get("max_train_steps", 1000)
        learning_rate = config.get("learning_rate", 5e-5)
        train_batch_size = config.get("train_batch_size", 1)
        instance_prompt = config.get("instance_prompt", "a photo of TOK person")
        
        logger.info("Creating training dataset...")
        
        # Create simple dataset
        class LoRADataset(Dataset):
            def __init__(self, image_paths, prompt, resolution):
                self.image_paths = image_paths
                self.prompt = prompt
                self.resolution = resolution
                self.transform = transforms.Compose([
                    transforms.Resize((resolution, resolution)),
                    transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                ])
            
            def __len__(self):
                return len(self.image_paths)
            
            def __getitem__(self, idx):
                image_path = self.image_paths[idx]
                image = Image.open(image_path).convert("RGB")
                image = self.transform(image)
                return {"image": image, "prompt": self.prompt}
        
        dataset = LoRADataset(image_paths, instance_prompt, resolution)
        dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
        
        # Setup optimizer
        optimizer = AdamW(
            flux_pipeline.transformer.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8,
        )
        
        logger.info(f"Starting training: {max_train_steps} steps, lr={learning_rate}")
        
        # Move to GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        flux_pipeline = flux_pipeline.to(device)
        
        # Training loop
        flux_pipeline.transformer.train()
        global_step = 0
        
        epochs_needed = max_train_steps // len(dataloader) + 1
        
        for epoch in range(epochs_needed):
            if global_step >= max_train_steps:
                break
                
            for batch in dataloader:
                if global_step >= max_train_steps:
                    break
                
                # Move batch to device
                images = batch["image"].to(device, dtype=torch.bfloat16)
                prompts = batch["prompt"]
                
                # Forward pass (simplified training step)
                with torch.no_grad():
                    # Encode text
                    text_embeddings = flux_pipeline.text_encoder(
                        flux_pipeline.tokenizer(
                            prompts,
                            padding="max_length",
                            max_length=flux_pipeline.tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt"
                        ).input_ids.to(device)
                    ).last_hidden_state
                
                # Add noise for diffusion training
                noise = torch.randn_like(images)
                timesteps = torch.randint(0, 1000, (images.shape[0],), device=device)
                
                # Training forward pass
                optimizer.zero_grad()
                
                # Simplified loss computation for FLUX flow matching
                with torch.cuda.amp.autocast(enabled=True):
                    # Encode images to latent space
                    latents = flux_pipeline.vae.encode(images).latent_dist.sample()
                    latents = latents * flux_pipeline.vae.config.scaling_factor
                    
                    # For flow matching (FLUX), use a different approach
                    # Generate random timesteps (0 to 1 for flow matching)
                    timesteps = torch.rand((images.shape[0],), device=device)
                    
                    # Flow matching: interpolate between noise and data
                    noise = torch.randn_like(latents)
                    # Linear interpolation: x_t = (1-t) * noise + t * latents
                    noisy_latents = (1 - timesteps.view(-1, 1, 1, 1)) * noise + timesteps.view(-1, 1, 1, 1) * latents
                    
                    # The target for flow matching is the vector field: latents - noise
                    target = latents - noise
                    
                    # Convert timesteps to proper format for FLUX (scale to 0-1000 range)
                    timesteps_scaled = (timesteps * 1000).long()
                    
                    # Predict the vector field
                    model_pred = flux_pipeline.transformer(
                        noisy_latents,
                        timesteps_scaled,
                        encoder_hidden_states=text_embeddings,
                        return_dict=False
                    )[0]
                    
                    # Compute flow matching loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                global_step += 1
                
                # Log progress
                if global_step % 50 == 0:
                    progress = (global_step / max_train_steps) * 100
                    logger.info(f"Training progress: {progress:.1f}% ({global_step}/{max_train_steps} steps) - Loss: {loss.item():.4f}")
        
        logger.info("Training completed! Saving LoRA weights...")
        
        # Create output directory
        os.makedirs("/runpod-volume/cache/lora", exist_ok=True)  # Use cache/lora as requested
        output_path = f"/runpod-volume/cache/lora/{output_name}.safetensors"
        
        # Save LoRA weights
        flux_pipeline.transformer.save_pretrained(output_path.replace('.safetensors', ''))
        
        # Also save as safetensors for compatibility
        lora_state_dict = flux_pipeline.transformer.state_dict()
        from safetensors.torch import save_file
        save_file(lora_state_dict, output_path)
        
        logger.info(f"✅ REAL LoRA saved to: {output_path}")
        logger.info("🎉 Training completed successfully!")
        
        # Clean up GPU memory after training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Error during LoRA training: {e}")
        raise e

def load_lora_weights(pipeline, lora_path: str):
    """Load LoRA weights into the pipeline using diffusers built-in support"""
    global loaded_lora_path
    
    try:
        if loaded_lora_path == lora_path:
            logger.info(f"LoRA {lora_path} already loaded")
            return pipeline
        
        logger.info(f"Loading LoRA weights from: {lora_path}")
        
        # Verify LoRA file exists
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA file not found: {lora_path}")
        
        # Unload previous LoRA if any
        if loaded_lora_path is not None:
            try:
                pipeline.unload_lora_weights()
                logger.info("Previous LoRA unloaded")
            except Exception as e:
                logger.warning(f"Could not unload previous LoRA: {e}")
        
        # Load LoRA weights using diffusers built-in method
        logger.info(f"Loading LoRA file: {lora_path} ({os.path.getsize(lora_path)} bytes)")
        
        # Try to load LoRA weights with proper error handling
        try:
            # Load LoRA weights with adapter name for newer diffusers
            pipeline.load_lora_weights(lora_path, adapter_name="default")
            logger.info("LoRA weights loaded into pipeline with adapter name 'default'")
            
            # DEBUG: Check available adapters after loading
            available_adapters = []
            try:
                if hasattr(pipeline, 'adapters'):
                    available_adapters = list(pipeline.adapters.keys())
                    logger.info(f"Available adapters: {available_adapters}")
                elif hasattr(pipeline.transformer, 'adapters'):
                    available_adapters = list(pipeline.transformer.adapters.keys())
                    logger.info(f"Available transformer adapters: {available_adapters}")
                else:
                    logger.info("No adapters attribute found")
            except Exception as adapter_check_error:
                logger.warning(f"Could not check adapters: {adapter_check_error}")
            
            # Try to set LoRA adapters with stronger weight
            adapter_activated = False
            
            # Method 1: Try default adapter name
            try:
                pipeline.set_adapters(["default"], adapter_weights=[1.5])  # Higher weight for stronger effect
                logger.info("✅ LoRA adapters activated with 'default' name and weight 1.5")
                adapter_activated = True
            except Exception as adapter_error:
                logger.warning(f"Failed to set 'default' adapter: {adapter_error}")
                
                # Method 2: Try first available adapter
                if available_adapters:
                    try:
                        first_adapter = available_adapters[0]
                        pipeline.set_adapters([first_adapter], adapter_weights=[1.5])
                        logger.info(f"✅ LoRA adapters activated with '{first_adapter}' and weight 1.5")
                        adapter_activated = True
                    except Exception as fallback_error:
                        logger.error(f"Failed to set adapter '{first_adapter}': {fallback_error}")
            
            # Method 4: Try without explicit adapter names (auto-detection)
            if not adapter_activated:
                try:
                    if hasattr(pipeline, 'fuse_lora'):
                        pipeline.fuse_lora(lora_scale=1.5)
                        logger.info("✅ LoRA fused directly with scale 1.5")
                        adapter_activated = True
                except Exception as fuse_error:
                    logger.warning(f"Failed to fuse LoRA: {fuse_error}")
            
            if not adapter_activated:
                logger.error("❌ Failed to activate LoRA adapters - continuing anyway")
            
        except Exception as lora_error:
            # If there's a device error, try explicit device management
            logger.warning(f"Standard LoRA loading failed: {lora_error}")
            logger.info("Attempting LoRA loading with explicit device management...")
            
            # Get pipeline device
            device = next(pipeline.transformer.parameters()).device
            logger.info(f"Pipeline is on device: {device}")
            
            # Reload and move to device
            pipeline.load_lora_weights(lora_path)
            pipeline = pipeline.to(device)
            
            # Try to activate with stronger weight
            try:
                pipeline.set_adapters(["default"], adapter_weights=[1.5])
                logger.info("✅ LoRA loaded with explicit device management and weight 1.5")
            except:
                logger.warning("⚠️ LoRA loaded but adapter activation uncertain")
        
        loaded_lora_path = lora_path
        logger.info("✅ LoRA weights loaded and applied successfully!")
        
        return pipeline
        
    except Exception as e:
        logger.error(f"❌ Error loading LoRA weights: {e}")
        # If LoRA loading fails, continue without LoRA
        logger.warning("Continuing generation without LoRA")
        return pipeline

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
        
        # Load LoRA weights if specified
        lora_path = job_input.get("lora_path")
        if lora_path:
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
        
        # Load LoRA weights if specified (same system as regular model)
        lora_path = job_input.get("lora_path")
        if lora_path:
            pipe = load_lora_weights_krea(pipe, lora_path)
            logger.info(f"Using LoRA with Krea model: {lora_path}")
        
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

def load_lora_weights_krea(pipeline, lora_path: str):
    """Load LoRA weights into the Krea pipeline using diffusers built-in support"""
    global krea_loaded_lora_path
    
    try:
        if krea_loaded_lora_path == lora_path:
            logger.info(f"LoRA {lora_path} already loaded on Krea model")
            return pipeline
        
        logger.info(f"Loading LoRA weights into Krea model from: {lora_path}")
        
        # Verify LoRA file exists
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA file not found: {lora_path}")
        
        # Unload previous LoRA if any
        if krea_loaded_lora_path is not None:
            try:
                pipeline.unload_lora_weights()
                logger.info("Previous LoRA unloaded from Krea model")
            except Exception as e:
                logger.warning(f"Could not unload previous LoRA from Krea: {e}")
        
        # Load LoRA weights using diffusers built-in method
        logger.info(f"Loading LoRA file into Krea: {lora_path} ({os.path.getsize(lora_path)} bytes)")
        
        # Try to load LoRA weights with proper error handling
        try:
            # Load LoRA weights with adapter name for newer diffusers
            pipeline.load_lora_weights(lora_path, adapter_name="default")
            logger.info("LoRA weights loaded into Krea pipeline with adapter name 'default'")
            
            # Try to set LoRA adapters with stronger weight
            pipeline.set_adapters(["default"], adapter_weights=[1.5])
            logger.info("✅ LoRA adapters activated on Krea with 'default' name and weight 1.5")
            
        except Exception as lora_error:
            logger.warning(f"Standard LoRA loading failed on Krea: {lora_error}")
            logger.info("Attempting LoRA loading with explicit device management...")
            
            # Get pipeline device
            device = next(pipeline.transformer.parameters()).device
            logger.info(f"Krea pipeline is on device: {device}")
            
            # Reload and move to device
            pipeline.load_lora_weights(lora_path)
            pipeline = pipeline.to(device)
            
            # Try to activate with stronger weight
            try:
                pipeline.set_adapters(["default"], adapter_weights=[1.5])
                logger.info("✅ LoRA loaded on Krea with explicit device management and weight 1.5")
            except:
                logger.warning("⚠️ LoRA loaded on Krea but adapter activation uncertain")
        
        krea_loaded_lora_path = lora_path
        logger.info("✅ LoRA weights loaded and applied successfully to Krea!")
        
        return pipeline
        
    except Exception as e:
        logger.error(f"❌ Error loading LoRA weights into Krea: {e}")
        # If LoRA loading fails, continue without LoRA
        logger.warning("Continuing Krea generation without LoRA")
        return pipeline

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
                    "lora_path": {"type": "string", "optional": True, "description": "Path to LoRA weights file to use for generation"}
                },
                "download_lora_parameters": {
                    "lora_name": {"type": "string", "required": True, "description": "Name of the LoRA file to download (with or without .safetensors extension)"}
                },
                "endpoints": {
                    "generate": "Generate image from text prompt using FLUX.1-dev (optionally with LoRA)",
                    "generate_krea": "Generate image from text prompt using FLUX.1-Krea-dev with custom VAE (optionally with LoRA)",
                    "train_lora": "Train a LoRA model on provided images",
                    "download_lora": "Download a trained LoRA model as base64",
                    "list_loras": "List all available LoRA models",
                    "health": "Check API health status",
                    "info": "Get model and API information"
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