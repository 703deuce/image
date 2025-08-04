import requests
import json
import base64
import io
from PIL import Image
from typing import Optional, Dict, Any, Union
import time

class FluxRunPodAPI:
    """
    Client for interacting with FLUX.1-dev RunPod Serverless API
    """
    
    def __init__(self, endpoint_id: str, api_key: str):
        """
        Initialize the API client
        
        Args:
            endpoint_id: RunPod endpoint ID
            api_key: RunPod API key
        """
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to the RunPod API"""
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")
    
    def generate_image(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 50,
        max_sequence_length: int = 512,
        seed: Optional[int] = None,
        output_format: str = "PNG",
        return_base64: bool = True,
        sync: bool = True,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Generate an image using FLUX.1-dev
        
        Args:
            prompt: Text description for image generation
            height: Image height in pixels (256-2048)
            width: Image width in pixels (256-2048)
            guidance_scale: Guidance scale for generation (0-20)
            num_inference_steps: Number of denoising steps (1-100)
            max_sequence_length: Maximum sequence length for text encoding (1-1024)
            seed: Random seed for reproducible generation
            output_format: Output image format (PNG or JPEG)
            return_base64: Return image as base64 string
            sync: Whether to wait for completion (True) or return job ID (False)
            timeout: Timeout in seconds for sync requests
        
        Returns:
            Dictionary containing the result or job ID
        """
        
        payload = {
            "input": {
                "endpoint": "generate",
                "prompt": prompt,
                "height": height,
                "width": width,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "max_sequence_length": max_sequence_length,
                "seed": seed,
                "output_format": output_format,
                "return_base64": return_base64
            }
        }
        
        if sync:
            return self._make_request("runsync", payload)
        else:
            return self._make_request("run", payload)
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a job
        
        Args:
            job_id: The job ID returned from an async request
        
        Returns:
            Dictionary containing job status and results
        """
        
        url = f"{self.base_url}/status/{job_id}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Status check failed: {e}")
    
    def wait_for_completion(self, job_id: str, timeout: int = 300, poll_interval: int = 5) -> Dict[str, Any]:
        """
        Wait for a job to complete
        
        Args:
            job_id: The job ID to wait for
            timeout: Maximum time to wait in seconds
            poll_interval: How often to check status in seconds
        
        Returns:
            Final job result
        """
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_job_status(job_id)
            
            if status.get("status") == "COMPLETED":
                return status
            elif status.get("status") == "FAILED":
                raise Exception(f"Job failed: {status.get('error', 'Unknown error')}")
            
            time.sleep(poll_interval)
        
        raise Exception(f"Job timed out after {timeout} seconds")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health status of the API
        
        Returns:
            Health status information
        """
        
        payload = {
            "input": {
                "endpoint": "health"
            }
        }
        
        return self._make_request("runsync", payload)
    
    def get_api_info(self) -> Dict[str, Any]:
        """
        Get information about the API and supported parameters
        
        Returns:
            API information and supported parameters
        """
        
        payload = {
            "input": {
                "endpoint": "info"
            }
        }
        
        return self._make_request("runsync", payload)
    
    def base64_to_image(self, base64_string: str) -> Image.Image:
        """
        Convert base64 string to PIL Image
        
        Args:
            base64_string: Base64 encoded image string
        
        Returns:
            PIL Image object
        """
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    
    def save_image_from_response(self, response: Dict[str, Any], filename: str) -> bool:
        """
        Save image from API response to file
        
        Args:
            response: API response containing image_base64
            filename: Output filename
        
        Returns:
            True if successful, False otherwise
        """
        
        try:
            if "output" in response and "image_base64" in response["output"]:
                image = self.base64_to_image(response["output"]["image_base64"])
                image.save(filename)
                return True
            else:
                print("No image data found in response")
                return False
        except Exception as e:
            print(f"Error saving image: {e}")
            return False

# Example usage and testing functions
def example_usage():
    """
    Example of how to use the FluxRunPodAPI
    """
    
    # Replace with your actual endpoint ID and API key
    ENDPOINT_ID = "your-endpoint-id"
    API_KEY = "your-api-key"
    
    # Initialize the API client
    api = FluxRunPodAPI(ENDPOINT_ID, API_KEY)
    
    try:
        # Check API health
        print("Checking API health...")
        health = api.health_check()
        print(f"Health status: {health}")
        
        # Get API information
        print("\nGetting API info...")
        info = api.get_api_info()
        print(f"API Info: {json.dumps(info, indent=2)}")
        
        # Generate an image
        print("\nGenerating image...")
        result = api.generate_image(
            prompt="A majestic dragon flying over a mystical forest at sunset",
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=50,
            seed=42
        )
        
        if result.get("status") == "COMPLETED" and result.get("output", {}).get("success"):
            print("Image generated successfully!")
            
            # Save the image
            if api.save_image_from_response(result, "generated_image.png"):
                print("Image saved as 'generated_image.png'")
            
            # Print generation parameters
            params = result.get("output", {}).get("parameters_used", {})
            print(f"Generation parameters: {json.dumps(params, indent=2)}")
            
        else:
            print(f"Generation failed: {result}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    example_usage()