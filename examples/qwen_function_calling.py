"""
Example: Qwen LLM Function Calling Integration

This example shows how to integrate the ImageGeneratorLLM with Qwen using
function calling. The LLM can request image generation as a tool/function.

Requirements:
- ImageGeneratorLLM REST API server running (python -m image_gen.server)
- Qwen LLM with function calling support
"""

import requests
import json
from typing import Dict, Any, List


# Function definitions for Qwen
FUNCTION_DEFINITIONS = [
    {
        "name": "generate_image",
        "description": "Generate an image from a text description using SDXL. Returns the path to the generated image file.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Detailed text description of the image to generate. Be specific about style, mood, composition, and details."
                },
                "width": {
                    "type": "integer",
                    "description": "Output width in pixels (default: 1024). Must be multiple of 8.",
                    "default": 1024
                },
                "height": {
                    "type": "integer",
                    "description": "Output height in pixels (default: 1024). Must be multiple of 8.",
                    "default": 1024
                },
                "num_inference_steps": {
                    "type": "integer",
                    "description": "Number of denoising steps. More steps = higher quality but slower. Range: 10-50 (default: 30)",
                    "default": 30
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducible results. Same seed + prompt = identical image.",
                    "default": None
                }
            },
            "required": ["prompt"]
        }
    },
    {
        "name": "generate_images_batch",
        "description": "Generate multiple images from a list of text descriptions. More efficient than calling generate_image multiple times.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of text descriptions for image generation"
                },
                "num_inference_steps": {
                    "type": "integer",
                    "description": "Number of denoising steps for all images (default: 30)",
                    "default": 30
                }
            },
            "required": ["prompts"]
        }
    }
]


class ImageGeneratorClient:
    """Client for calling ImageGeneratorLLM REST API."""

    def __init__(self, api_url: str = "http://localhost:8000"):
        """
        Initialize client.

        Args:
            api_url: Base URL of the ImageGeneratorLLM API
        """
        self.api_url = api_url

    def generate_image(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        seed: int = None
    ) -> Dict[str, Any]:
        """
        Generate a single image.

        Args:
            prompt: Text description
            width: Output width
            height: Output height
            num_inference_steps: Quality/speed tradeoff
            seed: Random seed for reproducibility

        Returns:
            Response with image_path and metadata
        """
        response = requests.post(
            f"{self.api_url}/generate",
            json={
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "seed": seed
            }
        )
        response.raise_for_status()
        return response.json()

    def generate_images_batch(
        self,
        prompts: List[str],
        num_inference_steps: int = 30
    ) -> Dict[str, Any]:
        """
        Generate multiple images.

        Args:
            prompts: List of text descriptions
            num_inference_steps: Quality/speed tradeoff

        Returns:
            Response with list of image_paths
        """
        response = requests.post(
            f"{self.api_url}/generate/batch",
            json={
                "prompts": prompts,
                "num_inference_steps": num_inference_steps
            }
        )
        response.raise_for_status()
        return response.json()

    def health_check(self) -> Dict[str, Any]:
        """Check API health and status."""
        response = requests.get(f"{self.api_url}/health")
        response.raise_for_status()
        return response.json()


def execute_function_call(
    function_name: str,
    function_args: Dict[str, Any],
    client: ImageGeneratorClient
) -> Dict[str, Any]:
    """
    Execute a function call from Qwen.

    Args:
        function_name: Name of the function to call
        function_args: Arguments for the function
        client: ImageGeneratorClient instance

    Returns:
        Function execution result
    """
    if function_name == "generate_image":
        result = client.generate_image(**function_args)
        return {
            "success": True,
            "image_path": result["image_path"],
            "width": result["width"],
            "height": result["height"],
            "message": f"Generated image: {result['image_path']}"
        }

    elif function_name == "generate_images_batch":
        result = client.generate_images_batch(**function_args)
        return {
            "success": True,
            "image_paths": result["image_paths"],
            "count": result["count"],
            "message": f"Generated {result['count']} images"
        }

    else:
        return {
            "success": False,
            "error": f"Unknown function: {function_name}"
        }


def example_conversation():
    """
    Example conversation with Qwen using function calling.

    This demonstrates how Qwen can request image generation via function calls.
    """
    print("=== Qwen + ImageGeneratorLLM Integration Example ===\n")

    # Initialize client
    client = ImageGeneratorClient()

    # Check API health
    print("1. Checking API health...")
    try:
        health = client.health_check()
        print(f"   ✓ API Status: {health['status']}")
        print(f"   ✓ Device: {health['device']}")
        print(f"   ✓ Model: {health['model_id']}\n")
    except Exception as e:
        print(f"   ✗ API not available: {e}")
        print("   Please start the server: python -m image_gen.server\n")
        return

    # Example 1: Single image generation
    print("2. Example: User asks Qwen to generate an image")
    print('   User: "Can you create an image of a serene mountain landscape at sunset?"\n')

    print("   Qwen determines it needs to call generate_image function...")
    function_call = {
        "name": "generate_image",
        "arguments": {
            "prompt": "a serene mountain landscape at sunset, photorealistic, golden hour lighting, snow-capped peaks",
            "num_inference_steps": 30
        }
    }
    print(f"   Function call: {json.dumps(function_call, indent=2)}\n")

    print("   Executing function call...")
    result = execute_function_call(
        function_call["name"],
        function_call["arguments"],
        client
    )

    if result["success"]:
        print(f"   ✓ {result['message']}")
        print(f"   ✓ Size: {result['width']}x{result['height']}\n")

        print(f"   Qwen's response: \"I've created a beautiful mountain landscape")
        print(f"   at sunset for you! The image has been saved to:")
        print(f"   {result['image_path']}\"\n")
    else:
        print(f"   ✗ Error: {result.get('error')}\n")

    # Example 2: Batch generation
    print("3. Example: User asks for multiple variations")
    print('   User: "Create 3 variations: a red apple, a blue ocean, a green forest"\n')

    print("   Qwen determines it needs to call generate_images_batch...")
    function_call = {
        "name": "generate_images_batch",
        "arguments": {
            "prompts": [
                "a fresh red apple on a wooden table, studio lighting, detailed",
                "a blue ocean wave, dynamic motion, photorealistic",
                "a lush green forest path, sunlight filtering through trees"
            ],
            "num_inference_steps": 20
        }
    }
    print(f"   Function call: {json.dumps(function_call, indent=2)}\n")

    print("   Executing batch generation...")
    result = execute_function_call(
        function_call["name"],
        function_call["arguments"],
        client
    )

    if result["success"]:
        print(f"   ✓ {result['message']}")
        for i, path in enumerate(result['image_paths'], 1):
            print(f"   ✓ Image {i}: {path}")
        print()

        print(f"   Qwen's response: \"I've created all 3 images for you!")
        print(f"   You can find them in the outputs directory.\"\n")
    else:
        print(f"   ✗ Error: {result.get('error')}\n")

    print("=== Integration Examples Complete ===")


if __name__ == "__main__":
    example_conversation()
