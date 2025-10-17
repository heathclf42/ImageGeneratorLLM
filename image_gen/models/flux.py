"""
Module: image_gen.models.flux
Purpose: FLUX.1 Schnell pipeline implementation for fast text-to-image generation
Dependencies: diffusers, torch, transformers
Author: Generated for ImageGeneratorLLM
Reference: See docs/architecture/FLUX_PIPELINE.md

FLUX.1 Schnell is an Apache 2.0 licensed model optimized for speed.
It generates high-quality images in 2-4 steps, making it ideal as the primary model.
"""

import torch
from diffusers import FluxPipeline, StableDiffusionXLPipeline
from PIL import Image
from typing import Optional, Tuple, Union
import logging
import time

from image_gen.config import get_config
from image_gen.utils.device import get_device

logger = logging.getLogger(__name__)


class FluxGenerator:
    """
    FLUX.1 Schnell text-to-image generator with MPS optimization.

    This class handles loading, caching, and inference for the FLUX.1 Schnell
    model, optimized for Apple Silicon with Metal Performance Shaders (MPS).

    FLUX.1 Schnell is designed for speed:
    - 2-4 steps for generation (vs 20-50 for other models)
    - No guidance scale (distilled model)
    - Apache 2.0 license (fully free for commercial use)

    Attributes:
        pipeline: Loaded FluxPipeline instance
        device: Compute device (mps, cuda, or cpu)
        model_id: Hugging Face model identifier
        config: Model configuration from config.py

    Example:
        >>> gen = FluxGenerator()
        >>> image = gen.generate("a cat astronaut on mars")
        >>> image.save("output.png")

        >>> # With custom parameters
        >>> image = gen.generate(
        ...     prompt="a futuristic city",
        ...     height=1024,
        ...     width=1024,
        ...     num_inference_steps=4
        ... )
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize FLUX generator with automatic model loading.

        On first run, this will download the model (~12GB) from Hugging Face.
        Subsequent runs will use the cached model.

        Args:
            device: Override device selection. If None, auto-detects MPS/CUDA/CPU.

        Raises:
            RuntimeError: If model fails to load
            ValueError: If device is invalid
        """
        self.config = get_config().get_model_config("flux")
        self.model_id = get_config().get_model_id("flux")
        self.device = get_device(device)

        logger.info(f"Initializing FLUX.1 Schnell on device: {self.device}")
        logger.info(f"Model ID: {self.model_id}")

        # Track loading time
        start_time = time.time()

        try:
            # Load pipeline with torch_dtype for memory efficiency
            logger.info("Loading FLUX pipeline (this may take a while on first run)...")

            # Auto-detect pipeline type based on model_id
            if "flux" in self.model_id.lower():
                pipeline_class = FluxPipeline
            else:
                pipeline_class = StableDiffusionXLPipeline

            # For MPS, we need to be careful with dtypes
            if self.device.type == "mps":
                # MPS works best with float16
                self.pipeline = pipeline_class.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    cache_dir=get_config().cache_dir
                )
                self.pipeline = self.pipeline.to(self.device)
            elif self.device.type == "cuda":
                # CUDA can handle float16 efficiently
                self.pipeline = pipeline_class.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    cache_dir=get_config().cache_dir
                )
                self.pipeline = self.pipeline.to(self.device)
            else:
                # CPU uses float32
                self.pipeline = pipeline_class.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32,
                    cache_dir=get_config().cache_dir
                )

            load_time = time.time() - start_time
            logger.info(f"✓ FLUX pipeline loaded successfully in {load_time:.1f}s")

        except Exception as e:
            logger.error(f"Failed to load FLUX pipeline: {e}")
            raise RuntimeError(f"Could not initialize FLUX model: {e}")

    def generate(
        self,
        prompt: str,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of the image to generate
            height: Output height in pixels (default: 1024)
            width: Output width in pixels (default: 1024)
            num_inference_steps: Number of denoising steps (default: 4)
                               More steps = higher quality but slower
            seed: Random seed for reproducibility (optional)

        Returns:
            PIL Image object

        Raises:
            ValueError: If prompt is empty or parameters are invalid
            RuntimeError: If generation fails

        Example:
            >>> gen = FluxGenerator()
            >>> image = gen.generate(
            ...     prompt="a serene lake at sunset",
            ...     num_inference_steps=4,
            ...     seed=42
            ... )
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Use config defaults if not specified
        height = height or self.config["default_size"][0]
        width = width or self.config["default_size"][1]
        num_inference_steps = num_inference_steps or self.config["default_steps"]

        # Validate dimensions (must be multiples of 8 for most diffusion models)
        if height % 8 != 0 or width % 8 != 0:
            logger.warning(f"Height and width should be multiples of 8. Got {height}x{width}")

        logger.info(f"Generating image: '{prompt[:50]}...' ({height}x{width}, {num_inference_steps} steps)")

        start_time = time.time()

        try:
            # Set seed for reproducibility
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
                logger.info(f"Using seed: {seed}")

            # Generate image
            # FLUX doesn't use guidance_scale, but SDXL does
            if "flux" in self.model_id.lower():
                output = self.pipeline(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                )
            else:
                # SDXL uses guidance_scale
                output = self.pipeline(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=self.config.get("guidance_scale", 7.5),
                    generator=generator,
                )

            image = output.images[0]
            gen_time = time.time() - start_time

            logger.info(f"✓ Image generated in {gen_time:.2f}s")

            return image

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Image generation failed: {e}")

    def generate_batch(
        self,
        prompts: list[str],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> list[Image.Image]:
        """
        Generate multiple images from a list of prompts.

        More efficient than calling generate() multiple times when processing
        multiple prompts in sequence.

        Args:
            prompts: List of text descriptions
            height: Output height in pixels (default: 1024)
            width: Output width in pixels (default: 1024)
            num_inference_steps: Number of denoising steps (default: 4)
            seed: Random seed for reproducibility (optional)

        Returns:
            List of PIL Image objects

        Example:
            >>> gen = FluxGenerator()
            >>> images = gen.generate_batch([
            ...     "a cat on mars",
            ...     "a dog in space",
            ...     "a bird underwater"
            ... ])
        """
        if not prompts:
            raise ValueError("Prompts list cannot be empty")

        logger.info(f"Generating {len(prompts)} images in batch")

        images = []
        for i, prompt in enumerate(prompts, 1):
            logger.info(f"[{i}/{len(prompts)}] Generating: {prompt[:50]}...")
            image = self.generate(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                seed=seed
            )
            images.append(image)

        return images

    def unload(self) -> None:
        """
        Unload the model from memory.

        Useful for memory management when switching between models.
        After calling this, you'll need to create a new FluxGenerator
        instance to generate images again.

        Example:
            >>> gen = FluxGenerator()
            >>> image = gen.generate("a cat")
            >>> gen.unload()  # Free up ~12GB of memory
        """
        logger.info("Unloading FLUX pipeline from memory")

        if hasattr(self, 'pipeline'):
            # Move to CPU and clear cache
            self.pipeline = self.pipeline.to("cpu")
            del self.pipeline

            # Clear GPU cache if using GPU
            if self.device.type in ["cuda", "mps"]:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                elif self.device.type == "mps":
                    torch.mps.empty_cache()

            logger.info("✓ FLUX pipeline unloaded")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.unload()
        except:
            pass


if __name__ == "__main__":
    # Test the FLUX generator
    import sys

    print("Testing FLUX.1 Schnell Generator")
    print("=" * 50)

    try:
        # Initialize
        print("\n1. Initializing FLUX generator...")
        gen = FluxGenerator()

        # Generate test image
        print("\n2. Generating test image...")
        prompt = "a cat astronaut on mars, digital art"
        image = gen.generate(prompt, num_inference_steps=4)

        # Save
        output_path = "test_flux_output.png"
        image.save(output_path)
        print(f"\n✓ Test image saved to: {output_path}")
        print(f"  Size: {image.size}")

        print("\n✓ All tests passed!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
