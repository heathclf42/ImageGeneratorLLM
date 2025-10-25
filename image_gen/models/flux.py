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
from diffusers import (
    FluxPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel
)
from PIL import Image
from typing import Optional, Tuple, Union, List
import logging
import time
import io
import base64

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

        # Store additional pipelines for img2img and controlnet
        self.img2img_pipeline = None
        self.controlnet_pipeline = None

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

    def generate_progressive(
        self,
        prompt: str,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        callback_steps: int = 1,
    ) -> Tuple[Image.Image, List[Image.Image]]:
        """
        Generate an image with progressive step-by-step visualization.

        This captures the SAME image at each denoising step, showing how
        the image evolves from noise to the final result.

        Args:
            prompt: Text description of the image to generate
            height: Output height in pixels (default: 1024)
            width: Output width in pixels (default: 1024)
            num_inference_steps: Number of denoising steps (default: 30)
            seed: Random seed for reproducibility (optional)
            callback_steps: Capture image every N steps (default: 1 for all steps)

        Returns:
            Tuple of (final_image, list_of_intermediate_images)

        Example:
            >>> gen = FluxGenerator()
            >>> final, intermediates = gen.generate_progressive(
            ...     prompt="a serene lake at sunset",
            ...     num_inference_steps=30,
            ...     callback_steps=5  # Capture every 5 steps
            ... )
            >>> # intermediates = [step_5, step_10, step_15, ...]
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Use config defaults if not specified
        height = height or self.config["default_size"][0]
        width = width or self.config["default_size"][1]
        num_inference_steps = num_inference_steps or self.config.get("default_steps", 30)

        logger.info(f"Generating progressive visualization: '{prompt[:50]}...' ({num_inference_steps} steps)")

        # Storage for intermediate images
        intermediate_images = []

        # Callback function to capture intermediate latents
        def step_callback(pipe, step_index, timestep, callback_kwargs):
            """Callback function called at each denoising step."""
            # Get the current latents
            latents = callback_kwargs.get("latents")

            if latents is not None and step_index % callback_steps == 0:
                # Decode latents to image
                with torch.no_grad():
                    # Scale latents back from latent space
                    latents_scaled = latents / pipe.vae.config.scaling_factor
                    # Decode to pixel space
                    image_tensor = pipe.vae.decode(latents_scaled).sample
                    # Convert to PIL Image
                    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
                    image_tensor = image_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
                    image = pipe.numpy_to_pil(image_tensor)[0]

                    intermediate_images.append((step_index, image))
                    logger.debug(f"Captured intermediate at step {step_index}/{num_inference_steps}")

            return callback_kwargs

        start_time = time.time()

        try:
            # Set seed for reproducibility
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
                logger.info(f"Using seed: {seed}")

            # Generate with callback
            if "flux" in self.model_id.lower():
                output = self.pipeline(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    callback_on_step_end=step_callback,
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
                    callback_on_step_end=step_callback,
                )

            final_image = output.images[0]
            gen_time = time.time() - start_time

            logger.info(f"✓ Progressive generation complete in {gen_time:.2f}s")
            logger.info(f"✓ Captured {len(intermediate_images)} intermediate steps")

            # Extract just the images (without step numbers) for return
            intermediate_list = [img for (step, img) in intermediate_images]

            return final_image, intermediate_list

        except Exception as e:
            logger.error(f"Progressive generation failed: {e}")
            raise RuntimeError(f"Progressive generation failed: {e}")

    def generate_img2img(
        self,
        prompt: str,
        init_image: Union[Image.Image, str],
        strength: float = 0.8,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate image from existing image (img2img).

        Args:
            prompt: Text description of desired modifications
            init_image: PIL Image or base64 encoded string
            strength: How much to transform (0.0-1.0, higher = more changes)
            height: Output height
            width: Output width
            num_inference_steps: Number of denoising steps
            seed: Random seed

        Returns:
            PIL Image object
        """
        # Decode base64 if needed
        if isinstance(init_image, str):
            if init_image.startswith('data:image'):
                init_image = init_image.split(',')[1]
            image_data = base64.b64decode(init_image)
            init_image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Lazy load img2img pipeline
        if self.img2img_pipeline is None:
            logger.info("Loading img2img pipeline...")
            if self.device.type == "mps":
                self.img2img_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    cache_dir=get_config().cache_dir
                ).to(self.device)
            elif self.device.type == "cuda":
                self.img2img_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    cache_dir=get_config().cache_dir
                ).to(self.device)
            else:
                self.img2img_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32,
                    cache_dir=get_config().cache_dir
                )

        # Use defaults
        num_inference_steps = num_inference_steps or self.config["default_steps"]

        # Generate
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        logger.info(f"Img2img: '{prompt[:50]}...' (strength={strength})")

        output = self.img2img_pipeline(
            prompt=prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=self.config.get("guidance_scale", 7.5),
            generator=generator,
        )

        return output.images[0]

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
    from pathlib import Path

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

        # Save to outputs directory
        output_dir = Path(get_config().output["directory"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "test_flux_output.png"

        image.save(output_path)
        print(f"\n✓ Test image saved to: {output_path}")
        print(f"  Size: {image.size}")

        print("\n✓ All tests passed!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
