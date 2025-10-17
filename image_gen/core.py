"""
Module: image_gen.core
Purpose: Core ImageGenerator class with unified model management
Dependencies: PIL, typing, pathlib
Author: Generated for ImageGeneratorLLM
Reference: See docs/architecture/CORE_DESIGN.md

This module provides the main ImageGenerator interface that manages all models
and provides a unified API for image generation tasks.
"""

from PIL import Image
from typing import Optional, List, Literal
from pathlib import Path
import logging
import subprocess
import platform

from image_gen.config import get_config
from image_gen.models.flux import FluxGenerator

logger = logging.getLogger(__name__)

ModelType = Literal["flux", "sdxl", "controlnet", "brushnet"]


class ImageGenerator:
    """
    Main image generator with unified model management.

    This class provides a high-level interface for all image generation tasks,
    managing model loading/unloading, output handling, and automatic preview.

    Features:
    - Lazy model loading (models load on first use)
    - Automatic model caching based on config
    - Automatic image preview in system viewer
    - Consistent output management
    - Support for multiple model types

    Attributes:
        config: Configuration instance
        _flux_generator: Cached FLUX/SDXL generator instance
        auto_preview: Whether to automatically open generated images

    Example:
        >>> gen = ImageGenerator(auto_preview=True)
        >>> image = gen.generate("a cat on mars")
        >>> # Image automatically opens in Preview.app

        >>> # Batch generation
        >>> images = gen.generate_batch([
        ...     "a red apple",
        ...     "a blue ocean"
        ... ])
    """

    def __init__(
        self,
        auto_preview: bool = False,
        device: Optional[str] = None
    ):
        """
        Initialize ImageGenerator.

        Args:
            auto_preview: Automatically open generated images in system viewer
            device: Override device selection (None = auto-detect)
        """
        self.config = get_config()
        self.auto_preview = auto_preview
        self.device = device

        # Lazy-loaded model instances
        self._flux_generator: Optional[FluxGenerator] = None

        logger.info("ImageGenerator initialized")
        if auto_preview:
            logger.info("Auto-preview enabled")

    @property
    def flux_generator(self) -> FluxGenerator:
        """
        Get FLUX/SDXL generator, loading it if needed.

        This uses lazy loading - the model is only loaded on first access.
        Subsequent accesses return the cached instance.

        Returns:
            FluxGenerator instance
        """
        if self._flux_generator is None:
            logger.info("Loading FLUX/SDXL generator (first use)...")
            self._flux_generator = FluxGenerator(device=self.device)
        return self._flux_generator

    def generate(
        self,
        prompt: str,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        model: ModelType = "flux",
        save_path: Optional[Path] = None,
        auto_save: bool = True,
    ) -> Image.Image:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of the image to generate
            height: Output height in pixels (default from config)
            width: Output width in pixels (default from config)
            num_inference_steps: Number of denoising steps (default from config)
            seed: Random seed for reproducibility (optional)
            model: Which model to use ("flux" for SDXL currently)
            save_path: Where to save the image (None = auto-generate)
            auto_save: Whether to save the image automatically

        Returns:
            PIL Image object

        Example:
            >>> gen = ImageGenerator()
            >>> image = gen.generate(
            ...     prompt="a serene lake at sunset",
            ...     num_inference_steps=30,
            ...     seed=42
            ... )
        """
        if model != "flux":
            raise NotImplementedError(
                f"Model '{model}' not yet implemented. Currently only 'flux' (SDXL) is available."
            )

        # Generate using FLUX/SDXL
        image = self.flux_generator.generate(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            seed=seed
        )

        # Handle saving
        if auto_save:
            if save_path is None:
                save_path = self._generate_output_path(prompt)

            save_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(save_path)
            logger.info(f"Image saved to: {save_path}")

            # Auto-preview if enabled
            if self.auto_preview:
                self._open_image(save_path)

        return image

    def generate_batch(
        self,
        prompts: List[str],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        model: ModelType = "flux",
        auto_save: bool = True,
    ) -> List[Image.Image]:
        """
        Generate multiple images from a list of prompts.

        Args:
            prompts: List of text descriptions
            height: Output height in pixels (default from config)
            width: Output width in pixels (default from config)
            num_inference_steps: Number of denoising steps (default from config)
            seed: Random seed for reproducibility (optional)
            model: Which model to use ("flux" for SDXL currently)
            auto_save: Whether to save images automatically

        Returns:
            List of PIL Image objects

        Example:
            >>> gen = ImageGenerator()
            >>> images = gen.generate_batch([
            ...     "a cat on mars",
            ...     "a dog in space",
            ...     "a bird underwater"
            ... ])
        """
        if model != "flux":
            raise NotImplementedError(
                f"Model '{model}' not yet implemented. Currently only 'flux' (SDXL) is available."
            )

        images = []
        for i, prompt in enumerate(prompts, 1):
            logger.info(f"Generating image {i}/{len(prompts)}")

            image = self.generate(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                seed=seed,
                model=model,
                auto_save=auto_save
            )
            images.append(image)

        return images

    def _generate_output_path(self, prompt: str) -> Path:
        """
        Generate output filename from prompt.

        Creates a safe filename by:
        - Taking first 50 chars of prompt
        - Removing special characters
        - Adding .png extension
        - Placing in output directory

        Args:
            prompt: The generation prompt

        Returns:
            Path object for the output file
        """
        # Create safe filename from prompt
        safe_name = "".join(
            c for c in prompt[:50]
            if c.isalnum() or c in (' ', '-', '_')
        ).strip()
        safe_name = safe_name.replace(' ', '_')

        # Ensure we have a valid filename
        if not safe_name:
            safe_name = "generated_image"

        output_dir = Path(self.config.output["directory"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Add counter if file exists
        base_path = output_dir / f"{safe_name}.png"
        counter = 1
        final_path = base_path

        while final_path.exists():
            final_path = output_dir / f"{safe_name}_{counter}.png"
            counter += 1

        return final_path

    def _open_image(self, image_path: Path) -> None:
        """
        Open image in system default viewer.

        Uses platform-specific commands:
        - macOS: open
        - Linux: xdg-open
        - Windows: start

        Args:
            image_path: Path to the image file
        """
        try:
            system = platform.system()

            if system == "Darwin":  # macOS
                subprocess.run(["open", str(image_path)], check=True)
            elif system == "Linux":
                subprocess.run(["xdg-open", str(image_path)], check=True)
            elif system == "Windows":
                subprocess.run(["start", str(image_path)], shell=True, check=True)
            else:
                logger.warning(f"Auto-preview not supported on {system}")

            logger.info(f"Opened image in system viewer: {image_path}")

        except Exception as e:
            logger.warning(f"Failed to open image: {e}")

    def unload_models(self) -> None:
        """
        Unload all models from memory.

        Useful for freeing GPU memory when done generating images.
        After calling this, models will be reloaded on next use.

        Example:
            >>> gen = ImageGenerator()
            >>> image = gen.generate("a cat")
            >>> gen.unload_models()  # Free ~7GB of memory
        """
        if self._flux_generator is not None:
            logger.info("Unloading FLUX/SDXL generator...")
            self._flux_generator.unload()
            self._flux_generator = None

        logger.info("All models unloaded")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.unload_models()
        except:
            pass


if __name__ == "__main__":
    # Test the ImageGenerator
    import sys

    print("Testing ImageGenerator")
    print("=" * 50)

    try:
        # Initialize with auto-preview
        print("\n1. Initializing ImageGenerator with auto-preview...")
        gen = ImageGenerator(auto_preview=True)

        # Generate test image
        print("\n2. Generating test image...")
        prompt = "a beautiful sunset over mountains, photorealistic"
        image = gen.generate(prompt, num_inference_steps=20)

        print(f"\n✓ Image generated successfully")
        print(f"  Size: {image.size}")
        print(f"  Image should have opened automatically!")

        # Test batch generation
        print("\n3. Testing batch generation (2 images)...")
        prompts = [
            "a red sports car",
            "a blue ocean wave"
        ]
        images = gen.generate_batch(prompts, num_inference_steps=15)

        print(f"\n✓ Generated {len(images)} images")

        # Cleanup
        print("\n4. Unloading models...")
        gen.unload_models()

        print("\n✓ All tests passed!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
