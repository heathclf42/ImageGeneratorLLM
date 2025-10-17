"""
ImageGeneratorLLM - Local image generation optimized for Apple Silicon

This package provides a unified interface for multiple state-of-the-art
image generation models with MPS (Metal Performance Shaders) optimization.

Main Components:
    - ImageGenerator: Main class for image generation
    - FLUX.1 Schnell: Fast text-to-image (primary model)
    - SDXL: High-quality generation
    - ControlNet Union: Structural guidance
    - BrushNet: Inpainting and editing

Example:
    >>> from image_gen import ImageGenerator
    >>> gen = ImageGenerator()
    >>> image, path = gen.generate("a cat on mars", model="flux")

For more information, see README.md and docs/
"""

__version__ = "0.1.0"

# Core exports will be added as we build
# from image_gen.core import ImageGenerator
# __all__ = ["ImageGenerator"]
