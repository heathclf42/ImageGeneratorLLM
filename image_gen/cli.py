"""
Module: image_gen.cli
Purpose: Command-line interface for image generation
Dependencies: click, pathlib
Author: Generated for ImageGeneratorLLM
Reference: See docs/QUICK_START.md

This module provides a user-friendly CLI for generating images without
needing to write Python code.
"""

import click
from pathlib import Path
from typing import Optional
import sys

from image_gen.core import ImageGenerator
from image_gen.config import get_config
from image_gen.utils.device import print_device_info


@click.group()
@click.version_option(version="0.1.0", prog_name="image-gen")
def cli():
    """
    ImageGeneratorLLM - Generate images from text prompts using SDXL.

    Examples:

    \b
      # Generate a single image
      image-gen generate "a serene mountain landscape"

    \b
      # Generate with custom settings
      image-gen generate "a cat on mars" --steps 40 --width 1024 --height 1024

    \b
      # Generate multiple images
      image-gen batch prompts.txt --steps 20

    \b
      # Check device info
      image-gen info
    """
    pass


@cli.command()
@click.argument("prompt")
@click.option(
    "--width", "-w",
    type=int,
    default=None,
    help="Output width in pixels (default: 1024)"
)
@click.option(
    "--height", "-h",
    type=int,
    default=None,
    help="Output height in pixels (default: 1024)"
)
@click.option(
    "--steps", "-s",
    type=int,
    default=None,
    help="Number of inference steps (default: 30, range: 10-50)"
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility"
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file path (default: auto-generated in outputs/)"
)
@click.option(
    "--no-preview",
    is_flag=True,
    help="Don't automatically open the image"
)
def generate(
    prompt: str,
    width: Optional[int],
    height: Optional[int],
    steps: Optional[int],
    seed: Optional[int],
    output: Optional[Path],
    no_preview: bool
):
    """
    Generate a single image from a text prompt.

    PROMPT: Text description of the image to generate

    \b
    Examples:
      image-gen generate "a serene mountain landscape at sunset"
      image-gen generate "cyberpunk city" --steps 40 --width 768
      image-gen generate "portrait of a wizard" --seed 42 --no-preview
    """
    try:
        click.echo(f"🎨 Generating: {prompt}")
        if steps:
            click.echo(f"   Steps: {steps}")
        if seed is not None:
            click.echo(f"   Seed: {seed}")
        if width or height:
            click.echo(f"   Size: {width or 1024}x{height or 1024}")

        # Initialize generator
        gen = ImageGenerator(auto_preview=not no_preview)

        # Generate image
        image = gen.generate(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            seed=seed,
            save_path=output,
            auto_save=True
        )

        click.echo(f"✓ Image generated successfully!")
        click.echo(f"  Size: {image.size}")

        if output:
            click.echo(f"  Saved to: {output}")
        else:
            # Path was auto-generated, show where it was saved
            config = get_config()
            output_dir = Path(config.output["directory"])
            click.echo(f"  Saved in: {output_dir}/")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("prompts_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--width", "-w",
    type=int,
    default=None,
    help="Output width in pixels (default: 1024)"
)
@click.option(
    "--height", "-h",
    type=int,
    default=None,
    help="Output height in pixels (default: 1024)"
)
@click.option(
    "--steps", "-s",
    type=int,
    default=None,
    help="Number of inference steps (default: 30)"
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility"
)
@click.option(
    "--no-preview",
    is_flag=True,
    help="Don't automatically open images"
)
def batch(
    prompts_file: Path,
    width: Optional[int],
    height: Optional[int],
    steps: Optional[int],
    seed: Optional[int],
    no_preview: bool
):
    """
    Generate multiple images from prompts in a text file.

    PROMPTS_FILE: Path to text file with one prompt per line

    \b
    File format (one prompt per line):
      a serene mountain landscape
      a futuristic cyberpunk city
      a portrait of a wise wizard

    \b
    Examples:
      image-gen batch prompts.txt
      image-gen batch prompts.txt --steps 20 --no-preview
    """
    try:
        # Read prompts
        with open(prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]

        if not prompts:
            click.echo("✗ No prompts found in file", err=True)
            sys.exit(1)

        click.echo(f"🎨 Generating {len(prompts)} images from {prompts_file}")
        if steps:
            click.echo(f"   Steps: {steps}")

        # Initialize generator
        gen = ImageGenerator(auto_preview=not no_preview)

        # Generate batch
        images = gen.generate_batch(
            prompts=prompts,
            width=width,
            height=height,
            num_inference_steps=steps,
            seed=seed,
            auto_save=True
        )

        click.echo(f"\n✓ Generated {len(images)} images successfully!")

        config = get_config()
        output_dir = Path(config.output["directory"])
        click.echo(f"  Saved in: {output_dir}/")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def info():
    """
    Display system and device information.

    Shows:
    - Device type (MPS/CUDA/CPU)
    - Available memory
    - Model cache location
    - Output directory
    """
    try:
        click.echo("=== ImageGeneratorLLM System Info ===\n")

        # Device info
        print_device_info()

        # Config info
        config = get_config()
        click.echo("\n--- Configuration ---")
        click.echo(f"Model: {config.get_model_id('flux')}")
        click.echo(f"Default steps: {config.models['flux']['default_steps']}")
        click.echo(f"Default size: {config.models['flux']['default_size']}")
        click.echo(f"Output directory: {config.output['directory']}")

        if config.cache_dir:
            click.echo(f"Model cache: {config.cache_dir}")
        else:
            click.echo("Model cache: ~/.cache/huggingface/ (default)")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--prompt", "-p",
    default="a test image of a colorful parrot",
    help="Test prompt to use"
)
@click.option(
    "--steps",
    type=int,
    default=10,
    help="Number of steps for quick test (default: 10)"
)
def test(prompt: str, steps: int):
    """
    Run a quick test to verify the setup is working.

    Generates a single test image with minimal steps for fast verification.

    \b
    Examples:
      image-gen test
      image-gen test --prompt "a red apple" --steps 15
    """
    try:
        click.echo("🧪 Running quick test...")
        click.echo(f"   Prompt: {prompt}")
        click.echo(f"   Steps: {steps} (fast mode)")

        gen = ImageGenerator(auto_preview=True)

        image = gen.generate(
            prompt=prompt,
            num_inference_steps=steps,
            auto_save=True
        )

        click.echo(f"\n✓ Test passed!")
        click.echo(f"  Generated {image.size[0]}x{image.size[1]} image")
        click.echo(f"  If you can see the image, setup is working correctly!")

    except Exception as e:
        click.echo(f"✗ Test failed: {e}", err=True)
        click.echo("\nPlease check:")
        click.echo("  1. Virtual environment is activated")
        click.echo("  2. All dependencies are installed")
        click.echo("  3. Device (MPS/CUDA/CPU) is available")
        sys.exit(1)


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
