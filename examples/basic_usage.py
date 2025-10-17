"""
Example: Basic Python API Usage

This example shows direct Python usage of the ImageGenerator class
without needing the REST API or CLI.
"""

from image_gen.core import ImageGenerator


def example_single_generation():
    """Generate a single image."""
    print("=== Single Image Generation ===\n")

    # Initialize generator with auto-preview
    gen = ImageGenerator(auto_preview=True)

    # Generate image
    print("Generating image...")
    image = gen.generate(
        prompt="a serene mountain landscape at sunset, photorealistic",
        num_inference_steps=30,
        seed=42  # For reproducibility
    )

    print(f"✓ Generated {image.size[0]}x{image.size[1]} image")
    print("  Image saved to outputs/ and opened automatically\n")


def example_batch_generation():
    """Generate multiple images."""
    print("=== Batch Image Generation ===\n")

    gen = ImageGenerator(auto_preview=False)

    prompts = [
        "a cozy coffee shop interior, warm lighting",
        "a futuristic cityscape at night, cyberpunk",
        "a peaceful zen garden with cherry blossoms"
    ]

    print(f"Generating {len(prompts)} images...")
    images = gen.generate_batch(
        prompts=prompts,
        num_inference_steps=20
    )

    print(f"✓ Generated {len(images)} images")
    print("  All images saved to outputs/\n")


def example_custom_settings():
    """Generate with custom size and settings."""
    print("=== Custom Settings ===\n")

    gen = ImageGenerator()

    print("Generating 768x768 image with 40 steps...")
    image = gen.generate(
        prompt="portrait of a wise old wizard, oil painting style",
        width=768,
        height=768,
        num_inference_steps=40,
        seed=123
    )

    print(f"✓ Generated {image.size[0]}x{image.size[1]} high-quality image\n")


def example_reproducibility():
    """Demonstrate seed-based reproducibility."""
    print("=== Reproducibility with Seeds ===\n")

    gen = ImageGenerator()

    seed = 42
    prompt = "a cute robot reading a book"

    print(f"Generating image 1 with seed {seed}...")
    image1 = gen.generate(prompt, seed=seed, num_inference_steps=20)

    print(f"Generating image 2 with same seed...")
    image2 = gen.generate(prompt, seed=seed, num_inference_steps=20)

    # Compare pixel data
    import numpy as np
    arr1 = np.array(image1)
    arr2 = np.array(image2)

    if np.array_equal(arr1, arr2):
        print("✓ Images are pixel-perfect identical!")
    else:
        diff_pct = (np.sum(arr1 != arr2) / arr1.size) * 100
        print(f"Images differ by {diff_pct:.2f}% (MPS non-determinism)")

    print()


if __name__ == "__main__":
    import sys

    print("\nImageGeneratorLLM - Basic Python Usage Examples")
    print("=" * 50)
    print()

    try:
        # Run all examples
        example_single_generation()
        example_custom_settings()
        example_reproducibility()
        example_batch_generation()

        print("=" * 50)
        print("All examples completed successfully!")
        print("\nCheck the outputs/ directory for generated images.")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
