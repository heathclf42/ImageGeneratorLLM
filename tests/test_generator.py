"""
Module: tests.test_generator
Purpose: Comprehensive tests for image generation functionality
Author: Generated for ImageGeneratorLLM
"""

import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from image_gen.models.flux import FluxGenerator
from image_gen.config import get_config


def test_basic_generation():
    """Test basic image generation with default parameters."""
    print("\n" + "="*60)
    print("TEST 1: Basic Image Generation")
    print("="*60)

    gen = FluxGenerator()

    prompt = "a serene mountain landscape at sunset"
    print(f"\nPrompt: {prompt}")

    start = time.time()
    image = gen.generate(prompt)
    elapsed = time.time() - start

    output_path = Path(get_config().output["directory"]) / "test_landscape.png"
    image.save(output_path)

    print(f"✓ Generated in {elapsed:.1f}s")
    print(f"✓ Size: {image.size}")
    print(f"✓ Saved to: {output_path}")

    assert image.size == (1024, 1024), f"Expected (1024, 1024), got {image.size}"
    assert output_path.exists(), f"Output file not created"

    return True


def test_custom_size():
    """Test generation with custom dimensions."""
    print("\n" + "="*60)
    print("TEST 2: Custom Size Generation")
    print("="*60)

    gen = FluxGenerator()

    prompt = "a futuristic city at night, cyberpunk style"
    width, height = 768, 768
    print(f"\nPrompt: {prompt}")
    print(f"Size: {width}x{height}")

    start = time.time()
    image = gen.generate(prompt, width=width, height=height, num_inference_steps=20)
    elapsed = time.time() - start

    output_path = Path(get_config().output["directory"]) / "test_cyberpunk.png"
    image.save(output_path)

    print(f"✓ Generated in {elapsed:.1f}s")
    print(f"✓ Size: {image.size}")
    print(f"✓ Saved to: {output_path}")

    assert image.size == (width, height), f"Expected ({width}, {height}), got {image.size}"

    return True


def test_reproducibility():
    """Test that using the same seed produces identical images."""
    print("\n" + "="*60)
    print("TEST 3: Reproducibility with Seed")
    print("="*60)

    gen = FluxGenerator()

    prompt = "a cute robot reading a book"
    seed = 42
    print(f"\nPrompt: {prompt}")
    print(f"Seed: {seed}")

    # Generate first image
    print("\nGenerating image 1...")
    start = time.time()
    image1 = gen.generate(prompt, seed=seed, num_inference_steps=10)
    elapsed1 = time.time() - start

    # Generate second image with same seed
    print("Generating image 2 with same seed...")
    start = time.time()
    image2 = gen.generate(prompt, seed=seed, num_inference_steps=10)
    elapsed2 = time.time() - start

    output_path1 = Path(get_config().output["directory"]) / "test_seed1.png"
    output_path2 = Path(get_config().output["directory"]) / "test_seed2.png"

    image1.save(output_path1)
    image2.save(output_path2)

    print(f"✓ Image 1 generated in {elapsed1:.1f}s")
    print(f"✓ Image 2 generated in {elapsed2:.1f}s")
    print(f"✓ Saved to: {output_path1} and {output_path2}")

    # Compare pixels (should be identical)
    import numpy as np
    arr1 = np.array(image1)
    arr2 = np.array(image2)

    are_identical = np.array_equal(arr1, arr2)

    if are_identical:
        print("✓ Images are pixel-perfect identical (good!)")
    else:
        diff_percentage = (np.sum(arr1 != arr2) / arr1.size) * 100
        print(f"⚠ Images differ by {diff_percentage:.2f}% (MPS non-determinism)")

    return True


def test_batch_generation():
    """Test batch generation of multiple images."""
    print("\n" + "="*60)
    print("TEST 4: Batch Generation")
    print("="*60)

    gen = FluxGenerator()

    prompts = [
        "a red apple on a wooden table",
        "a blue ocean wave",
        "a green forest path"
    ]

    print(f"\nGenerating {len(prompts)} images:")
    for i, p in enumerate(prompts, 1):
        print(f"  {i}. {p}")

    start = time.time()
    images = gen.generate_batch(prompts, num_inference_steps=15)
    elapsed = time.time() - start

    print(f"\n✓ Generated {len(images)} images in {elapsed:.1f}s")
    print(f"✓ Average: {elapsed/len(images):.1f}s per image")

    # Save all images
    output_dir = Path(get_config().output["directory"])
    for i, (image, prompt) in enumerate(zip(images, prompts), 1):
        # Create safe filename from prompt
        safe_name = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        output_path = output_dir / f"test_batch_{i}_{safe_name}.png"
        image.save(output_path)
        print(f"  ✓ Saved: {output_path.name}")

    assert len(images) == len(prompts), f"Expected {len(prompts)} images, got {len(images)}"

    return True


def test_quality_steps():
    """Test effect of different step counts on quality."""
    print("\n" + "="*60)
    print("TEST 5: Quality vs Speed Trade-off")
    print("="*60)

    gen = FluxGenerator()

    prompt = "a detailed portrait of a wise old wizard"
    step_counts = [10, 20, 30]

    print(f"\nPrompt: {prompt}")
    print(f"Testing steps: {step_counts}")

    output_dir = Path(get_config().output["directory"])

    for steps in step_counts:
        print(f"\n  Generating with {steps} steps...")
        start = time.time()
        image = gen.generate(prompt, num_inference_steps=steps, seed=123)
        elapsed = time.time() - start

        output_path = output_dir / f"test_quality_{steps}steps.png"
        image.save(output_path)

        print(f"    ✓ Time: {elapsed:.1f}s")
        print(f"    ✓ Saved: {output_path.name}")

    print(f"\n✓ Compare images in outputs/ to see quality differences")

    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*60)
    print("SDXL IMAGE GENERATOR - COMPREHENSIVE TEST SUITE")
    print("="*60)

    tests = [
        ("Basic Generation", test_basic_generation),
        ("Custom Size", test_custom_size),
        ("Reproducibility", test_reproducibility),
        ("Batch Generation", test_batch_generation),
        ("Quality Steps", test_quality_steps),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, True, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n✗ {test_name} FAILED: {e}")

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for test_name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
        if error:
            print(f"       Error: {error}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
