"""
Test script for progressive iteration step visualization.

This tests the new generate_progressive() method that captures
the SAME image at each denoising step.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from image_gen.core import ImageGenerator
from PIL import Image
import time


def test_progressive_generation():
    """Test progressive generation with step-by-step visualization."""
    print("\n" + "="*70)
    print("TEST: Progressive Iteration Step Visualization")
    print("="*70)

    print("\n1. Initializing ImageGenerator...")
    gen = ImageGenerator(auto_preview=False)

    prompt = "a peaceful zen garden with koi pond and cherry blossoms"
    num_steps = 20  # Fewer steps for faster testing
    callback_steps = 5  # Capture every 5 steps

    print(f"\n2. Generating with progressive visualization...")
    print(f"   Prompt: {prompt}")
    print(f"   Total steps: {num_steps}")
    print(f"   Capture interval: every {callback_steps} steps")

    start = time.time()

    try:
        # Access the flux_generator directly to use generate_progressive
        final_image, intermediate_images = gen.flux_generator.generate_progressive(
            prompt=prompt,
            num_inference_steps=num_steps,
            callback_steps=callback_steps,
            seed=42  # For reproducibility
        )

        elapsed = time.time() - start

        print(f"\n3. ✓ Generation complete in {elapsed:.2f}s")
        print(f"   Final image size: {final_image.size}")
        print(f"   Intermediate images captured: {len(intermediate_images)}")

        # Save progressive images
        output_dir = Path("outputs/progressive_test")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save final image
        final_path = output_dir / "final.png"
        final_image.save(final_path)
        print(f"\n4. Saved final image: {final_path}")

        # Save intermediate images
        print(f"\n5. Saving intermediate images:")
        for i, img in enumerate(intermediate_images):
            step_num = (i + 1) * callback_steps
            step_path = output_dir / f"step_{step_num:02d}.png"
            img.save(step_path)
            print(f"   - Step {step_num:2d}: {step_path}")

        print(f"\n6. ✓ All {len(intermediate_images)} intermediate images saved")

        # Verify they're actually different
        print(f"\n7. Verifying images are different (not duplicates)...")
        if len(intermediate_images) >= 2:
            img1_data = list(intermediate_images[0].getdata())
            img2_data = list(intermediate_images[1].getdata())
            final_data = list(final_image.getdata())

            if img1_data == img2_data:
                print("   ⚠️  WARNING: First two intermediates are identical!")
                return False
            else:
                print("   ✓ Intermediate images are different")

            if img2_data == final_data:
                print("   ⚠️  WARNING: Last intermediate matches final!")
                return False
            else:
                print("   ✓ Final image is different from intermediates")

        print("\n" + "="*70)
        print("✓ TEST PASSED: Progressive visualization works correctly")
        print("="*70)

        print(f"\nView results in: {output_dir}")

        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step_progression_visual():
    """
    Test that shows the actual progression from noise to image.
    Uses a small number of steps to make changes visible.
    """
    print("\n" + "="*70)
    print("TEST: Visual Step Progression (Noise → Image)")
    print("="*70)

    print("\n1. Initializing ImageGenerator...")
    gen = ImageGenerator(auto_preview=False)

    prompt = "a simple red circle on white background"
    num_steps = 10  # Very few steps to see dramatic changes
    callback_steps = 1  # Capture every step

    print(f"\n2. Generating with ALL steps captured...")
    print(f"   Prompt: {prompt}")
    print(f"   Steps: {num_steps} (capturing ALL)")

    start = time.time()

    try:
        final_image, all_steps = gen.flux_generator.generate_progressive(
            prompt=prompt,
            num_inference_steps=num_steps,
            callback_steps=callback_steps,
            seed=123,
            height=512,  # Smaller for faster generation
            width=512
        )

        elapsed = time.time() - start

        print(f"\n3. ✓ Generated {len(all_steps)} step images in {elapsed:.2f}s")

        # Save to dedicated directory
        output_dir = Path("outputs/visual_progression_test")
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n4. Saving all {len(all_steps)} step images...")
        for i, img in enumerate(all_steps):
            step_path = output_dir / f"evolution_step_{i+1:02d}_of_{num_steps:02d}.png"
            img.save(step_path)
            if i % 2 == 0:  # Print every other one to avoid clutter
                print(f"   - Step {i+1}: {step_path}")

        final_path = output_dir / f"evolution_step_{num_steps:02d}_FINAL.png"
        final_image.save(final_path)
        print(f"   - FINAL: {final_path}")

        print("\n" + "="*70)
        print("✓ TEST PASSED: All steps saved for visual inspection")
        print("="*70)
        print(f"\nView the progression in: {output_dir}")
        print("Open the images in order to see noise → final image evolution!")

        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# Progressive Visualization Test Suite")
    print("#"*70)

    results = []

    # Run tests
    results.append(("Progressive Generation", test_progressive_generation()))
    print("\n" + "-"*70 + "\n")
    results.append(("Visual Step Progression", test_step_progression_visual()))

    # Summary
    print("\n" + "#"*70)
    print("# TEST SUMMARY")
    print("#"*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print(f"\n✗ {total - passed} TEST(S) FAILED")
        sys.exit(1)
