"""
Demonstration of thermal management for long-running image generation tasks.

This script shows how to use the ThermalManager to prevent critical
thermal throttling during batch image generation.
"""

from image_gen.core import ImageGenerator
from image_gen.utils.thermal_manager import ThermalManager, BatchThermalManager
from image_gen.utils.output_manager import OutputManager
from pathlib import Path
import time


def demo_basic_thermal_management():
    """
    Demo: Basic thermal management with automatic throttle detection.
    """
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Thermal Management")
    print("=" * 70)

    # Create organized output directory
    output_mgr = OutputManager(session_name="thermal_demo_basic")
    print(f"📁 Output directory: {output_mgr.session_dir}\n")

    gen = ImageGenerator(auto_preview=False)
    thermal_mgr = ThermalManager(
        baseline_time_per_step=5.0,  # Expected ~5s/step from our analysis
        throttle_threshold=1.5,      # Trigger cooling if 50% slower
        cooling_duration=30,         # 30-second cooling breaks
    )

    prompt = "a beautiful sunset over mountains"
    steps_to_generate = [20, 25, 30, 35, 40, 45, 50]

    for i, steps in enumerate(steps_to_generate, 1):
        print(f"\n[{i}/{len(steps_to_generate)}] Generating with {steps} steps...")

        # Check if cooling break needed
        if thermal_mgr.should_cool():
            thermal_mgr.cooling_break()

        # Generate and time the operation
        start = time.time()
        image = gen.generate(
            prompt=prompt,
            num_inference_steps=steps,
            seed=42,
            save_path=output_mgr.get_output_path(f"sunset_{steps:02d}steps.png"),
            auto_save=True
        )
        elapsed = time.time() - start

        # Record timing for thermal analysis
        thermal_mgr.record_timing(elapsed)

        state = thermal_mgr.get_thermal_state()
        status = "🔥 THROTTLED" if state and state.is_throttled else "✓ NORMAL"
        print(f"  Time: {elapsed:.2f}s | {status}")

    # Print final statistics
    thermal_mgr.print_stats()


def demo_batch_thermal_management():
    """
    Demo: Batch processing with proactive cooling breaks.
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Batch Thermal Management")
    print("=" * 70)

    # Create organized output directory
    output_mgr = OutputManager(session_name="thermal_demo_batch")
    print(f"📁 Output directory: {output_mgr.session_dir}\n")

    gen = ImageGenerator(auto_preview=False)
    thermal_mgr = BatchThermalManager(
        batch_size=5,                # Cool every 5 images
        batch_cooling_duration=60,   # 60-second breaks between batches
        baseline_time_per_step=5.0,
        throttle_threshold=1.5,
    )

    prompt = "a majestic mountain landscape, detailed, professional photography"
    steps_per_image = 25

    total_images = 15

    print(f"\nGenerating {total_images} images with proactive batch cooling...")
    print(f"Batch size: {thermal_mgr.batch_size} images")
    print(f"Cooling duration: {thermal_mgr.batch_cooling_duration}s between batches\n")

    for i in range(1, total_images + 1):
        print(f"[{i}/{total_images}] Generating image {i}...")

        # Check if cooling break needed (batch or throttle-based)
        if thermal_mgr.should_cool():
            if thermal_mgr.current_batch_count >= thermal_mgr.batch_size:
                print(f"  📦 Batch complete ({thermal_mgr.batch_size} images)")
            thermal_mgr.cooling_break()

        # Generate and time
        start = time.time()
        image = gen.generate(
            prompt=prompt,
            num_inference_steps=steps_per_image,
            seed=42 + i,  # Different seed per image
            save_path=output_mgr.get_output_path(f"mountain_{i:03d}.png"),
            auto_save=True
        )
        elapsed = time.time() - start

        thermal_mgr.record_timing(elapsed)
        print(f"  ✓ Completed in {elapsed:.2f}s")

    # Print final statistics
    thermal_mgr.print_stats()


def demo_long_running_with_thermal_management():
    """
    Demo: Recreate the Lincoln study but with thermal management.

    This demonstrates how the thermal manager would have prevented
    the critical throttling we observed in the original study.
    """
    print("\n" + "=" * 70)
    print("DEMO 3: Long-Running Generation with Thermal Protection")
    print("=" * 70)

    # Create organized output directory
    output_mgr = OutputManager(session_name="lincoln_thermal_managed")
    print(f"📁 Output directory: {output_mgr.session_dir}\n")

    gen = ImageGenerator(auto_preview=False)

    # Use batch thermal manager for long runs
    thermal_mgr = BatchThermalManager(
        batch_size=10,                # Cool every 10 images
        batch_cooling_duration=60,    # 1-minute cooling breaks
        baseline_time_per_step=4.5,   # From our optimal range analysis
        throttle_threshold=1.3,       # More aggressive (30% slowdown triggers cooling)
        critical_threshold=2.0,
    )

    prompt = "a photo-realistic portrait of Abraham Lincoln, president of the United States, highly detailed, professional photography"

    # Smaller test set (first 20 images from original study)
    steps_to_generate = list(range(5, 101, 5))  # 5, 10, 15, ..., 100

    print(f"\nGenerating {len(steps_to_generate)} Lincoln portraits with thermal protection...")
    print(f"This simulates the first phase of our original study.\n")

    for i, steps in enumerate(steps_to_generate, 1):
        print(f"[{i}/{len(steps_to_generate)}] Generating with {steps} steps...")

        # Thermal management check
        if thermal_mgr.should_cool():
            if thermal_mgr.current_batch_count >= thermal_mgr.batch_size:
                print(f"  📦 Batch {thermal_mgr.cooling_breaks_taken + 1} complete")
            thermal_mgr.cooling_break()

        # Generate and time
        start = time.time()
        image = gen.generate(
            prompt=prompt,
            num_inference_steps=steps,
            seed=42,
            save_path=output_mgr.get_output_path(f"lincoln_{steps:04d}steps.png"),
            auto_save=True
        )
        elapsed = time.time() - start

        thermal_mgr.record_timing(elapsed)

        # Status reporting
        time_per_step_ms = (elapsed / steps) * 1000
        state = thermal_mgr.get_thermal_state()
        if state:
            perf_pct = state.performance_ratio * 100
            status = f"Performance: {perf_pct:.0f}%"
        else:
            status = "Establishing baseline..."

        print(f"  ✓ {elapsed:.2f}s ({time_per_step_ms:.0f}ms/step) | {status}")

    # Print comprehensive statistics
    thermal_mgr.print_stats()

    print("\n💡 EXPECTED OUTCOME:")
    print("  With thermal management, you should see:")
    print("  • Consistent ~4-5s/step performance throughout")
    print("  • Proactive cooling breaks preventing critical throttling")
    print("  • Much shorter total runtime than the original 8.8-hour session")
    print("  • No 50+ second/step emergency throttling events")


if __name__ == "__main__":
    print("\n🌡️  THERMAL MANAGEMENT DEMONSTRATION")
    print("=" * 70)
    print("This demonstrates three thermal management strategies:")
    print("1. Basic throttle detection and reactive cooling")
    print("2. Proactive batch-based cooling")
    print("3. Long-running task protection (Lincoln study remake)")
    print("=" * 70)

    choice = input("\nWhich demo? (1/2/3 or 'all'): ").strip()

    if choice == "1":
        demo_basic_thermal_management()
    elif choice == "2":
        demo_batch_thermal_management()
    elif choice == "3":
        demo_long_running_with_thermal_management()
    elif choice.lower() == "all":
        demo_basic_thermal_management()
        demo_batch_thermal_management()
        demo_long_running_with_thermal_management()
    else:
        print("Invalid choice. Please run again and select 1, 2, 3, or 'all'.")
