"""
Generate a giraffe image with progressive step saves.

This script generates the same giraffe image multiple times with increasing
step counts to show how the image progressively improves during generation.
"""

from image_gen.core import ImageGenerator
from image_gen.utils.output_manager import OutputManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Generate giraffe image, saving every 5 steps."""

    # Create organized output directory
    output_mgr = OutputManager(session_name="giraffe_progressive", create_subdirs=True)
    logger.info("=" * 70)
    logger.info("Progressive Giraffe Generation")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_mgr.session_dir}\n")

    # Initialize generator
    gen = ImageGenerator(auto_preview=False)

    # Giraffe prompt
    prompt = "a majestic giraffe standing in the African savanna, golden hour lighting, professional wildlife photography, detailed, 4k"

    # Use same seed for consistency
    seed = 42

    # Generate at steps: 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
    step_counts = list(range(5, 51, 5))

    logger.info(f"Generating giraffe with prompt:")
    logger.info(f'  "{prompt}"')
    logger.info(f"\nUsing seed: {seed}")
    logger.info(f"Saving at steps: {step_counts}\n")

    for i, steps in enumerate(step_counts, 1):
        logger.info(f"[{i}/{len(step_counts)}] Generating with {steps} steps...")

        # Generate image
        image = gen.generate(
            prompt=prompt,
            num_inference_steps=steps,
            seed=seed,
            save_path=output_mgr.get_output_path(f"giraffe_{steps:02d}steps.png"),
            auto_save=True
        )

        logger.info(f"  ✓ Saved: giraffe_{steps:02d}steps.png")

    logger.info("\n" + "=" * 70)
    logger.info("✓ All images generated!")
    logger.info(f"Output directory: {output_mgr.session_dir}")
    logger.info("\nYou can see the progression from noisy (5 steps) to refined (50 steps)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
