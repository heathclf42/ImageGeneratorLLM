"""
Generate an image and save EVERY denoising step.

This creates a complete progression showing how the image evolves
from pure noise to the final result, one step at a time.
"""

from image_gen.core import ImageGenerator
from image_gen.utils.output_manager import OutputManager
import torch
from datetime import datetime
from pathlib import Path
from PIL import Image
import numpy as np


def latents_to_image(pipe, latents):
    """Convert latents to a PIL image using the VAE decoder."""
    # Decode latents to image
    latents = latents / pipe.vae.config.scaling_factor
    with torch.no_grad():
        image = pipe.vae.decode(latents, return_dict=False)[0]

    # Convert to PIL
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image[0])


def generate_with_all_steps(
    prompt: str,
    total_steps: int = 50,
    seed: int = 42,
    width: int = 1024,
    height: int = 1024
):
    """Generate an image and save every single denoising step."""

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d")
    session_name = f"{prompt.split()[0].lower()}_allsteps_{timestamp}"
    output_mgr = OutputManager(session_name=session_name)

    print(f"\n🎨 Generating: '{prompt}'")
    print(f"📁 Output: {output_mgr.session_dir}")
    print(f"🔢 Steps: {total_steps} (saving ALL steps)")
    print(f"🌱 Seed: {seed}\n")

    # Initialize generator
    gen = ImageGenerator()

    # Get the pipeline
    pipe = gen.pipeline

    # Set random seed
    generator = torch.Generator(device=gen.device).manual_seed(seed)

    # Prepare prompt embeddings
    print("📊 Encoding prompt...")
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        device=gen.device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )

    # Prepare timesteps
    pipe.scheduler.set_timesteps(total_steps, device=gen.device)
    timesteps = pipe.scheduler.timesteps

    # Prepare latent variables
    print("🎲 Initializing random noise...")
    num_channels_latents = pipe.unet.config.in_channels
    latents = pipe.prepare_latents(
        1,  # batch_size
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        gen.device,
        generator,
    )

    # Add noise embeddings for SDXL
    add_text_embeds = pooled_prompt_embeds
    add_time_ids = pipe._get_add_time_ids(
        (height, width),  # original_size
        (0, 0),  # crops_coords_top_left
        (height, width),  # target_size
        dtype=prompt_embeds.dtype,
    )

    negative_add_time_ids = add_time_ids
    add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0).to(gen.device)
    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)

    # Denoising loop
    print(f"\n🔄 Starting denoising process ({total_steps} steps)...\n")

    for i, t in enumerate(timesteps):
        step_num = i + 1

        # Expand latents for classifier free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Prepare added inputs
        added_cond_kwargs = {
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids
        }

        # Predict noise
        with torch.no_grad():
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds]),
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

        # Perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        guidance_scale = 7.5
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Compute previous noisy sample
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Save this step
        image = latents_to_image(pipe, latents)
        output_path = output_mgr.get_output_path(f"step_{step_num:03d}_of_{total_steps:03d}.png")
        image.save(output_path)

        # Progress indicator
        progress = (step_num / total_steps) * 100
        bar_length = 40
        filled = int(bar_length * step_num / total_steps)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\r  [{bar}] Step {step_num:3d}/{total_steps} ({progress:5.1f}%) - Saved: step_{step_num:03d}_of_{total_steps:03d}.png", end='', flush=True)

    print("\n\n✅ Generation complete!")
    print(f"📁 All {total_steps} steps saved to: {output_mgr.session_dir}/images/")

    # Also save the final result with a nice name
    final_path = output_mgr.get_output_path(f"{prompt.replace(' ', '_')[:30]}_final.png")
    image.save(final_path)
    print(f"🎨 Final image: {final_path}")

    return image, output_mgr.session_dir


if __name__ == "__main__":
    # Generate giraffe with ALL steps saved
    generate_with_all_steps(
        prompt="a majestic giraffe standing in the African savanna, golden hour lighting",
        total_steps=50,
        seed=42,
        width=1024,
        height=1024
    )
