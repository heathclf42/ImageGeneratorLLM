# Quick Start Guide

## Overview

ImageGeneratorLLM is a local image generation system optimized for Apple Silicon (M4 Pro). Generate high-quality images from text descriptions using Stable Diffusion XL with Metal Performance Shaders acceleration.

## Installation

```bash
# Navigate to project
cd ImageGeneratorLLM

# Activate virtual environment
source venv/bin/activate

# All dependencies are already installed!
```

## Basic Usage

### Generate Your First Image

```python
from image_gen.models.flux import FluxGenerator

# Initialize generator (loads SDXL model)
gen = FluxGenerator()

# Generate image
image = gen.generate("a serene mountain landscape at sunset")

# Image automatically saved to outputs/ directory
```

**Expected time**: 15-30 seconds for generation

### Custom Parameters

```python
# Specify size and quality
image = gen.generate(
    prompt="a futuristic cyberpunk city",
    width=1024,
    height=1024,
    num_inference_steps=30,  # More steps = higher quality but slower
    seed=42  # For reproducible results
)

# Save to specific location
image.save("my_city.png")
```

### Batch Generation

```python
# Generate multiple images
prompts = [
    "a red sports car",
    "a blue ocean wave",
    "a green forest path"
]

images = gen.generate_batch(prompts, num_inference_steps=20)

# All images saved to outputs/ directory
```

## Parameters Reference

### `generate()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | *required* | Text description of image to generate |
| `height` | int | 1024 | Output height in pixels (must be multiple of 8) |
| `width` | int | 1024 | Output width in pixels (must be multiple of 8) |
| `num_inference_steps` | int | 30 | Number of denoising steps (10-50 typical) |
| `seed` | int | None | Random seed for reproducible results |

### Quality vs Speed Guide

| Steps | Quality | Time (M4 Pro) | Use Case |
|-------|---------|---------------|----------|
| 10 | Draft | ~8s | Quick previews |
| 20 | Good | ~15s | Standard generation |
| 30 | High | ~25s | Final outputs |
| 40-50 | Maximum | ~35-45s | Professional work |

## Common Workflows

### Quick Preview

```python
# Fast generation for iteration
gen = FluxGenerator()
image = gen.generate("my idea", num_inference_steps=10)
```

### High-Quality Final

```python
# Best quality for production
gen = FluxGenerator()
image = gen.generate(
    prompt="professional product photo of a watch",
    num_inference_steps=40,
    seed=42  # Keep seed for reproducibility
)
```

### Reproducible Generation

```python
# Generate identical images
seed = 12345

image1 = gen.generate("a cat", seed=seed)
image2 = gen.generate("a cat", seed=seed)

# image1 and image2 are identical
```

## Output Management

All generated images are automatically saved to:
```
ImageGeneratorLLM/outputs/
```

Files are named based on your prompt and timestamp:
```
outputs/test_landscape.png
outputs/test_cyberpunk.png
```

## Troubleshooting

### Out of Memory Error

**Solution**: Reduce image size or close other applications
```python
# Use smaller size
image = gen.generate("prompt", width=768, height=768)
```

### Slow Generation

**Possible causes**:
- Other GPU-intensive apps running
- First generation (model loading takes extra time)
- High step count

**Solutions**:
- Close other GPU apps
- Wait for first generation (subsequent ones are faster)
- Use fewer steps (10-20 for previews)

### Image Quality Issues

**Solution**: Increase inference steps
```python
# Better quality
image = gen.generate("prompt", num_inference_steps=40)
```

## Prompt Writing Tips

### Good Prompts

✓ **Descriptive and specific**:
```
"a serene mountain landscape at sunset, with pine trees in foreground,
golden hour lighting, photorealistic"
```

✓ **Include style and mood**:
```
"a futuristic city, cyberpunk style, neon lights, rainy night,
cinematic composition"
```

✓ **Mention camera/art style**:
```
"portrait of a wise old wizard, oil painting style, detailed,
dramatic lighting"
```

### Avoid

✗ Too vague: "a thing"
✗ Too many concepts: "a cat dog bird fish all together"
✗ Contradictory: "bright dark scene"

## Examples

### Landscapes
```python
prompts = [
    "a serene mountain lake at dawn, mist over water, photorealistic",
    "a desert at sunset with dramatic clouds, warm colors",
    "a tropical beach, turquoise water, palm trees, sunny day"
]
```

### Art Styles
```python
prompts = [
    "a portrait in the style of Van Gogh, swirling brushstrokes",
    "a landscape as a Japanese woodblock print, Mt. Fuji",
    "a still life painting in the style of Dutch masters"
]
```

### Concepts
```python
prompts = [
    "a futuristic robot reading a book in a library",
    "a steampunk airship flying through clouds",
    "a cozy coffee shop interior, warm lighting, rainy day outside"
]
```

## Performance Notes

### First Generation
- **Time**: 30-60 seconds (model loading)
- **Subsequent**: 15-30 seconds

### Memory Usage
- **Model in memory**: ~7-8GB
- **Peak during generation**: ~10GB
- **Recommended**: 16GB+ unified memory

### Optimization
The generator uses:
- ✓ Metal Performance Shaders (MPS)
- ✓ FP16 precision for efficiency
- ✓ Automatic device detection
- ✓ Model caching (no re-downloading)

## Next Steps

- **Run tests**: `python tests/test_generator.py`
- **Explore examples**: Check `examples/` directory
- **Read architecture**: See `docs/architecture/`
- **API documentation**: See `docs/api/`

## Getting Help

- Check `README.md` for overview
- See `docs/` for detailed documentation
- Review `tests/test_generator.py` for more examples
- Open GitHub issue for bugs

## Model Information

**Current Model**: Stable Diffusion XL (SDXL)
- **Size**: ~7GB
- **License**: OpenRAIL++-M (commercial use allowed)
- **Quality**: State-of-the-art for 2024
- **Speed**: Optimized for Apple Silicon

**Future**: FLUX.1 Schnell support (faster, requires HuggingFace authentication)
