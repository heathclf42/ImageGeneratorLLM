# ImageGeneratorLLM

Local image generation system optimized for Apple Silicon (M4 Pro) combining multiple state-of-the-art models with LLM integration.

## Features

### ✅ Currently Implemented

- **SDXL (Stable Diffusion XL)**: High-quality versatile image generation (currently active)
- **MPS Optimization**: Native Apple Silicon Metal Performance Shaders support
- **REST API Server**: FastAPI server with OpenAPI docs at /docs
- **CLI Interface**: Full-featured command-line interface
- **Python API**: Direct Python integration with `ImageGenerator` class
- **Auto-Preview**: Automatic image opening in system viewer
- **Smart Memory Management**: Lazy model loading and unloading
- **LLM Integration**: Function calling examples for Qwen and other LLMs
- **Batch Generation**: Process multiple prompts efficiently
- **Reproducible Results**: Seed-based generation for identical outputs

### 🔜 Planned Features

- **FLUX.1 Schnell**: Fast text-to-image (requires HuggingFace terms acceptance)
- **ControlNet Union**: Structural guidance (pose, depth, canny, etc.)
- **BrushNet**: Professional inpainting and image editing

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.13+
- 24GB+ unified memory recommended
- ~30-50GB disk space for models

## Quick Start

### Installation

```bash
# Clone or navigate to project directory
cd ImageGeneratorLLM

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Python API

```python
from image_gen.core import ImageGenerator

# Initialize with auto-preview
gen = ImageGenerator(auto_preview=True)

# Generate single image
image = gen.generate("a serene mountain landscape at sunset")
# → Saves to outputs/ and opens automatically

# Generate with custom settings
image = gen.generate(
    prompt="a portrait of a wise wizard",
    width=768,
    height=768,
    num_inference_steps=40,  # Higher steps = better quality
    seed=42  # For reproducible results
)

# Batch generation
images = gen.generate_batch([
    "a red apple",
    "a blue ocean",
    "a green forest"
], num_inference_steps=20)
```

### REST API Server

```bash
# Start server (opens API docs at http://localhost:8000/docs)
python -m image_gen.server

# Generate image via API
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a serene mountain landscape",
    "num_inference_steps": 30,
    "seed": 42
  }'

# Health check
curl http://localhost:8000/health
```

### CLI

```bash
# Generate single image
python -m image_gen.cli generate "a serene mountain landscape" --steps 30

# Generate from file of prompts
python -m image_gen.cli batch prompts.txt --steps 20

# Quick test
python -m image_gen.cli test

# System information
python -m image_gen.cli info
```

## Documentation

- **[Quick Start Guide](docs/QUICK_START.md)** - Get started in 5 minutes
- **[Codebase Reference](docs/CODEBASE_REFERENCE.md)** - Complete developer reference
- **[Examples](examples/README.md)** - Usage patterns and integration examples
  - Python API usage
  - Qwen LLM function calling integration
  - Batch processing examples

## Project Structure

```
ImageGeneratorLLM/
├── image_gen/              # Main package
│   ├── models/             # Model implementations (FLUX, SDXL, etc.)
│   ├── utils/              # Utilities (device detection, etc.)
│   ├── core.py             # Main ImageGenerator class
│   ├── config.py           # Configuration management
│   ├── server.py           # FastAPI REST server
│   └── cli.py              # Command-line interface
├── tests/                  # Test suite
├── examples/               # Usage examples
├── outputs/                # Generated images
├── docs/                   # Documentation
└── config/                 # Configuration files
```

## Model Details

### Currently Active: SDXL

- **Model**: Stable Diffusion XL Base 1.0
- **Size**: ~7GB
- **Quality**: Excellent, state-of-the-art for 2024
- **Speed on M4 Pro**:
  - 10 steps: ~8s (draft quality)
  - 20 steps: ~15s (good quality)
  - 30 steps: ~25s (high quality)
  - 40-50 steps: ~35-45s (maximum quality)
- **License**: OpenRAIL++-M (commercial use allowed)

### Memory Management

- **Lazy Loading**: Models only load when first used
- **Automatic Caching**: Keeps frequently used models in memory
- **Manual Control**: Unload models with `gen.unload_models()`
- **Memory Usage**:
  - Idle: 0GB (no models loaded)
  - SDXL loaded: 7-8GB
  - Peak during generation: 10-12GB
  - Recommended: 16GB+ unified memory

### Performance Benchmarks (M4 Pro, 24GB)

| Operation | Time | Notes |
|-----------|------|-------|
| Model loading | 12-15s | First time only |
| Generation (1024x1024, 10 steps) | ~8s | Draft quality |
| Generation (1024x1024, 20 steps) | ~15s | Good quality |
| Generation (1024x1024, 30 steps) | ~25s | High quality |
| Batch (3 images, 20 steps) | ~45s | Sequential processing |

## Integration with Qwen LLM

See `examples/qwen_function_calling.py` for complete integration examples.

The REST API is designed for LLM function calling:

```python
# Function definition for Qwen
{
    "name": "generate_image",
    "description": "Generate an image from text description using SDXL",
    "parameters": {
        "prompt": {"type": "string"},
        "num_inference_steps": {"type": "integer", "default": 30},
        "seed": {"type": "integer"}
    }
}

# Qwen calls the function
response = requests.post("http://localhost:8000/generate", json={
    "prompt": "a serene mountain landscape",
    "num_inference_steps": 30
})
```

## License

- Code: MIT License
- SDXL Model: OpenRAIL++-M License (commercial use allowed)
- All components free for commercial use

## Credits

- Stable Diffusion XL by Stability AI
- HuggingFace Diffusers library
- FastAPI framework
- Built with Claude Code
