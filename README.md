# ImageGeneratorLLM

Local image generation system optimized for Apple Silicon (M4 Pro) combining multiple state-of-the-art models with LLM integration.

## Features

- **FLUX.1 Schnell**: Fast text-to-image generation (2-4 steps, Apache 2.0 license)
- **SDXL**: High-quality versatile image generation
- **ControlNet Union**: Structural guidance (pose, depth, canny, etc.)
- **BrushNet**: Professional inpainting and image editing
- **MPS Optimization**: Native Apple Silicon Metal Performance Shaders support
- **Smart Memory Management**: Automatic model loading/unloading for 24GB unified memory
- **REST API**: FastAPI server for integration with any application
- **CLI Interface**: Simple command-line usage
- **LLM Integration**: Function calling support for Qwen and other LLMs

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

### Basic Usage

```python
from image_gen import ImageGenerator

# Initialize (auto-detects MPS)
gen = ImageGenerator()

# Generate image
image, path = gen.generate("a cat astronaut on mars", model="flux")
# → Saves to outputs/ and opens in Preview.app
```

### REST API Server

```bash
# Start server
python -m image_gen.server

# Make request
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat astronaut on mars", "model": "flux"}'
```

### CLI

```bash
# Generate and open image
image-gen "a cat astronaut on mars" --model flux
```

## Documentation

- [Architecture Overview](docs/architecture/)
- [API Reference](docs/api/)
- [Integration Guides](docs/guides/)
- [Codebase Reference](docs/CODEBASE_REFERENCE.md)

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

| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| FLUX.1 Schnell | ~12GB | ~5s | Fast generation, primary model |
| SDXL | ~7GB | ~15s | High quality, detailed images |
| ControlNet Union | ~5GB | +5s | Guided generation with structure |
| BrushNet | ~3GB | +5s | Inpainting and editing |

## Memory Management

The system uses intelligent model caching:
- **FLUX**: Kept in memory (primary model)
- **SDXL/ControlNet/BrushNet**: Load on-demand, unload after idle timeout
- **Total baseline**: ~12GB (FLUX only)
- **Peak usage**: ~19GB (when using multiple models)

## License

This project uses Apache 2.0 licensed models (FLUX.1 Schnell) and is fully free for commercial use.

## Credits

- FLUX.1 Schnell by Black Forest Labs
- Stable Diffusion XL by Stability AI
- ControlNet by Lvmin Zhang
- BrushNet by TencentARC
