# Codebase Reference

Complete reference for the ImageGeneratorLLM project structure, components, and development guidelines.

## Quick Navigation

### Core Components
- **Image Generation**: `image_gen/models/flux.py` - Main SDXL pipeline implementation
- **Configuration**: `image_gen/config.py` - All settings and model configurations
- **Device Management**: `image_gen/utils/device.py` - MPS/CUDA/CPU detection
- **Tests**: `tests/test_generator.py` - Comprehensive test suite

### Documentation
- **Quick Start**: `docs/QUICK_START.md` - Get started in 5 minutes
- **This File**: Complete codebase reference
- **README**: `README.md` - Project overview

### Output Locations
- **Generated Images**: `outputs/` - All generated images
- **Model Cache**: `~/.cache/huggingface/hub/` - Downloaded models (system-wide)
- **Virtual Environment**: `venv/` - Python dependencies

## Project Structure

```
ImageGeneratorLLM/
├── image_gen/                  # Main package
│   ├── __init__.py            # Package initialization
│   ├── config.py              # Configuration management
│   ├── core.py                # [PLANNED] Main ImageGenerator class
│   ├── server.py              # [PLANNED] FastAPI REST server
│   ├── cli.py                 # [PLANNED] Command-line interface
│   ├── models/                # Model implementations
│   │   ├── __init__.py       # Models package init
│   │   ├── flux.py           # FLUX/SDXL pipeline (IMPLEMENTED)
│   │   ├── sdxl.py           # [PLANNED] Dedicated SDXL
│   │   ├── controlnet.py     # [PLANNED] ControlNet Union
│   │   └── brushnet.py       # [PLANNED] BrushNet inpainting
│   └── utils/                 # Utility functions
│       ├── __init__.py       # Utils package init
│       └── device.py         # Device detection (IMPLEMENTED)
├── tests/                     # Test suite
│   └── test_generator.py     # Generator tests (IMPLEMENTED)
├── examples/                  # Usage examples
│   └── [PLANNED]             # Example scripts
├── outputs/                   # Generated images
│   ├── .gitkeep             # Tracked in git
│   └── *.png                # Ignored by git
├── docs/                      # Documentation
│   ├── QUICK_START.md        # User guide (IMPLEMENTED)
│   ├── CODEBASE_REFERENCE.md # This file (IMPLEMENTED)
│   ├── architecture/         # Architecture docs
│   ├── guides/               # How-to guides
│   ├── api/                  # API reference
│   └── decisions/            # Architecture Decision Records
├── config/                    # Configuration files
│   └── local.yaml            # Optional user config (git-ignored)
├── venv/                      # Virtual environment (git-ignored)
├── .gitignore                # Git ignore rules
├── README.md                  # Project overview
└── requirements.txt           # Python dependencies
```

## File Purposes

### Core Python Modules

#### `image_gen/config.py`
**Purpose**: Centralized configuration management

**Key Classes**:
- `Config`: Main configuration class with model settings, API config, output settings

**Key Functions**:
- `get_config()`: Get singleton config instance

**Configuration Sections**:
- `models`: Model-specific settings (model_id, keep_loaded, idle_timeout, default_steps)
- `api`: REST API settings (host, port, log_level)
- `output`: Output file settings (directory, auto_open, naming_pattern)
- `device`: Device override (None = auto-detect)
- `cache_dir`: Model cache location (None = use HuggingFace default)

**Example**:
```python
from image_gen.config import get_config

config = get_config()
model_id = config.get_model_id("flux")  # "stabilityai/stable-diffusion-xl-base-1.0"
should_keep = config.should_keep_loaded("flux")  # True
```

#### `image_gen/utils/device.py`
**Purpose**: Automatic device detection for PyTorch

**Key Functions**:
- `detect_device(prefer_device=None)`: Returns "mps", "cuda", or "cpu"
- `get_device(config_device=None)`: Returns torch.device object
- `get_device_info()`: Returns dict with device specs
- `print_device_info()`: Prints formatted device information

**Example**:
```python
from image_gen.utils.device import detect_device, get_device

device_str = detect_device()  # "mps" on Apple Silicon
device = get_device()  # torch.device("mps")
```

#### `image_gen/models/flux.py`
**Purpose**: FLUX/SDXL image generation pipeline

**Key Classes**:
- `FluxGenerator`: Main generator class

**Key Methods**:
- `__init__(device=None)`: Initialize and load model
- `generate(prompt, height, width, num_inference_steps, seed)`: Generate single image
- `generate_batch(prompts, ...)`: Generate multiple images
- `unload()`: Free model from memory

**Auto-Detection**:
- Automatically detects model type from model_id
- Uses `FluxPipeline` for FLUX models
- Uses `StableDiffusionXLPipeline` for SDXL models

**Example**:
```python
from image_gen.models.flux import FluxGenerator

gen = FluxGenerator()
image = gen.generate("a cat on mars", num_inference_steps=30, seed=42)
```

### Test Files

#### `tests/test_generator.py`
**Purpose**: Comprehensive test suite for image generation

**Test Functions**:
1. `test_basic_generation()`: Basic image generation with defaults
2. `test_custom_size()`: Custom dimensions (768x768)
3. `test_reproducibility()`: Seed-based reproducibility
4. `test_batch_generation()`: Multiple prompts
5. `test_quality_steps()`: Different step counts (10, 20, 30)

**Run Tests**:
```bash
source venv/bin/activate
python tests/test_generator.py
```

## Architecture Decisions

### 1. Why SDXL as Primary Model?

**Decision**: Use Stable Diffusion XL as the primary model initially

**Rationale**:
- No authentication required (unlike FLUX.1-schnell)
- Widely tested and proven
- Excellent quality
- Commercial-friendly license

**Trade-offs**:
- Slower than FLUX (30 steps vs 4 steps)
- Larger model (~7GB vs potential smaller alternatives)

**Future**: Add FLUX.1-schnell once user completes HuggingFace authentication

### 2. MPS (Metal Performance Shaders) Optimization

**Decision**: Use Apple's Metal Performance Shaders for GPU acceleration

**Rationale**:
- Native Apple Silicon support
- Excellent performance on M-series chips
- Automatically detected and used

**Implementation**:
- Auto-detection in `device.py`
- FP16 precision for memory efficiency
- Fallback to CPU if MPS unavailable

### 3. System-Wide Model Cache

**Decision**: Use HuggingFace's default cache (`~/.cache/huggingface/`)

**Rationale**:
- Share models across projects
- Standard location
- Managed by HuggingFace CLI

**Alternative Considered**: Project-local `./models/` directory
- **Pros**: Self-contained, easy to delete
- **Cons**: Duplicates models, uses project disk space

### 4. Configuration-Driven Design

**Decision**: All settings in `config.py`, no hardcoded values

**Benefits**:
- Easy to modify behavior
- Clear defaults
- Override via `config/local.yaml`
- Environment-specific settings

### 5. Organized Output Directory

**Decision**: All generated images go to `outputs/` directory

**Rationale**:
- Clean project root
- Easy to find all images
- Git-ignored (except .gitkeep)
- Consistent naming

## Import Standards

### Absolute Imports (Required)

✓ **Use absolute imports from package root**:
```python
from image_gen.config import get_config
from image_gen.models.flux import FluxGenerator
from image_gen.utils.device import detect_device
```

✗ **Never use relative imports**:
```python
from ..config import get_config  # DON'T DO THIS
from .device import detect_device  # DON'T DO THIS
```

### Import Order

1. **Standard library** (alphabetical)
```python
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
```

2. **Third-party** (alphabetical)
```python
import torch
from diffusers import FluxPipeline
from PIL import Image
```

3. **Local** (alphabetical)
```python
from image_gen.config import get_config
from image_gen.utils.device import get_device
```

## Coding Standards

### Type Hints (Required)

All functions must have type hints:
```python
def generate(
    self,
    prompt: str,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> Image.Image:
    """Generate an image from a text prompt."""
    ...
```

### Docstrings (Required)

All public functions and classes must have docstrings:
```python
class FluxGenerator:
    """
    FLUX.1 Schnell text-to-image generator with MPS optimization.

    This class handles loading, caching, and inference for the FLUX.1 Schnell
    model, optimized for Apple Silicon with Metal Performance Shaders (MPS).

    Attributes:
        pipeline: Loaded FluxPipeline instance
        device: Compute device (mps, cuda, or cpu)

    Example:
        >>> gen = FluxGenerator()
        >>> image = gen.generate("a cat on mars")
    """
```

### Module Headers (Required)

All modules must start with a header:
```python
"""
Module: image_gen.models.flux
Purpose: FLUX.1 Schnell pipeline implementation for fast text-to-image generation
Dependencies: diffusers, torch, transformers
Author: Generated for ImageGeneratorLLM
Reference: See docs/architecture/FLUX_PIPELINE.md
"""
```

## Testing Standards

### Test Coverage Requirements

Every feature must have corresponding tests:
- Model implementations → `tests/test_<model>.py`
- API endpoints → `tests/test_api.py`
- CLI commands → `tests/test_cli.py`
- Utilities → `tests/test_utils.py`

### Test Structure

```python
def test_feature_name():
    """Test description of what is being tested."""
    # Arrange
    setup_code()

    # Act
    result = function_to_test()

    # Assert
    assert result == expected_value

    return True
```

## Git Workflow

### Commit Message Format

```
<type>: <short summary>

<detailed description if needed>

- Bullet points for specific changes
- Reference issues: #123
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code refactoring
- `test`: Add/update tests
- `chore`: Maintenance

### Branch Strategy

- `master`: Stable, working code
- `develop`: Integration branch (when added)
- `feature/<name>`: New features
- `fix/<name>`: Bug fixes

## Environment Variables

Currently none. Future considerations:
- `HUGGINGFACE_TOKEN`: HF authentication token
- `IMAGE_GEN_CACHE_DIR`: Override model cache location
- `IMAGE_GEN_DEVICE`: Force specific device

## Performance Benchmarks

### M4 Pro (24GB) Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Model Loading | 12-15s | First time only |
| Generation (10 steps) | ~8s | Draft quality |
| Generation (20 steps) | ~15s | Good quality |
| Generation (30 steps) | ~25s | High quality |
| Generation (40 steps) | ~35s | Maximum quality |
| Batch (3 images, 20 steps) | ~45s | Sequential |

### Memory Usage

| State | Memory | Notes |
|-------|--------|-------|
| Idle | 0GB | No model loaded |
| Model Loaded | 7-8GB | SDXL in memory |
| Generating | 10-12GB | Peak during inference |

## Known Issues

1. **MPS Non-Determinism**: Seeds may not produce identical images on MPS
   - **Status**: Apple Silicon limitation
   - **Workaround**: Use CPU for reproducibility (slower)

2. **FLUX.1-schnell Authentication**: Requires HuggingFace account
   - **Status**: User must accept terms manually
   - **Workaround**: Using SDXL as primary model

## Future Roadmap

### Phase 1 (Current)
- [x] SDXL pipeline with MPS
- [x] Configuration system
- [x] Basic tests
- [x] Quick start documentation

### Phase 2 (Next)
- [ ] REST API server
- [ ] CLI interface
- [ ] Automatic image preview
- [ ] FLUX.1 support

### Phase 3 (Future)
- [ ] ControlNet integration
- [ ] BrushNet inpainting
- [ ] Qwen LLM integration
- [ ] Web UI

## Contributing Guidelines

1. **Follow coding standards**: Type hints, docstrings, imports
2. **Write tests**: Every feature needs tests
3. **Update docs**: Keep documentation current
4. **Commit regularly**: Small, logical commits
5. **Push to GitHub**: All work tracked in repository

## Getting Help

- **Quick Start**: See `docs/QUICK_START.md`
- **Examples**: Check `tests/test_generator.py` for working code
- **Issues**: Open GitHub issue for bugs/features
- **Architecture**: See `docs/architecture/` for design docs
