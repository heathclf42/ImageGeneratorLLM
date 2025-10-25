# ImageGeneratorLLM - Architecture Documentation

**Last Updated:** October 25, 2024
**Version:** 1.0.0

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Core Components](#core-components)
4. [API Reference](#api-reference)
5. [Configuration](#configuration)
6. [Testing Strategy](#testing-strategy)
7. [Code Standards](#code-standards)
8. [Design Decisions](#design-decisions)
9. [File Locations Reference](#file-locations-reference)

---

## Project Overview

ImageGeneratorLLM is a Python-based image generation framework built on Stable Diffusion XL (SDXL). It provides:
- **CLI interface** for command-line usage
- **REST API** for programmatic access
- **Real-time visualization server** for monitoring generation pipeline
- **Thermal management** for sustained performance on Apple Silicon
- **Flexible output organization** with automatic directory management

**Target Platform:** Primarily macOS with Apple Silicon (MPS), with fallback support for CUDA and CPU.

---

## Directory Structure

```
ImageGeneratorLLM/
├── image_gen/                  # Core package
│   ├── __init__.py
│   ├── cli.py                 # Command-line interface
│   ├── config.py              # Configuration management
│   ├── core.py                # ImageGenerator main class
│   ├── server.py              # FastAPI REST API
│   ├── visualization_server.py # Real-time UI (2723 lines)
│   ├── models/
│   │   ├── __init__.py
│   │   └── flux.py            # SDXL model implementation
│   └── utils/
│       ├── __init__.py
│       ├── device.py          # Device detection (MPS/CUDA/CPU)
│       ├── output_manager.py  # Output directory organization
│       └── thermal_manager.py # Thermal throttling management
├── docs/                       # Documentation
│   ├── ARCHITECTURE.md        # This file
│   ├── CODEBASE_REFERENCE.md  # Code navigation guide
│   ├── QUICK_START.md         # Getting started guide
│   ├── guides/                # User guides
│   │   └── THERMAL_MANAGEMENT_GUIDE.md
│   └── design/                # Design documents
│       └── COMPREHENSIVE_FLOWCHART_DESIGN.md
├── examples/                   # Example scripts
│   ├── README.md
│   ├── basic_usage.py
│   ├── qwen_function_calling.py
│   ├── thermal_management_demo.py
│   ├── advanced/              # Advanced examples
│   │   ├── save_all_denoising_steps.py
│   │   └── progressive_generation.py
│   └── image_editing/         # Image editing examples
│       └── remove_text_interactive.py
├── tests/                      # Test suite
│   └── test_generator.py
├── outputs/                    # Generated images (gitignored)
│   ├── experiments/           # Test outputs
│   ├── generated/             # Production outputs
│   └── analysis/              # Metrics and analysis data
├── venv/                       # Virtual environment (gitignored)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Main project documentation
```

---

## Core Components

### 1. ImageGenerator (`image_gen/core.py`)

**Purpose:** Main orchestration class for image generation

**Key Responsibilities:**
- Model initialization and management
- Device selection (MPS/CUDA/CPU)
- Output organization
- Thermal management integration

**Public API:**
```python
class ImageGenerator:
    def __init__(self, model: ModelType = "flux", device: Optional[str] = None)
    def generate(self, prompt: str, **kwargs) -> Image.Image
    def generate_batch(self, prompts: List[str], **kwargs) -> List[Image.Image]
```

**Usage Example:**
```python
from image_gen.core import ImageGenerator

gen = ImageGenerator(model="flux")
image = gen.generate("a majestic giraffe", num_inference_steps=30)
```

### 2. FluxGenerator (`image_gen/models/flux.py`)

**Purpose:** SDXL model wrapper (note: named "flux" but implements SDXL)

**Key Responsibilities:**
- HuggingFace pipeline loading
- Model inference
- Device management
- Memory optimization

**Public API:**
```python
class FluxGenerator:
    def __init__(self, model_id: str, device: str)
    def generate(self, prompt: str, **kwargs) -> Image.Image
    def cleanup(self)
```

**Note:** Despite the filename, this currently implements Stable Diffusion XL. The name is historical and may be updated when FLUX.1 Schnell support is added.

### 3. CLI (`image_gen/cli.py`)

**Purpose:** Command-line interface built with Click

**Available Commands:**
- `image-gen generate` - Generate single image
- `image-gen batch` - Generate multiple images
- `image-gen info` - Show system information

**Usage:**
```bash
# Generate image
image-gen generate "a majestic giraffe" --steps 30 --size 1024x1024

# Batch generation
image-gen batch prompts.txt --count 5
```

### 4. REST API (`image_gen/server.py`)

**Purpose:** FastAPI-based HTTP API

**Endpoints:**
- `GET /` - API documentation (auto-generated)
- `POST /generate` - Generate single image
- `POST /generate/batch` - Batch generation
- `GET /health` - Health check

**Usage:**
```bash
# Start server
uvicorn image_gen.server:app --host 0.0.0.0 --port 8000

# Generate image
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a majestic giraffe", "num_inference_steps": 30}'
```

### 5. Visualization Server (`image_gen/visualization_server.py`)

**Purpose:** Real-time web UI for monitoring generation pipeline

**Features:**
- Live pipeline visualization
- Performance metrics
- Model insights
- Multi-modality support (text2img, img2img, audio, video, LLM)

**Status:** Experimental - not integrated into main API

**Usage:**
```bash
python -m image_gen.visualization_server
# Open http://localhost:8000 in browser
```

### 6. Utility Modules

#### OutputManager (`image_gen/utils/output_manager.py`)
- Organizes outputs into dated directories
- Prevents filename collisions
- Supports custom output paths

#### ThermalManager (`image_gen/utils/thermal_manager.py`)
- Monitors system temperature (macOS only)
- Implements thermal throttling
- Provides cooling recommendations

#### Device (`image_gen/utils/device.py`)
- Auto-detects best available device (MPS/CUDA/CPU)
- Provides device selection logic
- Falls back gracefully

---

## API Reference

### ImageGenerator

#### `__init__(model: ModelType = "flux", device: Optional[str] = None)`
Initialize the image generator.

**Parameters:**
- `model`: Model type ("flux" for SDXL, "sdxl" also accepted)
- `device`: Device to use ("mps", "cuda", "cpu", or None for auto-detect)

#### `generate(...) -> Image.Image`
Generate a single image.

**Parameters:**
- `prompt` (str): Text description
- `height` (Optional[int]): Image height (default: 1024)
- `width` (Optional[int]): Image width (default: 1024)
- `num_inference_steps` (Optional[int]): Denoising steps (default: 30)
- `guidance_scale` (Optional[float]): Prompt adherence (default: 7.5)
- `seed` (Optional[int]): Random seed for reproducibility
- `save_path` (Optional[Path]): Custom save location
- `auto_save` (bool): Automatically save to outputs/ (default: True)

**Returns:** PIL Image object

#### `generate_batch(...) -> List[Image.Image]`
Generate multiple images.

**Parameters:**
- `prompts` (List[str]): List of text descriptions
- All other parameters same as `generate()`

**Returns:** List of PIL Image objects

---

## Configuration

### Config File Location
- Default: `image_gen/config.py`
- Environment-based overrides supported (future)

### Model Configuration

```python
DEFAULT_CONFIG = {
    "flux": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "default_steps": 30,
        "default_guidance": 7.5,
        "scheduler": "EulerDiscreteScheduler",
    }
}
```

### Device Selection Priority
1. User-specified device (if provided)
2. MPS (if available - Apple Silicon)
3. CUDA (if available - NVIDIA GPU)
4. CPU (fallback)

### Output Organization

**Default Structure:**
```
outputs/
└── YYYYMMDD_HHMMSS/
    ├── image_0001.png
    ├── image_0002.png
    └── metadata.json
```

**Custom Path:**
```python
gen.generate("prompt", save_path=Path("my_output/custom.png"))
```

---

## Testing Strategy

### Current Test Coverage

**Tested:**
- ✅ Core image generation
- ✅ Custom sizes
- ✅ Seed reproducibility
- ✅ Batch generation
- ✅ Quality settings

**Not Tested (TODO):**
- ❌ CLI commands
- ❌ REST API endpoints
- ❌ Output manager
- ❌ Thermal manager
- ❌ Device detection

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run with coverage
pytest --cov=image_gen tests/

# Run specific test
pytest tests/test_generator.py::test_basic_generation
```

### Test File Structure

```
tests/
├── test_generator.py       # Core generation tests
├── test_cli.py            # CLI command tests (TODO)
├── test_api.py            # REST API tests (TODO)
├── test_output_manager.py # Output organization (TODO)
└── test_thermal_manager.py # Thermal management (TODO)
```

---

## Code Standards

### 1. Type Hints

**Required** for all function signatures:

```python
def generate(
    self,
    prompt: str,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> Image.Image:
    pass
```

### 2. Docstrings

**Required** for all public classes, functions, and methods:

```python
def generate(self, prompt: str, **kwargs) -> Image.Image:
    """Generate an image from a text prompt.

    Args:
        prompt: Text description of desired image
        **kwargs: Additional generation parameters

    Returns:
        PIL Image object

    Example:
        >>> gen = ImageGenerator()
        >>> image = gen.generate("a majestic giraffe")
    """
```

**Format:** Google-style docstrings

### 3. Import Organization

**Order:**
1. Standard library imports
2. Third-party imports
3. Local/package imports

**Alphabetical within each group**

```python
# Standard library
import logging
from pathlib import Path
from typing import Optional, List

# Third-party
from PIL import Image
import torch

# Local
from image_gen.config import get_config
from image_gen.utils.device import get_device
```

### 4. Naming Conventions

- **Classes:** PascalCase (`ImageGenerator`)
- **Functions/Methods:** snake_case (`generate_image`)
- **Constants:** UPPER_SNAKE_CASE (`DEFAULT_CONFIG`)
- **Private methods:** Leading underscore (`_internal_method`)

### 5. File Organization

**Maximum file length:** ~500 lines (guideline, not strict)
- `visualization_server.py` exceeds this (2723 lines) - consider refactoring if it grows

---

## Design Decisions

### 1. Why SDXL over FLUX?

**Decision:** Use Stable Diffusion XL as primary model

**Reasoning:**
- More mature ecosystem
- Better documentation
- Wider community support
- FLUX.1 Schnell is planned for future

**Trade-offs:**
- SDXL is slower than FLUX (3-5 sec vs 1-2 sec per image)
- But SDXL has better quality at lower step counts

### 2. Why MPS (Apple Silicon) as Primary Target?

**Decision:** Optimize for Apple Silicon first

**Reasoning:**
- Developer's primary platform
- Good performance with Metal Performance Shaders (MPS)
- Growing market share

**Fallback Support:**
- CUDA (NVIDIA) fully supported
- CPU fallback available

### 3. Why Separate CLI, API, and Visualization Server?

**Decision:** Three separate interfaces instead of one monolithic app

**Reasoning:**
- **Separation of concerns:** Each interface has different use cases
- **Flexibility:** Users can choose their preferred interface
- **Maintainability:** Easier to test and update independently

**Trade-offs:**
- More code to maintain
- Need to keep APIs in sync

### 4. Why OutputManager Instead of Simple save()?

**Decision:** Centralized output management with date-based directories

**Reasoning:**
- Prevents filename collisions
- Automatic organization
- Easy to find recent outputs
- Supports custom paths when needed

**Trade-offs:**
- Slightly more complex than simple save()
- Users need to know directory structure

### 5. Why Real-time Visualization Server?

**Decision:** Build comprehensive web-based UI with pipeline visualization

**Reasoning:**
- Educational value (shows how diffusion works)
- Debugging aid (see which stage is slow)
- Impressive demo for presentations
- Multi-modality support (future-proofing)

**Trade-offs:**
- Large file (2723 lines)
- Experimental status (not production-ready)
- Maintenance overhead

---

## File Locations Reference

### Configuration Files
- **Main config:** `image_gen/config.py`
- **Model configs:** Embedded in `config.py` (DEFAULT_CONFIG dict)

### Templates
- **CLI help:** Embedded in `image_gen/cli.py` docstrings
- **API docs:** Auto-generated by FastAPI from Pydantic models
- **Visualization UI:** Embedded in `image_gen/visualization_server.py` (HTML in Python string)

### Icons and Assets
- **None currently** - visualization UI uses emoji and CSS-only graphics

### Test Data
- **Ground truth images:** Not yet implemented
- **Test prompts:** Embedded in `tests/test_generator.py`

### Output Locations
- **Default outputs:** `outputs/YYYYMMDD_HHMMSS/`
- **Experiments:** `outputs/experiments/` (manual organization)
- **Analysis data:** `outputs/analysis/*.json`

### Documentation
- **User guides:** `docs/guides/*.md`
- **Design docs:** `docs/design/*.md`
- **Code reference:** `docs/CODEBASE_REFERENCE.md`
- **This file:** `docs/ARCHITECTURE.md`

### Examples
- **Basic usage:** `examples/basic_usage.py`
- **Advanced:** `examples/advanced/*.py`
- **Image editing:** `examples/image_editing/*.py`

---

## Future Roadmap

### Planned Features

1. **FLUX.1 Schnell Support**
   - 1-2 second generation times
   - Update `image_gen/models/flux.py` or create new module

2. **ControlNet Integration**
   - Structure-guided generation
   - Pose, depth, edge control

3. **Image-to-Image**
   - Modify existing images
   - Style transfer

4. **Video Generation**
   - Text-to-video
   - Image-to-video animation

5. **Multi-GPU Support**
   - Parallel batch generation
   - Load balancing

### Technical Debt

1. **Rename flux.py → sdxl.py**
   - Clarify that it implements SDXL, not FLUX
   - Or implement actual FLUX support

2. **Split visualization_server.py**
   - Extract HTML/CSS/JS to separate files
   - Reduce file size from 2723 lines

3. **Add CLI/API Tests**
   - Comprehensive test coverage
   - Integration tests

4. **Improve Error Handling**
   - Better error messages
   - Graceful degradation

---

## Questions and Answers

### Q: Where do I find the model code?
**A:** `image_gen/models/flux.py` (implements SDXL)

### Q: How do I add a new model?
**A:**
1. Create new file in `image_gen/models/` (e.g., `controlnet.py`)
2. Implement generator class similar to `FluxGenerator`
3. Add config entry in `image_gen/config.py`
4. Update `ModelType` in `image_gen/core.py`

### Q: Where are generated images saved?
**A:** By default: `outputs/YYYYMMDD_HHMMSS/image_NNNN.png`

### Q: How do I change the default model settings?
**A:** Edit `DEFAULT_CONFIG` in `image_gen/config.py`

### Q: Why is generation slow?
**A:**
1. Check device (should be MPS or CUDA, not CPU)
2. Reduce num_inference_steps (try 20-25 instead of 50)
3. Check thermal throttling (see `docs/guides/THERMAL_MANAGEMENT_GUIDE.md`)

### Q: How do I add a new CLI command?
**A:**
1. Add function in `image_gen/cli.py`
2. Decorate with `@cli.command()`
3. Add Click arguments/options

### Q: How do I add a new API endpoint?
**A:**
1. Add route function in `image_gen/server.py`
2. Define Pydantic model for request/response
3. Add FastAPI decorator (`@app.post("/endpoint")`)

---

## Getting Help

- **Quick Start:** `docs/QUICK_START.md`
- **Code Navigation:** `docs/CODEBASE_REFERENCE.md`
- **User Guides:** `docs/guides/`
- **Issues:** GitHub Issues
- **Examples:** `examples/` directory

---

**Last Updated:** October 25, 2024
**Maintainer:** Brad Musick
**License:** (To be determined)