# ImageGeneratorLLM Examples

This directory contains example code demonstrating different ways to use ImageGeneratorLLM.

## Examples

### 1. Basic Python API (`basic_usage.py`)

Direct Python usage without REST API or CLI. Best for:
- Python scripts and applications
- Jupyter notebooks
- When you need direct access to PIL Image objects

```python
from image_gen.core import ImageGenerator

gen = ImageGenerator(auto_preview=True)
image = gen.generate("a serene mountain landscape")
```

**Run it:**
```bash
python examples/basic_usage.py
```

### 2. Qwen Function Calling (`qwen_function_calling.py`)

Integration with Qwen LLM using function calling. Best for:
- LLM-driven image generation
- Chatbots that can create images
- Agentic workflows

**Features:**
- Function definitions for Qwen
- REST API client
- Example conversation flow

**Requirements:**
- REST API server running: `python -m image_gen.server`

**Run it:**
```bash
# Terminal 1: Start API server
python -m image_gen.server

# Terminal 2: Run example
python examples/qwen_function_calling.py
```

## Usage Patterns

### Pattern 1: Python API (Direct)

**Pros:**
- Fastest (no HTTP overhead)
- Direct access to image data
- Easiest for Python scripts

**Cons:**
- Model stays in memory
- Single process

**Example:**
```python
from image_gen.core import ImageGenerator

gen = ImageGenerator()
image = gen.generate("a cat on mars")
image.save("output.png")
```

### Pattern 2: REST API

**Pros:**
- Memory isolation (model in separate process)
- Language-agnostic
- Can be called from anywhere
- Perfect for LLM integration

**Cons:**
- HTTP overhead
- Requires running server

**Example:**
```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "prompt": "a cat on mars",
    "num_inference_steps": 30
})
result = response.json()
print(result["image_path"])
```

### Pattern 3: CLI

**Pros:**
- No coding required
- Great for quick tests
- Easy to script

**Cons:**
- Less flexible than API

**Example:**
```bash
python -m image_gen.cli generate "a cat on mars" --steps 30
```

## Integration with Qwen

For integrating with your Qwen LLM in MessageAnalyzer:

1. **Start the REST API server** (in ImageGeneratorLLM directory):
   ```bash
   cd /Users/bradmusick/Documents/Projects/ImageGeneratorLLM
   source venv/bin/activate
   python -m image_gen.server
   ```

2. **Add function definitions to Qwen** (see `qwen_function_calling.py`):
   ```python
   FUNCTION_DEFINITIONS = [
       {
           "name": "generate_image",
           "description": "Generate an image from text description",
           "parameters": { ... }
       }
   ]
   ```

3. **Handle function calls** in your Qwen application:
   ```python
   if function_call["name"] == "generate_image":
       result = requests.post("http://localhost:8000/generate",
                            json=function_call["arguments"])
   ```

## Common Use Cases

### Use Case 1: Batch Generation from File

```bash
# Create prompts.txt with one prompt per line
echo "a red apple" > prompts.txt
echo "a blue ocean" >> prompts.txt
echo "a green forest" >> prompts.txt

# Generate all
python -m image_gen.cli batch prompts.txt --steps 20
```

### Use Case 2: Reproducible Results

```python
# Same seed = identical image
gen = ImageGenerator()
image1 = gen.generate("a cat", seed=42)
image2 = gen.generate("a cat", seed=42)
# image1 and image2 are identical
```

### Use Case 3: Quality Comparison

```python
gen = ImageGenerator()

# Draft quality (fast)
draft = gen.generate("portrait", num_inference_steps=10)

# High quality (slow)
final = gen.generate("portrait", num_inference_steps=50)
```

## Tips

1. **Memory Management**: Unload models when done to free GPU memory:
   ```python
   gen.unload_models()
   ```

2. **Auto-Preview**: Enable automatic image opening:
   ```python
   gen = ImageGenerator(auto_preview=True)
   ```

3. **Quality vs Speed**:
   - 10 steps: Draft quality (~8s on M4 Pro)
   - 20 steps: Good quality (~15s)
   - 30 steps: High quality (~25s)
   - 40-50 steps: Maximum quality (~35-45s)

4. **Optimal Sizes**: Use multiples of 8 for dimensions:
   - Standard: 1024x1024
   - Portrait: 768x1024
   - Landscape: 1024x768
   - Square small: 768x768

## Troubleshooting

**"API not available"**
- Start the server: `python -m image_gen.server`
- Check it's running: `curl http://localhost:8000/health`

**"Out of memory"**
- Reduce image size: `--width 768 --height 768`
- Close other GPU applications
- Unload models when done

**"Slow generation"**
- First generation loads model (extra 10-15s)
- Reduce steps: `--steps 10`
- Use smaller size

## Next Steps

- See `docs/QUICK_START.md` for detailed documentation
- See `docs/CODEBASE_REFERENCE.md` for architecture details
- Run tests: `python tests/test_generator.py`
