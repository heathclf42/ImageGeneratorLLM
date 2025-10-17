"""
Module: image_gen.server
Purpose: FastAPI REST server for image generation
Dependencies: fastapi, uvicorn, pydantic
Author: Generated for ImageGeneratorLLM
Reference: See docs/API_REFERENCE.md

This module provides a REST API server for generating images, designed to be
called by LLMs like Qwen for image generation tasks.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from pathlib import Path
import logging
import base64
from io import BytesIO

from image_gen.core import ImageGenerator
from image_gen.config import get_config

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ImageGeneratorLLM API",
    description="REST API for generating images from text prompts using SDXL",
    version="0.1.0"
)

# Global generator instance (lazy-loaded)
_generator: Optional[ImageGenerator] = None


def get_generator() -> ImageGenerator:
    """Get or create the global generator instance."""
    global _generator
    if _generator is None:
        logger.info("Initializing ImageGenerator...")
        _generator = ImageGenerator(auto_preview=False)
    return _generator


# Request/Response models
class GenerateRequest(BaseModel):
    """Request model for single image generation."""
    prompt: str = Field(..., description="Text description of the image to generate")
    width: Optional[int] = Field(None, description="Output width in pixels (default: 1024)")
    height: Optional[int] = Field(None, description="Output height in pixels (default: 1024)")
    num_inference_steps: Optional[int] = Field(None, description="Number of denoising steps (default: 30)")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    return_base64: bool = Field(False, description="Return image as base64 instead of file path")

    class Config:
        schema_extra = {
            "example": {
                "prompt": "a serene mountain landscape at sunset",
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 30,
                "seed": 42
            }
        }


class BatchGenerateRequest(BaseModel):
    """Request model for batch image generation."""
    prompts: List[str] = Field(..., description="List of text descriptions")
    width: Optional[int] = Field(None, description="Output width in pixels (default: 1024)")
    height: Optional[int] = Field(None, description="Output height in pixels (default: 1024)")
    num_inference_steps: Optional[int] = Field(None, description="Number of denoising steps (default: 30)")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

    class Config:
        schema_extra = {
            "example": {
                "prompts": [
                    "a red sports car",
                    "a blue ocean wave",
                    "a green forest path"
                ],
                "num_inference_steps": 20
            }
        }


class GenerateResponse(BaseModel):
    """Response model for image generation."""
    success: bool
    message: str
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None


class BatchGenerateResponse(BaseModel):
    """Response model for batch generation."""
    success: bool
    message: str
    image_paths: List[str] = []
    count: int = 0


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    device: str
    model_id: str


# API Routes
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "ImageGeneratorLLM API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.

    Returns server status and model information.
    """
    try:
        from image_gen.utils.device import detect_device

        config = get_config()
        gen = get_generator()

        return HealthResponse(
            status="healthy",
            model_loaded=gen._flux_generator is not None,
            device=detect_device(),
            model_id=config.get_model_id("flux")
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            device="unknown",
            model_id="unknown"
        )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate a single image from a text prompt.

    Args:
        request: Generation parameters

    Returns:
        Generated image path or base64 data

    Example:
        POST /generate
        {
            "prompt": "a serene mountain landscape",
            "num_inference_steps": 30,
            "seed": 42
        }
    """
    try:
        logger.info(f"Generating image: {request.prompt[:50]}...")

        gen = get_generator()

        # Generate image
        image = gen.generate(
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            seed=request.seed,
            auto_save=True
        )

        # Get the saved path
        config = get_config()
        output_dir = Path(config.output["directory"])

        # Find the most recently created file
        files = sorted(output_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            image_path = files[0]
        else:
            raise RuntimeError("Image was generated but file not found")

        response_data = {
            "success": True,
            "message": "Image generated successfully",
            "image_path": str(image_path),
            "width": image.size[0],
            "height": image.size[1],
            "seed": request.seed
        }

        # Optionally return base64
        if request.return_base64:
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            response_data["image_base64"] = image_base64

        return GenerateResponse(**response_data)

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/batch", response_model=BatchGenerateResponse)
async def generate_batch(request: BatchGenerateRequest):
    """
    Generate multiple images from a list of prompts.

    Args:
        request: Batch generation parameters

    Returns:
        List of generated image paths

    Example:
        POST /generate/batch
        {
            "prompts": [
                "a red apple",
                "a blue ocean",
                "a green forest"
            ],
            "num_inference_steps": 20
        }
    """
    try:
        logger.info(f"Batch generating {len(request.prompts)} images...")

        gen = get_generator()

        # Track file count before generation
        config = get_config()
        output_dir = Path(config.output["directory"])
        existing_files = set(output_dir.glob("*.png"))

        # Generate batch
        images = gen.generate_batch(
            prompts=request.prompts,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            seed=request.seed,
            auto_save=True
        )

        # Find newly created files
        all_files = set(output_dir.glob("*.png"))
        new_files = sorted(all_files - existing_files, key=lambda p: p.stat().st_mtime)

        image_paths = [str(f) for f in new_files]

        return BatchGenerateResponse(
            success=True,
            message=f"Generated {len(images)} images successfully",
            image_paths=image_paths,
            count=len(images)
        )

    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/image/{filename}")
async def get_image(filename: str):
    """
    Retrieve a generated image by filename.

    Args:
        filename: Image filename

    Returns:
        Image file

    Example:
        GET /image/my_image.png
    """
    try:
        config = get_config()
        output_dir = Path(config.output["directory"])
        image_path = output_dir / filename

        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")

        return FileResponse(image_path, media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload")
async def unload_models():
    """
    Unload models from memory.

    Useful for freeing GPU memory when done generating images.

    Returns:
        Success message
    """
    try:
        global _generator
        if _generator is not None:
            _generator.unload_models()
            _generator = None
            return {"success": True, "message": "Models unloaded successfully"}
        else:
            return {"success": True, "message": "No models were loaded"}

    except Exception as e:
        logger.error(f"Failed to unload models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False
):
    """
    Run the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to listen on
        reload: Enable auto-reload for development
    """
    import uvicorn

    logger.info(f"Starting ImageGeneratorLLM API server on {host}:{port}")

    uvicorn.run(
        "image_gen.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    # Run server in development mode
    import sys

    port = 8000
    if len(sys.argv) > 1:
        port = int(sys.argv[1])

    run_server(port=port, reload=True)
