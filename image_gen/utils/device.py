"""
Module: image_gen.utils.device
Purpose: Device detection and management for PyTorch with Apple Silicon MPS support
Dependencies: torch
Author: Generated for ImageGeneratorLLM
Reference: See docs/architecture/DEVICE_MANAGEMENT.md
"""

import torch
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def detect_device(prefer_device: Optional[str] = None) -> str:
    """
    Detect the best available compute device for PyTorch.

    Priority order:
    1. User-specified device (if provided and available)
    2. MPS (Metal Performance Shaders) for Apple Silicon
    3. CUDA for NVIDIA GPUs
    4. CPU fallback

    Args:
        prefer_device: Optional device preference ("mps", "cuda", "cpu")
                      If None, auto-detects the best available device.

    Returns:
        Device string: "mps", "cuda", or "cpu"

    Example:
        >>> device = detect_device()
        >>> print(f"Using device: {device}")
        Using device: mps

        >>> device = detect_device(prefer_device="cpu")
        >>> print(f"Using device: {device}")
        Using device: cpu
    """
    # If user specified a device, validate and use it
    if prefer_device:
        prefer_device = prefer_device.lower()

        if prefer_device == "mps":
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                logger.info("Using user-specified MPS device")
                return "mps"
            else:
                logger.warning("MPS requested but not available, falling back to auto-detection")

        elif prefer_device == "cuda":
            if torch.cuda.is_available():
                logger.info("Using user-specified CUDA device")
                return "cuda"
            else:
                logger.warning("CUDA requested but not available, falling back to auto-detection")

        elif prefer_device == "cpu":
            logger.info("Using user-specified CPU device")
            return "cpu"

        else:
            logger.warning(f"Unknown device '{prefer_device}', falling back to auto-detection")

    # Auto-detection
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        logger.info("Auto-detected MPS (Apple Silicon) device")
        return "mps"

    if torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(0)
        logger.info(f"Auto-detected CUDA device: {cuda_name}")
        return "cuda"

    logger.info("No GPU detected, using CPU")
    return "cpu"


def get_device(config_device: Optional[str] = None) -> torch.device:
    """
    Get PyTorch device object for model loading.

    Args:
        config_device: Optional device from configuration

    Returns:
        PyTorch device object ready for model.to(device)

    Example:
        >>> device = get_device()
        >>> model = model.to(device)
    """
    device_str = detect_device(config_device)
    return torch.device(device_str)


def get_device_info() -> dict:
    """
    Get detailed information about available compute devices.

    Returns:
        Dictionary with device availability and specifications

    Example:
        >>> info = get_device_info()
        >>> print(f"MPS available: {info['mps_available']}")
        >>> print(f"Total memory: {info['total_memory_gb']} GB")
    """
    info = {
        "mps_available": torch.backends.mps.is_available() and torch.backends.mps.is_built(),
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": torch.get_num_threads(),
        "current_device": detect_device(),
    }

    # Add CUDA-specific info if available
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_device_count"] = torch.cuda.device_count()
        # Memory in GB
        info["total_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    # For MPS, we can't query memory directly but can note the system
    elif info["mps_available"]:
        import platform
        info["platform"] = platform.platform()
        info["processor"] = platform.processor()

    return info


def print_device_info() -> None:
    """
    Print formatted device information to console.

    Example:
        >>> print_device_info()
        Device Information:
        ------------------
        Current Device: mps
        MPS Available: True
        CUDA Available: False
        CPU Threads: 10
        Platform: macOS-14.0-arm64-arm-64bit
        Processor: arm
    """
    info = get_device_info()

    print("\nDevice Information:")
    print("------------------")
    print(f"Current Device: {info['current_device']}")
    print(f"MPS Available: {info['mps_available']}")
    print(f"CUDA Available: {info['cuda_available']}")
    print(f"CPU Threads: {info['cpu_count']}")

    if "cuda_device_name" in info:
        print(f"CUDA Device: {info['cuda_device_name']}")
        print(f"CUDA Memory: {info['total_memory_gb']:.1f} GB")

    if "platform" in info:
        print(f"Platform: {info['platform']}")
        print(f"Processor: {info['processor']}")

    print()


if __name__ == "__main__":
    # Test device detection
    print_device_info()
