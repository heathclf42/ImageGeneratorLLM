"""
Module: image_gen.config
Purpose: Configuration management for ImageGeneratorLLM
Dependencies: pyyaml, pathlib
Author: Generated for ImageGeneratorLLM
Reference: See docs/architecture/CONFIGURATION.md
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import os

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Directory paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_CACHE_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"
DOCS_DIR = PROJECT_ROOT / "docs"

# Ensure directories exist
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class Config:
    """
    Configuration manager for ImageGeneratorLLM.

    Handles all configuration including model settings, API configuration,
    memory management, and output formatting.

    Attributes:
        models (Dict[str, Any]): Model-specific configuration
        api (Dict[str, Any]): API server configuration
        output (Dict[str, Any]): Output file configuration
        device (str): Compute device override (None for auto-detection)

    Example:
        >>> config = Config()
        >>> config.models["flux"]["keep_loaded"]
        True
        >>> config.get_model_id("flux")
        'black-forest-labs/FLUX.1-schnell'
    """

    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize configuration with default values and optional overrides.

        Args:
            config_file: Optional path to YAML config file for overrides
        """
        # Default configuration
        self.models: Dict[str, Any] = {
            "flux": {
                "model_id": "black-forest-labs/FLUX.1-schnell",
                "keep_loaded": True,  # Primary model, always in memory
                "idle_timeout": None,  # Never unload
                "default_steps": 4,
                "default_size": (1024, 1024),
                "guidance_scale": 0.0,  # Schnell doesn't use guidance
            },
            "sdxl": {
                "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "keep_loaded": False,  # Load on-demand
                "idle_timeout": 300,  # Unload after 5 min idle
                "default_steps": 30,
                "default_size": (1024, 1024),
                "guidance_scale": 7.5,
            },
            "controlnet": {
                "model_id": "xinsir/controlnet-union-sdxl-1.0",
                "keep_loaded": False,
                "idle_timeout": 180,  # 3 min idle
                "default_steps": 30,
                "guidance_scale": 7.5,
            },
            "brushnet": {
                "model_id": "TencentARC/BrushNet",
                "keep_loaded": False,
                "idle_timeout": 180,
                "default_steps": 30,
                "guidance_scale": 7.5,
            }
        }

        self.api: Dict[str, Any] = {
            "host": "0.0.0.0",
            "port": 8000,
            "reload": False,  # Set True for development
            "log_level": "info",
        }

        self.output: Dict[str, Any] = {
            "directory": str(OUTPUTS_DIR),
            "auto_open": True,  # Automatically open in Preview.app
            "naming_pattern": "{timestamp}_{model}_{prompt_short}",
            "max_filename_length": 50,
            "image_format": "PNG",
            "jpeg_quality": 95,  # If using JPEG
        }

        # Device configuration (None = auto-detect)
        self.device: Optional[str] = None

        # Cache directory for models
        self.cache_dir: str = str(MODELS_CACHE_DIR)

        # Load overrides from file if provided
        if config_file and config_file.exists():
            self._load_overrides(config_file)

    def _load_overrides(self, config_file: Path) -> None:
        """
        Load configuration overrides from YAML file.

        Args:
            config_file: Path to YAML configuration file
        """
        with open(config_file, 'r') as f:
            overrides = yaml.safe_load(f)

        if overrides:
            # Deep merge overrides into existing config
            for key, value in overrides.items():
                if hasattr(self, key) and isinstance(getattr(self, key), dict):
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, value)

    def get_model_id(self, model_name: str) -> str:
        """
        Get Hugging Face model ID for a given model name.

        Args:
            model_name: Model name (flux, sdxl, controlnet, brushnet)

        Returns:
            Hugging Face model identifier string

        Raises:
            KeyError: If model name is not recognized
        """
        if model_name not in self.models:
            raise KeyError(f"Unknown model: {model_name}. Available: {list(self.models.keys())}")
        return self.models[model_name]["model_id"]

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get full configuration for a specific model.

        Args:
            model_name: Model name (flux, sdxl, controlnet, brushnet)

        Returns:
            Dictionary of model configuration settings

        Raises:
            KeyError: If model name is not recognized
        """
        if model_name not in self.models:
            raise KeyError(f"Unknown model: {model_name}. Available: {list(self.models.keys())}")
        return self.models[model_name].copy()

    def should_keep_loaded(self, model_name: str) -> bool:
        """
        Check if a model should be kept in memory.

        Args:
            model_name: Model name to check

        Returns:
            True if model should stay loaded, False otherwise
        """
        return self.models.get(model_name, {}).get("keep_loaded", False)

    def get_idle_timeout(self, model_name: str) -> Optional[int]:
        """
        Get idle timeout in seconds for a model.

        Args:
            model_name: Model name to check

        Returns:
            Timeout in seconds, or None for no timeout
        """
        return self.models.get(model_name, {}).get("idle_timeout")


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """
    Get global configuration instance (singleton pattern).

    Returns:
        Shared Config instance

    Example:
        >>> from image_gen.config import get_config
        >>> config = get_config()
        >>> print(config.models["flux"]["model_id"])
    """
    global _config_instance
    if _config_instance is None:
        # Check for local config override
        local_config = CONFIG_DIR / "local.yaml"
        _config_instance = Config(local_config if local_config.exists() else None)
    return _config_instance
