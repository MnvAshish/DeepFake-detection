"""
config_loader.py - Loads and validates the YAML configuration file.

Provides a simple interface to access config values throughout the project.
Uses a singleton pattern to avoid repeatedly reading from disk.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Singleton config store
# ─────────────────────────────────────────────────────────────────────────────
_config: Optional[Dict] = None
_config_path: Optional[Path] = None


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file. Caches result after first load.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    global _config, _config_path

    path = Path(config_path)

    # Return cached config if already loaded from same path
    if _config is not None and _config_path == path:
        return _config

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{path}'. "
            "Ensure you are running from the project root directory."
        )

    logger.info(f"Loading configuration from: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Validate required top-level keys
    required_keys = ["paths", "data", "training", "models", "ensemble", "inference"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: '{key}'")

    _config = config
    _config_path = path
    logger.info("Configuration loaded successfully.")
    return _config


def get_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Get the configuration (load if not already loaded).

    Args:
        config_path (str): Path to config file (only used on first call).

    Returns:
        dict: Configuration dictionary.
    """
    return load_config(config_path)


def get_value(key_path: str, config_path: str = "config/config.yaml") -> Any:
    """
    Get a nested config value using dot-notation.

    Example:
        get_value("training.batch_size") → 32
        get_value("paths.model_save_dir") → "models/saved"

    Args:
        key_path (str): Dot-separated path to config key.
        config_path (str): Path to config YAML.

    Returns:
        Any: The config value.

    Raises:
        KeyError: If the key path doesn't exist in config.
    """
    config = load_config(config_path)
    keys = key_path.split(".")
    value = config

    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(
                f"Config key '{key_path}' not found. "
                f"Failed at segment '{key}'."
            )
        value = value[key]

    return value


def ensure_directories(config_path: str = "config/config.yaml") -> None:
    """
    Create all required directories defined in config paths section.

    Args:
        config_path (str): Path to config YAML.
    """
    config = load_config(config_path)
    paths = config.get("paths", {})

    dirs_to_create = [
        paths.get("model_save_dir", "models/saved"),
        paths.get("log_dir", "logs"),
        paths.get("output_dir", "outputs"),
        paths.get("processed_frames_dir", "data/processed"),
    ]

    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {dir_path}")

    logger.info("All required directories verified/created.")
