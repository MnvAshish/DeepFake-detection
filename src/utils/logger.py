"""
logger.py - Centralized logging setup for the Deepfake Detection System.

Uses Python's built-in logging module with optional file output.
All modules import `get_logger(__name__)` to get a named logger.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def get_logger(name: str, log_level: str = "INFO", log_to_file: bool = True) -> logging.Logger:
    """
    Create and return a configured logger instance.

    Args:
        name (str): Logger name, typically __name__ of calling module.
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR).
        log_to_file (bool): Whether to also write logs to a file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    # Set logging level
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Define log format
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler - always enabled
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # File handler - optional
    if log_to_file:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"deepfake_detector_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


# Module-level default logger
logger = get_logger("deepfake_detector")
