"""
helpers.py - General utility functions for the Deepfake Detection System.

Includes device detection, seed setting, timing, checkpoint saving/loading,
metric computation, and result visualization helpers.
"""

import os
import random
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Device Utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_device(device_preference: str = "auto") -> torch.device:
    """
    Determine the best available compute device.

    Args:
        device_preference (str): "auto", "cpu", "cuda", or "mps".

    Returns:
        torch.device: The selected device.
    """
    if device_preference == "cpu":
        device = torch.device("cpu")
    elif device_preference == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
    elif device_preference == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            logger.warning("MPS requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
    else:  # auto
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    return device


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for full reproducibility.

    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set to {seed}")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint Management
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    save_path: str,
    extra_info: Optional[Dict] = None,
) -> None:
    """
    Save a model checkpoint to disk.

    Args:
        model (nn.Module): The model to save.
        optimizer: The optimizer state.
        epoch (int): Current epoch number.
        val_acc (float): Validation accuracy at this checkpoint.
        save_path (str): Path to save the .pth file.
        extra_info (dict): Optional extra metadata to store.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra_info:
        checkpoint.update(extra_info)

    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved: {save_path} (epoch={epoch}, val_acc={val_acc:.4f})")


def load_checkpoint(
    model: nn.Module,
    load_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, int, float]:
    """
    Load a model checkpoint from disk.

    Args:
        model (nn.Module): The model architecture (weights will be loaded in).
        load_path (str): Path to the .pth checkpoint file.
        optimizer: Optional optimizer to restore state for.
        device: Device to map tensors to.

    Returns:
        Tuple[model, epoch, val_acc]
    """
    if not Path(load_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    if device is None:
        device = get_device()

    logger.info(f"Loading checkpoint from: {load_path}")
    checkpoint = torch.load(load_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    val_acc = checkpoint.get("val_acc", 0.0)

    logger.info(f"Loaded checkpoint: epoch={epoch}, val_acc={val_acc:.4f}")
    return model, epoch, val_acc


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities for class 1 (optional, for AUC).

    Returns:
        dict: Dictionary of metric names to values.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = 0.0

    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = "Model") -> None:
    """Pretty-print a metrics dictionary."""
    logger.info(f"\n{'='*50}")
    logger.info(f"  {model_name} Evaluation Metrics")
    logger.info(f"{'='*50}")
    for name, value in metrics.items():
        logger.info(f"  {name:<20}: {value:.4f}")
    logger.info(f"{'='*50}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    model_name: str,
    save_dir: str = "outputs",
) -> str:
    """
    Plot and save training/validation loss and accuracy curves.

    Args:
        train_losses: Per-epoch training losses.
        val_losses: Per-epoch validation losses.
        train_accs: Per-epoch training accuracies.
        val_accs: Per-epoch validation accuracies.
        model_name: Name of the model for the title/filename.
        save_dir: Directory to save the plot.

    Returns:
        str: Path to saved figure.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} - Training History", fontsize=14, fontweight="bold")

    # Loss plot
    axes[0].plot(epochs, train_losses, "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, val_losses, "r-o", label="Val Loss", markersize=4)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, train_accs, "b-o", label="Train Acc", markersize=4)
    axes[1].plot(epochs, val_accs, "r-o", label="Val Acc", markersize=4)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(save_dir) / f"{model_name.lower()}_training_history.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Training history plot saved: {save_path}")
    return str(save_path)


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    model_name: str,
    save_dir: str = "outputs",
) -> str:
    """
    Plot and save a confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: List of class name strings.
        model_name: Name of the model.
        save_dir: Directory to save the plot.

    Returns:
        str: Path to saved figure.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_title(f"{model_name} - Confusion Matrix")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    save_path = Path(save_dir) / f"{model_name.lower()}_confusion_matrix.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Confusion matrix saved: {save_path}")
    return str(save_path)


# ─────────────────────────────────────────────────────────────────────────────
# JSON Serialization
# ─────────────────────────────────────────────────────────────────────────────

def save_results_json(results: Dict, save_path: str) -> None:
    """Save results dictionary as JSON."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {save_path}")


def load_results_json(load_path: str) -> Dict:
    """Load results dictionary from JSON."""
    with open(load_path, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────────────────────────────────

class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self, label: str = "Block"):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        logger.info(f"[Timer] {self.label}: {self.elapsed:.3f}s")


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"
