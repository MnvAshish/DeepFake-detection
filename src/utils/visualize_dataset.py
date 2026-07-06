"""
visualize_dataset.py - Visualize dataset statistics and sample face crops.

Generates:
  1. Class distribution bar chart (train/val/test splits)
  2. Sample image grid (real vs fake face crops side by side)
  3. Image statistics (mean, std, brightness distribution)
  4. Summary JSON report

Usage:
    python src/utils/visualize_dataset.py
    python src/utils/visualize_dataset.py --data_dir data --output_dir outputs
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from src.utils.logger import get_logger

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Statistics
# ─────────────────────────────────────────────────────────────────────────────

def count_images(data_dir: str) -> Dict[str, Dict[str, int]]:
    """
    Count images per split per class.

    Args:
        data_dir (str): Root data directory containing train/val/test splits.

    Returns:
        dict: Nested dict {split: {class_name: count}}.
    """
    data_path = Path(data_dir)
    counts = {}

    for split in ["train", "val", "test"]:
        split_dir = data_path / split
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            continue

        counts[split] = {}
        for label in ["real", "fake"]:
            label_dir = split_dir / label
            if label_dir.exists():
                n = sum(
                    1 for f in label_dir.iterdir()
                    if f.suffix.lower() in IMAGE_EXTENSIONS
                )
                counts[split][label] = n
                logger.info(f"  {split}/{label}: {n} images")
            else:
                counts[split][label] = 0

    return counts


def compute_image_stats(
    image_paths: List[Path],
    max_samples: int = 500,
) -> Dict:
    """
    Compute pixel statistics for a sample of images.

    Args:
        image_paths: List of image file paths.
        max_samples: Maximum images to sample for statistics.

    Returns:
        dict: mean, std, min, max per channel, plus brightness histogram.
    """
    if len(image_paths) > max_samples:
        indices = np.random.choice(len(image_paths), max_samples, replace=False)
        sampled = [image_paths[i] for i in indices]
    else:
        sampled = image_paths

    all_means = []
    all_stds = []
    all_brightness = []

    for path in sampled:
        try:
            img = cv2.imread(str(path))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            all_means.append(img_rgb.mean(axis=(0, 1)))   # (3,)
            all_stds.append(img_rgb.std(axis=(0, 1)))     # (3,)
            # Brightness = mean of grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            all_brightness.append(float(gray.mean()))

        except Exception as e:
            logger.debug(f"Could not process {path}: {e}")
            continue

    if not all_means:
        return {}

    means_arr = np.array(all_means)   # (N, 3)
    stds_arr = np.array(all_stds)     # (N, 3)

    return {
        "channel_mean": means_arr.mean(axis=0).tolist(),       # [R, G, B]
        "channel_std": stds_arr.mean(axis=0).tolist(),
        "overall_mean": float(means_arr.mean()),
        "overall_std": float(stds_arr.mean()),
        "brightness_mean": float(np.mean(all_brightness)),
        "brightness_std": float(np.std(all_brightness)),
        "brightness_histogram": np.histogram(
            all_brightness, bins=20, range=(0, 1)
        )[0].tolist(),
        "samples_analyzed": len(all_means),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting Functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(
    counts: Dict[str, Dict[str, int]],
    save_dir: str = "outputs",
) -> str:
    """
    Bar chart of class distribution across train/val/test.

    Args:
        counts: Nested dict from count_images().
        save_dir: Output directory.

    Returns:
        str: Path to saved figure.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    splits = list(counts.keys())
    real_counts = [counts[s].get("real", 0) for s in splits]
    fake_counts = [counts[s].get("fake", 0) for s in splits]
    total_counts = [r + f for r, f in zip(real_counts, fake_counts)]

    x = np.arange(len(splits))
    width = 0.3

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dataset Distribution", fontsize=14, fontweight="bold")

    # ── Left: Stacked bar chart ──────────────────────────────────────────────
    ax = axes[0]
    bars_real = ax.bar(x, real_counts, width * 2, label="Real", color="#00C853", alpha=0.85)
    bars_fake = ax.bar(x, fake_counts, width * 2, bottom=real_counts,
                       label="Fake", color="#FF4B4B", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in splits], fontsize=11)
    ax.set_ylabel("Number of Images")
    ax.set_title("Images per Split (Real vs Fake)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Add count labels
    for bar_r, bar_f, r, f in zip(bars_real, bars_fake, real_counts, fake_counts):
        ax.text(bar_r.get_x() + bar_r.get_width() / 2, r / 2,
                str(r), ha="center", va="center", fontweight="bold", color="white", fontsize=9)
        ax.text(bar_f.get_x() + bar_f.get_width() / 2, r + f / 2,
                str(f), ha="center", va="center", fontweight="bold", color="white", fontsize=9)

    # ── Right: Class balance per split ──────────────────────────────────────
    ax2 = axes[1]
    for i, split in enumerate(splits):
        r = counts[split].get("real", 0)
        f = counts[split].get("fake", 0)
        total = r + f
        if total == 0:
            continue
        real_pct = r / total * 100
        fake_pct = f / total * 100

        ax2.barh(i - 0.15, real_pct, 0.3, color="#00C853", alpha=0.85, label="Real" if i == 0 else "")
        ax2.barh(i + 0.15, fake_pct, 0.3, color="#FF4B4B", alpha=0.85, label="Fake" if i == 0 else "")
        ax2.text(real_pct + 0.5, i - 0.15, f"{real_pct:.1f}%", va="center", fontsize=9)
        ax2.text(fake_pct + 0.5, i + 0.15, f"{fake_pct:.1f}%", va="center", fontsize=9)

    ax2.set_yticks(range(len(splits)))
    ax2.set_yticklabels([s.upper() for s in splits])
    ax2.set_xlabel("Percentage (%)")
    ax2.set_title("Class Balance per Split")
    ax2.set_xlim(0, 115)
    ax2.legend()
    ax2.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    save_path = Path(save_dir) / "dataset_distribution.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Distribution plot saved: {save_path}")
    return str(save_path)


def plot_sample_images(
    data_dir: str,
    split: str = "train",
    n_samples: int = 5,
    save_dir: str = "outputs",
    image_size: int = 112,
) -> str:
    """
    Create a side-by-side grid of real vs fake sample face crops.

    Args:
        data_dir: Root data directory.
        split: Which split to sample from.
        n_samples: Number of samples per class.
        save_dir: Output directory.
        image_size: Display size for each thumbnail.

    Returns:
        str: Path to saved figure.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    def load_samples(label: str) -> List[np.ndarray]:
        """Load n_samples random images for a given class."""
        label_dir = Path(data_dir) / split / label
        if not label_dir.exists():
            return []

        all_imgs = [
            f for f in label_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not all_imgs:
            return []

        selected = np.random.choice(all_imgs, min(n_samples, len(all_imgs)), replace=False)
        images = []
        for path in selected:
            try:
                img = cv2.imread(str(path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (image_size, image_size))
                images.append(img_resized)
            except Exception:
                continue
        return images

    real_imgs = load_samples("real")
    fake_imgs = load_samples("fake")

    if not real_imgs and not fake_imgs:
        logger.warning(f"No images found in {data_dir}/{split}/")
        return ""

    # Ensure both lists have same length
    n = min(len(real_imgs), len(fake_imgs), n_samples)
    real_imgs = real_imgs[:n]
    fake_imgs = fake_imgs[:n]

    if n == 0:
        return ""

    fig = plt.figure(figsize=(n * 2.5, 6))
    gs = gridspec.GridSpec(2, n, figure=fig, hspace=0.4, wspace=0.1)

    fig.suptitle(
        f"Sample Face Crops — {split.upper()} split  "
        f"(Top: REAL | Bottom: FAKE)",
        fontsize=12, fontweight="bold", y=1.02
    )

    for col in range(n):
        # Real row
        ax_real = fig.add_subplot(gs[0, col])
        ax_real.imshow(real_imgs[col])
        ax_real.axis("off")
        if col == 0:
            ax_real.set_ylabel("REAL", fontsize=10, color="#00C853",
                               fontweight="bold", labelpad=5)
        ax_real.set_title(f"#{col+1}", fontsize=8, color="#888")

        # Fake row
        ax_fake = fig.add_subplot(gs[1, col])
        ax_fake.imshow(fake_imgs[col])
        ax_fake.axis("off")
        if col == 0:
            ax_fake.set_ylabel("FAKE", fontsize=10, color="#FF4B4B",
                               fontweight="bold", labelpad=5)

    # Add colored border effect using insets
    for col in range(n):
        # Green border for real
        ax_real = fig.axes[col * 2]  # Note: axes are indexed differently
        for spine in fig.add_subplot(gs[0, col]).spines.values():
            spine.set_edgecolor("#00C853")
            spine.set_linewidth(2)

    plt.tight_layout()
    save_path = Path(save_dir) / f"sample_images_{split}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Sample images plot saved: {save_path}")
    return str(save_path)


def plot_brightness_distribution(
    stats_real: Dict,
    stats_fake: Dict,
    save_dir: str = "outputs",
) -> str:
    """
    Histogram comparing brightness distributions of real vs fake images.

    Args:
        stats_real: Output from compute_image_stats() for real images.
        stats_fake: Output from compute_image_stats() for fake images.
        save_dir: Output directory.

    Returns:
        str: Path to saved figure.
    """
    if not stats_real or not stats_fake:
        return ""

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Image Statistics: Real vs Fake", fontsize=14, fontweight="bold")

    # ── Brightness histogram ─────────────────────────────────────────────────
    ax = axes[0]
    bins = np.linspace(0, 1, 21)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = bins[1] - bins[0]

    real_hist = np.array(stats_real.get("brightness_histogram", [0] * 20))
    fake_hist = np.array(stats_fake.get("brightness_histogram", [0] * 20))

    # Normalize to fractions
    if real_hist.sum() > 0:
        real_hist = real_hist / real_hist.sum()
    if fake_hist.sum() > 0:
        fake_hist = fake_hist / fake_hist.sum()

    ax.bar(bin_centers - width / 4, real_hist, width / 2,
           label=f"Real (μ={stats_real['brightness_mean']:.3f})",
           color="#00C853", alpha=0.75)
    ax.bar(bin_centers + width / 4, fake_hist, width / 2,
           label=f"Fake (μ={stats_fake['brightness_mean']:.3f})",
           color="#FF4B4B", alpha=0.75)

    ax.set_xlabel("Brightness (normalized)")
    ax.set_ylabel("Fraction of Images")
    ax.set_title("Brightness Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Channel mean comparison ──────────────────────────────────────────────
    ax2 = axes[1]
    channels = ["Red", "Green", "Blue"]
    x = np.arange(3)
    width2 = 0.3

    real_means = stats_real.get("channel_mean", [0, 0, 0])
    fake_means = stats_fake.get("channel_mean", [0, 0, 0])
    real_stds = stats_real.get("channel_std", [0, 0, 0])
    fake_stds = stats_fake.get("channel_std", [0, 0, 0])

    channel_colors = ["#FF6B6B", "#51CF66", "#4DABF7"]

    for i, (ch, color) in enumerate(zip(channels, channel_colors)):
        ax2.bar(i - 0.15, real_means[i], 0.3, yerr=real_stds[i],
                color=color, alpha=0.85, capsize=4,
                label="Real" if i == 0 else "")
        ax2.bar(i + 0.15, fake_means[i], 0.3, yerr=fake_stds[i],
                color=color, alpha=0.45, capsize=4, hatch="///",
                label="Fake" if i == 0 else "")

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#888", label="Real (solid)"),
        Patch(facecolor="#888", alpha=0.4, hatch="///", label="Fake (hatched)"),
    ]
    ax2.legend(handles=legend_elements)
    ax2.set_xticks(x)
    ax2.set_xticklabels(channels)
    ax2.set_ylabel("Mean Pixel Value (0-1)")
    ax2.set_title("Channel Mean Comparison")
    ax2.set_ylim(0, 0.8)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = Path(save_dir) / "image_statistics.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Image statistics plot saved: {save_path}")
    return str(save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_visualization(
    data_dir: str = "data",
    output_dir: str = "outputs",
    n_samples: int = 5,
) -> Dict:
    """
    Generate all dataset visualization plots and stats.

    Args:
        data_dir: Root data directory.
        output_dir: Output directory for plots.
        n_samples: Sample images per class per visualization.

    Returns:
        dict: Summary statistics.
    """
    logger.info("=" * 60)
    logger.info("  Dataset Visualization")
    logger.info("=" * 60)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # ── Count images ──────────────────────────────────────────────────────────
    logger.info("\nCounting images...")
    counts = count_images(data_dir)

    if not counts:
        logger.error(
            f"No images found in {data_dir}. "
            "Run prepare_dataset.py first."
        )
        return {}

    # ── Distribution plot ─────────────────────────────────────────────────────
    logger.info("\nGenerating distribution plot...")
    dist_path = plot_class_distribution(counts, save_dir=output_dir)

    # ── Sample images ─────────────────────────────────────────────────────────
    for split in ["train", "val", "test"]:
        if split in counts:
            logger.info(f"\nGenerating sample images for {split}...")
            plot_sample_images(
                data_dir=data_dir,
                split=split,
                n_samples=n_samples,
                save_dir=output_dir,
            )

    # ── Image statistics ──────────────────────────────────────────────────────
    logger.info("\nComputing image statistics (this may take a moment)...")

    train_real_dir = Path(data_dir) / "train" / "real"
    train_fake_dir = Path(data_dir) / "train" / "fake"

    stats_real = {}
    stats_fake = {}

    if train_real_dir.exists():
        real_paths = [
            f for f in train_real_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        stats_real = compute_image_stats(real_paths)
        logger.info(f"  Real stats: mean={stats_real.get('overall_mean', 0):.3f}, "
                    f"brightness={stats_real.get('brightness_mean', 0):.3f}")

    if train_fake_dir.exists():
        fake_paths = [
            f for f in train_fake_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        stats_fake = compute_image_stats(fake_paths)
        logger.info(f"  Fake stats: mean={stats_fake.get('overall_mean', 0):.3f}, "
                    f"brightness={stats_fake.get('brightness_mean', 0):.3f}")

    if stats_real and stats_fake:
        plot_brightness_distribution(stats_real, stats_fake, save_dir=output_dir)

    # ── Save summary JSON ─────────────────────────────────────────────────────
    summary = {
        "counts": counts,
        "total_images": sum(
            sum(split_counts.values())
            for split_counts in counts.values()
        ),
        "stats_real": stats_real,
        "stats_fake": stats_fake,
    }

    summary_path = Path(output_dir) / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nDataset summary saved to: {summary_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    total = summary["total_images"]
    logger.info("\n" + "=" * 50)
    logger.info("  DATASET SUMMARY")
    logger.info("=" * 50)
    for split, split_counts in counts.items():
        r = split_counts.get("real", 0)
        f = split_counts.get("fake", 0)
        logger.info(f"  {split:<8} | real={r:>6}, fake={f:>6}, total={r+f:>6}")
    logger.info(f"  {'TOTAL':<8} | {total:>20}")
    logger.info("=" * 50)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize dataset statistics and sample images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", default="data", help="Root data directory.")
    parser.add_argument("--output_dir", default="outputs", help="Output directory.")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of sample images per class.")

    args = parser.parse_args()
    run_visualization(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
    )
