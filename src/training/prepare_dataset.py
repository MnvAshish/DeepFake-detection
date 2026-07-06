"""
prepare_dataset.py - Convert raw video files into a face-cropped image dataset.

This script processes raw videos from data/raw/ (organized by real/fake) and:
  1. Extracts frames every N frames
  2. Detects faces using MediaPipe
  3. Saves cropped face images to data/train/, data/val/, data/test/
  4. Splits data according to config ratios

Expected input structure:
    data/raw/
    ├── real/    ← real video files (.mp4, .avi, .mov, etc.)
    └── fake/    ← fake/deepfake video files

Output structure:
    data/train/real/*.jpg
    data/train/fake/*.jpg
    data/val/real/*.jpg
    data/val/fake/*.jpg
    data/test/real/*.jpg
    data/test/fake/*.jpg

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --raw_dir data/raw --output_dir data
    python prepare_dataset.py --frame_interval 5 --max_frames 50
"""

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.frame_extractor import FaceExtractor
from src.utils.config_loader import load_config, ensure_directories
from src.utils.helpers import set_seed
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Supported video file formats
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v", ".webm"}


def find_videos(directory: str) -> List[Path]:
    """
    Recursively find all video files in a directory.

    Args:
        directory (str): Root directory to search.

    Returns:
        List[Path]: Sorted list of video file paths.
    """
    videos = []
    dir_path = Path(directory)
    if not dir_path.exists():
        logger.warning(f"Directory not found: {directory}")
        return []

    for ext in VIDEO_EXTENSIONS:
        videos.extend(dir_path.rglob(f"*{ext}"))
        videos.extend(dir_path.rglob(f"*{ext.upper()}"))

    return sorted(videos)


def split_videos(
    videos: List[Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Randomly split video list into train/val/test sets.

    Args:
        videos: List of video paths.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation (rest goes to test).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_videos, val_videos, test_videos).
    """
    random.seed(seed)
    shuffled = videos.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_vids = shuffled[:train_end]
    val_vids = shuffled[train_end:val_end]
    test_vids = shuffled[val_end:]

    return train_vids, val_vids, test_vids


def process_video_split(
    videos: List[Path],
    split: str,
    label: str,
    output_dir: str,
    extractor: FaceExtractor,
    target_size: Tuple[int, int] = (224, 224),
) -> Dict[str, int]:
    """
    Process a list of videos and extract face crops to a split directory.

    Args:
        videos: List of video paths to process.
        split: "train", "val", or "test".
        label: "real" or "fake".
        output_dir: Base output directory.
        extractor: FaceExtractor instance.
        target_size: Face crop dimensions.

    Returns:
        dict: Processing statistics.
    """
    save_dir = Path(output_dir) / split / label
    save_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    failed_videos = 0

    for idx, video_path in enumerate(videos):
        logger.info(
            f"Processing [{idx+1}/{len(videos)}] {split}/{label}: {video_path.name}"
        )

        try:
            saved_paths = extractor.extract_faces_to_disk(
                video_path=str(video_path),
                output_dir=str(Path(output_dir) / split),
                target_size=target_size,
                label=label,
            )
            total_saved += len(saved_paths)

            if len(saved_paths) == 0:
                logger.warning(f"No faces extracted from: {video_path.name}")

        except Exception as e:
            logger.error(f"Failed to process {video_path.name}: {e}")
            failed_videos += 1
            continue

    return {
        "videos_processed": len(videos) - failed_videos,
        "videos_failed": failed_videos,
        "faces_saved": total_saved,
    }


def prepare_dataset(
    raw_dir: str = "data/raw",
    output_dir: str = "data",
    frame_interval: int = 10,
    max_frames: int = 30,
    face_margin: float = 0.3,
    target_size: Tuple[int, int] = (224, 224),
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict:
    """
    Full dataset preparation pipeline.

    Args:
        raw_dir: Directory containing 'real/' and 'fake/' video subdirs.
        output_dir: Base output directory for train/val/test splits.
        frame_interval: Extract every N-th frame.
        max_frames: Maximum frames per video.
        face_margin: Padding around detected face (fraction of face size).
        target_size: Output face crop dimensions (width, height).
        train_ratio: Fraction of videos for training.
        val_ratio: Fraction of videos for validation.
        seed: Random seed.

    Returns:
        dict: Processing statistics per split and class.
    """
    logger.info("=" * 60)
    logger.info("  Dataset Preparation Pipeline")
    logger.info("=" * 60)

    # Initialize face extractor
    extractor = FaceExtractor(
        frame_interval=frame_interval,
        max_frames=max_frames,
        face_margin=face_margin,
    )

    all_stats = {}

    for label in ["real", "fake"]:
        label_dir = Path(raw_dir) / label
        logger.info(f"\nProcessing '{label}' videos from: {label_dir}")

        videos = find_videos(str(label_dir))
        if not videos:
            logger.warning(
                f"No videos found in {label_dir}. "
                f"Add your {label} video files there."
            )
            continue

        logger.info(f"Found {len(videos)} {label} videos")

        # Split into train/val/test
        train_vids, val_vids, test_vids = split_videos(
            videos, train_ratio, val_ratio, seed
        )
        logger.info(
            f"Split: train={len(train_vids)}, "
            f"val={len(val_vids)}, test={len(test_vids)}"
        )

        label_stats = {}

        for split, split_videos in [
            ("train", train_vids),
            ("val", val_vids),
            ("test", test_vids),
        ]:
            if not split_videos:
                logger.warning(f"No videos for {split}/{label} split.")
                continue

            stats = process_video_split(
                videos=split_videos,
                split=split,
                label=label,
                output_dir=output_dir,
                extractor=extractor,
                target_size=target_size,
            )
            label_stats[split] = stats
            logger.info(
                f"  {split:<5}: processed={stats['videos_processed']}, "
                f"failed={stats['videos_failed']}, "
                f"faces_saved={stats['faces_saved']}"
            )

        all_stats[label] = label_stats

    extractor.close()

    # Print final summary
    logger.info("\n" + "=" * 60)
    logger.info("  Dataset Preparation Complete")
    logger.info("=" * 60)
    for label, label_stats in all_stats.items():
        for split, stats in label_stats.items():
            logger.info(
                f"  {label}/{split}: {stats.get('faces_saved', 0)} images saved"
            )

    # Verify output directory structure
    _verify_dataset(output_dir)

    return all_stats


def _verify_dataset(output_dir: str) -> None:
    """Count and log images in each split/class directory."""
    logger.info("\nDataset verification:")
    total_images = 0
    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            dir_path = Path(output_dir) / split / label
            if dir_path.exists():
                count = len(list(dir_path.glob("*.jpg"))) + len(list(dir_path.glob("*.png")))
                logger.info(f"  {split}/{label}: {count} images")
                total_images += count
    logger.info(f"  TOTAL: {total_images} images")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare face image dataset from raw videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw_dir", default="data/raw", help="Raw video directory.")
    parser.add_argument("--output_dir", default="data", help="Output base directory.")
    parser.add_argument("--frame_interval", type=int, default=10)
    parser.add_argument("--max_frames", type=int, default=30)
    parser.add_argument("--face_margin", type=float, default=0.3)
    parser.add_argument("--target_size", type=int, default=224)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Config file (overrides CLI args if provided).",
    )

    args = parser.parse_args()

    # Load from config if available
    try:
        config = load_config(args.config)
        frame_interval = config["frame_extraction"]["frame_interval"]
        max_frames = config["frame_extraction"]["max_frames_per_video"]
        face_margin = config["frame_extraction"]["face_margin"]
        train_ratio = config["data"]["train_split"]
        val_ratio = config["data"]["val_split"]
        seed = config["training"]["seed"]
        logger.info("Using settings from config file.")
    except Exception:
        frame_interval = args.frame_interval
        max_frames = args.max_frames
        face_margin = args.face_margin
        train_ratio = args.train_ratio
        val_ratio = args.val_ratio
        seed = args.seed

    prepare_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        frame_interval=frame_interval,
        max_frames=max_frames,
        face_margin=face_margin,
        target_size=(args.target_size, args.target_size),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
