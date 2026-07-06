"""
real_dataset_prep.py — Full real-world dataset preprocessing pipeline.

Converts raw real/fake videos or images from data/raw/ into a
train/val/test face-image dataset ready for model training.

Handles both:
  - VIDEO inputs  → frame extraction → face detection → crop → save
  - IMAGE inputs  → face detection   → crop           → save

Also handles:
  - Class balancing (cap majority class)
  - Train/val/test splitting
  - Duplicate detection (hash-based)
  - Quality filtering (blur, tiny images)

Usage:
    python src/data/real_dataset_prep.py
    python src/data/real_dataset_prep.py --raw_dir data/raw --max_per_class 5000
    python src/data/real_dataset_prep.py --skip_face_detection  # use full image
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.frame_extractor import FaceExtractor, extract_face_from_image
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# Quality Checks
# ─────────────────────────────────────────────────────────────────────────────

def is_blurry(image: np.ndarray, threshold: float = 50.0) -> bool:
    """
    Check if an image is too blurry using Laplacian variance.
    Low variance = blurry. Returns True if blurry.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < threshold


def image_hash(image: np.ndarray) -> str:
    """Perceptual hash string for duplicate detection."""
    small = cv2.resize(image, (16, 16))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY) if small.ndim == 3 else small
    return hashlib.md5(gray.tobytes()).hexdigest()


def is_too_dark_or_bright(image: np.ndarray,
                           min_mean: float = 15.0,
                           max_mean: float = 240.0) -> bool:
    """Check if image is too dark or overexposed."""
    mean_val = float(image.mean())
    return mean_val < min_mean or mean_val > max_mean


# ─────────────────────────────────────────────────────────────────────────────
# File Scanning
# ─────────────────────────────────────────────────────────────────────────────

def scan_raw_directory(raw_dir: str) -> Tuple[List[Path], List[Path], List[Path], List[Path]]:
    """
    Scan raw directory and separate into video/image files per class.

    Returns:
        (real_videos, fake_videos, real_images, fake_images)
    """
    raw_path = Path(raw_dir)
    real_videos, fake_videos = [], []
    real_images, fake_images = [], []

    for label, video_list, image_list in [
        ("real", real_videos, real_images),
        ("fake", fake_videos, fake_images),
    ]:
        label_dir = raw_path / label
        if not label_dir.exists():
            logger.warning(f"Directory not found: {label_dir}")
            continue
        for f in sorted(label_dir.rglob("*")):
            if f.suffix.lower() in VIDEO_EXTENSIONS:
                video_list.append(f)
            elif f.suffix.lower() in IMAGE_EXTENSIONS:
                image_list.append(f)

    logger.info(
        f"Raw data found: real_videos={len(real_videos)}, fake_videos={len(fake_videos)}, "
        f"real_images={len(real_images)}, fake_images={len(fake_images)}"
    )
    return real_videos, fake_videos, real_images, fake_images


# ─────────────────────────────────────────────────────────────────────────────
# Processing
# ─────────────────────────────────────────────────────────────────────────────

def process_videos(
    video_paths: List[Path],
    label: str,
    output_dir: Path,
    extractor: FaceExtractor,
    target_size: Tuple[int, int],
    skip_face_detection: bool,
    max_frames_per_video: int,
    frame_interval: int,
    seen_hashes: Set[str],
    blur_threshold: float = 50.0,
) -> int:
    """Process a list of video files, extract faces, and save to output_dir/label/."""
    save_dir = output_dir / label
    save_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for idx, video_path in enumerate(video_paths):
        logger.info(f"  Video [{idx+1}/{len(video_paths)}] {video_path.name}")
        try:
            if skip_face_detection:
                from src.preprocessing.frame_extractor import extract_raw_frames
                frames = extract_raw_frames(
                    str(video_path),
                    frame_interval=frame_interval,
                    max_frames=max_frames_per_video,
                    target_size=target_size,
                )
            else:
                frames = extractor.extract_faces_from_video(
                    str(video_path), target_size=target_size
                )

            for fi, frame in enumerate(frames):
                if frame is None or frame.size == 0:
                    continue
                if is_blurry(frame, blur_threshold):
                    continue
                if is_too_dark_or_bright(frame):
                    continue
                h = image_hash(frame)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                fname = f"{video_path.stem}_{fi:05d}.jpg"
                fpath = save_dir / fname
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(fpath), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1

        except Exception as e:
            logger.warning(f"  Failed to process {video_path.name}: {e}")

    return saved


def process_images(
    image_paths: List[Path],
    label: str,
    output_dir: Path,
    extractor: FaceExtractor,
    target_size: Tuple[int, int],
    skip_face_detection: bool,
    seen_hashes: Set[str],
    blur_threshold: float = 50.0,
) -> int:
    """Process a list of image files, detect/crop faces, and save to output_dir/label/."""
    save_dir = output_dir / label
    save_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for idx, img_path in enumerate(image_paths):
        try:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            if skip_face_detection:
                faces = [cv2.resize(rgb, target_size)]
            else:
                faces = extract_face_from_image(
                    rgb,
                    extractor=extractor,
                    target_size=target_size,
                    fallback_full_image=True,
                )

            for fi, face in enumerate(faces):
                if is_blurry(face, blur_threshold):
                    continue
                if is_too_dark_or_bright(face):
                    continue
                h = image_hash(face)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                fname = f"{img_path.stem}_face{fi:03d}.jpg"
                fpath = save_dir / fname
                bgr_out = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(fpath), bgr_out, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1

        except Exception as e:
            logger.warning(f"  Failed to process {img_path.name}: {e}")

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Train/Val/Test Split
# ─────────────────────────────────────────────────────────────────────────────

def split_into_sets(
    processed_dir: Path,
    output_dir: Path,
    label: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> Dict[str, int]:
    """
    Split processed face images into train/val/test directories.

    Args:
        processed_dir: Source directory with all face images for one class.
        output_dir: Root data directory (train/val/test will be created under it).
        label: "real" or "fake".
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        seed: Random seed.
        max_samples: Cap total samples (for balancing).

    Returns:
        Dict with counts per split.
    """
    import random
    random.seed(seed)

    all_files = sorted([
        f for f in processed_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    if not all_files:
        logger.warning(f"No files found in {processed_dir}")
        return {"train": 0, "val": 0, "test": 0}

    random.shuffle(all_files)

    if max_samples and len(all_files) > max_samples:
        logger.info(f"  Capping {label} from {len(all_files)} to {max_samples} samples")
        all_files = all_files[:max_samples]

    n = len(all_files)
    train_end = int(n * train_ratio)
    val_end   = train_end + int(n * val_ratio)

    splits = {
        "train": all_files[:train_end],
        "val":   all_files[train_end:val_end],
        "test":  all_files[val_end:],
    }

    counts = {}
    for split_name, files in splits.items():
        dst_dir = output_dir / split_name / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            dst = dst_dir / f.name
            if not dst.exists():
                import shutil
                shutil.copy2(str(f), str(dst))
        counts[split_name] = len(files)
        logger.info(f"  {split_name}/{label}: {len(files)} files")

    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_real_dataset_prep(
    raw_dir: str = "data/raw",
    output_dir: str = "data",
    processed_dir: str = "data/processed",
    target_size: Tuple[int, int] = (224, 224),
    frame_interval: int = 10,
    max_frames_per_video: int = 30,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    skip_face_detection: bool = False,
    max_per_class: Optional[int] = None,
    blur_threshold: float = 50.0,
) -> Dict:
    """
    Full real-world dataset preparation pipeline.

    Steps:
      1. Scan raw_dir for video/image files by class
      2. Extract faces from each file
      3. Filter blurry/dark/duplicate frames
      4. Split into train/val/test
      5. Report final counts

    Returns:
        Dict: Statistics on processed files.
    """
    logger.info("=" * 60)
    logger.info("  Real Dataset Preparation Pipeline")
    logger.info("=" * 60)

    # Initialize face extractor
    extractor = FaceExtractor(
        frame_interval=frame_interval,
        max_frames=max_frames_per_video,
        face_margin=0.3,
        min_face_size=50,
    )

    processed_path = Path(processed_dir)
    output_path    = Path(output_dir)
    raw_path       = Path(raw_dir)

    real_videos, fake_videos, real_images, fake_images = scan_raw_directory(raw_dir)

    if not any([real_videos, fake_videos, real_images, fake_images]):
        logger.error(
            f"No data found in {raw_dir}!\n"
            "Run: python src/data/dataset_downloader.py --dataset celebdf_v2\n"
            "  or place videos/images in data/raw/real/ and data/raw/fake/"
        )
        return {}

    total_stats = {}

    for label, videos, images in [
        ("real", real_videos, real_images),
        ("fake", fake_videos, fake_images),
    ]:
        if not videos and not images:
            logger.warning(f"No {label} files found. Skipping.")
            continue

        logger.info(f"\nProcessing '{label}': {len(videos)} videos, {len(images)} images")
        proc_label_dir = processed_path / label
        proc_label_dir.mkdir(parents=True, exist_ok=True)

        seen_hashes: Set[str] = set()

        vid_saved = process_videos(
            video_paths=videos,
            label=label,
            output_dir=processed_path,
            extractor=extractor,
            target_size=target_size,
            skip_face_detection=skip_face_detection,
            max_frames_per_video=max_frames_per_video,
            frame_interval=frame_interval,
            seen_hashes=seen_hashes,
            blur_threshold=blur_threshold,
        ) if videos else 0

        img_saved = process_images(
            image_paths=images,
            label=label,
            output_dir=processed_path,
            extractor=extractor,
            target_size=target_size,
            skip_face_detection=skip_face_detection,
            seen_hashes=seen_hashes,
            blur_threshold=blur_threshold,
        ) if images else 0

        total_saved = vid_saved + img_saved
        logger.info(f"  Total {label} faces saved: {total_saved}")

        if total_saved == 0:
            logger.warning(f"No {label} faces extracted! Skipping split.")
            continue

        counts = split_into_sets(
            processed_dir=proc_label_dir,
            output_dir=output_path,
            label=label,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            max_samples=max_per_class,
        )
        total_stats[label] = {"processed": total_saved, **counts}

    extractor.close()

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("  Dataset Preparation Complete")
    logger.info("=" * 60)
    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            count = total_stats.get(label, {}).get(split, 0)
            logger.info(f"  {split}/{label}: {count}")

    # Balance check
    train_real = total_stats.get("real", {}).get("train", 0)
    train_fake = total_stats.get("fake", {}).get("train", 0)
    if train_real > 0 and train_fake > 0:
        ratio = min(train_real, train_fake) / max(train_real, train_fake)
        if ratio < 0.5:
            logger.warning(
                f"Class imbalance detected (ratio={ratio:.2f}). "
                "Consider setting max_per_class or use_weighted_sampler=true in config."
            )

    return total_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess real deepfake dataset into face images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw_dir",      default="data/raw",       help="Raw data directory.")
    parser.add_argument("--output_dir",   default="data",           help="Output base directory.")
    parser.add_argument("--processed_dir",default="data/processed", help="Intermediate processed dir.")
    parser.add_argument("--target_size",  type=int, default=224,    help="Face crop size (square).")
    parser.add_argument("--frame_interval", type=int, default=10)
    parser.add_argument("--max_frames",   type=int, default=30)
    parser.add_argument("--train_ratio",  type=float, default=0.70)
    parser.add_argument("--val_ratio",    type=float, default=0.15)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--max_per_class",type=int, default=None,   help="Cap samples per class.")
    parser.add_argument("--skip_face_detection", action="store_true",
                        help="Use full frame instead of face crop.")
    parser.add_argument("--blur_threshold", type=float, default=50.0)
    parser.add_argument("--config",       default="config/config.yaml")

    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        fe  = cfg.get("frame_extraction", {})
        frame_interval     = fe.get("frame_interval", args.frame_interval)
        max_frames         = fe.get("max_frames_per_video", args.max_frames)
        max_per_class      = cfg.get("data", {}).get("max_samples_per_class", args.max_per_class)
    except Exception:
        frame_interval = args.frame_interval
        max_frames     = args.max_frames
        max_per_class  = args.max_per_class

    run_real_dataset_prep(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        processed_dir=args.processed_dir,
        target_size=(args.target_size, args.target_size),
        frame_interval=frame_interval,
        max_frames_per_video=max_frames,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        skip_face_detection=args.skip_face_detection,
        max_per_class=max_per_class,
        blur_threshold=args.blur_threshold,
    )
