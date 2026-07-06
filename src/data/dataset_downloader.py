"""
dataset_downloader.py — Real deepfake dataset downloader and organizer.

Supports:
  1. Celeb-DF v2   — public, no registration, Google Drive hosted
  2. FaceForensics++ — requires access request (script guides user)
  3. DFDC          — requires Kaggle API credentials
  4. Custom        — any user-provided directory of real/fake videos or images

After download, all datasets are normalized to the same structure:
    data/raw/real/  ← original/real videos or images
    data/raw/fake/  ← manipulated/deepfake videos or images

Usage:
    python src/data/dataset_downloader.py --dataset celebdf_v2
    python src/data/dataset_downloader.py --dataset faceforensics --ff_path /path/to/ff++
    python src/data/dataset_downloader.py --dataset custom --real_dir /my/real --fake_dir /my/fake
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Celeb-DF v2 Google Drive folder ID (public)
CELEBDF_V2_GDRIVE_ID = "1iLx76rsj1x6Y4pKKQDKBsTTWjBXqxMm0"
# Individual video list file
CELEBDF_V2_LIST_ID   = "1koRZcB_sMxFoqRwMk2xQAW8zp7eqE6dn"


# ─────────────────────────────────────────────────────────────────────────────
# Celeb-DF v2
# ─────────────────────────────────────────────────────────────────────────────

def download_celebdf_v2(output_dir: str = "data/raw") -> bool:
    """
    Download Celeb-DF v2 dataset from Google Drive using gdown.

    Celeb-DF v2 contains:
      - 590 real YouTube videos  → data/raw/real/
      - 5639 synthesized videos  → data/raw/fake/

    Args:
        output_dir: Base directory for raw data.

    Returns:
        bool: True on success.
    """
    try:
        import gdown
    except ImportError:
        logger.error("gdown not installed. Run: pip install gdown")
        return False

    output_path = Path(output_dir)
    real_dir = output_path / "real"
    fake_dir = output_path / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path("data/tmp_celebdf")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  Downloading Celeb-DF v2 dataset")
    logger.info("  Source: Google Drive (public)")
    logger.info("  Real videos: ~590 | Fake videos: ~5639")
    logger.info("=" * 60)

    # Download the full folder
    url = f"https://drive.google.com/drive/folders/{CELEBDF_V2_GDRIVE_ID}"
    logger.info(f"Downloading from: {url}")
    logger.info("This may take 30-90 minutes depending on connection speed...")

    try:
        gdown.download_folder(
            url=url,
            output=str(tmp_dir),
            quiet=False,
            use_cookies=False,
        )
    except Exception as e:
        logger.error(f"gdown folder download failed: {e}")
        logger.info("Attempting alternative: downloading via file list...")
        return _download_celebdf_alternative(output_dir)

    # Organize downloaded files
    logger.info("Organizing Celeb-DF v2 into real/fake directories...")
    return _organize_celebdf_v2(tmp_dir, real_dir, fake_dir)


def _download_celebdf_alternative(output_dir: str) -> bool:
    """
    Alternative Celeb-DF v2 download when folder download fails.
    Downloads individual components via known IDs.
    """
    import gdown

    output_path = Path(output_dir)
    real_dir = output_path / "real"
    fake_dir = output_path / "fake"

    # Known public file IDs for Celeb-DF v2 components
    components = [
        # (gdrive_file_id, local_filename, is_real)
        ("1jQ7d9nmGcmt8g-J9tDqMJp4nE6J4sCVQ", "YouTube-real.zip", True),
        ("1FLEHFzxJIHopTVcLJGTy7hfST3dj5W_O", "Celeb-synthesis.zip", False),
    ]

    for file_id, filename, is_real in components:
        dst = output_path / filename
        logger.info(f"Downloading {filename}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        try:
            gdown.download(url, str(dst), quiet=False)
            if dst.exists():
                logger.info(f"Extracting {filename}...")
                target = real_dir if is_real else fake_dir
                with zipfile.ZipFile(dst, 'r') as z:
                    z.extractall(target)
                dst.unlink()
                logger.info(f"  -> {target}")
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            return False

    return True


def _organize_celebdf_v2(tmp_dir: Path, real_dir: Path, fake_dir: Path) -> bool:
    """
    Move Celeb-DF v2 files from tmp into proper real/fake directories.
    Celeb-DF v2 structure: YouTube-real/, Celeb-synthesis/
    """
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

    real_count = 0
    fake_count = 0

    # YouTube-real → real/
    for src_dir in tmp_dir.rglob("YouTube-real"):
        if src_dir.is_dir():
            for f in src_dir.rglob("*"):
                if f.suffix.lower() in video_extensions:
                    dst = real_dir / f.name
                    shutil.move(str(f), str(dst))
                    real_count += 1

    # Celeb-synthesis → fake/
    for src_dir in tmp_dir.rglob("Celeb-synthesis"):
        if src_dir.is_dir():
            for f in src_dir.rglob("*"):
                if f.suffix.lower() in video_extensions:
                    dst = fake_dir / f.name
                    shutil.move(str(f), str(dst))
                    fake_count += 1

    # Fallback: any videos not in named subdirs
    if real_count == 0 and fake_count == 0:
        logger.warning("Named subdirs not found. Scanning all downloaded files...")
        for f in tmp_dir.rglob("*"):
            if f.suffix.lower() in video_extensions:
                # Heuristic: files with 'id' in name are real, others fake
                if "id" in f.stem.lower() and not "synthesis" in str(f.parent).lower():
                    shutil.move(str(f), str(real_dir / f.name))
                    real_count += 1
                else:
                    shutil.move(str(f), str(fake_dir / f.name))
                    fake_count += 1

    shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(f"Celeb-DF v2 organized: {real_count} real, {fake_count} fake videos")

    if real_count == 0 and fake_count == 0:
        logger.error("No videos were found/moved. Check the download.")
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# FaceForensics++
# ─────────────────────────────────────────────────────────────────────────────

def setup_faceforensics(
    ff_path: str,
    manipulation: str = "Deepfakes",
    compression: str = "c23",
    output_dir: str = "data/raw",
) -> bool:
    """
    Organize a locally downloaded FaceForensics++ dataset.

    FF++ must be requested from: https://github.com/ondyari/FaceForensics
    After approval you receive a download script. Run it, then point --ff_path here.

    Expected FF++ structure:
        ff_path/
        ├── original_sequences/youtube/c23/videos/
        └── manipulated_sequences/Deepfakes/c23/videos/

    Args:
        ff_path: Path to the FF++ root directory.
        manipulation: Which manipulation type to use (Deepfakes, Face2Face, etc.)
        compression: c0/c23/c40
        output_dir: Where to write real/fake symlinks/copies.

    Returns:
        bool: True on success.
    """
    ff_root = Path(ff_path)
    if not ff_root.exists():
        logger.error(f"FaceForensics++ path not found: {ff_path}")
        return False

    real_src = ff_root / "original_sequences" / "youtube" / compression / "videos"
    fake_src = ff_root / "manipulated_sequences" / manipulation / compression / "videos"

    if not real_src.exists():
        # Try alternative structure
        real_src = ff_root / "original_sequences" / "youtube" / "videos"
    if not fake_src.exists():
        fake_src = ff_root / "manipulated_sequences" / manipulation / "videos"

    if not real_src.exists():
        logger.error(f"Real videos not found at: {real_src}")
        logger.info("Expected: original_sequences/youtube/{compression}/videos/")
        return False

    if not fake_src.exists():
        logger.error(f"Fake videos not found at: {fake_src}")
        logger.info(f"Expected: manipulated_sequences/{manipulation}/{compression}/videos/")
        return False

    output_path = Path(output_dir)
    real_out = output_path / "real"
    fake_out = output_path / "fake"
    real_out.mkdir(parents=True, exist_ok=True)
    fake_out.mkdir(parents=True, exist_ok=True)

    video_exts = {".mp4", ".avi", ".mov"}

    real_count = 0
    for f in real_src.iterdir():
        if f.suffix.lower() in video_exts:
            dst = real_out / f.name
            if not dst.exists():
                shutil.copy2(str(f), str(dst))
            real_count += 1

    fake_count = 0
    for f in fake_src.iterdir():
        if f.suffix.lower() in video_exts:
            dst = fake_out / f.name
            if not dst.exists():
                shutil.copy2(str(f), str(dst))
            fake_count += 1

    logger.info(f"FaceForensics++ organized: {real_count} real, {fake_count} fake")
    return real_count > 0 and fake_count > 0


# ─────────────────────────────────────────────────────────────────────────────
# DFDC
# ─────────────────────────────────────────────────────────────────────────────

def setup_dfdc(dfdc_path: str, output_dir: str = "data/raw") -> bool:
    """
    Organize a locally downloaded DFDC dataset.

    DFDC download via Kaggle API:
        kaggle competitions download -c deepfake-detection-challenge

    DFDC structure includes metadata.json with is_fake labels per video.

    Args:
        dfdc_path: Path to DFDC root directory (contains metadata.json and .mp4 files).
        output_dir: Where to write organized real/fake videos.

    Returns:
        bool: True on success.
    """
    dfdc_root = Path(dfdc_path)

    if not dfdc_root.exists():
        logger.error(f"DFDC path not found: {dfdc_path}")
        return False

    output_path = Path(output_dir)
    real_out = output_path / "real"
    fake_out = output_path / "fake"
    real_out.mkdir(parents=True, exist_ok=True)
    fake_out.mkdir(parents=True, exist_ok=True)

    # Find all metadata.json files (DFDC has one per chunk)
    metadata_files = list(dfdc_root.rglob("metadata.json"))
    if not metadata_files:
        logger.error("No metadata.json found in DFDC directory.")
        return False

    real_count = 0
    fake_count = 0

    for meta_file in metadata_files:
        with open(meta_file) as f:
            metadata = json.load(f)

        video_dir = meta_file.parent
        for video_name, info in metadata.items():
            src = video_dir / video_name
            if not src.exists():
                continue
            is_fake = info.get("label", "REAL") == "FAKE"
            dst_dir = fake_out if is_fake else real_out
            dst = dst_dir / video_name
            if not dst.exists():
                shutil.copy2(str(src), str(dst))
            if is_fake:
                fake_count += 1
            else:
                real_count += 1

    logger.info(f"DFDC organized: {real_count} real, {fake_count} fake")
    return real_count > 0 or fake_count > 0


# ─────────────────────────────────────────────────────────────────────────────
# Custom Dataset
# ─────────────────────────────────────────────────────────────────────────────

def setup_custom(real_dir: str, fake_dir: str, output_dir: str = "data/raw") -> bool:
    """
    Use a custom user-provided dataset. Simply validates and symlinks/copies.

    Accepts both VIDEO files (.mp4/.avi/.mov) and IMAGE files (.jpg/.png).

    Args:
        real_dir: Directory with real videos or images.
        fake_dir: Directory with fake videos or images.
        output_dir: Target output directory.
    """
    real_src = Path(real_dir)
    fake_src = Path(fake_dir)

    if not real_src.exists():
        logger.error(f"Real directory not found: {real_dir}")
        return False
    if not fake_src.exists():
        logger.error(f"Fake directory not found: {fake_dir}")
        return False

    valid_exts = {".mp4", ".avi", ".mov", ".mkv", ".jpg", ".jpeg", ".png", ".webm"}

    output_path = Path(output_dir)
    real_out = output_path / "real"
    fake_out = output_path / "fake"
    real_out.mkdir(parents=True, exist_ok=True)
    fake_out.mkdir(parents=True, exist_ok=True)

    real_count = 0
    for f in real_src.rglob("*"):
        if f.suffix.lower() in valid_exts:
            dst = real_out / f.name
            if not dst.exists():
                shutil.copy2(str(f), str(dst))
            real_count += 1

    fake_count = 0
    for f in fake_src.rglob("*"):
        if f.suffix.lower() in valid_exts:
            dst = fake_out / f.name
            if not dst.exists():
                shutil.copy2(str(f), str(dst))
            fake_count += 1

    logger.info(f"Custom dataset: {real_count} real, {fake_count} fake files copied.")
    return real_count > 0 or fake_count > 0


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_dataset(raw_dir: str = "data/raw") -> Tuple[int, int]:
    """
    Count real/fake files in the raw data directory.

    Returns:
        (real_count, fake_count)
    """
    valid_exts = {".mp4", ".avi", ".mov", ".mkv", ".jpg", ".jpeg", ".png"}
    raw_path = Path(raw_dir)

    real_count = sum(
        1 for f in (raw_path / "real").rglob("*") if f.suffix.lower() in valid_exts
    ) if (raw_path / "real").exists() else 0

    fake_count = sum(
        1 for f in (raw_path / "fake").rglob("*") if f.suffix.lower() in valid_exts
    ) if (raw_path / "fake").exists() else 0

    logger.info(f"Dataset verified: {real_count} real, {fake_count} fake files")
    return real_count, fake_count


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and organize a real deepfake dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Celeb-DF v2 (recommended, fully automated):
  python src/data/dataset_downloader.py --dataset celebdf_v2

  # FaceForensics++ (manual download required first):
  python src/data/dataset_downloader.py --dataset faceforensics --ff_path /path/to/ff++

  # DFDC (Kaggle API required):
  python src/data/dataset_downloader.py --dataset dfdc --dfdc_path /path/to/dfdc

  # Custom dataset (your own videos/images):
  python src/data/dataset_downloader.py --dataset custom \\
         --real_dir /my/real/videos --fake_dir /my/fake/videos

  # Verify existing data:
  python src/data/dataset_downloader.py --verify
        """,
    )
    parser.add_argument("--dataset", choices=["celebdf_v2", "faceforensics", "dfdc", "custom"],
                        help="Dataset to download/organize.")
    parser.add_argument("--output_dir", default="data/raw", help="Raw data output directory.")
    parser.add_argument("--ff_path", help="Path to FaceForensics++ root directory.")
    parser.add_argument("--ff_manipulation", default="Deepfakes",
                        choices=["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"])
    parser.add_argument("--ff_compression", default="c23", choices=["c0", "c23", "c40"])
    parser.add_argument("--dfdc_path", help="Path to DFDC root directory.")
    parser.add_argument("--real_dir", help="Custom: path to real videos/images.")
    parser.add_argument("--fake_dir", help="Custom: path to fake videos/images.")
    parser.add_argument("--verify", action="store_true", help="Verify existing dataset.")

    args = parser.parse_args()

    if args.verify:
        r, f = verify_dataset(args.output_dir)
        print(f"\nDataset at '{args.output_dir}':")
        print(f"  Real files: {r}")
        print(f"  Fake files: {f}")
        print(f"  Total:      {r + f}")
        sys.exit(0 if (r > 0 and f > 0) else 1)

    if not args.dataset:
        parser.print_help()
        sys.exit(1)

    success = False
    if args.dataset == "celebdf_v2":
        success = download_celebdf_v2(args.output_dir)
    elif args.dataset == "faceforensics":
        if not args.ff_path:
            logger.error("--ff_path is required for faceforensics.")
            sys.exit(1)
        success = setup_faceforensics(args.ff_path, args.ff_manipulation,
                                       args.ff_compression, args.output_dir)
    elif args.dataset == "dfdc":
        if not args.dfdc_path:
            logger.error("--dfdc_path is required for dfdc.")
            sys.exit(1)
        success = setup_dfdc(args.dfdc_path, args.output_dir)
    elif args.dataset == "custom":
        if not args.real_dir or not args.fake_dir:
            logger.error("--real_dir and --fake_dir are required for custom.")
            sys.exit(1)
        success = setup_custom(args.real_dir, args.fake_dir, args.output_dir)

    if success:
        r, f = verify_dataset(args.output_dir)
        print(f"\n✓ Dataset ready: {r} real, {f} fake files in '{args.output_dir}'")
        print("Next step: python src/data/real_dataset_prep.py")
    else:
        print("\n✗ Dataset setup failed. Check the logs above.")
        sys.exit(1)
