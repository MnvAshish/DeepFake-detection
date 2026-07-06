"""
dataset.py - PyTorch Dataset and DataLoader factory for deepfake detection.

Expects data organized as:
    data/
    ├── train/
    │   ├── real/   ← real face images (.jpg/.png)
    │   └── fake/   ← fake face images (.jpg/.png)
    ├── val/
    │   ├── real/
    │   └── fake/
    └── test/
        ├── real/
        └── fake/

Supports:
  - Standard torchvision transforms (resize, normalize)
  - Albumentations augmentations for training
  - Weighted random sampling to handle class imbalance
  - Both 224x224 (ResNet/VGG) and 299x299 (InceptionV3) sizes
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Class mapping: folder name → integer label
CLASS_TO_IDX = {"real": 0, "fake": 1}
IDX_TO_CLASS = {0: "real", 1: "fake"}

# Supported image file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Class
# ─────────────────────────────────────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """
    Custom PyTorch Dataset for binary deepfake/real classification.

    Loads face-cropped images from a directory structure with
    'real/' and 'fake/' subdirectories.
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        use_albumentations: bool = False,
    ):
        """
        Args:
            data_dir (str): Root directory containing 'real/' and 'fake/' subfolders.
            transform: torchvision transforms or albumentations pipeline.
            use_albumentations (bool): If True, expects albumentations transform.
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.use_albumentations = use_albumentations and ALBUMENTATIONS_AVAILABLE

        # Gather all image paths and labels
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

        logger.info(
            f"Dataset loaded from: {self.data_dir} | "
            f"Total samples: {len(self.samples)} | "
            f"Real: {self.class_counts.get(0, 0)}, "
            f"Fake: {self.class_counts.get(1, 0)}"
        )

    def _load_samples(self) -> None:
        """Scan data directory and build (image_path, label) pairs."""
        self.class_counts: Dict[int, int] = {0: 0, 1: 0}

        for class_name, label in CLASS_TO_IDX.items():
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                logger.warning(
                    f"Class directory not found: {class_dir}. "
                    f"Expected '{class_name}/' inside '{self.data_dir}'."
                )
                continue

            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((img_path, label))
                    self.class_counts[label] += 1

        if len(self.samples) == 0:
            raise ValueError(
                f"No images found in {self.data_dir}. "
                "Ensure the directory contains 'real/' and 'fake/' subdirectories "
                "with image files."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and transform a single sample.

        Returns:
            Tuple[Tensor, int]: (image_tensor, label)
                image_tensor shape: (C, H, W)
                label: 0 for real, 1 for fake
        """
        img_path, label = self.samples[idx]

        # Load image
        try:
            if self.use_albumentations:
                # Albumentations expects numpy arrays
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            # Return a black image as fallback
            if self.use_albumentations:
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        # Apply transforms
        if self.transform is not None:
            if self.use_albumentations:
                augmented = self.transform(image=image)
                image = augmented["image"]
            else:
                image = self.transform(image)

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse frequency class weights for loss function.

        Returns:
            Tensor of shape (num_classes,) with per-class weights.
        """
        total = len(self.samples)
        weights = []
        for label in range(2):
            count = self.class_counts.get(label, 1)
            weights.append(total / (2.0 * count))
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> List[float]:
        """
        Compute per-sample weights for WeightedRandomSampler (handles imbalance).

        Returns:
            List of float weights, one per sample.
        """
        class_weights = self.get_class_weights()
        sample_weights = [
            class_weights[label].item()
            for _, label in self.samples
        ]
        return sample_weights


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(
    split: str = "train",
    image_size: int = 224,
    mean: List[float] = None,
    std: List[float] = None,
) -> transforms.Compose:
    """
    Build torchvision transform pipelines for train/val/test splits.

    ImageNet normalization stats are used by default since we load
    pretrained weights trained on ImageNet.

    Args:
        split (str): "train", "val", or "test".
        image_size (int): Target image size (both H and W).
        mean (list): Normalization mean per channel.
        std (list): Normalization std per channel.

    Returns:
        transforms.Compose: The transform pipeline.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.3, hue=0.1
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def get_albumentations_transforms(
    split: str = "train",
    image_size: int = 224,
    mean: Tuple = (0.485, 0.456, 0.406),
    std: Tuple = (0.229, 0.224, 0.225),
):
    """
    Build Albumentations augmentation pipelines (stronger augmentation).

    Args:
        split (str): "train", "val", or "test".
        image_size (int): Target image size.
        mean (tuple): Normalization mean.
        std (tuple): Normalization std.

    Returns:
        Albumentations Compose pipeline.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("albumentations is not installed. Run: pip install albumentations")

    if split == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.ElasticTransform(alpha=1, sigma=50, p=0.2),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloader(
    data_dir: str,
    split: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_weighted_sampler: bool = True,
    use_albumentations: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for the given data split.

    Args:
        data_dir (str): Root directory with 'real/' and 'fake/' folders.
        split (str): "train", "val", or "test".
        batch_size (int): Samples per batch.
        image_size (int): Target image size for transforms.
        num_workers (int): Parallel data loading workers.
        pin_memory (bool): Pin tensors in memory for faster GPU transfer.
        use_weighted_sampler (bool): Balance classes by oversampling minority.
        use_albumentations (bool): Use albumentations augmentation pipeline.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    if use_albumentations and ALBUMENTATIONS_AVAILABLE:
        transform = get_albumentations_transforms(split, image_size)
    else:
        transform = get_transforms(split, image_size)
        use_albumentations = False

    dataset = DeepfakeDataset(
        data_dir=data_dir,
        transform=transform,
        use_albumentations=use_albumentations,
    )

    # Set up sampler
    sampler = None
    shuffle = (split == "train")

    if split == "train" and use_weighted_sampler:
        sample_weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False  # Sampler handles randomization

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=(split == "train"),  # Drop incomplete last batch during training
    )

    logger.info(
        f"DataLoader created | split={split}, samples={len(dataset)}, "
        f"batches={len(dataloader)}, batch_size={batch_size}, "
        f"weighted_sampler={use_weighted_sampler and split == 'train'}"
    )

    return dataloader


def get_all_dataloaders(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_albumentations: bool = False,
) -> Dict[str, DataLoader]:
    """
    Create train, val, and test DataLoaders in one call.

    Args:
        train_dir (str): Training data directory.
        val_dir (str): Validation data directory.
        test_dir (str): Test data directory.
        batch_size (int): Batch size for all loaders.
        image_size (int): Image size for transforms.
        num_workers (int): DataLoader workers.
        pin_memory (bool): Pin memory flag.
        use_albumentations (bool): Use albumentations pipeline.

    Returns:
        dict: {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    loaders = {}

    for split, data_dir in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
        try:
            loaders[split] = get_dataloader(
                data_dir=data_dir,
                split=split,
                batch_size=batch_size,
                image_size=image_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                use_weighted_sampler=(split == "train"),
                use_albumentations=use_albumentations,
            )
        except ValueError as e:
            logger.warning(f"Could not create {split} DataLoader: {e}")

    return loaders
