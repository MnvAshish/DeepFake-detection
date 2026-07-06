"""
train_pipeline.py — Orchestrates training of all three ensemble models.

New features:
  - Resume training support (per model)
  - TensorBoard log directory per model
  - Removes all dummy/synthetic data references
  - Validates real dataset exists before starting
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.models import build_model
from src.preprocessing.dataset import get_dataloader, get_all_dataloaders
from src.training.trainer import Trainer
from src.utils.config_loader import load_config, ensure_directories
from src.utils.helpers import (
    get_device, set_seed, load_checkpoint,
    compute_metrics, print_metrics,
    plot_confusion_matrix, save_results_json, Timer,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _validate_dataset(config: Dict) -> bool:
    """Check that real training data exists before starting."""
    paths    = config["paths"]
    train_dir = Path(paths["train_dir"])
    has_data = False
    for label in ["real", "fake"]:
        d = train_dir / label
        if d.exists():
            count = sum(1 for f in d.iterdir()
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if count > 0:
                has_data = True
                logger.info(f"  train/{label}: {count} images")

    if not has_data:
        logger.error(
            "\n" + "=" * 60 + "\n"
            "  NO TRAINING DATA FOUND\n"
            "=" * 60 + "\n"
            "  Run one of:\n"
            "  1. Download Celeb-DF v2:\n"
            "       python src/data/dataset_downloader.py --dataset celebdf_v2\n"
            "       python src/data/real_dataset_prep.py\n\n"
            "  2. Use your own videos:\n"
            "       Place real videos → data/raw/real/\n"
            "       Place fake videos → data/raw/fake/\n"
            "       python src/data/real_dataset_prep.py\n\n"
            "  3. Use existing face images directly:\n"
            "       Place images → data/train/real/ and data/train/fake/\n"
            + "=" * 60
        )
    return has_data


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader,
    device: torch.device,
    model_name: str,
    class_names: List[str],
) -> Dict:
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs  = F.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    print_metrics(metrics, model_name)

    try:
        plot_confusion_matrix(all_labels, all_preds, class_names, model_name, "outputs")
    except Exception as e:
        logger.warning(f"Confusion matrix failed: {e}")

    return metrics


def train_single_model(
    model_key: str,
    config: Dict,
    device: torch.device,
    train_loader,
    val_loader,
    test_loader,
    resume: bool = False,
) -> Dict:
    model_cfg  = config["models"][model_key]
    save_path  = model_cfg["save_path"]
    resume_path = model_cfg.get("resume_path") if resume else None
    class_names = config["data"]["class_names"]

    tb_dir = str(
        Path(config.get("paths", {}).get("tensorboard_dir", "runs")) / model_key
    )

    logger.info(f"\n{'#'*60}\n  Training: {model_key.upper()}\n{'#'*60}")

    model = build_model(
        model_name=model_key,
        num_classes=config["data"]["num_classes"],
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.5),
        freeze_backbone=model_cfg.get("freeze_layers", False),
    ).to(device)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        model_save_path=save_path,
        resume_path=resume_path,
        tb_log_dir=tb_dir,
    )

    with Timer(f"{model_key} training"):
        history = trainer.train()

    # Evaluate best checkpoint
    try:
        model, _, _ = load_checkpoint(model, save_path, device=device)
    except FileNotFoundError:
        logger.warning("No saved checkpoint found for evaluation.")

    test_metrics = evaluate_model(model, test_loader, device, model_key.upper(), class_names)
    return {**history, "test_metrics": test_metrics}


def run_training_pipeline(
    config_path: str = "config/config.yaml",
    models_to_train: Optional[List[str]] = None,
    resume: bool = False,
) -> Dict:
    config = load_config(config_path)
    ensure_directories(config_path)
    set_seed(config["training"].get("seed", 42))
    device = get_device(config["inference"].get("device", "auto"))

    if not _validate_dataset(config):
        sys.exit(1)

    if models_to_train is None:
        models_to_train = ["resnet50", "vgg16", "inceptionv3"]

    paths    = config["paths"]
    data_cfg = config["data"]
    train_cfg = config["training"]

    # Auto-detect resume from config
    if not resume:
        resume = train_cfg.get("resume", False)

    all_results = {}

    # ── ResNet50 + VGG16 (224×224) ────────────────────────────────────────────
    standard = [m for m in models_to_train if m != "inceptionv3"]
    if standard:
        loaders = get_all_dataloaders(
            train_dir=paths["train_dir"],
            val_dir=paths["val_dir"],
            test_dir=paths["test_dir"],
            batch_size=train_cfg["batch_size"],
            image_size=data_cfg["image_size"],
            num_workers=data_cfg["num_workers"],
            pin_memory=data_cfg["pin_memory"],
        )
        for mk in standard:
            if mk not in config["models"]:
                continue
            res = train_single_model(mk, config, device,
                                     loaders["train"], loaders["val"], loaders["test"],
                                     resume=resume)
            all_results[mk] = res

    # ── InceptionV3 (299×299) ─────────────────────────────────────────────────
    if "inceptionv3" in models_to_train:
        loaders299 = get_all_dataloaders(
            train_dir=paths["train_dir"],
            val_dir=paths["val_dir"],
            test_dir=paths["test_dir"],
            batch_size=train_cfg["batch_size"],
            image_size=data_cfg["inception_size"],
            num_workers=data_cfg["num_workers"],
            pin_memory=data_cfg["pin_memory"],
        )
        res = train_single_model("inceptionv3", config, device,
                                  loaders299["train"], loaders299["val"], loaders299["test"],
                                  resume=resume)
        all_results["inceptionv3"] = res

    # ── Save results ──────────────────────────────────────────────────────────
    save_results_json(all_results, "outputs/training_results.json")

    logger.info("\n" + "="*60 + "\n  TRAINING SUMMARY\n" + "="*60)
    for mn, r in all_results.items():
        logger.info(
            f"  {mn:<15} | best_val={r.get('best_val_acc',0):.4f} | "
            f"test_acc={r.get('test_metrics',{}).get('accuracy',0):.4f}"
        )
    logger.info("="*60)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model", default=None,
                        choices=["resnet50", "vgg16", "inceptionv3", "all"])
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint.")
    args = parser.parse_args()

    models = None
    if args.model and args.model != "all":
        models = [args.model]

    run_training_pipeline(
        config_path=args.config,
        models_to_train=models,
        resume=args.resume,
    )
