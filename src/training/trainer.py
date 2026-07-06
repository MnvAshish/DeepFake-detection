"""
trainer.py — Core training engine for the Deepfake Detection System.

Handles:
  - Training loop with gradient accumulation
  - Validation loop
  - Automatic Mixed Precision (AMP)
  - Early stopping
  - Learning rate scheduling with warmup
  - TensorBoard logging
  - Resume training from checkpoint
  - Best AND last epoch checkpoint saving
  - Label smoothing
  - InceptionV3 auxiliary output
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.logger import get_logger
from src.utils.helpers import (
    save_checkpoint,
    load_checkpoint,
    compute_metrics,
    plot_training_history,
    format_time,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 1e-4, verbose: bool = True):
        self.patience   = patience
        self.min_delta  = min_delta
        self.verbose    = verbose
        self.counter    = 0
        self.best_loss  = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping counter: {self.counter}/{self.patience} "
                    f"(best loss: {self.best_loss:.4f})"
                )
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info("Early stopping triggered.")
        return self.should_stop


# ─────────────────────────────────────────────────────────────────────────────
# TensorBoard Writer (lazy import so it's optional)
# ─────────────────────────────────────────────────────────────────────────────

class TBWriter:
    """Thin wrapper around SummaryWriter that does nothing when disabled."""

    def __init__(self, log_dir: Optional[str], enabled: bool = True):
        self.writer  = None
        self.enabled = enabled and log_dir is not None
        if self.enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                Path(log_dir).mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(log_dir=log_dir)
                logger.info(f"TensorBoard logging → {log_dir}")
                logger.info(f"  Launch with: tensorboard --logdir {log_dir}")
            except ImportError:
                logger.warning("tensorboard not installed — disabling TB logging.")
                self.enabled = False

    def add_scalar(self, tag: str, value: float, step: int):
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict, step: int):
        if self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def add_figure(self, tag: str, figure, step: int):
        if self.writer:
            self.writer.add_figure(tag, figure, step)

    def close(self):
        if self.writer:
            self.writer.close()


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Full training pipeline for a single model.

    New features vs original:
      - TensorBoard: scalars every batch + epoch summaries
      - Resume: loads from resume_path and continues from saved epoch
      - LR warmup: linear ramp for warmup_epochs before main scheduler
      - Label smoothing: passed to CrossEntropyLoss
      - Last checkpoint: always saves last epoch (for resume)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: Dict,
        model_save_path: str,
        resume_path: Optional[str] = None,
        tb_log_dir: Optional[str] = None,
    ):
        self.model           = model.to(device)
        self.train_loader    = train_loader
        self.val_loader      = val_loader
        self.device          = device
        self.config          = config
        self.model_save_path = model_save_path
        self.model_name      = getattr(model, "model_name", "Model")
        self.is_inception    = "inception" in self.model_name.lower()

        tc = config.get("training", {})
        self.num_epochs     = tc.get("num_epochs", 25)
        self.lr             = tc.get("learning_rate", 1e-4)
        self.weight_decay   = tc.get("weight_decay", 1e-4)
        self.grad_clip      = tc.get("grad_clip_value", 1.0)
        self.use_amp        = tc.get("use_amp", True) and device.type == "cuda"
        self.patience       = tc.get("early_stopping_patience", 7)
        self.warmup_epochs  = tc.get("warmup_epochs", 2)
        self.label_smoothing = tc.get("label_smoothing", 0.1)
        self.log_every_n    = config.get("logging", {}).get("log_every_n_batches", 50)

        # Last checkpoint path (for resume)
        self.last_ckpt_path = str(Path(model_save_path).parent /
                                   (Path(model_save_path).stem.replace("_best", "_last") + ".pth"))

        # Class weights
        try:
            cw = train_loader.dataset.get_class_weights().to(device)
            logger.info(f"Class weights: {cw.tolist()}")
        except AttributeError:
            cw = None

        self.criterion = nn.CrossEntropyLoss(
            weight=cw,
            label_smoothing=self.label_smoothing,
        )

        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler_type = tc.get("lr_scheduler", "cosine")
        self.scheduler = self._build_scheduler(scheduler_type, tc)
        self.scaler    = GradScaler() if self.use_amp else None
        self.early_stopping = EarlyStopping(patience=self.patience)

        # TensorBoard
        tb_enabled = config.get("logging", {}).get("tensorboard", True)
        if tb_log_dir is None:
            tb_log_dir = config.get("paths", {}).get("tensorboard_dir", "runs")
            tb_log_dir = str(Path(tb_log_dir) / self.model_name)
        self.tb = TBWriter(log_dir=tb_log_dir, enabled=tb_enabled)

        # History
        self.train_losses: List[float] = []
        self.val_losses:   List[float] = []
        self.train_accs:   List[float] = []
        self.val_accs:     List[float] = []
        self.best_val_acc  = 0.0
        self.start_epoch   = 1

        # Resume from checkpoint
        self._maybe_resume(resume_path, tc)

        logger.info(
            f"Trainer ready [{self.model_name}] | "
            f"epochs={self.num_epochs}, start_epoch={self.start_epoch}, "
            f"lr={self.lr}, AMP={'ON' if self.use_amp else 'OFF'}, "
            f"TB={'ON' if self.tb.enabled else 'OFF'}"
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_scheduler(self, scheduler_type: str, tc: Dict):
        if scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.num_epochs, eta_min=1e-6
            )
        elif scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=tc.get("lr_step_size", 7),
                gamma=tc.get("lr_gamma", 0.1),
            )
        elif scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=3
            )
        else:
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.num_epochs, eta_min=1e-6
            )

    def _maybe_resume(self, resume_path: Optional[str], tc: Dict):
        """Load checkpoint for resume if configured."""
        should_resume = tc.get("resume", False)
        path = resume_path or tc.get("resume_from")

        # Auto-detect last checkpoint
        if should_resume and path is None:
            if Path(self.last_ckpt_path).exists():
                path = self.last_ckpt_path
            elif Path(self.model_save_path).exists():
                path = self.model_save_path

        if path and Path(path).exists():
            logger.info(f"Resuming training from: {path}")
            try:
                self.model, epoch, val_acc = load_checkpoint(
                    self.model, path,
                    optimizer=self.optimizer,
                    device=self.device,
                )
                self.start_epoch   = epoch + 1
                self.best_val_acc  = val_acc
                logger.info(f"Resumed from epoch {epoch}, best_val_acc={val_acc:.4f}")
            except Exception as e:
                logger.warning(f"Could not resume from {path}: {e}. Starting fresh.")
        elif should_resume:
            logger.info("Resume=true but no checkpoint found — starting from epoch 1.")

    def _warmup_lr(self, epoch: int):
        """Apply linear LR warmup for first warmup_epochs."""
        if epoch <= self.warmup_epochs:
            warmup_factor = epoch / max(self.warmup_epochs, 1)
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.lr * warmup_factor

    def _forward(self, images: torch.Tensor):
        return self.model(images)

    def _compute_loss(self, outputs, labels: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, tuple):
            primary, aux = outputs
            loss = self.criterion(primary, labels)
            if aux is not None:
                loss = loss + 0.4 * self.criterion(aux, labels)
            return loss
        return self.criterion(outputs, labels)

    # ── Training / Validation epochs ──────────────────────────────────────────

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        global_step = (epoch - 1) * len(self.train_loader)

        pbar = tqdm(self.train_loader,
                    desc=f"Train [{self.model_name}] E{epoch}",
                    leave=False, ncols=100)

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with autocast():
                    outputs = self._forward(images)
                    loss    = self._compute_loss(outputs, labels)
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self._forward(images)
                loss    = self._compute_loss(outputs, labels)
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            preds  = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            step     = global_step + batch_idx

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

            # TensorBoard batch-level logging
            if batch_idx % self.log_every_n == 0:
                self.tb.add_scalar(f"{self.model_name}/batch_loss", loss.item(), step)
                self.tb.add_scalar(
                    f"{self.model_name}/lr",
                    self.optimizer.param_groups[0]["lr"],
                    step,
                )

        return total_loss / len(self.train_loader), correct / total

    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(
            self.val_loader,
            desc=f"Val   [{self.model_name}] E{epoch}",
            leave=False, ncols=100,
        ):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.use_amp:
                with autocast():
                    logits = self.model(images)
                    loss   = self.criterion(logits, labels)
            else:
                logits = self.model(images)
                loss   = self.criterion(logits, labels)

            total_loss += loss.item()
            preds  = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        return total_loss / len(self.val_loader), correct / total

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self) -> Dict:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Training: {self.model_name}  (epoch {self.start_epoch}→{self.num_epochs})")
        logger.info(f"{'='*60}\n")

        t_start = time.time()

        for epoch in range(self.start_epoch, self.num_epochs + 1):
            t_epoch = time.time()

            # LR warmup
            if epoch <= self.warmup_epochs:
                self._warmup_lr(epoch)

            train_loss, train_acc = self.train_epoch(epoch)
            val_loss,   val_acc   = self.validate_epoch(epoch)

            # Scheduler step (skip warmup epochs for non-plateau schedulers)
            if epoch > self.warmup_epochs:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # History
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - t_epoch

            logger.info(
                f"Epoch [{epoch:3d}/{self.num_epochs}] | "
                f"Train {train_loss:.4f}/{train_acc:.4f} | "
                f"Val {val_loss:.4f}/{val_acc:.4f} | "
                f"LR {current_lr:.2e} | {format_time(epoch_time)}"
            )

            # TensorBoard epoch-level
            self.tb.add_scalars(f"{self.model_name}/Loss",
                                 {"train": train_loss, "val": val_loss}, epoch)
            self.tb.add_scalars(f"{self.model_name}/Accuracy",
                                 {"train": train_acc, "val": val_acc}, epoch)
            self.tb.add_scalar(f"{self.model_name}/LR", current_lr, epoch)

            # Save best checkpoint
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_acc,
                    self.model_save_path,
                    extra_info={
                        "model_name": self.model_name,
                        "train_loss": train_loss,
                        "val_loss":   val_loss,
                    },
                )
                logger.info(f"  ✓ Best model saved  val_acc={val_acc:.4f}")

            # Always save last checkpoint (for resume)
            save_checkpoint(
                self.model, self.optimizer, epoch, val_acc,
                self.last_ckpt_path,
                extra_info={"model_name": self.model_name, "is_last": True},
            )

            # Early stopping
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping at epoch {epoch}.")
                break

        total_time = time.time() - t_start
        logger.info(
            f"\nDone [{self.model_name}] | best_val_acc={self.best_val_acc:.4f} | "
            f"time={format_time(total_time)}"
        )

        # Save training history plot
        try:
            plot_training_history(
                self.train_losses, self.val_losses,
                self.train_accs,   self.val_accs,
                self.model_name,   save_dir="outputs",
            )
        except Exception as e:
            logger.warning(f"Could not save training plot: {e}")

        self.tb.close()

        return {
            "model_name":     self.model_name,
            "best_val_acc":   self.best_val_acc,
            "epochs_trained": len(self.train_losses),
            "train_losses":   self.train_losses,
            "val_losses":     self.val_losses,
            "train_accs":     self.train_accs,
            "val_accs":       self.val_accs,
        }
