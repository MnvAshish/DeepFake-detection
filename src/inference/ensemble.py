"""
ensemble.py - Ensemble prediction module for deepfake detection.

Combines predictions from ResNet50, VGG16, and InceptionV3 using:
  1. Soft Voting   - Average of class probabilities (recommended)
  2. Weighted Avg  - Weighted average with per-model weights
  3. Hard Voting   - Majority vote of predicted classes

All methods return a probability for class 1 (fake), which is thresholded
to produce the final binary prediction.

Design rationale:
  - Soft voting preserves confidence calibration better than hard voting
  - Weighted avg is useful when models have different known accuracies
  - Ensemble reduces variance and improves generalization
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnsembleMethod(Enum):
    SOFT_VOTING = "soft_voting"
    WEIGHTED_AVG = "weighted_avg"
    HARD_VOTING = "hard_voting"


class EnsemblePredictor:
    """
    Combines predictions from multiple models using configurable ensemble methods.

    Designed for the deepfake detection pipeline where each model outputs
    class-level probabilities (after softmax).

    Usage:
        ensemble = EnsemblePredictor(
            method="soft_voting",
            weights={"resnet50": 0.4, "vgg16": 0.3, "inceptionv3": 0.3}
        )
        result = ensemble.predict(frame_predictions_per_model)
    """

    def __init__(
        self,
        method: str = "soft_voting",
        weights: Optional[Dict[str, float]] = None,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize the EnsemblePredictor.

        Args:
            method (str): Ensemble strategy. One of:
                          "soft_voting", "weighted_avg", "hard_voting".
            weights (dict): Per-model weights for weighted_avg.
                            Keys must match model names passed to predict().
                            Values must sum to 1.0.
            confidence_threshold (float): Threshold for final classification.
                                          Probability >= threshold → FAKE.
        """
        self.method = EnsembleMethod(method)
        self.weights = weights or {}
        self.confidence_threshold = confidence_threshold

        # Validate weights
        if self.method == EnsembleMethod.WEIGHTED_AVG and self.weights:
            total = sum(self.weights.values())
            if abs(total - 1.0) > 0.01:
                logger.warning(
                    f"Ensemble weights sum to {total:.4f}, expected 1.0. "
                    "Normalizing weights..."
                )
                self.weights = {k: v / total for k, v in self.weights.items()}

        logger.info(
            f"EnsemblePredictor initialized | method={method} | "
            f"threshold={confidence_threshold}"
        )
        if self.weights:
            logger.info(f"Weights: {self.weights}")

    def predict(
        self,
        model_probabilities: Dict[str, np.ndarray],
    ) -> Dict:
        """
        Produce ensemble prediction from per-model frame probabilities.

        Args:
            model_probabilities: Dict mapping model_name → array of shape
                                  (N, 2) where N=number of frames, columns
                                  are [prob_real, prob_fake].
                                  Example:
                                  {
                                    "resnet50":    np.array([[0.8, 0.2], [0.7, 0.3], ...]),
                                    "vgg16":       np.array([[0.9, 0.1], ...]),
                                    "inceptionv3": np.array([[0.85, 0.15], ...]),
                                  }

        Returns:
            dict: {
                "label": "REAL" or "FAKE",
                "label_idx": 0 or 1,
                "confidence": float (confidence in the predicted class),
                "fake_probability": float (overall prob of being fake),
                "real_probability": float (overall prob of being real),
                "frame_level_fake_probs": List[float] per frame,
                "per_model_probs": Dict per model,
                "method": str ensemble method used,
                "num_frames": int total frames used,
            }
        """
        if not model_probabilities:
            raise ValueError("model_probabilities is empty. At least one model required.")

        model_names = list(model_probabilities.keys())
        n_models = len(model_names)

        # Validate and convert to numpy
        probs_arrays = {}
        min_frames = float("inf")
        for name, probs in model_probabilities.items():
            arr = np.array(probs, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 2)
            if arr.shape[1] != 2:
                raise ValueError(
                    f"Model '{name}' probabilities must have shape (N, 2), got {arr.shape}"
                )
            probs_arrays[name] = arr
            min_frames = min(min_frames, arr.shape[0])

        # Aggregate using chosen method
        if self.method == EnsembleMethod.SOFT_VOTING:
            final_probs = self._soft_voting(probs_arrays)
        elif self.method == EnsembleMethod.WEIGHTED_AVG:
            final_probs = self._weighted_average(probs_arrays)
        elif self.method == EnsembleMethod.HARD_VOTING:
            final_probs = self._hard_voting(probs_arrays)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

        # final_probs shape: (N, 2) — averaged across models and frames
        # Aggregate frame-level predictions into video-level
        video_level_probs = final_probs.mean(axis=0)  # (2,)

        fake_prob = float(video_level_probs[1])
        real_prob = float(video_level_probs[0])

        # Threshold-based classification
        label_idx = 1 if fake_prob >= self.confidence_threshold else 0
        label = "FAKE" if label_idx == 1 else "REAL"

        # Confidence = probability of the predicted class
        confidence = fake_prob if label_idx == 1 else real_prob

        # Per-frame fake probabilities (for timeline visualization)
        frame_fake_probs = final_probs[:, 1].tolist()

        # Per-model summary
        per_model_summary = {}
        for name, arr in probs_arrays.items():
            avg_fake_prob = float(arr[:, 1].mean())
            per_model_summary[name] = {
                "fake_probability": avg_fake_prob,
                "real_probability": 1.0 - avg_fake_prob,
                "prediction": "FAKE" if avg_fake_prob >= self.confidence_threshold else "REAL",
            }

        result = {
            "label": label,
            "label_idx": label_idx,
            "confidence": confidence,
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "frame_level_fake_probs": frame_fake_probs,
            "per_model_probs": per_model_summary,
            "method": self.method.value,
            "num_frames": len(frame_fake_probs),
            "num_models": n_models,
        }

        logger.info(
            f"Ensemble result: {label} | "
            f"fake_prob={fake_prob:.4f} | "
            f"confidence={confidence:.4f} | "
            f"frames={len(frame_fake_probs)}"
        )

        return result

    def _soft_voting(self, probs_arrays: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Soft voting: simple mean of all model probability arrays.

        For frames with different counts, we resample to the minimum count.
        Each model's probability output is averaged frame-by-frame.

        Returns:
            np.ndarray: Shape (N, 2) averaged probabilities.
        """
        # Stack all probability arrays along a new axis, then mean
        # Handle different frame counts by aligning lengths
        aligned = self._align_frame_counts(probs_arrays)
        stacked = np.stack(list(aligned.values()), axis=0)  # (num_models, N, 2)
        return stacked.mean(axis=0)  # (N, 2)

    def _weighted_average(self, probs_arrays: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Weighted average ensemble using per-model weights.

        Falls back to equal weights if model name not found in self.weights.

        Returns:
            np.ndarray: Shape (N, 2) weighted probabilities.
        """
        aligned = self._align_frame_counts(probs_arrays)
        model_names = list(aligned.keys())

        # Get weights, default to 1.0 if not specified
        raw_weights = [self.weights.get(name, 1.0) for name in model_names]
        total_weight = sum(raw_weights)
        normalized_weights = [w / total_weight for w in raw_weights]

        weighted_sum = None
        for name, weight in zip(model_names, normalized_weights):
            contribution = aligned[name] * weight
            if weighted_sum is None:
                weighted_sum = contribution
            else:
                weighted_sum += contribution

        return weighted_sum

    def _hard_voting(self, probs_arrays: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Hard voting: majority class vote across models per frame.

        Converts each model's prediction to one-hot, then averages.

        Returns:
            np.ndarray: Shape (N, 2) vote fractions.
        """
        aligned = self._align_frame_counts(probs_arrays)
        votes = []
        for arr in aligned.values():
            # Convert to one-hot predictions
            predicted_classes = arr.argmax(axis=1)  # (N,)
            one_hot = np.zeros_like(arr)
            for i, cls in enumerate(predicted_classes):
                one_hot[i, cls] = 1.0
            votes.append(one_hot)

        # Average one-hot votes = fraction of models voting for each class
        stacked = np.stack(votes, axis=0)
        return stacked.mean(axis=0)

    @staticmethod
    def _align_frame_counts(probs_arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Ensure all arrays have the same number of frames by trimming to minimum.

        This handles cases where different models process different numbers
        of frames (e.g., InceptionV3 at 299x299 might process fewer in a batch).
        """
        if len(probs_arrays) <= 1:
            return probs_arrays

        min_frames = min(arr.shape[0] for arr in probs_arrays.values())

        if min_frames == 0:
            raise ValueError("One or more models produced 0 frame predictions.")

        return {name: arr[:min_frames] for name, arr in probs_arrays.items()}


def aggregate_frame_predictions(
    frame_probs: List[float],
    method: str = "mean",
    top_k: Optional[int] = None,
) -> float:
    """
    Aggregate a list of per-frame fake probabilities into a single video score.

    Different aggregation methods handle different scenarios:
    - mean: Best for videos where most frames may be fake
    - max: Conservative (if any frame looks fake, mark as fake)
    - median: Robust to outlier frames
    - top_k_mean: Average only the most suspicious frames

    Args:
        frame_probs: List of per-frame fake probability scores.
        method: "mean", "max", "median", or "top_k_mean".
        top_k: Number of top frames for top_k_mean (default: len//2).

    Returns:
        float: Video-level fake probability.
    """
    if not frame_probs:
        raise ValueError("frame_probs is empty.")

    probs = np.array(frame_probs)

    if method == "mean":
        return float(probs.mean())
    elif method == "max":
        return float(probs.max())
    elif method == "median":
        return float(np.median(probs))
    elif method == "top_k_mean":
        k = top_k or max(1, len(probs) // 2)
        top_probs = np.sort(probs)[-k:]
        return float(top_probs.mean())
    else:
        raise ValueError(f"Unknown aggregation method: '{method}'")
