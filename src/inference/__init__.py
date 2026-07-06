"""
src/inference/__init__.py - Inference module exports.
"""

from src.inference.ensemble import EnsemblePredictor, EnsembleMethod, aggregate_frame_predictions
from src.inference.predictor import DeepfakeDetector, predict_frames, load_trained_model

__all__ = [
    "EnsemblePredictor",
    "EnsembleMethod",
    "aggregate_frame_predictions",
    "DeepfakeDetector",
    "predict_frames",
    "load_trained_model",
]
