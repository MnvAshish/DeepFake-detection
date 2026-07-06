"""
predictor.py — Inference engine supporting both IMAGE and VIDEO modes.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.inference.ensemble import EnsemblePredictor
from src.models import build_model
from src.preprocessing.frame_extractor import (
    FaceExtractor, extract_raw_frames, preprocess_image_for_inference
)
from src.utils.config_loader import load_config
from src.utils.helpers import get_device, load_checkpoint, format_time
from src.utils.logger import get_logger

logger = get_logger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_inference_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_trained_model(
    model_key: str,
    checkpoint_path: str,
    num_classes: int = 2,
    device: Optional[torch.device] = None,
) -> nn.Module:
    if device is None:
        device = get_device()
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train with: python src/training/train_pipeline.py"
        )
    logger.info(f"Loading {model_key.upper()} from {checkpoint_path}")
    model = build_model(model_key, num_classes=num_classes, pretrained=False, dropout=0.0)
    model, _, _ = load_checkpoint(model, checkpoint_path, device=device)
    return model.to(device).eval()


@torch.no_grad()
def predict_frames(
    model: nn.Module,
    frames: List[np.ndarray],
    device: torch.device,
    batch_size: int = 16,
    image_size: int = 224,
) -> np.ndarray:
    if not frames:
        raise ValueError("frames list is empty.")
    model.eval()
    transform = get_inference_transform(image_size)
    all_probs = []
    for i in range(0, len(frames), batch_size):
        batch = []
        for fr in frames[i:i + batch_size]:
            if fr is None:
                continue
            batch.append(transform(Image.fromarray(fr.astype(np.uint8))))
        if not batch:
            continue
        t = torch.stack(batch).to(device, non_blocking=True)
        probs = F.softmax(model(t), dim=1)
        all_probs.append(probs.cpu().numpy())
    if not all_probs:
        raise RuntimeError("Inference produced no outputs.")
    return np.concatenate(all_probs, axis=0)


class DeepfakeDetector:
    """
    Main inference class — supports IMAGE mode and VIDEO mode.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config  = load_config(config_path)
        self.device  = get_device(self.config["inference"].get("device", "auto"))
        self.models: Dict[str, nn.Module] = {}
        self.model_loaded: Dict[str, bool] = {}

        fe = self.config["frame_extraction"]
        self.face_extractor = FaceExtractor(
            frame_interval=fe["frame_interval"],
            max_frames=fe["max_frames_per_video"],
            min_frames=fe["min_frames"],
            face_margin=fe["face_margin"],
            min_face_size=fe["min_face_size"],
        )

        ens = self.config["ensemble"]
        self.ensemble = EnsemblePredictor(
            method=ens["method"],
            weights=ens.get("weights", {}),
            confidence_threshold=ens["confidence_threshold"],
        )

        self._load_models()

    def _load_models(self):
        for name, size_key in [
            ("resnet50", "image_size"),
            ("vgg16",    "image_size"),
            ("inceptionv3", "inception_size"),
        ]:
            ckpt = self.config["models"][name]["save_path"]
            try:
                self.models[name]       = load_trained_model(name, ckpt,
                                            self.config["data"]["num_classes"],
                                            self.device)
                self.model_loaded[name] = True
                logger.info(f"✓ {name.upper()} loaded")
            except FileNotFoundError as e:
                logger.warning(f"⚠ {name}: {e}")
                self.model_loaded[name] = False

        if not any(self.model_loaded.values()):
            raise RuntimeError(
                "No models loaded. Train first:\n"
                "  python src/training/train_pipeline.py"
            )

    @property
    def available_models(self) -> List[str]:
        return [n for n, v in self.model_loaded.items() if v]

    # ── IMAGE MODE ────────────────────────────────────────────────────────────

    def predict_image(
        self,
        image_input,
        return_gradcam: bool = True,
        progress_callback=None,
    ) -> Dict:
        """
        Predict a single image (file path, PIL Image, or numpy RGB array).

        Returns full result dict including per-model scores, ensemble result,
        and optional Grad-CAM heatmaps per model.
        """
        start = time.time()

        def _prog(f, msg):
            if progress_callback:
                progress_callback(f, msg)

        _prog(0.05, "Detecting face in image...")

        # Preprocess: detect face, return list of crops
        cfg_img  = self.config["inference"].get("image_mode", {})
        do_face  = cfg_img.get("face_detection", True)
        img_size_224 = self.config["data"]["image_size"]
        img_size_299 = self.config["data"]["inception_size"]

        crops_224, face_detected = preprocess_image_for_inference(
            image_input, target_size=img_size_224,
            face_margin=self.config["frame_extraction"].get("image_face_margin", 0.2),
            fallback_full_image=True,
        )
        crops_299, _ = preprocess_image_for_inference(
            image_input, target_size=img_size_299,
            face_margin=self.config["frame_extraction"].get("image_face_margin", 0.2),
            fallback_full_image=True,
        )

        if not face_detected:
            logger.warning("No face detected in image — using full image.")

        _prog(0.2, "Running model inference...")

        model_sizes  = {"resnet50": img_size_224, "vgg16": img_size_224, "inceptionv3": img_size_299}
        crops_by_size = {img_size_224: crops_224, img_size_299: crops_299}

        model_probs: Dict[str, np.ndarray] = {}
        batch_size = self.config["inference"]["batch_size"]

        for i, name in enumerate(self.available_models):
            _prog(0.2 + 0.5 * (i / max(len(self.available_models), 1)),
                  f"Running {name.upper()}...")
            sz    = model_sizes.get(name, img_size_224)
            crops = crops_by_size.get(sz, crops_224)
            try:
                probs = predict_frames(self.models[name], crops, self.device, batch_size, sz)
                model_probs[name] = probs
            except Exception as e:
                logger.error(f"{name} failed: {e}")

        if not model_probs:
            raise RuntimeError("All models failed on this image.")

        _prog(0.75, "Computing ensemble...")
        result = self.ensemble.predict(model_probs)

        # Grad-CAM per model
        gradcam_maps: Dict[str, np.ndarray] = {}
        if return_gradcam and cfg_img.get("return_gradcam", True):
            _prog(0.80, "Generating Grad-CAM heatmaps...")
            display_crop = crops_224[0] if crops_224 else None
            if display_crop is not None:
                from src.utils.gradcam import GradCAM, apply_heatmap_overlay
                from src.utils.gradcam import MODEL_TARGET_LAYERS
                for name in self.available_models:
                    sz    = model_sizes.get(name, img_size_224)
                    crops = crops_by_size.get(sz, crops_224)
                    layer = MODEL_TARGET_LAYERS.get(
                        type(self.models[name]).__name__, None
                    )
                    try:
                        cam_obj = GradCAM(self.models[name], layer, self.device)
                        tf = get_inference_transform(sz)
                        t  = tf(Image.fromarray(crops[0].astype(np.uint8))).unsqueeze(0)
                        hmap, pred_cls, conf = cam_obj.generate(t)
                        cam_obj.remove_hooks()
                        # Resize heatmap to 224 for display
                        hmap_resized = cv2.resize(hmap, (img_size_224, img_size_224))
                        overlay = apply_heatmap_overlay(display_crop, hmap_resized)
                        gradcam_maps[name] = {
                            "heatmap":  hmap_resized,
                            "overlay":  overlay,
                            "pred_cls": pred_cls,
                            "conf":     conf,
                        }
                    except Exception as e:
                        logger.warning(f"Grad-CAM failed for {name}: {e}")

        result["mode"]           = "image"
        result["face_detected"]  = face_detected
        result["face_crop"]      = crops_224[0] if crops_224 else None
        result["gradcam_maps"]   = gradcam_maps
        result["processing_time"] = time.time() - start
        result["processing_time_str"] = format_time(result["processing_time"])

        _prog(1.0, "Done!")
        return result

    # ── VIDEO MODE ────────────────────────────────────────────────────────────

    def predict_video(
        self,
        video_path: str,
        progress_callback=None,
    ) -> Dict:
        start = time.time()

        def _prog(f, msg):
            if progress_callback:
                progress_callback(f, msg)

        _prog(0.05, "Extracting frames and detecting faces...")

        try:
            face_frames = self.face_extractor.extract_faces_from_video(str(video_path))
        except Exception as e:
            logger.error(f"Face extraction error: {e}")
            face_frames = []

        if len(face_frames) < self.config["frame_extraction"]["min_frames"]:
            logger.warning("Too few faces — falling back to raw frames.")
            _prog(0.15, "No faces — using raw frames...")
            try:
                face_frames = extract_raw_frames(
                    str(video_path),
                    frame_interval=self.config["frame_extraction"]["frame_interval"],
                    max_frames=self.config["frame_extraction"]["max_frames_per_video"],
                    target_size=(224, 224),
                )
            except Exception as e:
                raise RuntimeError(f"Frame extraction failed: {e}")

        if not face_frames:
            raise RuntimeError("No frames could be extracted from video.")

        _prog(0.2, f"Extracted {len(face_frames)} frames. Running models...")

        model_sizes = {
            "resnet50":    self.config["data"]["image_size"],
            "vgg16":       self.config["data"]["image_size"],
            "inceptionv3": self.config["data"]["inception_size"],
        }
        model_probs: Dict[str, np.ndarray] = {}
        batch_size = self.config["inference"]["batch_size"]

        for i, name in enumerate(self.available_models):
            _prog(0.2 + 0.6 * (i / max(len(self.available_models), 1)),
                  f"Running {name.upper()}...")
            sz = model_sizes.get(name, 224)
            try:
                probs = predict_frames(self.models[name], face_frames, self.device, batch_size, sz)
                model_probs[name] = probs
            except Exception as e:
                logger.error(f"{name} failed: {e}")

        if not model_probs:
            raise RuntimeError("All model inferences failed.")

        _prog(0.85, "Computing ensemble...")
        result = self.ensemble.predict(model_probs)
        result["mode"]               = "video"
        result["video_path"]         = str(video_path)
        result["video_name"]         = Path(video_path).name
        result["processing_time"]    = time.time() - start
        result["processing_time_str"]= format_time(result["processing_time"])
        _prog(1.0, "Done!")
        return result

    def predict_video_from_frames(self, frames: List[np.ndarray]) -> Dict:
        model_sizes = {
            "resnet50": self.config["data"]["image_size"],
            "vgg16":    self.config["data"]["image_size"],
            "inceptionv3": self.config["data"]["inception_size"],
        }
        model_probs = {}
        for name in self.available_models:
            sz = model_sizes.get(name, 224)
            try:
                model_probs[name] = predict_frames(
                    self.models[name], frames, self.device,
                    self.config["inference"]["batch_size"], sz,
                )
            except Exception as e:
                logger.error(f"{name}: {e}")
        if not model_probs:
            raise RuntimeError("All inferences failed.")
        return self.ensemble.predict(model_probs)

    def close(self):
        self.face_extractor.close()

    def __enter__(self): return self
    def __exit__(self, *a): self.close()
