"""
gradcam.py - Gradient-weighted Class Activation Mapping (Grad-CAM) visualization.

Grad-CAM produces heatmaps that highlight which regions of an input face image
most influenced the model's deepfake prediction. This helps interpret and
debug model decisions.

How Grad-CAM works:
  1. Run forward pass to get prediction
  2. Backpropagate gradient of the target class score to the last conv layer
  3. Global average pool the gradients → per-channel weights
  4. Weighted sum of activation maps → coarse heatmap
  5. ReLU + normalize → overlay on input image

Supports: ResNet50, VGG16 (last conv layers are auto-detected).
InceptionV3 support is partial (uses last Inception block).

Usage:
    from src.utils.gradcam import GradCAM, visualize_gradcam
    cam = GradCAM(model, target_layer_name="layer4")
    heatmap = cam.generate(image_tensor, target_class=1)
    vis = visualize_gradcam(image_np, heatmap)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Target Layer Auto-detection
# ─────────────────────────────────────────────────────────────────────────────

# Map model class names to the layer name most suitable for Grad-CAM
# (typically the last convolutional block before global pooling)
MODEL_TARGET_LAYERS = {
    "ResNet50Classifier": "features.7",        # layer4 (last residual block)
    "VGG16Classifier":    "features.28",       # last conv layer (relu after conv 5-3)
    "InceptionV3Classifier": "model.Mixed_7c", # last Inception block
}


def get_target_layer(model: nn.Module, layer_name: Optional[str] = None) -> Optional[nn.Module]:
    """
    Retrieve a named submodule from a model by dot-separated path.

    Args:
        model (nn.Module): The model to search.
        layer_name (str): Dot-separated layer path, e.g. "features.7".
                          If None, auto-detects from model class name.

    Returns:
        nn.Module or None if not found.
    """
    if layer_name is None:
        model_class = type(model).__name__
        layer_name = MODEL_TARGET_LAYERS.get(model_class)
        if layer_name is None:
            logger.warning(
                f"No default Grad-CAM layer for model class '{model_class}'. "
                "Please specify layer_name explicitly."
            )
            return None
        logger.info(f"Auto-detected Grad-CAM layer for {model_class}: '{layer_name}'")

    # Traverse the model's module tree using the dot path
    try:
        parts = layer_name.split(".")
        module = model
        for part in parts:
            if part.isdigit():
                module = list(module.children())[int(part)]
            else:
                module = getattr(module, part)
        return module
    except (AttributeError, IndexError, StopIteration) as e:
        logger.error(f"Could not find layer '{layer_name}' in model: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GradCAM Class
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Registers forward and backward hooks on a target convolutional layer
    to capture activations and gradients for visualization.

    Reference:
        Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks
        via Gradient-based Localization." ICCV 2017.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer_name: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize GradCAM.

        Args:
            model (nn.Module): The model to visualize (must be in eval mode).
            target_layer_name (str): Dot-separated path to the target conv layer.
                                      Auto-detected if None.
            device: Compute device.
        """
        self.model = model
        self.device = device or torch.device("cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        # Internal storage for hooks
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks: List = []

        # Find target layer
        self.target_layer = get_target_layer(model, target_layer_name)
        if self.target_layer is None:
            raise ValueError(
                f"Could not locate target layer '{target_layer_name}'. "
                "Check the layer path using: dict(model.named_modules())"
            )

        # Register hooks
        self._register_hooks()

        layer_id = target_layer_name or MODEL_TARGET_LAYERS.get(type(model).__name__, "?")
        logger.info(f"GradCAM initialized on layer: '{layer_id}'")

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""

        def forward_hook(module, input, output):
            """Capture feature maps from forward pass."""
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            """Capture gradients from backward pass."""
            self._gradients = grad_output[0].detach()

        fwd_handle = self.target_layer.register_forward_hook(forward_hook)
        bwd_handle = self.target_layer.register_full_backward_hook(backward_hook)

        self._hooks = [fwd_handle, bwd_handle]

    def remove_hooks(self) -> None:
        """Remove all registered hooks (call when done to free resources)."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate a Grad-CAM heatmap for the given input image tensor.

        Args:
            input_tensor (Tensor): Shape (1, C, H, W) — a single image.
                                    Must already be normalized.
            target_class (int): Class index for which to generate CAM.
                                  If None, uses the predicted class.

        Returns:
            Tuple:
              - heatmap (np.ndarray): Normalized heatmap, shape (H, W), values [0, 1].
              - predicted_class (int): The class index used for visualization.
              - confidence (float): Softmax confidence of that class.
        """
        assert input_tensor.ndim == 4, "Input must be (1, C, H, W)"
        assert input_tensor.shape[0] == 1, "Batch size must be 1 for Grad-CAM"

        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(False)

        # Clear any previous activations/gradients
        self._activations = None
        self._gradients = None

        # ── Forward pass ──────────────────────────────────────────────────────
        self.model.zero_grad()

        logits = self.model(input_tensor)
        # Handle InceptionV3 tuple output in eval mode (should return tensor)
        if isinstance(logits, tuple):
            logits = logits[0]

        probs = F.softmax(logits, dim=1)
        predicted_class = int(probs.argmax(dim=1).item())
        confidence = float(probs[0, predicted_class].item())

        if target_class is None:
            target_class = predicted_class

        # ── Backward pass ─────────────────────────────────────────────────────
        # Backpropagate the score for the target class
        self.model.zero_grad()
        class_score = logits[0, target_class]
        class_score.backward(retain_graph=False)

        # ── Grad-CAM computation ──────────────────────────────────────────────
        if self._gradients is None or self._activations is None:
            logger.error("Hooks did not capture gradients/activations. "
                         "Check that the target layer has conv output.")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3])), predicted_class, confidence

        # Global average pooling of gradients → alpha weights
        # gradients shape: (1, C, H', W')
        gradients = self._gradients  # (1, C, H', W')
        activations = self._activations  # (1, C, H', W')

        # Alpha = mean of gradients over spatial dimensions
        alpha = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (alpha * activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')

        # ReLU: only keep positive contributions
        cam = F.relu(cam)

        # Resize to input image size
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()  # (H, W)

        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, predicted_class, confidence

    def __del__(self):
        """Clean up hooks on deletion."""
        self.remove_hooks()


# ─────────────────────────────────────────────────────────────────────────────
# Visualization Helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_heatmap_overlay(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on top of the original image.

    Args:
        image_rgb (np.ndarray): Original RGB image (H, W, 3), values [0, 255].
        heatmap (np.ndarray): Grad-CAM heatmap (H, W), values [0, 1].
        colormap (int): OpenCV colormap for the heatmap (default: JET).
        alpha (float): Blend factor: 0 = pure image, 1 = pure heatmap.

    Returns:
        np.ndarray: Blended RGB overlay image (H, W, 3), values [0, 255].
    """
    # Convert heatmap to uint8 and apply colormap
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    # Resize heatmap to match image if needed
    if colored_heatmap_rgb.shape[:2] != image_rgb.shape[:2]:
        h, w = image_rgb.shape[:2]
        colored_heatmap_rgb = cv2.resize(colored_heatmap_rgb, (w, h))

    # Blend with original image
    image_float = image_rgb.astype(np.float32)
    heatmap_float = colored_heatmap_rgb.astype(np.float32)
    overlay = (1 - alpha) * image_float + alpha * heatmap_float
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay


def visualize_gradcam(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    title: str = "",
    predicted_class: int = 0,
    confidence: float = 0.0,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Create a 3-panel visualization: original | heatmap | overlay.

    Args:
        image_rgb (np.ndarray): Original RGB image (H, W, 3).
        heatmap (np.ndarray): Grad-CAM heatmap (H, W), values [0, 1].
        title (str): Figure title.
        predicted_class (int): 0=real, 1=fake.
        confidence (float): Confidence score.
        save_path (str): If provided, save figure to this path.

    Returns:
        np.ndarray: The overlay image as numpy array.
    """
    class_name = "FAKE" if predicted_class == 1 else "REAL"
    overlay = apply_heatmap_overlay(image_rgb, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    title_str = title or f"Grad-CAM Visualization — Predicted: {class_name} ({confidence:.1%})"
    fig.suptitle(title_str, fontsize=12, fontweight="bold")

    # Panel 1: Original image
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Face Crop", fontsize=10)
    axes[0].axis("off")

    # Panel 2: Raw heatmap
    heatmap_plot = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap\n(Red = most influential)", fontsize=10)
    axes[1].axis("off")
    plt.colorbar(heatmap_plot, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Overlay
    axes[2].imshow(overlay)
    color = "#FF4B4B" if predicted_class == 1 else "#00C853"
    axes[2].set_title(
        f"Overlay\nPred: {class_name} ({confidence:.1%})",
        fontsize=10,
        color=color,
        fontweight="bold",
    )
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Grad-CAM visualization saved: {save_path}")

    plt.close()
    return overlay


def generate_gradcam_for_video(
    model: nn.Module,
    face_frames: List[np.ndarray],
    target_layer_name: Optional[str] = None,
    device: Optional[torch.device] = None,
    n_frames: int = 4,
    image_size: int = 224,
    save_path: Optional[str] = None,
) -> List[np.ndarray]:
    """
    Generate Grad-CAM visualizations for n_frames from a video's face crops.

    Args:
        model (nn.Module): Trained model.
        face_frames (list): List of RGB face crop arrays from extract_faces.
        target_layer_name (str): Target layer for Grad-CAM.
        device: Compute device.
        n_frames (int): How many frames to visualize.
        image_size (int): Model input size.
        save_path (str): If provided, save combined grid to this path.

    Returns:
        List[np.ndarray]: Overlay images for each selected frame.
    """
    from torchvision import transforms as T

    if device is None:
        device = torch.device("cpu")

    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Select evenly-spaced frames
    if len(face_frames) > n_frames:
        indices = np.linspace(0, len(face_frames) - 1, n_frames, dtype=int)
        selected_frames = [face_frames[i] for i in indices]
    else:
        selected_frames = face_frames[:n_frames]

    cam = GradCAM(model, target_layer_name=target_layer_name, device=device)
    overlays = []

    for frame_idx, frame in enumerate(selected_frames):
        try:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(frame.astype(np.uint8))
            tensor = transform(pil_img).unsqueeze(0)

            heatmap, pred_class, conf = cam.generate(tensor, target_class=None)

            # Resize frame for display
            display_frame = cv2.resize(frame, (image_size, image_size))
            overlay = apply_heatmap_overlay(display_frame, heatmap)
            overlays.append(overlay)

        except Exception as e:
            logger.error(f"Grad-CAM failed for frame {frame_idx}: {e}")
            continue

    cam.remove_hooks()

    # Save combined grid if requested
    if save_path and overlays:
        _save_gradcam_grid(overlays, selected_frames, save_path, image_size)

    return overlays


def _save_gradcam_grid(
    overlays: List[np.ndarray],
    original_frames: List[np.ndarray],
    save_path: str,
    image_size: int = 224,
) -> None:
    """Save a grid of original + overlay pairs."""
    n = len(overlays)
    if n == 0:
        return

    fig, axes = plt.subplots(2, n, figsize=(n * 3, 6))
    if n == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle("Grad-CAM Frame Analysis", fontsize=12, fontweight="bold")

    for col in range(n):
        # Top row: original
        orig = cv2.resize(original_frames[col], (image_size, image_size))
        axes[0, col].imshow(orig)
        axes[0, col].set_title(f"Frame {col+1}\n(Original)", fontsize=8)
        axes[0, col].axis("off")

        # Bottom row: overlay
        axes[1, col].imshow(overlays[col])
        axes[1, col].set_title("Grad-CAM", fontsize=8)
        axes[1, col].axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Grad-CAM grid saved: {save_path}")
