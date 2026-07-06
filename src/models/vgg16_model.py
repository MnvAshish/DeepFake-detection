"""
vgg16_model.py - VGG16 with transfer learning for deepfake detection.

Architecture:
  - Pretrained VGG16 backbone (ImageNet weights)
  - Replaced FC classifier: Linear(25088, 4096) → ReLU → Dropout →
                            Linear(4096, 1024) → ReLU → Dropout → Linear(1024, 2)
  - Option to freeze conv feature layers

VGG16 is a 16-layer deep network using 3x3 conv filters throughout.
Key properties:
  - ~138M parameters (much larger than ResNet50)
  - High accuracy but computationally expensive
  - Strong texture features — valuable for detecting face blending artifacts
  - Simpler architecture, easier to understand
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VGG16Classifier(nn.Module):
    """
    VGG16-based binary classifier for deepfake detection.

    The original VGG16 classifier (3 FC layers) is replaced with a
    custom head better suited for binary classification.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_backbone: bool = False,
    ):
        """
        Initialize VGG16Classifier.

        Args:
            num_classes (int): Output classes (2 for binary detection).
            pretrained (bool): Load ImageNet pretrained weights.
            dropout (float): Dropout probability in the classification head.
            freeze_backbone (bool): If True, freeze convolutional layers.
        """
        super(VGG16Classifier, self).__init__()

        self.num_classes = num_classes
        self.model_name = "VGG16"

        # Load pretrained VGG16
        if pretrained:
            logger.info("Loading VGG16 with pretrained ImageNet weights...")
            weights = VGG16_Weights.IMAGENET1K_V1
            backbone = models.vgg16(weights=weights)
        else:
            logger.info("Initializing VGG16 with random weights...")
            backbone = models.vgg16(weights=None)

        # Keep the convolutional feature extractor (features + avgpool)
        # VGG16 conv output: (batch, 512, 7, 7) → after avgpool: (batch, 512, 7, 7)
        self.features = backbone.features
        self.avgpool = backbone.avgpool  # Adaptive avg pooling to (7, 7)

        # VGG16 flattened feature size: 512 * 7 * 7 = 25088
        vgg_flat_size = 512 * 7 * 7  # = 25088

        # Custom classification head (lighter than original VGG classifier)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(vgg_flat_size, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(1024, num_classes),
        )

        # Optional: Freeze convolutional backbone
        if freeze_backbone:
            self._freeze_backbone()

        # Initialize custom head weights
        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"VGG16Classifier ready | "
            f"Total params: {total_params:,} | "
            f"Trainable: {trainable_params:,}"
        )

    def _freeze_backbone(self) -> None:
        """Freeze all VGG16 convolutional layers."""
        for param in self.features.parameters():
            param.requires_grad = False
        logger.info("VGG16 convolutional backbone frozen.")

    def _unfreeze_all(self) -> None:
        """Unfreeze everything for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All VGG16 layers unfrozen for fine-tuning.")

    def _init_weights(self) -> None:
        """Initialize custom classifier head using Kaiming init."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): Input images of shape (batch, 3, 224, 224).

        Returns:
            Tensor: Class logits of shape (batch, num_classes).
        """
        # Convolutional feature extraction
        x = self.features(x)       # (batch, 512, 7, 7)
        x = self.avgpool(x)         # (batch, 512, 7, 7)

        # Classification head
        logits = self.classifier(x)  # (batch, num_classes)

        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 1024-dimensional feature embeddings.

        Args:
            x (Tensor): Input images of shape (batch, 3, 224, 224).

        Returns:
            Tensor: Embeddings of shape (batch, 1024).
        """
        x = self.features(x)
        x = self.avgpool(x)
        # Pass through all but the last linear layer
        for layer in self.classifier[:-1]:
            x = layer(x)
        return x


def build_vgg16(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.5,
    freeze_backbone: bool = False,
) -> VGG16Classifier:
    """
    Factory function to create a VGG16Classifier instance.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Use ImageNet pretrained weights.
        dropout (float): Dropout probability.
        freeze_backbone (bool): Freeze feature layers.

    Returns:
        VGG16Classifier: Configured model instance.
    """
    return VGG16Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
