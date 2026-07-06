"""
resnet50_model.py - ResNet50 with transfer learning for deepfake detection.

Architecture:
  - Pretrained ResNet50 backbone (ImageNet weights)
  - Replaced FC head: GlobalAvgPool → Dropout → Linear(2048, 512) → ReLU → Dropout → Linear(512, 2)
  - Optional layer freezing for fine-tuning strategies

ResNet50 is a 50-layer residual network. The residual (skip) connections
allow gradients to flow more easily during backpropagation. Key properties:
  - ~25M parameters
  - Strong performance on image classification tasks
  - Good balance of speed and accuracy
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResNet50Classifier(nn.Module):
    """
    ResNet50-based binary classifier for deepfake detection.

    The final fully-connected layer is replaced with a custom head
    that outputs 2-class logits (real=0, fake=1).
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_backbone: bool = False,
    ):
        """
        Initialize ResNet50Classifier.

        Args:
            num_classes (int): Output classes (2 for binary deepfake detection).
            pretrained (bool): Load ImageNet pretrained weights.
            dropout (float): Dropout probability in classification head.
            freeze_backbone (bool): If True, freeze all backbone layers
                                     and only train the classification head.
        """
        super(ResNet50Classifier, self).__init__()

        self.num_classes = num_classes
        self.model_name = "ResNet50"

        # Load pretrained ResNet50 backbone
        if pretrained:
            logger.info("Loading ResNet50 with pretrained ImageNet weights...")
            weights = ResNet50_Weights.IMAGENET1K_V2
            backbone = models.resnet50(weights=weights)
        else:
            logger.info("Initializing ResNet50 with random weights...")
            backbone = models.resnet50(weights=None)

        # Extract feature layers (everything except the final FC)
        # ResNet50 output: (batch, 2048) after global avg pool
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        # ^ This gives: conv1 → bn1 → relu → maxpool → layer1 → layer2 → layer3 → layer4 → avgpool

        in_features = backbone.fc.in_features  # 2048 for ResNet50

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),                          # (batch, 2048, 1, 1) → (batch, 2048)
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

        # Optional: Freeze backbone layers
        if freeze_backbone:
            self._freeze_backbone()

        # Initialize custom head weights
        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"ResNet50Classifier ready | "
            f"Total params: {total_params:,} | "
            f"Trainable: {trainable_params:,}"
        )

    def _freeze_backbone(self) -> None:
        """Freeze all backbone layers (only train the classifier head)."""
        for param in self.features.parameters():
            param.requires_grad = False
        logger.info("ResNet50 backbone frozen. Only classifier head will be trained.")

    def _unfreeze_all(self) -> None:
        """Unfreeze all layers for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All ResNet50 layers unfrozen for fine-tuning.")

    def _init_weights(self) -> None:
        """Initialize custom classifier head weights using Kaiming initialization."""
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
        # Extract features through backbone
        features = self.features(x)
        # features shape: (batch, 2048, 1, 1)

        # Classify
        logits = self.classifier(features)
        # logits shape: (batch, 2)

        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings (before the final classification layer).

        Useful for visualization (t-SNE, UMAP) or transfer learning.

        Args:
            x (Tensor): Input images of shape (batch, 3, 224, 224).

        Returns:
            Tensor: Feature embeddings of shape (batch, 512).
        """
        features = self.features(x)
        # Pass through all but the last linear layer
        embeddings = self.classifier[:-1](features)
        return embeddings


def build_resnet50(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.5,
    freeze_backbone: bool = False,
) -> ResNet50Classifier:
    """
    Factory function to create a ResNet50Classifier instance.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Use ImageNet pretrained weights.
        dropout (float): Dropout probability.
        freeze_backbone (bool): Freeze feature layers.

    Returns:
        ResNet50Classifier: Configured model instance.
    """
    return ResNet50Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
