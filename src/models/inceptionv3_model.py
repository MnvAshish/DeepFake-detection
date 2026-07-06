"""
inceptionv3_model.py - InceptionV3 with transfer learning for deepfake detection.

Architecture:
  - Pretrained InceptionV3 backbone (ImageNet weights)
  - Replaced auxiliary and final FC heads
  - Uses 299x299 input (unlike ResNet/VGG which use 224x224)
  - Handles auxiliary output during training (important for convergence)

InceptionV3 uses parallel convolutional pathways (Inception modules)
that capture features at multiple scales simultaneously.
Key properties:
  - ~24M parameters
  - Requires 299x299 input
  - Very efficient at capturing multi-scale spatial features
  - Auxiliary classifier helps gradient flow to earlier layers
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import Inception_V3_Weights

from src.utils.logger import get_logger

logger = get_logger(__name__)


class InceptionV3Classifier(nn.Module):
    """
    InceptionV3-based binary classifier for deepfake detection.

    IMPORTANT: InceptionV3 requires 299x299 input images (not 224x224).
    It also returns an auxiliary output during training, which must be
    handled in the training loop.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_backbone: bool = False,
    ):
        """
        Initialize InceptionV3Classifier.

        Args:
            num_classes (int): Output classes.
            pretrained (bool): Use ImageNet pretrained weights.
            dropout (float): Dropout in classification head.
            freeze_backbone (bool): Freeze all layers except the final head.
        """
        super(InceptionV3Classifier, self).__init__()

        self.num_classes = num_classes
        self.model_name = "InceptionV3"
        self.input_size = 299  # InceptionV3 requires 299x299

        # Load pretrained InceptionV3
        if pretrained:
            logger.info("Loading InceptionV3 with pretrained ImageNet weights...")
            weights = Inception_V3_Weights.IMAGENET1K_V1
            backbone = models.inception_v3(weights=weights, aux_logits=True)
        else:
            logger.info("Initializing InceptionV3 with random weights...")
            backbone = models.inception_v3(weights=None, aux_logits=True)

        # Replace the primary classification head
        # Original: Linear(2048, 1000)  → New: custom head
        in_features = backbone.fc.in_features  # 2048

        # Primary classification head
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

        # Replace auxiliary classifier head (used only during training)
        if backbone.AuxLogits is not None:
            aux_in_features = backbone.AuxLogits.fc.in_features  # 768
            backbone.AuxLogits.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(aux_in_features, num_classes),
            )

        self.model = backbone

        # Optional: freeze backbone
        if freeze_backbone:
            self._freeze_backbone()

        # Initialize custom head weights
        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"InceptionV3Classifier ready | "
            f"Total params: {total_params:,} | "
            f"Trainable: {trainable_params:,} | "
            f"Input size: {self.input_size}x{self.input_size}"
        )

    def _freeze_backbone(self) -> None:
        """Freeze all backbone layers except the custom heads."""
        for name, param in self.model.named_parameters():
            if "fc" not in name and "AuxLogits" not in name:
                param.requires_grad = False
        logger.info("InceptionV3 backbone frozen. Only classification heads will train.")

    def _unfreeze_all(self) -> None:
        """Unfreeze everything for full fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("All InceptionV3 layers unfrozen.")

    def _init_weights(self) -> None:
        """Initialize custom classification head weights."""
        for module in self.model.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        During TRAINING: returns (primary_logits, aux_logits) tuple.
        During EVAL:     returns primary_logits only.

        Args:
            x (Tensor): Input images of shape (batch, 3, 299, 299).

        Returns:
            Training: Tuple[Tensor, Tensor] — (primary_logits, aux_logits)
            Inference: Tensor — primary_logits of shape (batch, num_classes)
        """
        if self.training:
            # InceptionV3 returns InceptionOutputs namedtuple during training
            outputs = self.model(x)
            if hasattr(outputs, "logits"):
                primary = outputs.logits
                aux = outputs.aux_logits
            else:
                primary, aux = outputs, None
            return primary, aux
        else:
            # During eval, it returns just the logits tensor
            outputs = self.model(x)
            if hasattr(outputs, "logits"):
                return outputs.logits
            return outputs


def build_inceptionv3(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.5,
    freeze_backbone: bool = False,
) -> InceptionV3Classifier:
    """
    Factory function to create an InceptionV3Classifier instance.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Use ImageNet pretrained weights.
        dropout (float): Dropout probability.
        freeze_backbone (bool): Freeze feature layers.

    Returns:
        InceptionV3Classifier: Configured model instance.
    """
    return InceptionV3Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
