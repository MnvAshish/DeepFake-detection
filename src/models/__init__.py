"""
src/models/__init__.py - Model module exports.
"""

from src.models.resnet50_model import ResNet50Classifier, build_resnet50
from src.models.vgg16_model import VGG16Classifier, build_vgg16
from src.models.inceptionv3_model import InceptionV3Classifier, build_inceptionv3

__all__ = [
    "ResNet50Classifier",
    "build_resnet50",
    "VGG16Classifier",
    "build_vgg16",
    "InceptionV3Classifier",
    "build_inceptionv3",
]


def build_model(model_name: str, **kwargs):
    """
    Build a model by name.

    Args:
        model_name (str): One of "resnet50", "vgg16", "inceptionv3".
        **kwargs: Passed to the model constructor.

    Returns:
        nn.Module: The configured model.

    Raises:
        ValueError: If model_name is not recognized.
    """
    builders = {
        "resnet50": build_resnet50,
        "vgg16": build_vgg16,
        "inceptionv3": build_inceptionv3,
    }

    name = model_name.lower()
    if name not in builders:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Choose from: {list(builders.keys())}"
        )

    return builders[name](**kwargs)
