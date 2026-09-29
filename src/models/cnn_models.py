"""
Convolutional Neural Network (CNN) architectures and transfer learning utilities
for Pediatric ASD facial screening in NeuroGraph-ASD.
Provides ResNet-50 and MobileNet-V2 backbones customized with clinical classification heads.
"""

from typing import Tuple, List
import torch
import torch.nn as nn
from torchvision import models, transforms


# Target diagnostic classes in Zenodo Pediatric ASD Dataset
# Index 0 corresponds to 'autism' (ASD)
# Index 1 corresponds to 'tipical' (Typical Control)
CLASS_NAMES: List[str] = ["autism", "tipical"]


def get_cnn_transforms() -> transforms.Compose:
    """
    Standard image transformation pipeline for PyTorch vision models:
    1. Resizes input portrait images to (224, 224) resolution required by ImageNet backbones.
    2. Converts PIL / numpy images into normalized torch float32 tensors in range [0.0, 1.0].
    3. Standardizes color channels with ImageNet mean and standard deviation.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def build_resnet50(pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    """
    Constructs a ResNet-50 transfer learning architecture for ASD detection:
    - Pretrained ImageNet weights are loaded to retain low-level visual representations.
    - Early convolutional blocks (conv1, bn1, layer1, layer2, layer3) are frozen to prevent
      gradient distortion and catastrophic forgetting.
    - Deep feature representations (layer4) and the custom classification head remain trainable
      for fine-tuning on pediatric facial morphology.
    - The original 1000-class linear head is replaced with Dropout(0.3) + Linear(2048, num_classes).

    Args:
        pretrained: If True, initializes with torchvision 'DEFAULT' pretrained weights.
        num_classes: Number of target categories (default: 2 -> ASD vs Typical).

    Returns:
        Configured PyTorch nn.Module.
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    # Freeze early layers to retain generic feature representations (edges, textures, contours)
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4 to allow high-level clinical feature adaptation
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace final linear head with dropout regularization and binary diagnostic projection
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model


def build_mobilenet_v2(pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    """
    Constructs a lightweight MobileNet-V2 transfer learning architecture:
    - Pretrained ImageNet inverted residual blocks enable resource-efficient inference.
    - Early feature layers are frozen while the trailing blocks and classifier head are trainable.
    - Replaces classifier head with Dropout(0.3) + Linear(1280, num_classes).

    Args:
        pretrained: If True, initializes with torchvision 'DEFAULT' weights.
        num_classes: Number of target categories (default: 2 -> ASD vs Typical).

    Returns:
        Configured PyTorch nn.Module.
    """
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    # Freeze early inverted residual feature extraction layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the final 3 inverted residual blocks for domain specialization
    for param in model.features[-3:].parameters():
        param.requires_grad = True

    # Extract input feature dimension from default classifier
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model
