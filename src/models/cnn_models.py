"""
Convolutional Neural Network (CNN) architectures and transfer learning utilities
for Pediatric ASD facial screening in NeuroGraph-ASD.
Provides Mod-Stem ResNet-50 + LSTM Hybrid, baseline ResNet-50, MobileNet-V2,
and Vision Transformer (ViT) backbones customized with clinical classification heads.
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
from torchvision import models, transforms


# Target diagnostic classes in Zenodo Pediatric ASD Dataset
# Index 0 corresponds to 'autism' (ASD)
# Index 1 corresponds to 'tipical' (Typical Control)
CLASS_NAMES: List[str] = ["autism", "tipical"]


def get_cnn_transforms(is_training: bool = False) -> transforms.Compose:
    """
    Standard image transformation pipeline for PyTorch vision models:
    - Normalization parameters are standard ImageNet mean and std.
    - If is_training is True, applies modest data augmentation (random horizontal flips,
      subtle random rotation up to 10 degrees) to prevent overfitting on facial angles.
    - Resizes to (224, 224) resolution.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class ModStem(nn.Module):
    """
    Modified Stem (Mod-Stem) replacing standard ResNet-50 7x7 conv.
    
    Why: The original ResNet-50 7x7 convolution with stride 2 rapidly downsamples
    the input by 4x, causing severe loss of fine-grained morphological details.
    By replacing it with three consecutive 3x3 convolutions (stride 2 on the first,
    stride 1 on the subsequent two), we retain low-level edge and texture boundaries
    around delicate pediatric facial landmarks (eye spacing, philtrum width).
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class ModStemResNet50LSTM(nn.Module):
    """
    Hybrid Deep Learning Architecture combining:
    1. Modified Stem (Mod-Stem) ResNet-50 for high-resolution spatial feature extraction.
    2. Bidirectional LSTM (Bi-LSTM) modeling spatial sequence dependencies across facial regions.
    
    Why: Static ConvNets evaluate local patches independently, whereas autism facial phenotypes
    depend on relational proportions (e.g., inter-ocular distance relative to midface breadth).
    This architecture slices the 2048-dimensional deep feature map into vertical anatomical tiers
    (forehead -> eyes -> nose -> philtrum -> mouth -> chin) and processes them sequentially with a Bi-LSTM.
    """
    def __init__(
        self,
        num_classes: int = 2,
        lstm_hidden_dim: int = 128,
        num_lstm_layers: int = 1,
        pretrained_backbone: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained_backbone else None
        base_resnet = models.resnet50(weights=weights)

        # 1. Mod-Stem replaces standard 7x7 conv
        self.stem = ModStem(in_channels=3, out_channels=64)
        self.maxpool = base_resnet.maxpool

        # 2. Residual Blocks (Transfer Learning Backbone)
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4  # Shape: (B, 2048, 7, 7)

        # 3. Anatomical Regional Sequence Pooling along vertical facial axis (7 regions)
        self.region_pool = nn.AdaptiveAvgPool2d((7, 1))

        # 4. Bidirectional LSTM Sequence Modeling
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        # 5. Clinical Projection Head
        lstm_out_dim = lstm_hidden_dim * 2  # Bidirectional doubles output channels
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through Mod-Stem
        x = self.stem(x)
        x = self.maxpool(x)

        # Deep residual extraction
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (B, 2048, 7, 7)

        # Extract 7 vertical anatomical facial regions
        x = self.region_pool(x)  # (B, 2048, 7, 1)
        x = x.squeeze(-1).permute(0, 2, 1)  # (B, 7, 2048) -> Sequence of 7 tokens

        # Process sequential morphological relationships via Bi-LSTM
        lstm_out, _ = self.lstm(x)  # (B, 7, 256)

        # Global temporal average pooling across the 7 regions
        out_pooled = torch.mean(lstm_out, dim=1)  # (B, 256)

        # Diagnostic classification logits
        logits = self.classifier(out_pooled)
        return logits


def build_modstem_resnet50_lstm(
    pretrained: bool = True,
    num_classes: int = 2,
    freeze_early: bool = True
) -> nn.Module:
    """
    Factory function to construct the Mod-Stem ResNet-50 + LSTM Hybrid model.
    """
    model = ModStemResNet50LSTM(
        num_classes=num_classes,
        lstm_hidden_dim=128,
        num_lstm_layers=1,
        pretrained_backbone=pretrained,
        dropout=0.3
    )

    if freeze_early:
        # Freeze only layer1; allow layer2, layer3, and layer4 to adapt to ModStem activations
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.stem.parameters():
            param.requires_grad = True
        for param in model.layer2.parameters():
            param.requires_grad = True
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.lstm.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True

    return model


def build_resnet50(pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    """
    Constructs a baseline ResNet-50 transfer learning architecture for ASD detection.
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model


def build_mobilenet_v2(pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    """
    Constructs a lightweight MobileNet-V2 transfer learning architecture.
    """
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.features[-3:].parameters():
        param.requires_grad = True

    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model


def build_vit(pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    """
    Constructs a Vision Transformer (ViT-B/16) transfer learning model.
    Why: Included in our candidate benchmark matrix to directly evaluate
    the performance and computational trade-offs of pure transformer vision models
    against our Mod-Stem hybrid approach on medical pediatric cohorts.
    """
    weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
    model = models.vit_b_16(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last transformer encoder block for fair fine-tuning (matching layer4 in ResNet)
    for param in model.encoder.layers[-1].parameters():
        param.requires_grad = True

    in_features = model.heads.head.in_features  # 768
    model.heads.head = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model
