"""
Unit and integration tests for NeuroGraph-ASD CNN deep learning pipelines.
Follows the 'No-Mistakes Pipeline' to verify model structure, input tensor compatibility,
and output logit dimensions BEFORE running training or deployment.
"""

import pytest
import torch
import torch.nn as nn
from src.models.cnn_models import (
    build_modstem_resnet50_lstm,
    build_resnet50,
    build_densenet121,
    build_efficientnet_b0,
    build_mobilenet_v2,
    build_vit,
    get_cnn_transforms,
    CLASS_NAMES
)


def test_classes_definition():
    """
    Verifies that the clinical diagnostic classes match the Zenodo ASD dataset layout.
    Index 0: 'autism' (ASD)
    Index 1: 'tipical' (Typical Control)
    """
    assert len(CLASS_NAMES) == 2, "Expected exactly 2 target classes for binary ASD classification."
    assert CLASS_NAMES[0] in ["autism", "ASD"]
    assert CLASS_NAMES[1] in ["tipical", "Typical", "control"]


def test_resnet50_pipeline_tensor_dimensions():
    """
    Validates that ResNet-50:
    1. Instantiates properly with transfer learning head.
    2. Accepts standard [1, 3, 224, 224] normalized image tensors.
    3. Outputs [1, 2] logits.
    4. Evaluates softmax probabilities summing to 1.0.
    """
    # Instantiate without downloading weights in unit tests to ensure fast, deterministic testing
    model = build_resnet50(pretrained=False, num_classes=2)
    model.eval()

    # Create dummy tensor representing a single 224x224 RGB image
    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        logits = model(dummy_input)

    # Verify output shape is [1, 2]
    assert logits.shape == (1, 2), f"Expected ResNet-50 output shape [1, 2], but got {logits.shape}"

    # Verify softmax probabilities
    probs = torch.softmax(logits, dim=-1)
    assert probs.shape == (1, 2)
    assert torch.isclose(probs.sum(dim=-1), torch.tensor([1.0]), atol=1e-5), "Softmax probabilities must sum to 1.0"


def test_mobilenet_v2_pipeline_tensor_dimensions():
    """
    Validates that MobileNet-V2:
    1. Instantiates properly with lightweight inverted residual architecture.
    2. Accepts [1, 3, 224, 224] normalized image tensors.
    3. Outputs [1, 2] logits.
    4. Evaluates batching capabilities on [4, 3, 224, 224] inputs.
    """
    model = build_mobilenet_v2(pretrained=False, num_classes=2)
    model.eval()

    # Single-item inference test [1, 3, 224, 224]
    single_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        single_logits = model(single_input)

    assert single_logits.shape == (1, 2), f"Expected MobileNet-V2 output shape [1, 2], got {single_logits.shape}"

    # Batch inference test [4, 3, 224, 224]
    batch_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        batch_logits = model(batch_input)

    assert batch_logits.shape == (4, 2), f"Expected MobileNet-V2 batch shape [4, 2], got {batch_logits.shape}"


def test_cnn_transforms_pipeline():
    """
    Verifies that image preprocessing transformation pipeline correctly resizes and normalizes PIL / Tensor data.
    """
    from PIL import Image
    # Create arbitrary test image
    img = Image.new("RGB", (320, 240), color=(128, 64, 192))
    transform = get_cnn_transforms()
    transformed_tensor = transform(img)

    # Tensor should be [3, 224, 224]
    assert transformed_tensor.shape == (3, 224, 224), f"Expected [3, 224, 224], got {transformed_tensor.shape}"
    assert transformed_tensor.dtype == torch.float32, "Transformed image must be float32 tensor"


def test_modstem_resnet50_lstm_architecture_and_unfrozen_layer1():
    """
    Validates Mod-Stem ResNet-50 + Bi-LSTM:
    1. Instantiates with freeze_early=False, unfreeze_layer1=True by default.
    2. layer1 parameters have requires_grad=True.
    3. Custom 3-conv ModStem, Bi-LSTM, and projection head have requires_grad=True.
    4. Handles batch inference [2, 3, 224, 224] and produces [2, 2] logits.
    """
    model = build_modstem_resnet50_lstm(pretrained=False, num_classes=2)
    model.eval()

    # Verify layer1 parameters are unfrozen and trainable
    layer1_grads = [p.requires_grad for p in model.layer1.parameters()]
    assert len(layer1_grads) > 0, "model.layer1 must contain parameters"
    assert all(layer1_grads), "All layer1 parameters must have requires_grad=True by default"

    # Verify stem, lstm, and classifier are trainable
    assert all(p.requires_grad for p in model.stem.parameters()), "ModStem parameters must be trainable"
    assert all(p.requires_grad for p in model.lstm.parameters()), "Bi-LSTM parameters must be trainable"
    assert all(p.requires_grad for p in model.classifier.parameters()), "Classifier parameters must be trainable"

    # Forward pass tensor test
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (2, 2), f"Expected output shape [2, 2], got {logits.shape}"
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5), "Softmax probabilities must sum to 1.0"


def test_densenet121_pipeline():
    """
    Validates DenseNet-121 instantiates and processes [1, 3, 224, 224] to [1, 2] logits.
    """
    model = build_densenet121(pretrained=False, num_classes=2)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)
    assert logits.shape == (1, 2), f"Expected DenseNet-121 output shape [1, 2], got {logits.shape}"


def test_efficientnet_b0_pipeline():
    """
    Validates EfficientNet-B0 instantiates and processes [1, 3, 224, 224] to [1, 2] logits.
    """
    model = build_efficientnet_b0(pretrained=False, num_classes=2)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)
    assert logits.shape == (1, 2), f"Expected EfficientNet-B0 output shape [1, 2], got {logits.shape}"


def test_vit_pipeline():
    """
    Validates ViT-B/16 instantiates and processes [1, 3, 224, 224] to [1, 2] logits.
    """
    model = build_vit(pretrained=False, num_classes=2)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)
    assert logits.shape == (1, 2), f"Expected ViT-B/16 output shape [1, 2], got {logits.shape}"
