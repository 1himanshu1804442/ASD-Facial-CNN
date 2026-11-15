"""
Convolutional Neural Network (CNN) Training Pipeline for NeuroGraph-ASD.
Trains and exports authentic PyTorch weights for Mod-Stem ResNet-50 + LSTM,
ResNet-50, MobileNet-V2, and Vision Transformer (ViT) on the Zenodo Pediatric ASD Dataset.
"""

import os
import sys
import time
import copy
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import roc_auc_score, f1_score

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
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_data_loaders(
    data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Constructs PyTorch DataLoaders for Train, Valid, and Test splits.
    Applies data augmentation to training data and deterministic transforms to eval splits.
    
    Args:
        data_dir: Root path to 'ASD Data' folder.
        batch_size: Mini-batch size.
        num_workers: DataLoader worker processes (0 is optimal for Windows compatibility).
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_transform = get_cnn_transforms(is_training=True)
    eval_transform = get_cnn_transforms(is_training=False)

    train_path = data_dir / "Train"
    val_path = data_dir / "valid"
    test_path = data_dir / "Test"

    if not train_path.exists():
        raise FileNotFoundError(f"Training directory not found at: {train_path}")

    # Load ImageFolders
    train_dataset = ImageFolder(root=str(train_path), transform=train_transform)
    val_dataset = ImageFolder(root=str(val_path), transform=eval_transform)
    test_dataset = ImageFolder(root=str(test_path), transform=eval_transform)

    logger.info(
        f"Loaded datasets from {data_dir}:\n"
        f"  - Training samples:   {len(train_dataset)} (classes: {train_dataset.class_to_idx})\n"
        f"  - Validation samples: {len(val_dataset)} (classes: {val_dataset.class_to_idx})\n"
        f"  - Testing samples:    {len(test_dataset)} (classes: {test_dataset.class_to_idx})"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Trains a CNN model for one full epoch over the training dataset.
    
    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def validate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluates model performance on validation/test dataset without gradient computation.
    
    Returns:
        Tuple of (loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total if total > 0 else 0.0
    val_acc = correct / total if total > 0 else 0.0
    return val_loss, val_acc


def evaluate_test_metrics(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """
    Computes comprehensive clinical metrics on held-out test cohort:
    Accuracy, Sensitivity, Specificity, Precision, F1-Score, AUROC, and Latency per image.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_asd_probs = []
    model_inference_time_ms = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            t0 = time.perf_counter()
            outputs = model(images)
            model_inference_time_ms += (time.perf_counter() - t0) * 1000.0
            probs = torch.softmax(outputs, dim=-1)
            _, preds = torch.max(outputs, 1)

            # Class 0: autism (ASD)
            all_asd_probs.extend(probs[:, 0].cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    total = len(all_labels)
    latency_per_img_ms = round(model_inference_time_ms / max(total, 1), 2)

    tp = sum(1 for p, y in zip(all_preds, all_labels) if p == 0 and y == 0)  # ASD true positive
    fn = sum(1 for p, y in zip(all_preds, all_labels) if p == 1 and y == 0)  # ASD false negative
    tn = sum(1 for p, y in zip(all_preds, all_labels) if p == 1 and y == 1)  # Typical true negative
    fp = sum(1 for p, y in zip(all_preds, all_labels) if p == 0 and y == 1)  # Typical false positive

    acc = (tp + tn) / total if total > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    # Binary ground truth where 1 = ASD (index 0 in ImageFolder)
    y_true_asd = [1 if y == 0 else 0 for y in all_labels]
    try:
        auc_roc = float(roc_auc_score(y_true_asd, all_asd_probs)) if len(set(y_true_asd)) > 1 else 0.0
    except Exception:
        auc_roc = 0.0

    metrics = {
        "accuracy": round(acc, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "f1_score": round(f1, 4),
        "auc_roc": round(auc_roc, 4),
        "inference_latency_ms": latency_per_img_ms,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "samples_evaluated": total,
    }
    logger.info(
        f"Test Benchmark Evaluation:\n"
        f"  - Accuracy:    {metrics['accuracy'] * 100:.2f}%\n"
        f"  - Sensitivity: {metrics['sensitivity'] * 100:.2f}%\n"
        f"  - Specificity: {metrics['specificity'] * 100:.2f}%\n"
        f"  - Precision:   {metrics['precision'] * 100:.2f}%\n"
        f"  - F1-Score:    {metrics['f1_score'] * 100:.2f}%\n"
        f"  - AUROC:       {metrics['auc_roc'] * 100:.2f}%\n"
        f"  - Latency:     {metrics['inference_latency_ms']} ms/image"
    )
    return metrics


def train_and_export_cnn(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_path: Path,
    epochs: int = 5,
    lr: float = 1e-4,
    device: torch.device = None
) -> nn.Module:
    """
    Trains a model for pediatric ASD facial image classification
    and exports authentic PyTorch model weights to disk.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"--- Initializing {model_name.upper()} Architecture on {device} ---")

    # Instantiate requested model architecture
    norm_name = model_name.lower().replace("-", "_")
    if norm_name in ["modstem_lstm", "modstem_resnet50_lstm", "hybrid_resnet50_lstm"]:
        model = build_modstem_resnet50_lstm(pretrained=True, num_classes=2)
    elif norm_name in ["resnet50", "resnet_50"]:
        model = build_resnet50(pretrained=True, num_classes=2)
    elif norm_name in ["densenet121", "densenet_121", "densenet"]:
        model = build_densenet121(pretrained=True, num_classes=2)
    elif norm_name in ["efficientnet_b0", "efficientnetb0", "efficientnet"]:
        model = build_efficientnet_b0(pretrained=True, num_classes=2)
    elif norm_name in ["mobilenet_v2", "mobilenetv2"]:
        model = build_mobilenet_v2(pretrained=True, num_classes=2)
    elif norm_name in ["vit", "vit_b_16", "vision_transformer"]:
        model = build_vit(pretrained=True, num_classes=2)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    model = model.to(device)

    # Filter trainable parameters only
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(
        f"Model {model_name} total parameters: {sum(p.numel() for p in model.parameters()):,} | "
        f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}"
    )

    optimizer = optim.Adam(trainable_params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_weights = None

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start

        logger.info(
            f"[{model_name.upper()}] Epoch {epoch}/{epochs} ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc * 100:.2f}%"
        )

        # Track best weights based on validation accuracy (deep copy to avoid shallow reference mutation)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    total_training_time = time.time() - start_time
    logger.info(f"Completed {epochs} epochs for {model_name} in {total_training_time:.1f}s. Best Val Acc: {best_val_acc * 100:.2f}%")

    # Ensure target directory exists and save state dictionary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if best_weights is not None:
        torch.save(best_weights, output_path)
        model.load_state_dict(best_weights)
    else:
        torch.save(model.state_dict(), output_path)

    logger.info(f"Saved authentic weights for {model_name} to: {output_path}")
    return model


def run_pipeline(
    data_dir_str: str = "data/zenodo/extracted/ASD Data",
    models_dir_str: str = "models",
    results_dir_str: str = "results",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    train_vit: bool = True
):
    """
    Executes complete training and export pipeline for all 4 candidate architectures:
    1. Mod-Stem ResNet-50 + LSTM (Proposed Primary Model)
    2. ResNet-50 (Residual Baseline)
    3. MobileNet-V2 (Lightweight Triage)
    4. Vision Transformer ViT-B/16 (High Parameter Benchmark)
    
    Saves authentic model weights to models/ and updates results/real_benchmark_metrics.json.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / data_dir_str if not Path(data_dir_str).is_absolute() else Path(data_dir_str)
    
    # Check if local Downloads dataset exists as fallback if project data path not yet extracted
    if not data_dir.exists():
        fallback_data = Path(r"C:\Users\hy180\Downloads\Zenodo_ASD_Dataset\ASD Data")
        if fallback_data.exists():
            data_dir = fallback_data

    models_dir = project_root / models_dir_str if not Path(models_dir_str).is_absolute() else Path(models_dir_str)
    results_dir = project_root / results_dir_str if not Path(results_dir_str).is_absolute() else Path(results_dir_str)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Dataset root: {data_dir}")
    logger.info(f"Models directory: {models_dir}")
    logger.info(f"Results directory: {results_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))
        logger.info(f"Configured PyTorch CPU threads: {torch.get_num_threads()}")

    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=0
    )

    MODEL_META = {
        "modstem_lstm": ("Mod-Stem ResNet-50 + LSTM (Proposed Hybrid)", "Proposed Architecture (Yadav et al., 2026)"),
        "resnet50": ("ResNet-50 (Residual Baseline)", "Khan et al. [11] / Monani et al. [13]"),
        "densenet121": ("DenseNet-121 (Dense Feature-Reuse CNN)", "Khan et al. [11] (96.0% Acc on ASD)"),
        "efficientnet_b0": ("EfficientNet-B0 (Compound Scaled CNN)", "Rahman & Subashini [2] (96.6% AUC on Facial Screening)"),
        "mobilenet_v2": ("MobileNet-V2 (Lightweight Mobile Triage)", "Beary et al. [1] / Alkahtani et al. [7]"),
        "vit": ("Vision Transformer ViT-B/16 (High Parameter Benchmark)", "Revathi & Arunarani [3] (ViT Attention on ASD)"),
    }

    def package_metrics(model_key: str, model: nn.Module) -> Dict[str, Any]:
        arch_title, citation = MODEL_META.get(model_key, (model_key, "Literature Review"))
        params_m = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
        raw_metrics = evaluate_test_metrics(model, test_loader, device)
        return {
            "architecture": arch_title,
            "literature_citation": citation,
            "parameters_m": params_m,
            **raw_metrics
        }

    benchmark_results = {}

    # 1. Train and save Mod-Stem ResNet-50 + LSTM (Proposed Primary Model)
    modstem_path = models_dir / "modstem_resnet50_lstm_asd.pth"
    modstem_model = train_and_export_cnn(
        model_name="modstem_lstm",
        train_loader=train_loader,
        val_loader=val_loader,
        output_path=modstem_path,
        epochs=epochs,
        lr=lr,
        device=device
    )
    benchmark_results["modstem_lstm"] = package_metrics("modstem_lstm", modstem_model)

    # 2. Train and save baseline ResNet-50
    resnet_path = models_dir / "resnet50_asd.pth"
    resnet_model = train_and_export_cnn(
        model_name="resnet50",
        train_loader=train_loader,
        val_loader=val_loader,
        output_path=resnet_path,
        epochs=epochs,
        lr=lr,
        device=device
    )
    benchmark_results["resnet50"] = package_metrics("resnet50", resnet_model)

    # 3. Train and save DenseNet-121 (Literature Paper [11] Khan et al., 2024)
    densenet_path = models_dir / "densenet121_asd.pth"
    densenet_model = train_and_export_cnn(
        model_name="densenet121",
        train_loader=train_loader,
        val_loader=val_loader,
        output_path=densenet_path,
        epochs=epochs,
        lr=lr,
        device=device
    )
    benchmark_results["densenet121"] = package_metrics("densenet121", densenet_model)

    # 4. Train and save EfficientNet-B0 (Literature Paper [2] Rahman & Subashini, 2022)
    effnet_path = models_dir / "efficientnet_b0_asd.pth"
    effnet_model = train_and_export_cnn(
        model_name="efficientnet_b0",
        train_loader=train_loader,
        val_loader=val_loader,
        output_path=effnet_path,
        epochs=epochs,
        lr=lr,
        device=device
    )
    benchmark_results["efficientnet_b0"] = package_metrics("efficientnet_b0", effnet_model)

    # 5. Train and save MobileNet-V2
    mobilenet_path = models_dir / "mobilenetv2_asd.pth"
    mobilenet_model = train_and_export_cnn(
        model_name="mobilenet_v2",
        train_loader=train_loader,
        val_loader=val_loader,
        output_path=mobilenet_path,
        epochs=epochs,
        lr=lr,
        device=device
    )
    benchmark_results["mobilenet_v2"] = package_metrics("mobilenet_v2", mobilenet_model)

    # 6. Train and save Vision Transformer (ViT-B/16) (Literature Paper [3] Revathi & Arunarani, 2026)
    if train_vit:
        vit_path = models_dir / "vit_asd.pth"
        vit_model = train_and_export_cnn(
            model_name="vit",
            train_loader=train_loader,
            val_loader=val_loader,
            output_path=vit_path,
            epochs=epochs,
            lr=lr,
            device=device
        )
        benchmark_results["vit"] = package_metrics("vit", vit_model)

    # Save real benchmark metrics to JSON file
    metrics_json_path = results_dir / "real_benchmark_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    logger.info(f"Exported empirical benchmark metrics to: {metrics_json_path}")
    logger.info("All CNN, Hybrid, and ViT models trained and evaluated successfully!")


if __name__ == "__main__":
    run_pipeline()
