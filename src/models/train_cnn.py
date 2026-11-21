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
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure project root is in sys.path for direct CLI or Colab execution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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
            all_asd_probs.extend(probs[:, 0].cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

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
    epochs: int = 20,
    lr: float = 1e-4,
    patience: int = 5,
    device: torch.device = None
) -> nn.Module:
    """
    Trains a model for pediatric ASD facial image classification with early stopping,
    Cosine Annealing learning rate scheduling, differential learning rates for hybrid models,
    and exports authentic PyTorch model weights to disk.
    
    Args:
        model_name: Name of architecture (e.g. 'modstem_lstm', 'resnet50').
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader for early stopping evaluation.
        output_path: Destination .pth file path.
        epochs: Maximum training epochs (default: 20).
        lr: Base learning rate (default: 1e-4).
        patience: Early stopping patience epochs before halting (default: 5).
        device: PyTorch device (CUDA or CPU).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"--- Initializing {model_name.upper()} Architecture on {device} ---")

    # Instantiate requested model architecture
    norm_name = model_name.lower().replace("-", "_")
    if norm_name in ["modstem_lstm", "modstem_resnet50_lstm", "hybrid_resnet50_lstm"]:
        model = build_modstem_resnet50_lstm(pretrained=True, num_classes=2, freeze_early=False, unfreeze_layer1=True)
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

    # Filter trainable parameters and set up differential learning rates
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(
        f"Model {model_name} total parameters: {sum(p.numel() for p in model.parameters()):,} | "
        f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}"
    )

    if norm_name in ["modstem_lstm", "modstem_resnet50_lstm", "hybrid_resnet50_lstm"]:
        # Differential learning rates:
        # Pretrained backbone layers (layer1-layer4) trained at 1e-4 to prevent catastrophic forgetting
        # Freshly initialized layers (Mod-Stem, Bi-LSTM, classification head) trained at 5e-4
        backbone_params = []
        for blk in [model.layer1, model.layer2, model.layer3, model.layer4]:
            backbone_params.extend([p for p in blk.parameters() if p.requires_grad])

        fresh_params = []
        for blk in [model.stem, model.lstm, model.classifier]:
            fresh_params.extend([p for p in blk.parameters() if p.requires_grad])

        optimizer = optim.Adam([
            {"params": backbone_params, "lr": lr, "weight_decay": 1e-4},
            {"params": fresh_params, "lr": 5e-4, "weight_decay": 1e-4}
        ])
        logger.info(
            f"Configured differential learning rates for {model_name}: "
            f"backbone ({sum(p.numel() for p in backbone_params):,} params) @ lr={lr}, "
            f"fresh layers ({sum(p.numel() for p in fresh_params):,} params) @ lr=5e-4, weight_decay=1e-4"
        )
    else:
        optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)

    # Cosine Annealing Learning Rate Scheduler stepping once per epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_weights = None
    patience_counter = 0

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - epoch_start

        logger.info(
            f"[{model_name.upper()}] Epoch {epoch}/{epochs} ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc * 100:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # Early Stopping tracking best weights on validation accuracy/loss
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"[{model_name.upper()}] EarlyStopping counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info(
                    f"[{model_name.upper()}] Early stopping triggered at epoch {epoch}. "
                    f"Restoring best weights (Val Acc: {best_val_acc * 100:.2f}%, Val Loss: {best_val_loss:.4f})."
                )
                break

    total_training_time = time.time() - start_time
    logger.info(
        f"Completed training for {model_name} in {total_training_time:.1f}s. "
        f"Best Val Acc: {best_val_acc * 100:.2f}%, Val Loss: {best_val_loss:.4f}"
    )

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
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    patience: int = 5,
    model_choice: str = "all",
    train_vit: bool = True
):
    """
    Executes training and evaluation pipeline for candidate architectures:
    1. Mod-Stem ResNet-50 + Spatial Bi-LSTM (Proposed Primary Model)
    2. ResNet-50 (Residual Baseline)
    3. DenseNet-121 (Dense Feature-Reuse CNN)
    4. EfficientNet-B0 (Compound Scaled CNN)
    5. MobileNet-V2 (Lightweight Mobile Triage)
    6. Vision Transformer ViT-B/16 (High Parameter Benchmark)
    
    Supports training a single model or all 6, saving weights to models/ and
    updating results/real_benchmark_metrics.json.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_data_path = Path(data_dir_str) if Path(data_dir_str).is_absolute() else (project_root / data_dir_str)

    # Auto-discover directory containing the 'Train' split
    candidate_paths = [
        raw_data_path,
        project_root / "ASD Data",
        Path("ASD Data"),
        project_root / "data" / "zenodo" / "extracted" / "ASD Data",
        project_root / "data" / "zenodo" / "ASD Data",
        project_root / "data" / "ASD Data",
        project_root / "data",
        Path("/content/data/zenodo/extracted/ASD Data"),
        Path("/content/data/ASD Data"),
        Path("/content/ASD Data"),
        Path(r"C:\Users\hy180\Downloads\Zenodo_ASD_Dataset\ASD Data"),
        Path(r"C:\Users\hy180\Downloads\Zenodo_ASD_Dataset"),
    ]
    data_dir = raw_data_path
    for cand in candidate_paths:
        if cand.exists() and (cand / "Train").exists():
            data_dir = cand
            break

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

    MODEL_CONFIGS = {
        "modstem_lstm": {
            "file": "modstem_resnet50_lstm_asd.pth",
            "title": "Mod-Stem ResNet-50 + Spatial Bi-LSTM (Proposed)",
            "citation": "Proposed Architecture (This Work, 2026)",
        },
        "resnet50": {
            "file": "resnet50_asd.pth",
            "title": "ResNet-50 (Residual Baseline)",
            "citation": "Khan et al. [11] / Monani et al. [13]",
        },
        "densenet121": {
            "file": "densenet121_asd.pth",
            "title": "DenseNet-121 (Dense Feature-Reuse CNN)",
            "citation": "Khan et al. [11] (96.0% Acc on ASD)",
        },
        "efficientnet_b0": {
            "file": "efficientnet_b0_asd.pth",
            "title": "EfficientNet-B0 (Compound Scaled CNN)",
            "citation": "Rahman & Subashini [2] (96.6% AUC on Facial Screening)",
        },
        "mobilenet_v2": {
            "file": "mobilenetv2_asd.pth",
            "title": "MobileNet-V2 (Lightweight Mobile Triage)",
            "citation": "Beary et al. [1] / Alkahtani et al. [7]",
        },
        "vit": {
            "file": "vit_asd.pth",
            "title": "Vision Transformer ViT-B/16 (High Parameter Benchmark)",
            "citation": "Revathi & Arunarani [3] (ViT Attention on ASD)",
        },
    }

    # Load existing benchmark metrics so single-model runs augment rather than overwrite
    metrics_json_path = results_dir / "real_benchmark_metrics.json"
    benchmark_results = {}
    if metrics_json_path.exists():
        try:
            with open(metrics_json_path, "r", encoding="utf-8") as f:
                benchmark_results = json.load(f)
            logger.info(f"Loaded {len(benchmark_results)} existing model benchmarks from {metrics_json_path}")
        except Exception as e:
            logger.warning(f"Could not load existing benchmark metrics: {e}")

    def package_metrics(model_key: str, model: nn.Module) -> Dict[str, Any]:
        cfg = MODEL_CONFIGS.get(model_key, {})
        arch_title = cfg.get("title", model_key)
        citation = cfg.get("citation", "Literature Review")
        params_m = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
        raw_metrics = evaluate_test_metrics(model, test_loader, device)
        return {
            "architecture": arch_title,
            "literature_citation": citation,
            "parameters_m": params_m,
            **raw_metrics
        }

    # Determine models to train based on CLI / function argument
    norm_choice = model_choice.lower().replace("-", "_")
    if norm_choice == "all":
        models_to_train = ["modstem_lstm", "resnet50", "densenet121", "efficientnet_b0", "mobilenet_v2"]
        if train_vit:
            models_to_train.append("vit")
    elif norm_choice in ["modstem_lstm", "modstem_resnet50_lstm", "hybrid_resnet50_lstm"]:
        models_to_train = ["modstem_lstm"]
    elif norm_choice in ["resnet50", "resnet_50"]:
        models_to_train = ["resnet50"]
    elif norm_choice in ["densenet121", "densenet_121", "densenet"]:
        models_to_train = ["densenet121"]
    elif norm_choice in ["efficientnet_b0", "efficientnetb0", "efficientnet"]:
        models_to_train = ["efficientnet_b0"]
    elif norm_choice in ["mobilenet_v2", "mobilenetv2"]:
        models_to_train = ["mobilenet_v2"]
    elif norm_choice in ["vit", "vit_b_16", "vision_transformer"]:
        models_to_train = ["vit"]
    else:
        raise ValueError(f"Unknown model choice: {model_choice}. Choose from: {list(MODEL_CONFIGS.keys())} or 'all'")

    logger.info(f"Target models scheduled for training ({len(models_to_train)}): {models_to_train}")

    for m_key in models_to_train:
        cfg = MODEL_CONFIGS[m_key]
        model_out_path = models_dir / cfg["file"]
        trained_model = train_and_export_cnn(
            model_name=m_key,
            train_loader=train_loader,
            val_loader=val_loader,
            output_path=model_out_path,
            epochs=epochs,
            lr=lr,
            patience=patience,
            device=device
        )
        benchmark_results[m_key] = package_metrics(m_key, trained_model)

        # Save benchmark metrics incrementally after each model completes
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, f, indent=2)
        logger.info(f"Updated benchmark results for {m_key} in: {metrics_json_path}")

    logger.info(f"Benchmark training complete! Exported metrics to: {metrics_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroGraph-ASD CNN/Hybrid Benchmark Training Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "modstem_lstm", "resnet50", "densenet121", "efficientnet_b0", "mobilenet_v2", "vit"],
        help="Specify architecture to train ('all' or single model name)"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument("--batch-size", type=int, default=32, help="DataLoader batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate (default: 1e-4)")
    parser.add_argument("--data-dir", type=str, default="data/zenodo/extracted/ASD Data", help="Path to ASD dataset")
    parser.add_argument("--models-dir", type=str, default="models", help="Output directory for .pth models")
    parser.add_argument("--results-dir", type=str, default="results", help="Output directory for benchmark metrics JSON")
    parser.add_argument("--skip-vit", action="store_true", help="Skip training ViT when running --model all")

    args = parser.parse_args()

    run_pipeline(
        data_dir_str=args.data_dir,
        models_dir_str=args.models_dir,
        results_dir_str=args.results_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        model_choice=args.model,
        train_vit=not args.skip_vit
    )
