#!/usr/bin/env python3
"""
CUDA-enabled PyTorch + timm benchmarking pipeline for comparing pretrained
vision models on binary Mooney animal face classification (Cat vs. Dog).
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[INFO] Using CUDA GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("[WARN] CUDA not available — falling back to CPU.")
    return device


def build_dataloaders(
    data_dir: Path,
    model_cfg: dict,
    batch_size: int = 32,
    num_workers: int = 4,
    custom_test_dir: Path | None = None,
):
    """Creates train, val, and test DataLoaders matched to the timm model's expected input size."""
    input_size = model_cfg["input_size"][-1]
    mean = model_cfg["mean"]
    std = model_cfg["std"]

    # Augmentations preserve strictly binary 0/255 values via nearest-neighbor interpolation
    # and horizontal flips (avoid heavy color jitter since Mooney images are binary)
    train_tf = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)
    test_path = custom_test_dir if custom_test_dir is not None else (data_dir / "test")
    test_ds = datasets.ImageFolder(test_path, transform=eval_tf)

    # Compute inverse-frequency class weights to handle ~1:2 Cat/Dog imbalance
    targets = np.array(train_ds.targets)
    class_counts = np.bincount(targets)
    class_weights = len(targets) / (len(class_counts) * class_counts.astype(np.float32))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin
    )

    return train_loader, val_loader, test_loader, train_ds.classes, class_weights_tensor


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return running_loss / max(1, total_samples)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> dict:
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of class 1 ('dog')

        preds = torch.argmax(logits, dim=1)
        running_loss += loss.item() * images.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "loss": running_loss / max(1, len(y_true)),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
    }


def run_experiment(
    model_name: str,
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    num_workers: int = 4,
    freeze_backbone: bool = False,
    custom_test_dir: Path | None = None,
) -> dict:
    print(f"\n{'=' * 70}")
    print(f"Training & Evaluating Model: {model_name}")
    print(f"{'=' * 70}")

    model = timm.create_model(model_name, pretrained=True, num_classes=2)
    model_cfg = timm.data.resolve_model_data_config(model)

    if freeze_backbone:
        # Linear probe: freeze all layers except classification head
        for param in model.parameters():
            param.requires_grad = False
        for param in model.get_classifier().parameters():
            param.requires_grad = True
        print("[INFO] Mode: Linear Probe (backbone frozen, training classifier head only)")
    else:
        print("[INFO] Mode: Full Fine-Tuning")

    model.to(device)

    train_loader, val_loader, test_loader, classes, class_weights = build_dataloaders(
        data_dir=data_dir,
        model_cfg=model_cfg,
        batch_size=batch_size,
        num_workers=num_workers,
        custom_test_dir=custom_test_dir,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    best_ckpt_path = ckpt_dir / f"{safe_name}_best.pt"

    best_val_bal_acc = -1.0
    best_epoch = 0
    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_bal_acc": val_metrics["balanced_accuracy"],
            "val_f1": val_metrics["macro_f1"],
        })

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val BalAcc: {val_metrics['balanced_accuracy']*100:.2f}% | "
            f"Val F1: {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["balanced_accuracy"] > best_val_bal_acc:
            best_val_bal_acc = val_metrics["balanced_accuracy"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_ckpt_path)

    elapsed_sec = time.time() - start_time

    # Load best validation checkpoint for final unbiased TEST evaluation
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_loader, criterion, device, use_amp)

    print(f"\n[TEST RESULTS — {model_name} (Best Epoch {best_epoch})]")
    print(f"  Classes           : {classes} (0={classes[0]}, 1={classes[1]})")
    print(f"  Test Accuracy     : {test_metrics['accuracy']*100:.2f}%")
    print(f"  Test Balanced Acc : {test_metrics['balanced_accuracy']*100:.2f}%")
    print(f"  Test Macro F1     : {test_metrics['macro_f1']:.4f}")
    print(f"  Test ROC-AUC      : {test_metrics['roc_auc']:.4f}")
    print(f"  Confusion Matrix  : {test_metrics['confusion_matrix']}")

    # Save training history
    pd.DataFrame(history).to_csv(output_dir / f"{safe_name}_history.csv", index=False)

    return {
        "model": model_name,
        "mode": "linear_probe" if freeze_backbone else "full_finetune",
        "best_epoch": best_epoch,
        "val_balanced_acc": round(best_val_bal_acc * 100, 2),
        "test_accuracy": round(test_metrics["accuracy"] * 100, 2),
        "test_balanced_acc": round(test_metrics["balanced_accuracy"] * 100, 2),
        "test_macro_f1": round(test_metrics["macro_f1"], 4),
        "test_roc_auc": round(test_metrics["roc_auc"], 4),
        "confusion_matrix": json.dumps(test_metrics["confusion_matrix"]),
        "train_time_sec": round(elapsed_sec, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark multiple pretrained timm vision models on Mooney Cat vs. Dog faces."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to dataset directory containing train/, val/, and test/ folders.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Optional override test directory (e.g., train on RGB faces, test on Mooney faces).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./mooney_benchmark_results"),
        help="Directory to save model checkpoints, training curves, and summary CSV.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "resnet50",
            "convnext_small.fb_in22k_ft_in1k",
            "vit_base_patch16_224",
            "swin_base_patch4_window7_224",
            "vit_base_patch14_dinov2.lvd142m",
        ],
        help="List of timm model names to benchmark.",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per model (default: 15).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate (default: 1e-4).")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay (default: 1e-2).")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker threads (default: 4).")
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze pretrained backbone and train only the linear classifier head.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")

    args = parser.parse_args()
    set_seed(args.seed)
    device = get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for model_name in args.models:
        result = run_experiment(
            model_name=model_name,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            freeze_backbone=args.freeze_backbone,
            custom_test_dir=args.test_dir,
        )
        summary_rows.append(result)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = args.output_dir / "benchmark_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"\n{'=' * 70}")
    print("FINAL BENCHMARK LEADERBOARD (Sorted by Test Balanced Accuracy)")
    print(f"{'=' * 70}")
    print(summary_df.sort_values(by="test_balanced_acc", ascending=False).to_string(index=False))
    print(f"\nSaved full summary table to: {summary_csv.resolve()}")


if __name__ == "__main__":
    main()