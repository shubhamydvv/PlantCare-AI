"""Unified training engine for PlantCare AI models."""

from __future__ import annotations
import os
import time
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score

from src.config import (
    CONFIG,
    MODELS_DIR,
    METRICS_DIR,
    get_device,
    set_seed,
)
from src.preprocessing import load_or_create_splits, get_dataloaders
from src.models import get_model, count_parameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("PlantCare.Train")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluates validation loss, accuracy, and macro F1 score."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    total = len(all_targets)
    epoch_loss = running_loss / max(1, total)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return epoch_loss, epoch_acc, epoch_f1


def train_model(
    model_name: str = "resnet50",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    seed: int = 42,
    dry_run: bool = False,
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main training routine for a specified model architecture.
    """
    set_seed(seed)
    device = torch.device(device_name) if device_name else get_device()
    logger.info(f"Starting training run for '{model_name}' on device: {device}")

    # Lookup architecture config from train_config.yaml
    model_configs = {m["name"]: m for m in CONFIG.get("models", {}).get("architectures", [])}
    arch_cfg = model_configs.get(model_name, {})

    # Resolve hyperparameters
    num_epochs = epochs or (1 if dry_run else arch_cfg.get("epochs", 10))
    b_size = batch_size or (4 if dry_run else arch_cfg.get("batch_size", 32))
    learning_rate = lr or arch_cfg.get("lr", 0.0005)
    w_decay = weight_decay or arch_cfg.get("weight_decay", 0.0001)
    label_smoothing = CONFIG.get("training", {}).get("label_smoothing", 0.05)
    patience = 2 if dry_run else CONFIG.get("training", {}).get("early_stopping_patience", 3)

    # Load dataset splits
    splits = load_or_create_splits(seed=seed)
    num_classes = splits["num_classes"]
    class_names = splits["class_names"]

    # If dry run, slice splits to 10 samples
    if dry_run:
        logger.info("Executing dry run with reduced subset for verification...")
        splits["train"]["image_paths"] = splits["train"]["image_paths"][:16]
        splits["train"]["labels"] = splits["train"]["labels"][:16]
        splits["val"]["image_paths"] = splits["val"]["image_paths"][:8]
        splits["val"]["labels"] = splits["val"]["labels"][:8]
        splits["test"]["image_paths"] = splits["test"]["image_paths"][:8]
        splits["test"]["labels"] = splits["test"]["labels"][:8]

    train_loader, val_loader, _ = get_dataloaders(splits, batch_size=b_size, num_workers=0)

    # Initialize model
    model = get_model(model_name, num_classes=num_classes, pretrained=True)
    model.to(device)
    param_counts = count_parameters(model)
    logger.info(f"Model parameters: {param_counts['total_params']:,} total ({param_counts['trainable_params']:,} trainable)")

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=w_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = model_dir / "best_model.pth"

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    patience_counter = 0
    history = []
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        ep_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        ep_duration = time.time() - ep_start

        logger.info(
            f"Epoch [{epoch}/{num_epochs}] ({ep_duration:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro-F1: {val_f1:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_f1": float(val_f1),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "duration_sec": float(ep_duration),
        })

        # Save best model based on validation macro F1 (and lower loss on tie)
        if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_loss < best_val_loss):
            best_val_f1 = val_f1
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_name": model_name,
                "num_classes": num_classes,
                "class_names": class_names,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": float(val_f1),
                "val_acc": float(val_acc),
                "hyperparameters": {
                    "lr": learning_rate,
                    "batch_size": b_size,
                    "weight_decay": w_decay,
                    "seed": seed,
                }
            }, best_ckpt_path)
            logger.info(f"==> Checkpoint saved: {best_ckpt_path} (Best Val F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience and not dry_run:
                logger.info(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
                break

    total_training_time = time.time() - start_time
    logger.info(f"Completed training '{model_name}' in {total_training_time:.1f}s. Best Val F1: {best_val_f1:.4f}")

    # Append run to run_log.json
    run_log_file = METRICS_DIR / "run_log.json"
    run_entry = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "seed": seed,
        "device": str(device),
        "dry_run": dry_run,
        "epochs_trained": len(history),
        "total_time_sec": round(total_training_time, 2),
        "hyperparameters": {
            "lr": learning_rate,
            "batch_size": b_size,
            "weight_decay": w_decay,
            "label_smoothing": label_smoothing,
        },
        "best_metrics": {
            "val_f1": round(float(best_val_f1), 4),
            "val_loss": round(float(best_val_loss), 4),
        },
        "checkpoint_path": str(best_ckpt_path.resolve()),
        "history": history,
    }

    logs = []
    if run_log_file.exists():
        try:
            with open(run_log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []
    logs.append(run_entry)
    with open(run_log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    return run_entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PlantCare AI Model Training Script")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "efficientnet_b0", "vit_b_16", "cnn_vit_hybrid"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Run fast verification with small subset")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        dry_run=args.dry_run,
        device_name=args.device,
    )
