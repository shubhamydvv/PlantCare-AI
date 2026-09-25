"""Evaluation, robustness benchmarking, confidence calibration, and comparison plotting."""

from __future__ import annotations
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)

from src.config import (
    CONFIG,
    MODELS_DIR,
    METRICS_DIR,
    GRAPHS_DIR,
    PROCESSED_DATA_DIR,
    ROBUSTNESS_DATA_DIR,
    get_device,
    set_seed,
)
from src.preprocessing import get_val_test_transforms, load_or_create_splits
from src.robustness import load_or_generate_robustness_manifest, ROBUSTNESS_TRANSFORMS
from src.models import get_model, load_model_checkpoint

logger = logging.getLogger("PlantCare.Evaluate")


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Computes Brier score, Expected Calibration Error (ECE), and bin statistics for reliability diagrams.
    """
    num_classes = y_prob.shape[1]
    # Multi-class Brier score
    y_true_onehot = np.eye(num_classes)[y_true]
    brier = float(np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1)))

    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = predictions == y_true

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_data = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_data.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "accuracy": float(accuracy_in_bin),
                "confidence": float(avg_confidence_in_bin),
                "count": int(np.sum(in_bin)),
            })
        else:
            bin_data.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "accuracy": 0.0,
                "confidence": float((bin_lower + bin_upper) / 2),
                "count": 0,
            })

    return brier, float(ece), {"bins": bin_data}


def evaluate_dataset(
    model: torch.nn.Module,
    image_paths: List[str],
    labels: List[int],
    device: torch.device,
    image_size: int = 224,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Evaluates a single model on an image list and computes accuracy, precision, recall, F1, latency, and calibration.
    """
    model.eval()
    transform = get_val_test_transforms(image_size)

    all_preds = []
    all_probs = []
    all_targets = []
    latencies_ms = []

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]

            batch_tensors = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    tensor = transform(img)
                except Exception:
                    tensor = torch.zeros((3, image_size, image_size))
                batch_tensors.append(tensor)

            if not batch_tensors:
                continue

            inputs = torch.stack(batch_tensors).to(device)

            start_t = time.perf_counter()
            outputs = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency_ms = ((time.perf_counter() - start_t) / len(batch_paths)) * 1000.0

            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(batch_labels)
            latencies_ms.extend([latency_ms] * len(batch_paths))

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    mean_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0

    brier, ece, calib_info = compute_calibration_metrics(y_true, y_prob)

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec, 4),
        "recall_macro": round(rec, 4),
        "f1_macro": round(f1, 4),
        "mean_latency_ms": round(mean_latency, 2),
        "brier_score": round(brier, 4),
        "expected_calibration_error": round(ece, 4),
        "calibration_info": calib_info,
        "sample_count": len(y_true),
    }


def run_full_evaluation(
    model_names: Optional[List[str]] = None,
    device_name: Optional[str] = None,
    seed: int = 42,
    require_real_data: bool = True,
    require_checkpoints: bool = True,
) -> Dict[str, Any]:
    """
    Runs complete evaluation suite across clean test set and all 9 robustness variants for all specified models.

    Args:
        require_real_data: If True (default), raises an error when the test set has fewer than
            500 images, which is the signature of mock/synthetic data being used instead of
            the real PlantVillage dataset. Set False only for deliberate dry-run testing.
        require_checkpoints: If True (default), raises an error when a trained checkpoint is
            missing rather than silently falling back to an untrained pretrained baseline.
            The fallback produces near-random accuracy (7-13%) that looks like a trained result.
            Set False only if you explicitly want to evaluate ImageNet pretrained baselines.
    """
    set_seed(seed)
    device = torch.device(device_name) if device_name else get_device()
    target_models = model_names or ["resnet50", "efficientnet_b0", "vit_b_16"]

    splits = load_or_create_splits(seed=seed)
    num_classes = splits["num_classes"]

    # Guard: detect mock/synthetic data masquerading as real PlantVillage
    test_size = len(splits["test"]["image_paths"])
    if require_real_data and test_size < 500:
        raise RuntimeError(
            f"\n\nEVALUATION ABORTED — test set has only {test_size} images.\n"
            f"This is the signature of synthetic mock data (15 images/class x 38 classes = 570 total).\n"
            f"The real PlantVillage dataset produces ~8,100 test images (15% of ~54,300 total).\n"
            f"\nAction required:\n"
            f"  1. Download PlantVillage from Kaggle: https://www.kaggle.com/datasets/emmarex/plantdisease\n"
            f"  2. Extract class folders into data/raw/\n"
            f"  3. Delete data/processed/split_info.json (stale mock splits)\n"
            f"  4. Re-run evaluation\n"
            f"\nTo bypass this guard (dry-run only): pass require_real_data=False"
        )

    robustness_manifest = load_or_generate_robustness_manifest(seed=seed)

    results: Dict[str, Any] = {
        "clean_evaluations": {},
        "robustness_evaluations": {},
        "summary_table": [],
    }

    clean_test_paths = splits["test"]["image_paths"]
    clean_test_labels = splits["test"]["labels"]

    for model_name in target_models:
        ckpt_path = MODELS_DIR / model_name / "best_model.pth"
        if not ckpt_path.exists():
            if require_checkpoints:
                raise FileNotFoundError(
                    f"\n\nEVALUATION ABORTED — no trained checkpoint found for '{model_name}'\n"
                    f"Expected: {ckpt_path}\n"
                    f"\nThe previous run evaluated an untrained ImageNet pretrained model, which\n"
                    f"produces near-random accuracy (7-13%) on PlantVillage (38 classes).\n"
                    f"\nAction required: train the model first:\n"
                    f"  python -m src.train --model {model_name}\n"
                    f"  (or use the Colab training script: notebooks/colab_training_script.py)\n"
                    f"\nTo bypass this guard: pass require_checkpoints=False"
                )
            logger.warning(
                f"No checkpoint for '{model_name}' at {ckpt_path}. "
                f"Evaluating ImageNet pretrained baseline (NOT fine-tuned — results will be near-random)."
            )
            model = get_model(model_name, num_classes=num_classes, pretrained=True)
            model.to(device)
        else:
            model = load_model_checkpoint(model_name, num_classes=num_classes, checkpoint_path=ckpt_path, device=device)

        logger.info(f"--- Evaluating {model_name} on Clean Test Set ---")
        clean_res = evaluate_dataset(model, clean_test_paths, clean_test_labels, device=device)
        results["clean_evaluations"][model_name] = clean_res

        # Add to summary
        results["summary_table"].append({
            "Model": model_name,
            "Condition": "Clean (Baseline)",
            "Accuracy": clean_res["accuracy"],
            "Macro F1": clean_res["f1_macro"],
            "Macro Recall": clean_res["recall_macro"],
            "Latency (ms)": clean_res["mean_latency_ms"],
            "ECE": clean_res["expected_calibration_error"],
            "Brier Score": clean_res["brier_score"],
        })

        # Evaluate Robustness Variants
        results["robustness_evaluations"][model_name] = {}
        for variant_name, variant_info in robustness_manifest["variants"].items():
            logger.info(f"Evaluating {model_name} on Robustness Variant: {variant_name}")
            var_res = evaluate_dataset(
                model,
                variant_info["image_paths"],
                variant_info["labels"],
                device=device,
            )
            results["robustness_evaluations"][model_name][variant_name] = var_res

            results["summary_table"].append({
                "Model": model_name,
                "Condition": variant_name,
                "Accuracy": var_res["accuracy"],
                "Macro F1": var_res["f1_macro"],
                "Macro Recall": var_res["recall_macro"],
                "Latency (ms)": var_res["mean_latency_ms"],
                "ECE": var_res["expected_calibration_error"],
                "Brier Score": var_res["brier_score"],
            })

    # Save metrics to JSON & CSV
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "clean_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(results["clean_evaluations"], f, indent=2)

    with open(METRICS_DIR / "robustness_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(results["robustness_evaluations"], f, indent=2)

    df_summary = pd.DataFrame(results["summary_table"])
    df_summary.to_csv(METRICS_DIR / "comparison_table.csv", index=False)
    logger.info(f"Saved complete comparative evaluation table to {METRICS_DIR / 'comparison_table.csv'}")

    # Generate charts
    generate_evaluation_charts(results)
    return results


def generate_evaluation_charts(results: Dict[str, Any], output_dir: Path | str = GRAPHS_DIR) -> None:
    """Generates comparative visualization charts for the research paper / results chapter."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")

    df = pd.DataFrame(results["summary_table"])

    # 1. Clean vs Degraded F1-Score Bar Chart
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Condition", y="Macro F1", hue="Model")
    plt.xticks(rotation=35, ha="right")
    plt.title("Model Macro F1-Score Across Clean and Degraded Test Sets (Robustness)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(out_path / "accuracy_f1_comparison.png", dpi=300)
    plt.close()

    # 2. Latency vs Macro F1 Scatter Plot
    clean_df = df[df["Condition"] == "Clean (Baseline)"]
    plt.figure(figsize=(8, 5))
    for _, row in clean_df.iterrows():
        plt.scatter(row["Latency (ms)"], row["Macro F1"], s=200, label=row["Model"])
        plt.annotate(
            row["Model"],
            (row["Latency (ms)"], row["Macro F1"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontweight="bold",
        )
    plt.xlabel("Mean Inference Latency per Image (ms)", fontsize=11)
    plt.ylabel("Clean Macro F1-Score", fontsize=11)
    plt.title("Accuracy vs. Inference Latency Trade-Off", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(out_path / "latency_vs_f1_scatter.png", dpi=300)
    plt.close()

    # 3. Calibration Reliability Diagrams
    fig, axes = plt.subplots(1, len(results["clean_evaluations"]), figsize=(15, 4.5), sharey=True)
    if len(results["clean_evaluations"]) == 1:
        axes = [axes]

    for ax, (model_name, res) in zip(axes, results["clean_evaluations"].items()):
        bins = res["calibration_info"]["bins"]
        confidences = [b["confidence"] for b in bins]
        accuracies = [b["accuracy"] for b in bins]

        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        ax.plot(confidences, accuracies, marker="o", linewidth=2, label=f"ECE: {res['expected_calibration_error']:.4f}")
        ax.set_title(f"{model_name} Reliability", fontsize=12)
        ax.set_xlabel("Mean Predicted Confidence")
        ax.set_ylabel("Empirical Accuracy")
        ax.legend(loc="upper left")

    plt.suptitle("Confidence Calibration Curves (Reliability Diagram)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path / "reliability_calibration_diagram.png", dpi=300)
    plt.close()

    logger.info(f"Generated evaluation figures saved in {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = run_full_evaluation()
    print("Full evaluation suite completed.")
