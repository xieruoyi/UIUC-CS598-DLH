"""
Example usage of SEERDataset + SEERSurvivalPrediction.

This script expects a preprocessed file at:
    <root>/processed/seer_ml_ready.csv

Ablation study:
1. Train Logistic Regression using all features.
2. Train Logistic Regression after removing 'year_dx' if present
   (otherwise remove the last feature as a fallback).

This script demonstrates:
- dataset loading
- task processing
- simple train/test split
- baseline model training
- feature ablation comparison

Example:
    python examples/seer_survival_prediction_lr.py --root "C:/Users/xieru/Desktop/CS 598 DLH"
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pyhealth.datasets import SEERDataset
from pyhealth.tasks.seer_survival_prediction import SEERSurvivalPrediction


def evaluate_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float]:
    """Train and evaluate a Logistic Regression model."""
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)

    auc = roc_auc_score(y_test, prob)
    acc = accuracy_score(y_test, pred)
    return auc, acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SEER survival prediction baseline and feature ablation."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing processed/seer_ml_ready.csv",
    )
    args = parser.parse_args()

    print("Loading dataset...")
    print(f"Using root: {args.root}")

    dataset = SEERDataset(
        root=args.root,
        tables=["seer"],
    )

    task = SEERSurvivalPrediction()
    samples = dataset.set_task(task)

    print("Total samples:", len(samples))

    X = np.asarray([samples[i]["features"] for i in range(len(samples))])
    y = np.asarray([int(samples[i]["label"].item()) for i in range(len(samples))])

    print("Feature dimension:", X.shape[1])
    print("Positive ratio:", y.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # -------------------------------
    # Baseline: all features
    # -------------------------------
    print("\nTraining baseline model (all features)...")
    auc_full, acc_full = evaluate_model(X_train, X_test, y_train, y_test)

    print(f"AUROC (all features): {auc_full:.6f}")
    print(f"Accuracy (all features): {acc_full:.6f}")

    # -------------------------------
    # Ablation: remove year_dx if present
    # -------------------------------
    feature_names = task.feature_names or []
    print("\nFeature names:")
    print(feature_names)

    if "year_dx" in feature_names:
        remove_idx = feature_names.index("year_dx")
        removed_feature = "year_dx"
    else:
        remove_idx = X.shape[1] - 1
        removed_feature = (
            feature_names[remove_idx] if feature_names else f"feature_{remove_idx}"
        )

    print(f"\nTraining ablation model (remove feature: {removed_feature})...")

    X_train_ab = np.delete(X_train, remove_idx, axis=1)
    X_test_ab = np.delete(X_test, remove_idx, axis=1)

    auc_ab, acc_ab = evaluate_model(X_train_ab, X_test_ab, y_train, y_test)

    print(f"AUROC (feature removed): {auc_ab:.6f}")
    print(f"Accuracy (feature removed): {acc_ab:.6f}")

    # -------------------------------
    # Comparison
    # -------------------------------
    print("\n========== RESULTS ==========")
    print(f"Removed feature: {removed_feature}")
    print(f"Full feature count: {X_train.shape[1]}")
    print(f"Ablation feature count: {X_train_ab.shape[1]}")
    print(f"Full feature AUROC: {auc_full:.6f}")
    print(f"Ablation AUROC: {auc_ab:.6f}")
    print(f"AUROC difference: {auc_full - auc_ab:.6f}")
    print(f"Full feature Accuracy: {acc_full:.6f}")
    print(f"Ablation Accuracy: {acc_ab:.6f}")
    print(f"Accuracy difference: {acc_full - acc_ab:.6f}")


if __name__ == "__main__":
    main()