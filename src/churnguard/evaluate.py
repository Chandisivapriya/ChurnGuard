"""Metrics and cost-based threshold selection."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def expected_cost(y_true, y_proba, threshold: float, fn_cost: float, fp_cost: float) -> float:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    _, fp, fn, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fn * fn_cost + fp * fp_cost)


def best_threshold(y_true, y_proba, fn_cost: float, fp_cost: float) -> float:
    """Pick the probability cut-off that minimises total business cost."""
    grid = np.linspace(0.05, 0.95, 91)
    costs = [expected_cost(y_true, y_proba, t, fn_cost, fp_cost) for t in grid]
    return float(grid[int(np.argmin(costs))])
