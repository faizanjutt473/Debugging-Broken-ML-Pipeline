"""
utils.py
--------
Shared helpers: metric calculation, pretty printing and simple data diagnostics.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    """Return a full set of classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def print_metrics(title: str, metrics: Dict[str, float], y_true, y_pred) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    for name, value in metrics.items():
        print(f"{name:<10}: {value:.4f}")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"Confusion matrix -> TN={tn}  FP={fp}  FN={fn}  TP={tp}")


# ----------------------------------------------------------------------
# Diagnostics used to DETECT flaws (referenced in the report)
# ----------------------------------------------------------------------
def find_leaky_features(X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> List[str]:
    """Flag features whose absolute correlation with the target is suspiciously high."""
    corr = X.select_dtypes("number").corrwith(y).abs().sort_values(ascending=False)
    return corr[corr > threshold].index.tolist()


def report_data_quality(df: pd.DataFrame, target: str) -> None:
    """Print duplicates, missing values and class balance."""
    print("\n--- Data quality diagnostics ---")
    print(f"Rows                : {len(df)}")
    print(f"Duplicate rows      : {df.duplicated().sum()}")
    print(f"Missing values      : {int(df.isna().sum().sum())}")
    print("Class distribution  :")
    print(df[target].value_counts(normalize=True).round(4).to_string())
