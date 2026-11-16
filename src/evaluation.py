"""Shared metrics for level classification and 0-100 score regression."""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error)

from src.config import REPORTS, score_to_level, LEVEL2ID


def level_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }


def score_metrics(s_true, s_pred) -> dict:
    s_pred = np.clip(s_pred, 0, 100)
    return {
        "mae": mean_absolute_error(s_true, s_pred),
        "rmse": float(np.sqrt(mean_squared_error(s_true, s_pred))),
        "spearman": float(spearmanr(s_true, s_pred).statistic),
        "band_accuracy": float(np.mean([score_to_level(a) == score_to_level(b) for a, b in zip(s_true, s_pred)])),
    }


def full_report(y_true, y_pred, s_true, s_pred) -> dict:
    r = level_metrics(y_true, y_pred)
    r.update(score_metrics(s_true, s_pred))
    r["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}


def save_metrics(name: str, metrics: dict) -> None:
    path = REPORTS / "metrics.json"
    allm = json.loads(path.read_text()) if path.exists() else {}
    allm[name] = metrics
    path.write_text(json.dumps(allm, indent=2, default=float))


def levels_from_scores(scores) -> np.ndarray:
    return np.array([LEVEL2ID[score_to_level(s)] for s in scores])
