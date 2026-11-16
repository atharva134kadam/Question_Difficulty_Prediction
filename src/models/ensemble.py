"""Weighted ensemble of LightGBM + DistilBERT + BiLSTM.

1. Level probabilities: weighted average, weights grid-searched on VALIDATION macro-F1.
2. Score: alpha * (weighted regression score) + (1 - alpha) * (probability-weighted
   expected score), alpha tuned on validation MAE.
3. Consistency: the final score is clipped into the band of the predicted level
   (Easy 0-33, Medium 34-66, Hard 67-100) so level and score never contradict.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from src.config import DATA_PROC, LEVEL_CENTERS, MODELS, REPORTS
from src.evaluation import full_report, save_metrics

SOURCES = {
    "gbm": (MODELS / "gbm_{}_probs.npy", MODELS / "gbm_{}_scores.npy"),
    "bert": (MODELS / "distilbert_difficulty" / "{}_probs.npy", MODELS / "distilbert_difficulty" / "{}_scores.npy"),
    "bilstm": (MODELS / "bilstm" / "{}_probs.npy", MODELS / "bilstm" / "{}_scores.npy"),
}
BANDS = [(0.0, 33.3), (33.4, 66.6), (66.7, 100.0)]


def combine(probs: dict, scores: dict, w: dict, alpha: float):
    P = sum(w[k] * probs[k] for k in w)
    P = P / P.sum(1, keepdims=True)
    reg = sum(w[k] * scores[k] for k in w) / sum(w.values())
    ev = P @ np.array(LEVEL_CENTERS)
    s = alpha * reg + (1 - alpha) * ev
    lvl = P.argmax(1)
    lo = np.array([BANDS[l][0] for l in lvl]); hi = np.array([BANDS[l][1] for l in lvl])
    return P, lvl, np.clip(s, lo, hi)


def _load(split):
    probs, scores = {}, {}
    for k, (pp, sp) in SOURCES.items():
        p, s = pp.with_name(pp.name.format(split)), sp.with_name(sp.name.format(split))
        if p.exists():
            probs[k], scores[k] = np.load(p), np.load(s)
    return probs, scores


def fit():
    va, te = pd.read_parquet(DATA_PROC / "val.parquet"), pd.read_parquet(DATA_PROC / "test.parquet")
    pv, sv = _load("val")
    names = list(pv)
    print("[ensemble] members:", names)
    grid = np.arange(0, 1.01, 0.1)
    best = None
    for ws in itertools.product(grid, repeat=len(names)):
        if abs(sum(ws) - 1) > 1e-6:
            continue
        w = dict(zip(names, ws))
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            _, lvl, s = combine(pv, sv, w, alpha)
            m = full_report(va.label, lvl, va.score, s)
            key = (round(m["macro_f1"], 4), -m["mae"])
            if best is None or key > best[0]:
                best = (key, w, alpha)
    _, w, alpha = best
    cfg = {"weights": {k: float(v) for k, v in w.items()}, "alpha": alpha}
    (MODELS / "ensemble.json").write_text(json.dumps(cfg, indent=2))
    pt, st = _load("test")
    P, lvl, s = combine(pt, st, w, alpha)
    m = full_report(te.label, lvl, te.score, s)
    save_metrics("model::Ensemble (final)", m)
    print("[ensemble] config", cfg)
    print("[ensemble] TEST", {k: v for k, v in m.items() if k != "confusion_matrix"})

    # per-source analysis on the test set
    te = te.assign(pred_label=lvl, pred_score=s)
    per = te.groupby("group").apply(lambda d: pd.Series({
        "n": len(d), "target_score": d.score.iloc[0], "mean_pred_score": d.pred_score.mean(),
        "level_accuracy": (d.pred_label == d.label).mean()}), include_groups=False).sort_values("target_score")
    per.round(3).to_csv(REPORTS / "per_source_results.csv")
    te[["uid", "source", "group", "question", "level", "score", "pred_label", "pred_score"]].to_csv(
        REPORTS / "test_predictions.csv", index=False)
    print(per.round(3))
    return cfg


if __name__ == "__main__":
    fit()
