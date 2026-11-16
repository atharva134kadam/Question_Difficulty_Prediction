"""LightGBM difficulty model on interpretable features.

Inputs = 57 hand-crafted linguistic features (readability, lexical, syntactic, Bloom, tone)
       + 4 stacked features from the TF-IDF model (out-of-fold P(easy/medium/hard) and score)
Outputs = level classifier + 0-100 score regressor. Explained with SHAP.
"""
from __future__ import annotations

import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold

from src.config import DATA_PROC, MODELS, REPORTS, SEED
from src.evaluation import full_report, save_metrics
from src.features.vectorizers import TfidfWordChar

STACK_COLS = ["tfidf_p_easy", "tfidf_p_medium", "tfidf_p_hard", "tfidf_score"]


def _load(split):
    d = pd.read_parquet(DATA_PROC / f"{split}.parquet")
    f = pd.read_parquet(DATA_PROC / f"features_{split}.parquet").drop(columns="uid")
    return d, f


def oof_tfidf(train: pd.DataFrame) -> np.ndarray:
    """Out-of-fold TF-IDF predictions so the GBM never sees leaked train predictions."""
    oof = np.zeros((len(train), 4))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for k, (a, b) in enumerate(skf.split(train, train.label)):
        v = TfidfWordChar()
        Xa = v.fit_transform(train.question.iloc[a])
        Xb = v.transform(train.question.iloc[b])
        oof[b, :3] = LogisticRegression(max_iter=2000, C=2.0).fit(Xa, train.label.iloc[a]).predict_proba(Xb)
        oof[b, 3] = Ridge(alpha=1.0).fit(Xa, train.score.iloc[a]).predict(Xb)
        print(f"  oof fold {k + 1}/5", flush=True)
    return oof


def tfidf_stack(texts, base=None) -> np.ndarray:
    base = base or joblib.load(MODELS / "baseline_tfidf.joblib")
    X = base["vectorizer"].transform(list(texts))
    return np.column_stack([base["clf"].predict_proba(X), base["reg"].predict(X)])


def train():
    (tr, ftr), (va, fva), (te, fte) = _load("train"), _load("val"), _load("test")
    base = joblib.load(MODELS / "baseline_tfidf.joblib")
    Xtr = pd.concat([ftr, pd.DataFrame(oof_tfidf(tr), columns=STACK_COLS)], axis=1)
    Xva = pd.concat([fva, pd.DataFrame(tfidf_stack(va.question, base), columns=STACK_COLS)], axis=1)
    Xte = pd.concat([fte, pd.DataFrame(tfidf_stack(te.question, base), columns=STACK_COLS)], axis=1)

    params = dict(n_estimators=2000, learning_rate=0.03, num_leaves=63, min_child_samples=30,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=SEED, verbose=-1, n_jobs=4)
    clf = lgb.LGBMClassifier(**params).fit(Xtr, tr.label, eval_set=[(Xva, va.label)],
                                           callbacks=[lgb.early_stopping(100, verbose=False)])
    reg = lgb.LGBMRegressor(**params).fit(Xtr, tr.score, eval_set=[(Xva, va.score)],
                                          callbacks=[lgb.early_stopping(100, verbose=False)])
    joblib.dump({"clf": clf, "reg": reg, "columns": list(Xtr.columns)}, MODELS / "lightgbm.joblib")

    for split, X in (("val", Xva), ("test", Xte)):
        np.save(MODELS / f"gbm_{split}_probs.npy", clf.predict_proba(X))
        np.save(MODELS / f"gbm_{split}_scores.npy", reg.predict(X))

    # Linguistic-only GBM (ablation: how far do interpretable features alone go?)
    clf_l = lgb.LGBMClassifier(**params).fit(ftr, tr.label, eval_set=[(fva, va.label)],
                                             callbacks=[lgb.early_stopping(100, verbose=False)])
    reg_l = lgb.LGBMRegressor(**params).fit(ftr, tr.score, eval_set=[(fva, va.score)],
                                            callbacks=[lgb.early_stopping(100, verbose=False)])
    m_l = full_report(te.label, clf_l.predict(fte), te.score, reg_l.predict(fte))
    save_metrics("model::LightGBM (linguistic features only)", m_l)

    # TF-IDF baseline (for the comparison table)
    p_base = tfidf_stack(te.question, base)
    save_metrics("model::TF-IDF + LogReg/Ridge (baseline)",
                 full_report(te.label, p_base[:, :3].argmax(1), te.score, p_base[:, 3]))

    m = full_report(te.label, clf.predict(Xte), te.score, reg.predict(Xte))
    save_metrics("model::LightGBM (linguistic + stacked TF-IDF)", m)
    print("[gbm] linguistic-only TEST", {k: v for k, v in m_l.items() if k != "confusion_matrix"})
    print("[gbm] stacked TEST       ", {k: v for k, v in m.items() if k != "confusion_matrix"})

    imp = pd.Series(reg.booster_.feature_importance("gain"), index=Xtr.columns).sort_values(ascending=False)
    imp.to_csv(REPORTS / "gbm_feature_importance.csv", header=["gain"])
    shap_summary(reg, Xte.sample(1000, random_state=SEED))


def shap_summary(reg, X):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    from src.config import FIGURES
    sv = shap.TreeExplainer(reg).shap_values(X)
    plt.figure()
    shap.summary_plot(sv, X, show=False, max_display=20)
    plt.title("SHAP: features driving the difficulty score")
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_summary.png", dpi=130)
    plt.close()
    mean_abs = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)
    mean_abs.to_csv(REPORTS / "shap_mean_abs.csv", header=["mean_abs_shap"])
    print("[gbm] top SHAP features:", mean_abs.head(10).round(2).to_dict())


if __name__ == "__main__":
    train()
