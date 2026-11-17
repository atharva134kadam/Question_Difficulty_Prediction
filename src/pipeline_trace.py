"""Step-by-step trace of one prediction, for the app's "Pipeline" visualisation.

Runs the real inference path once and records the intermediate output of every stage:
input -> language/translation -> cleaning -> tokenisation -> TF-IDF vectors -> linguistic
features -> info extraction -> stage-1 TF-IDF model -> LightGBM -> DistilBERT -> BiLSTM
(explainer) -> weighted ensemble -> final level + score.
"""
from __future__ import annotations

import numpy as np

from src.config import LEVEL_CENTERS, LEVELS
from src.features.linguistic import FEATURE_GROUPS
from src.models.ensemble import BANDS
from src.nlp.cleaning import preprocess_steps


def _top_terms(vec, X, k=8):
    row = X.tocsr()
    if row.nnz == 0:
        return []
    names = vec.get_feature_names_out()
    idx = np.argsort(-row.data)[:k]
    return [(str(names[row.indices[i]]), round(float(row.data[i]), 3)) for i in idx]


def trace(pred, text: str) -> dict:
    r = pred.predict(text, explain=True)
    q = r["cleaned"]

    # ---- CO1 preprocessing (the classic NLP pipeline shown for teaching / used by topics & W2V)
    pre = preprocess_steps(r["english"])

    # ---- vectorisation (TF-IDF word 1-2 grams + char 3-5 grams)
    v = pred.base["vectorizer"]
    Xw, Xc = v.word.transform([q]), v.char.transform([q])
    tfidf = {"word_dim": len(v.word.vocabulary_), "char_dim": len(v.char.vocabulary_),
             "word_nonzero": int(Xw.nnz), "char_nonzero": int(Xc.nnz),
             "top_word": _top_terms(v.word, Xw), "top_char": _top_terms(v.char, Xc)}

    # ---- stage-1 text model (LogReg + Ridge on TF-IDF) -> 4 numbers stacked into LightGBM
    X = v.transform([q])
    p1, s1 = pred.base["clf"].predict_proba(X)[0], float(pred.base["reg"].predict(X)[0])
    stage1 = {"probabilities": {l: round(float(p1[i]), 3) for i, l in enumerate(LEVELS)}, "score": round(s1, 1)}

    # ---- features grouped
    f = r["features"]
    groups = {g: {c: round(f[c], 3) for c in cols if c in f} for g, cols in FEATURE_GROUPS.items()}

    # ---- re-run members to show their individual probabilities
    feats, Xg, _ = pred._gbm_inputs([q])
    probs, scores = pred._members([q], Xg)
    members = {k: {"probabilities": {l: round(float(probs[k][0, i]), 3) for i, l in enumerate(LEVELS)},
                   "score": round(float(scores[k][0]), 1), "level": LEVELS[int(probs[k][0].argmax())]}
               for k in probs}

    # ---- DistilBERT sub-word tokens
    bert = None
    if pred.bert:
        tok = pred.bert[1]
        pieces = tok.tokenize(q)[:94]
        bert = {"tokens": ["[CLS]"] + pieces + ["[SEP]"], "n_tokens": len(pieces) + 2, "max_len": 96}

    # ---- ensemble arithmetic, written out
    w_all = pred.cfg["weights"]
    w = {k: float(v) for k, v in w_all.items() if k in probs and v > 0}
    alpha = float(pred.cfg["alpha"])
    P = sum(w[k] * probs[k][0] for k in w)
    P = P / P.sum()
    reg = sum(w[k] * float(scores[k][0]) for k in w) / sum(w.values())
    ev = float(P @ np.array(LEVEL_CENTERS))
    raw = alpha * reg + (1 - alpha) * ev
    lvl = int(P.argmax())
    lo, hi = BANDS[lvl]
    ensemble = {"weights": {k: round(float(v), 2) for k, v in w_all.items()}, "alpha": alpha,
                "probabilities": {l: round(float(P[i]), 3) for i, l in enumerate(LEVELS)},
                "regression_score": round(reg, 2), "expected_score": round(ev, 2), "raw_score": round(raw, 2),
                "band": [lo, hi], "clipped": not (lo <= raw <= hi), "final_score": r["score"], "level": r["level"]}

    return {
        "input": text,
        "language": r["language"], "english": r["english"],
        "cleaned": q,
        "preprocess": pre,
        "tfidf": tfidf,
        "features": groups, "n_features": len(f),
        "info": r.get("info"), "sentiment": r.get("sentiment"), "topic": r.get("topic"),
        "stage1": stage1,
        "members": members,
        "shap": r.get("shap"),
        "bert": bert,
        "attention": r.get("attention"),
        "ensemble": ensemble,
        "result": {k: r[k] for k in ("level", "score", "confidence", "probabilities")},
    }
