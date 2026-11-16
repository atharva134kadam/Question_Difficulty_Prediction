"""CO2 - Sentiment analysis.

Three approaches evaluated on SST-2 (Stanford Sentiment Treebank) validation set:
  1. VADER                         - lexicon + rule based (no training)
  2. TF-IDF + Multinomial Naive Bayes
  3. TF-IDF + Logistic Regression  - selected model, saved for the app

Then applied to our questions: does question tone (polarity / negativity)
relate to difficulty? (results in reports/sentiment_vs_difficulty.csv)
"""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB

from src.config import DATA_PROC, DATA_RAW, MODELS, REPORTS

MODEL_PATH = MODELS / "sentiment_lr.joblib"


def train_and_evaluate() -> dict:
    tr = pd.read_parquet(DATA_RAW / "sst2" / "train.parquet")
    va = pd.read_parquet(DATA_RAW / "sst2" / "validation.parquet")  # SST-2 test labels are hidden
    results = {}

    from nltk.sentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
    vp = [int(vader.polarity_scores(s)["compound"] >= 0) for s in va.sentence]
    results["VADER (rule-based)"] = {"accuracy": accuracy_score(va.label, vp), "macro_f1": f1_score(va.label, vp, average="macro")}

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr, Xva = vec.fit_transform(tr.sentence), vec.transform(va.sentence)

    nb = MultinomialNB(alpha=0.5).fit(Xtr, tr.label)
    p = nb.predict(Xva)
    results["TF-IDF + Naive Bayes"] = {"accuracy": accuracy_score(va.label, p), "macro_f1": f1_score(va.label, p, average="macro")}

    lr = LogisticRegression(max_iter=3000, C=4.0).fit(Xtr, tr.label)
    p = lr.predict(Xva)
    results["TF-IDF + Logistic Regression"] = {"accuracy": accuracy_score(va.label, p), "macro_f1": f1_score(va.label, p, average="macro")}
    results["classification_report_lr"] = classification_report(va.label, p, target_names=["negative", "positive"], output_dict=True)

    joblib.dump({"vectorizer": vec, "model": lr}, MODEL_PATH)
    (REPORTS / "sentiment_metrics.json").write_text(json.dumps(results, indent=2))
    for k, v in results.items():
        if isinstance(v, dict) and "macro_f1" in v:
            print(f"[sentiment] {k:30s} acc={v['accuracy']:.4f} f1={v['macro_f1']:.4f}")
    return results


@lru_cache(maxsize=1)
def _load():
    from nltk.sentiment import SentimentIntensityAnalyzer
    return joblib.load(MODEL_PATH), SentimentIntensityAnalyzer()


def analyze(text: str) -> dict:
    m, vader = _load()
    pos = float(m["model"].predict_proba(m["vectorizer"].transform([text]))[0, 1])
    v = vader.polarity_scores(text)
    label = "positive" if v["compound"] >= 0.05 else "negative" if v["compound"] <= -0.05 else "neutral"
    return {"ml_positive_prob": round(pos, 3), "vader_compound": v["compound"], "vader_label": label,
            "vader_pos": v["pos"], "vader_neg": v["neg"], "vader_neu": v["neu"]}


def sentiment_vs_difficulty() -> pd.DataFrame:
    """Apply sentiment to the question corpus and relate it to difficulty."""
    df = pd.read_parquet(DATA_PROC / "test.parquet")
    m, vader = _load()
    df["ml_positive_prob"] = m["model"].predict_proba(m["vectorizer"].transform(df.question))[:, 1]
    df["vader_compound"] = [vader.polarity_scores(t)["compound"] for t in df.question]
    df["tone"] = pd.cut(df.vader_compound, [-1.01, -0.05, 0.05, 1.01], labels=["negative", "neutral", "positive"])
    out = df.groupby("level").agg(n=("uid", "size"), vader_mean=("vader_compound", "mean"),
                                  ml_pos_mean=("ml_positive_prob", "mean"),
                                  pct_negative=("tone", lambda s: (s == "negative").mean()),
                                  pct_neutral=("tone", lambda s: (s == "neutral").mean()))
    out.to_csv(REPORTS / "sentiment_vs_difficulty.csv")
    print(out)
    return out


if __name__ == "__main__":
    train_and_evaluate()
    sentiment_vs_difficulty()
