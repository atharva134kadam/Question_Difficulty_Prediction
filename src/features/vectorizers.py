"""CO1 - Vectorisation of unstructured question text.

Compares five representations on the same downstream classifier:
  1. Bag-of-Words (counts, unigrams)
  2. TF-IDF word 1-2 grams
  3. TF-IDF word 1-2 grams + character 3-5 grams
  4. Word2Vec (trained locally on our corpus, mean-pooled)
  5. Hand-crafted linguistic features (readability, syntax, Bloom, tone)
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PROC, MODELS, REPORTS, SEED
from src.evaluation import full_report, save_metrics
from src.nlp.cleaning import preprocess


class TfidfWordChar:
    """Word (1-2) + character (3-5) TF-IDF stacked horizontally."""

    def __init__(self, max_word=40000, max_char=60000):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_word,
                                    sublinear_tf=True, lowercase=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                    max_features=max_char, sublinear_tf=True)

    def fit_transform(self, texts):
        return hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()

    def transform(self, texts):
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()


class W2VEncoder:
    """Train Word2Vec on processed tokens and mean-pool word vectors."""

    def __init__(self, dim=100):
        self.dim = dim
        self.model = None

    def fit(self, token_lists):
        self.model = Word2Vec(token_lists, vector_size=self.dim, window=5, min_count=2,
                              workers=4, sg=1, epochs=10, seed=SEED)
        return self

    def transform(self, token_lists):
        kv = self.model.wv
        out = np.zeros((len(token_lists), self.dim), dtype=np.float32)
        for i, toks in enumerate(token_lists):
            vecs = [kv[t] for t in toks if t in kv]
            if vecs:
                out[i] = np.mean(vecs, axis=0)
        return out


def _load():
    d = {s: pd.read_parquet(DATA_PROC / f"{s}.parquet") for s in ("train", "val", "test")}
    f = {s: pd.read_parquet(DATA_PROC / f"features_{s}.parquet").drop(columns="uid") for s in d}
    return d, f


def run():
    d, f = _load()
    tr, te = d["train"], d["test"]
    toks = {s: [preprocess(t) for t in d[s].question] for s in ("train", "test")}
    joined = {s: [" ".join(t) for t in toks[s]] for s in toks}

    experiments = {}

    bow = CountVectorizer(min_df=2, max_features=30000)
    experiments["BoW (counts)"] = (bow.fit_transform(joined["train"]), bow.transform(joined["test"]))

    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000, sublinear_tf=True)
    experiments["TF-IDF word 1-2gram"] = (tf.fit_transform(joined["train"]), tf.transform(joined["test"]))

    twc = TfidfWordChar()
    experiments["TF-IDF word+char"] = (twc.fit_transform(tr.question), twc.transform(te.question))

    w2v = W2VEncoder().fit(toks["train"])
    w2v.model.save(str(MODELS / "word2vec.model"))
    experiments["Word2Vec (mean)"] = (w2v.transform(toks["train"]), w2v.transform(toks["test"]))

    sc = StandardScaler()
    experiments["Linguistic features"] = (sc.fit_transform(f["train"]), sc.transform(f["test"]))

    rows = []
    for name, (Xtr, Xte) in experiments.items():
        t = time.time()
        clf = LogisticRegression(max_iter=2000, C=2.0)
        clf.fit(Xtr, tr.label)
        reg = Ridge(alpha=1.0).fit(Xtr, tr.score)
        m = full_report(te.label, clf.predict(Xte), te.score, reg.predict(Xte))
        m["dims"] = int(Xtr.shape[1])
        m["seconds"] = round(time.time() - t, 1)
        rows.append({"vectorizer": name, **{k: v for k, v in m.items() if k != "confusion_matrix"}})
        save_metrics(f"vec::{name}", m)
        print(f"[vec] {name:24s} acc={m['accuracy']:.3f} f1={m['macro_f1']:.3f} mae={m['mae']:.2f}")

    table = pd.DataFrame(rows)
    table.to_csv(REPORTS / "vectorizer_comparison.csv", index=False)

    # Persist the TF-IDF word+char baseline (used later in the ensemble + app)
    Xtr, _ = experiments["TF-IDF word+char"]
    clf = LogisticRegression(max_iter=2000, C=2.0).fit(Xtr, tr.label)
    reg = Ridge(alpha=1.0).fit(Xtr, tr.score)
    joblib.dump({"vectorizer": twc, "clf": clf, "reg": reg}, MODELS / "baseline_tfidf.joblib")
    return table


if __name__ == "__main__":
    # import through the package so pickled classes reference src.features.vectorizers, not __main__
    from src.features import vectorizers as _v
    print(_v.run())
