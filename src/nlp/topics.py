"""CO3 - Topic modelling of the question corpus.

  * LDA (gensim, probabilistic) - number of topics chosen by C_v coherence
  * NMF (scikit-learn, on TF-IDF) - matrix factorisation alternative
Then: average difficulty per topic, and a topic for any new question.
"""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import DATA_PROC, MODELS, REPORTS, SEED
from src.nlp.cleaning import preprocess

EXTRA_STOP = {"which", "what", "following", "how", "why", "who", "when", "where", "find", "one",
              "would", "use", "make", "get", "two", "let", "number", "value", "answer", "express",
              "passage", "most", "best", "likely", "true", "statement", "not", "all", "only", "more",
              "no", "few", "least"}


def tokens(text: str) -> list[str]:
    return [t for t in preprocess(text) if t.isalpha() and len(t) > 2 and t not in EXTRA_STOP]


def train(k_values=(8, 10, 12, 15, 20), sample=20000):
    df = pd.read_parquet(DATA_PROC / "train.parquet").sample(sample, random_state=SEED)
    docs = [tokens(t) for t in df.question]
    dic = Dictionary(docs)
    dic.filter_extremes(no_below=10, no_above=0.4)
    corpus = [dic.doc2bow(d) for d in docs]

    coh = {}
    models = {}
    for k in k_values:
        lda = LdaModel(corpus, id2word=dic, num_topics=k, passes=8, random_state=SEED, alpha="auto", eta="auto")
        c = CoherenceModel(model=lda, texts=docs, dictionary=dic, coherence="c_v", processes=1).get_coherence()
        coh[k] = c
        models[k] = lda
        print(f"[lda] k={k:2d}  coherence C_v={c:.4f}", flush=True)
    best_k = max(coh, key=coh.get)
    lda = models[best_k]
    lda.save(str(MODELS / "lda.model"))
    dic.save(str(MODELS / "lda.dict"))

    # dominant topic & difficulty per topic
    dom = [max(lda.get_document_topics(b, minimum_probability=0), key=lambda x: x[1])[0] for b in corpus]
    df["lda_topic"] = dom
    topics = []
    for t in range(best_k):
        sub = df[df.lda_topic == t]
        topics.append({"topic": t,
                       "top_words": [w for w, _ in lda.show_topic(t, 10)],
                       "n_questions": int(len(sub)),
                       "mean_score": round(float(sub.score.mean()), 1) if len(sub) else None,
                       "pct_hard": round(float((sub.level == "Hard").mean()), 3) if len(sub) else None,
                       "main_sources": sub.source.value_counts().head(3).to_dict()})

    # NMF
    texts = [" ".join(d) for d in docs]
    tf = TfidfVectorizer(max_features=8000, min_df=5, max_df=0.4)
    X = tf.fit_transform(texts)
    nmf = NMF(n_components=best_k, random_state=SEED, init="nndsvda", max_iter=400).fit(X)
    vocab = np.array(tf.get_feature_names_out())
    nmf_topics = [{"topic": i, "top_words": vocab[np.argsort(c)[::-1][:10]].tolist()} for i, c in enumerate(nmf.components_)]
    df["nmf_topic"] = nmf.transform(X).argmax(1)
    for t in nmf_topics:
        sub = df[df.nmf_topic == t["topic"]]
        t["n_questions"] = int(len(sub))
        t["mean_score"] = round(float(sub.score.mean()), 1) if len(sub) else None
    nmf_coh = CoherenceModel(topics=[t["top_words"] for t in nmf_topics], texts=docs, dictionary=dic,
                             coherence="c_v", processes=1).get_coherence()
    joblib.dump({"vectorizer": tf, "nmf": nmf}, MODELS / "nmf.joblib")

    out = {"lda_coherence_by_k": coh, "best_k": best_k, "lda_coherence": coh[best_k],
           "nmf_coherence": nmf_coh, "lda_topics": topics, "nmf_topics": nmf_topics}
    (REPORTS / "topics.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"[topics] best k={best_k}  LDA C_v={coh[best_k]:.4f}  NMF C_v={nmf_coh:.4f}")
    for t in sorted(topics, key=lambda x: x["mean_score"] or 0):
        print(f"  T{t['topic']:2d} score={t['mean_score']}  {' '.join(t['top_words'][:7])}")
    return out


@lru_cache(maxsize=1)
def _load():
    lda = LdaModel.load(str(MODELS / "lda.model"))
    dic = Dictionary.load(str(MODELS / "lda.dict"))
    info = json.loads((REPORTS / "topics.json").read_text())
    return lda, dic, {t["topic"]: t for t in info["lda_topics"]}


def topic_of(text: str) -> dict:
    lda, dic, info = _load()
    dist = sorted(lda.get_document_topics(dic.doc2bow(tokens(text)), minimum_probability=0.05), key=lambda x: -x[1])
    if not dist:
        return {"topic": None, "probability": 0, "top_words": [], "topic_mean_score": None}
    t, p = dist[0]
    return {"topic": int(t), "probability": round(float(p), 3), "top_words": info[t]["top_words"][:6],
            "topic_mean_score": info[t]["mean_score"],
            "distribution": [(int(a), round(float(b), 3)) for a, b in dist[:3]]}


if __name__ == "__main__":
    train()
