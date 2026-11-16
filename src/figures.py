"""Generate all report figures into reports/figures/."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import DATA_PROC, FIGURES, LEVELS, REPORTS, SEED

PAL = {"Easy": "#2e9d5b", "Medium": "#e0a100", "Hard": "#d6453d"}
sns.set_theme(style="whitegrid", font_scale=0.95)


def _save(name):
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=130)
    plt.close()


def data_figures():
    m = pd.read_csv(DATA_PROC / "modelling_set.csv")
    plt.figure(figsize=(10, 4.5))
    order = m.groupby("group").score.first().sort_values().index
    sns.countplot(data=m, x="group", hue="level", order=order, palette=PAL, hue_order=LEVELS, dodge=False)
    plt.xticks(rotation=45, ha="right"); plt.title("Modelling set: questions per source group (ordered by anchor score)")
    _save("data_distribution.png")

    f = pd.read_parquet(DATA_PROC / "features_train.parquet")
    tr = pd.read_parquet(DATA_PROC / "train.parquet")
    f["level"] = tr.level.values
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.8))
    for a, col, t in zip(ax, ["fk_grade", "rare_word_ratio", "n_words", "max_tree_depth"],
                         ["Flesch-Kincaid grade", "Rare-word ratio", "Words", "Max parse-tree depth"]):
        sns.boxplot(data=f, x="level", y=col, order=LEVELS, palette=PAL, ax=a, showfliers=False, hue="level", legend=False)
        a.set_title(t); a.set_xlabel("")
    _save("readability_by_level.png")

    b = f.groupby(["bloom_level", "level"]).size().unstack().reindex(columns=LEVELS)
    b.index = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"][:len(b)]
    (b.div(b.sum(1), axis=0)).plot(kind="bar", stacked=True, color=[PAL[l] for l in LEVELS], figsize=(7, 4))
    plt.title("Difficulty mix by detected Bloom's level"); plt.ylabel("share"); plt.xticks(rotation=0)
    _save("bloom_vs_difficulty.png")


def embedding_tsne(n=2500):
    from gensim.models import Word2Vec
    from sklearn.manifold import TSNE

    from src.config import MODELS
    from src.features.vectorizers import W2VEncoder
    from src.nlp.cleaning import preprocess
    te = pd.read_parquet(DATA_PROC / "test.parquet").sample(n, random_state=SEED)
    enc = W2VEncoder(); enc.model = Word2Vec.load(str(MODELS / "word2vec.model"))
    X = enc.transform([preprocess(t) for t in te.question])
    Z = TSNE(2, random_state=SEED, perplexity=40, init="pca").fit_transform(X)
    plt.figure(figsize=(7, 6))
    for l in LEVELS:
        mk = te.level.values == l
        plt.scatter(Z[mk, 0], Z[mk, 1], s=6, alpha=0.6, c=PAL[l], label=l)
    plt.legend(markerscale=3); plt.title("t-SNE of Word2Vec question embeddings"); plt.xticks([]); plt.yticks([])
    _save("embedding_tsne.png")


def model_figures():
    mt = json.loads((REPORTS / "metrics.json").read_text())
    pref = "orig::" if any(k.startswith("orig::") for k in mt) else "model::"   # original 4,183-question test set
    rows = [{"model": k.split("::", 1)[1], **{m: v for m, v in d.items() if m != "confusion_matrix"}}
            for k, d in mt.items() if k.startswith(pref)]
    md = pd.DataFrame(rows).sort_values("macro_f1")
    md.to_csv(REPORTS / "model_comparison.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].barh(md.model, md.macro_f1, color="#4c72b0"); ax[0].set_title("Level: macro-F1 (higher is better)")
    for i, v in enumerate(md.macro_f1):
        ax[0].text(v + 0.005, i, f"{v:.3f}", va="center")
    ax[0].set_xlim(0, 1)
    ax[1].barh(md.model, md.mae, color="#dd8452"); ax[1].set_title("Score: MAE in points (lower is better)")
    for i, v in enumerate(md.mae):
        ax[1].text(v + 0.1, i, f"{v:.2f}", va="center")
    ax[1].set_yticklabels([])
    _save("model_comparison.png")
    if pref + "Ensemble (final)" in mt:
        cm = np.array(mt[pref + "Ensemble (final)"]["confusion_matrix"])
        plt.figure(figsize=(4.8, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LEVELS, yticklabels=LEVELS)
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Final ensemble: confusion matrix")
        _save("confusion_matrix_ensemble.png")
    tp = REPORTS / "test_predictions.csv"
    if tp.exists():
        d = pd.read_csv(tp)
        d = d[d.source != "synthetic_arith"]
        plt.figure(figsize=(7, 4))
        for l in LEVELS:
            sns.kdeplot(d[d.level == l].pred_score, fill=True, color=PAL[l], label=f"true {l}", clip=(0, 100))
        for x in (33.3, 66.7):
            plt.axvline(x, ls="--", c="grey")
        plt.xlabel("predicted score"); plt.legend(); plt.title("Predicted score distribution by true level")
        _save("score_distribution.png")


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["data", "tsne", "models"]
    if "data" in what: data_figures()
    if "tsne" in what: embedding_tsne()
    if "models" in what: model_figures()
    print("figures:", sorted(p.name for p in FIGURES.glob("*.png")))
