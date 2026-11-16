"""Question Difficulty Predictor - single entry point used by CLI, app and notebooks.

    from src.predict import get_predictor
    p = get_predictor()
    p.predict("Prove that there are infinitely many primes.")
    -> {"level": "Hard", "score": 81.4, "confidence": 0.86, "probabilities": {...}, ...}

CLI:
    python -m src.predict "your question"
    python -m src.predict --file question_paper.pdf      (pdf / docx / txt / csv)
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.config import LEVELS, MODELS
from src.features.linguistic import extract_features
from src.models.ensemble import combine
from src.nlp.cleaning import clean_text


class DifficultyPredictor:
    def __init__(self, use_bert: bool = True, use_bilstm: bool = True):
        self.base = joblib.load(MODELS / "baseline_tfidf.joblib")
        g = joblib.load(MODELS / "lightgbm.joblib")
        self.gbm_clf, self.gbm_reg, self.gbm_cols = g["clf"], g["reg"], g["columns"]
        cfg_path = MODELS / "ensemble.json"
        self.cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {"weights": {"gbm": 1.0}, "alpha": 0.5}
        self.bert = self.bilstm = None
        w = self.cfg["weights"]
        if use_bert and w.get("bert", 0) > 0 and (MODELS / "distilbert_difficulty" / "heads.pt").exists():
            from src.models.transformer import load_trained
            self.bert = load_trained()
        if use_bilstm and (MODELS / "bilstm" / "model.pt").exists():
            from src.models.bilstm import load_trained as lb
            self.bilstm = lb()
        self._explainer = None
        self.members = ["gbm"] + (["bert"] if self.bert else []) + (["bilstm"] if self.bilstm and w.get("bilstm", 0) > 0 else [])

    # ------------------------------------------------------------------
    def _gbm_inputs(self, questions):
        feats = extract_features(questions)
        X = self.base["vectorizer"].transform(questions)
        stack = np.column_stack([self.base["clf"].predict_proba(X), self.base["reg"].predict(X)])
        Xg = pd.concat([feats, pd.DataFrame(stack, columns=["tfidf_p_easy", "tfidf_p_medium", "tfidf_p_hard", "tfidf_score"])], axis=1)
        return feats, Xg[self.gbm_cols], stack

    def _members(self, questions, Xg):
        probs = {"gbm": self.gbm_clf.predict_proba(Xg)}
        scores = {"gbm": self.gbm_reg.predict(Xg)}
        if self.bert:
            from src.models.transformer import predict as bp
            probs["bert"], scores["bert"] = bp(*self.bert, questions)
        if self.bilstm:
            from src.models.bilstm import predict as lp
            probs["bilstm"], scores["bilstm"] = lp(*self.bilstm, questions)
        return probs, scores

    def predict_batch(self, texts: list[str]) -> pd.DataFrame:
        """Fast batch prediction (no explanations). Non-English text is translated first."""
        from src.multilingual.pipeline import DEVANAGARI, to_english
        eng = [to_english(t)["english"] if DEVANAGARI.search(t) else t for t in texts]
        qs = [clean_text(t) or t for t in eng]
        feats, Xg, stack = self._gbm_inputs(qs)
        probs, scores = self._members(qs, Xg)
        w = {k: v for k, v in self.cfg["weights"].items() if k in probs and v > 0}
        P, lvl, s = combine(probs, scores, w, self.cfg["alpha"])
        out = pd.DataFrame({"question": texts, "english": eng, "level": [LEVELS[i] for i in lvl],
                            "score": np.round(s, 1), "confidence": P.max(1).round(3)})
        for i, l in enumerate(LEVELS):
            out[f"p_{l.lower()}"] = P[:, i].round(3)
        return pd.concat([out, feats[["fk_grade", "bloom_level", "rare_word_ratio", "n_words"]].reset_index(drop=True)], axis=1)

    def predict(self, text: str, explain: bool = True) -> dict:
        from src.multilingual.pipeline import to_english
        ml = to_english(text)
        q = clean_text(ml["english"]) or ml["english"]
        feats, Xg, stack = self._gbm_inputs([q])
        probs, scores = self._members([q], Xg)
        w = {k: v for k, v in self.cfg["weights"].items() if k in probs and v > 0}
        P, lvl, s = combine(probs, scores, w, self.cfg["alpha"])
        res = {
            "question": text,
            "cleaned": q,
            "language": {k: ml[k] for k in ("lang", "name", "translated")},
            "english": ml["english"],
            "level": LEVELS[int(lvl[0])],
            "score": round(float(s[0]), 1),
            "confidence": round(float(P[0].max()), 3),
            "probabilities": {l: round(float(P[0, i]), 3) for i, l in enumerate(LEVELS)},
            "member_scores": {k: round(float(v[0]), 1) for k, v in scores.items()},
            "member_levels": {k: LEVELS[int(v[0].argmax())] for k, v in probs.items()},
            "features": {k: float(v) for k, v in feats.iloc[0].items()},
        }
        if explain:
            from src.nlp.info_extraction import extract
            from src.nlp.sentiment import analyze
            from src.nlp.topics import topic_of
            res["info"] = extract(q)
            res["sentiment"] = analyze(q)
            try:
                res["topic"] = topic_of(q)
            except Exception:
                res["topic"] = None
            res["shap"] = self.shap_contributions(Xg)
            if self.bilstm:
                from src.models.bilstm import attention
                res["attention"] = attention(*self.bilstm, q)
        return res

    def shap_contributions(self, Xg, top=8) -> list[tuple[str, float, float]]:
        import shap
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.gbm_reg)
        sv = self._explainer.shap_values(Xg)[0]
        idx = np.argsort(-np.abs(sv))[:top]
        return [(self.gbm_cols[i], round(float(sv[i]), 2), round(float(Xg.iloc[0, i]), 3)) for i in idx]


@lru_cache(maxsize=1)
def get_predictor() -> DifficultyPredictor:
    return DifficultyPredictor()


def _print(r: dict):
    bar = "█" * int(r["score"] / 5) + "░" * (20 - int(r["score"] / 5))
    print(f"\nQuestion : {r['question']}")
    if r["language"]["translated"]:
        print(f"Language : {r['language']['name']}  ->  {r['english']}")
    print(f"Level    : {r['level'].upper()}   (confidence {r['confidence']:.0%})")
    print(f"Score    : {r['score']:.1f}/100  {bar}")
    print(f"Probs    : " + "  ".join(f"{k} {v:.0%}" for k, v in r["probabilities"].items()))
    if "info" in r:
        i = r["info"]
        print(f"Bloom    : {i['bloom_level']}   type: {i['question_type']}")
        print(f"Readab.  : FK grade {r['features']['fk_grade']:.1f}, rare words {r['features']['rare_word_ratio']:.0%}")
        print(f"Keywords : {', '.join(i['keyphrases_yake'][:5])}")
        print("Top SHAP : " + ", ".join(f"{n} ({v:+.1f})" for n, v, _ in r["shap"][:5]))


def main():
    ap = argparse.ArgumentParser(description="Predict question difficulty (Easy/Medium/Hard + 0-100 score)")
    ap.add_argument("question", nargs="*")
    ap.add_argument("--file", help="pdf / docx / txt / csv question paper")
    ap.add_argument("--out", help="save batch results to CSV")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    p = get_predictor()
    if a.file:
        from src.data.extract import extract_questions
        qs = extract_questions(a.file)
        df = p.predict_batch(qs)
        print(df[["question", "level", "score", "confidence"]].to_string(max_colwidth=70))
        print("\nPaper summary:", df.level.value_counts().to_dict(), f"| mean score {df.score.mean():.1f}")
        if a.out:
            df.to_csv(a.out, index=False)
    else:
        r = p.predict(" ".join(a.question) or input("Question: "))
        print(json.dumps(r, indent=2, default=str) if a.json else "", end="")
        if not a.json:
            _print(r)


if __name__ == "__main__":
    main()
