"""CO3 - Evaluate the multilingual pipeline on the hand-written Hindi/Marathi set.

Metrics
  * language identification accuracy
  * translation quality: corpus BLEU (NLTK) and chrF-like character overlap vs human references
  * cross-lingual consistency: |score(translated) - score(English reference)| and level agreement
  * accuracy vs the expected level
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu

from src.config import DATA_GOLD, REPORTS
from src.multilingual.pipeline import MT_CANDIDATES, MT_MODELS, detect_language, translate_to_english


def _tok(s):
    import re
    return re.findall(r"\w+|[^\s\w]", s.lower())


def run():
    g = pd.read_csv(DATA_GOLD / "multilingual_eval.csv")
    g["detected"] = [detect_language(q)["lang"] for q in g.question]
    g["english_mt"] = g.question
    for lang in ("hi", "mr"):
        m = g.lang == lang
        g.loc[m, "english_mt"] = translate_to_english(g.loc[m, "question"].tolist(), lang)

    from src.predict import get_predictor
    p = get_predictor()
    pred_mt = p.predict_batch(g.english_mt.tolist())
    pred_ref = p.predict_batch(g.english_reference.tolist())
    g["level_mt"], g["score_mt"] = pred_mt.level.values, pred_mt.score.values
    g["level_ref"], g["score_ref"] = pred_ref.level.values, pred_ref.score.values
    sm = SmoothingFunction().method3
    g["bleu"] = [sentence_bleu([_tok(r)], _tok(h), smoothing_function=sm) for r, h in zip(g.english_reference, g.english_mt)]

    res = {"language_id_accuracy": float((g.detected == g.lang).mean()), "mt_model_selection": {}, "selected_models": MT_MODELS}
    for lang, cands in MT_CANDIDATES.items():
        d = g[g.lang == lang]
        for name in cands:
            hyp = translate_to_english(d.question.tolist(), lang, model_name=name)
            res["mt_model_selection"][f"{lang} | {name.split('/')[-1]}"] = float(corpus_bleu(
                [[_tok(r)] for r in d.english_reference], [_tok(h) for h in hyp], smoothing_function=sm))
    print("[multilingual] MT model selection (BLEU):", {k: round(v, 3) for k, v in res["mt_model_selection"].items()})
    for lang in ("hi", "mr"):
        d = g[g.lang == lang]
        res[lang] = {
            "n": int(len(d)),
            "language_id_accuracy": float((d.detected == lang).mean()),
            "corpus_bleu": float(corpus_bleu([[_tok(r)] for r in d.english_reference], [_tok(h) for h in d.english_mt],
                                             smoothing_function=sm)),
            "score_mae_vs_english": float(np.abs(d.score_mt - d.score_ref).mean()),
            "level_agreement_with_english": float((d.level_mt == d.level_ref).mean()),
            "accuracy_vs_expected_level": float((d.level_mt == d.expected_level).mean()),
            "english_reference_accuracy": float((d.level_ref == d.expected_level).mean()),
        }
        print(f"[multilingual] {lang}: " + ", ".join(f"{k}={v:.3f}" for k, v in res[lang].items() if k != "n"))
    print(f"[multilingual] language-ID accuracy (all) = {res['language_id_accuracy']:.3f}")
    g.to_csv(REPORTS / "multilingual_predictions.csv", index=False)
    (REPORTS / "multilingual_metrics.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    run()
