"""CO4 - Prompt-engineering evaluation.

Compares three prompt strategies for the local LLM as a difficulty judge
(zero-shot, few-shot, chain-of-thought) against the trained NLP ensemble on the same
balanced sample of test questions. Also measures how often LLM rewrites move the
difficulty in the requested direction (verified by the trained model).
Outputs: reports/llm_prompt_eval.json, reports/llm_prompt_predictions.csv
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src.config import DATA_PROC, LEVEL2ID, REPORTS, SEED
from src.evaluation import full_report
from src.llm import genai


def run(n_per_level: int = 30, n_rewrite: int = 12, strategies=("zero_shot", "few_shot", "chain_of_thought", "score_first"), do_rewrite=True):
    assert genai.is_available(), "Start Ollama first:  ollama serve  (model qwen2.5:1.5b)"
    te = pd.read_parquet(DATA_PROC / "test.parquet")
    te = te[te.question.str.len() < 600]
    sample = te.groupby("level").sample(n_per_level, random_state=SEED)
    sample = sample.reset_index(drop=True)

    from src.predict import get_predictor
    pred = get_predictor()
    nlp = pred.predict_batch(sample.question.tolist())
    results = {"NLP ensemble (trained)": full_report(sample.label, nlp.level.map(LEVEL2ID), sample.score, nlp.score)}
    rows = sample[["uid", "group", "question", "level", "score"]].copy()
    rows["nlp_level"], rows["nlp_score"] = nlp.level.values, nlp.score.values

    prev = REPORTS / "llm_prompt_eval.json"
    old = json.loads(prev.read_text()) if prev.exists() else {}
    old_rows = pd.read_csv(REPORTS / "llm_prompt_predictions.csv") if (REPORTS / "llm_prompt_predictions.csv").exists() else None
    for strat in strategies:
        t0 = time.time()
        out = []
        for q in sample.question:
            try:
                out.append(genai.rate(q, strat))
            except Exception as e:  # keep going on a single failure
                out.append({"level": "Medium", "score": 50.0, "reasoning": f"error: {e}"})
        lv = [LEVEL2ID[o["level"]] for o in out]
        sc = [o["score"] for o in out]
        m = full_report(sample.label, lv, sample.score, sc)
        m["seconds_per_question"] = round((time.time() - t0) / len(sample), 2)
        m["predicted_level_distribution"] = pd.Series([o["level"] for o in out]).value_counts().to_dict()
        results[f"LLM {strat}"] = m
        rows[f"llm_{strat}_level"], rows[f"llm_{strat}_score"] = [o["level"] for o in out], sc
        print(f"[llm] {strat:17s} acc={m['accuracy']:.3f} f1={m['macro_f1']:.3f} qwk={m['qwk']:.3f} "
              f"mae={m['mae']:.1f}  {m['seconds_per_question']}s/q", flush=True)

    # keep results of strategies not re-run in this call
    for k, v in old.items():
        results.setdefault(k, v)
    if old_rows is not None and len(old_rows) == len(rows):
        for c in old_rows.columns:
            if c not in rows:
                rows[c] = old_rows[c].values
    # level derived from the LLM's own numeric score (tests whether its scores carry signal)
    from src.config import score_to_level
    for strat in ("zero_shot", "few_shot", "chain_of_thought", "score_first"):
        if f"llm_{strat}_score" in rows and f"LLM {strat}" in results:
            lv = [LEVEL2ID[score_to_level(x)] for x in rows[f"llm_{strat}_score"]]
            results[f"LLM {strat}"]["accuracy_level_from_score"] = round(float(np.mean(np.array(lv) == sample.label.values)), 4)
    if not do_rewrite:
        rows.to_csv(REPORTS / "llm_prompt_predictions.csv", index=False)
        (REPORTS / "llm_prompt_eval.json").write_text(json.dumps(results, indent=2, default=float))
        return results

    # Controlled rewriting: easier for Hard/Medium, harder for Easy/Medium
    rw = []
    pool = sample.sample(frac=1, random_state=1)
    for direction, lv in (("harder", ["Easy", "Medium"]), ("easier", ["Hard", "Medium"])):
        for q in pool[pool.level.isin(lv)].question.head(n_rewrite):
            try:
                rw.append(genai.rewrite(q, direction, pred))
            except Exception as e:
                print("rewrite error", e)
    rw_df = pd.DataFrame([{**{k: r[k] for k in ("original", "rewritten", "direction", "score_change", "verified")},
                           "before_level": r["before"]["level"], "after_level": r["after"]["level"]} for r in rw])
    rw_df.to_csv(REPORTS / "llm_rewrite_results.csv", index=False)
    results["rewrite"] = {
        "n": len(rw_df),
        "success_rate": float(rw_df.verified.mean()),
        "by_direction": rw_df.groupby("direction").agg(success=("verified", "mean"),
                                                       mean_change=("score_change", "mean")).round(3).to_dict("index"),
    }
    print(f"[llm] rewrite success rate={results['rewrite']['success_rate']:.2%}", results["rewrite"]["by_direction"])

    rows.to_csv(REPORTS / "llm_prompt_predictions.csv", index=False)
    (REPORTS / "llm_prompt_eval.json").write_text(json.dumps(results, indent=2, default=float))
    return results


if __name__ == "__main__":
    import sys
    if sys.argv[1:]:   # e.g.  python -m src.llm.evaluate_prompts score_first rewrite
        args = sys.argv[1:]
        run(strategies=[a for a in args if a != "rewrite"], do_rewrite="rewrite" in args)
    else:
        run()
