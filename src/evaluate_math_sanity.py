"""Sanity check: trivially easy symbolic maths vs genuinely hard maths (hand-written set).

    python -m src.evaluate_math_sanity [tag]
Writes reports/math_sanity_<tag>.csv and adds a summary to reports/math_sanity.json
"""
import json
import sys

import pandas as pd

from src.config import DATA_GOLD, REPORTS
from src.predict import get_predictor


def run(tag="after"):
    g = pd.read_csv(DATA_GOLD / "easy_math_sanity.csv")
    df = get_predictor().predict_batch(g.question.tolist())
    g["level"], g["score"] = df.level.values, df.score.values
    g.to_csv(REPORTS / f"math_sanity_{tag}.csv", index=False)
    summ = {lvl: {"n": int((g.expected_level == lvl).sum()),
                  "accuracy": float((g[g.expected_level == lvl].level == lvl).mean()),
                  "mean_score": float(g[g.expected_level == lvl].score.mean())} for lvl in ("Easy", "Hard")}
    path = REPORTS / "math_sanity.json"
    allr = json.loads(path.read_text()) if path.exists() else {}
    allr[tag] = summ
    path.write_text(json.dumps(allr, indent=2))
    print(f"[sanity:{tag}]", json.dumps(summ))
    print(g[g.level != g.expected_level][["question", "expected_level", "level", "score"]].to_string())


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "after")
