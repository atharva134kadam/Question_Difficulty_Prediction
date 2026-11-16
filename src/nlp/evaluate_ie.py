"""CO2 - Evaluate information extraction components against hand-labelled gold data."""
import json

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import DATA_GOLD, REPORTS
from src.nlp.blooms import BLOOM_LEVELS, bloom_level
from src.nlp.info_extraction import UNIT


def evaluate_blooms(file="blooms_gold.csv"):
    g = pd.read_csv(DATA_GOLD / file)
    g["pred"] = [bloom_level(q)[0] for q in g.question]
    rep = classification_report(g.bloom_level, g.pred, labels=BLOOM_LEVELS, output_dict=True, zero_division=0)
    res = {"accuracy": accuracy_score(g.bloom_level, g.pred),
           "exact_or_adjacent": float(sum(abs(BLOOM_LEVELS.index(a) - BLOOM_LEVELS.index(b)) <= 1
                                          for a, b in zip(g.bloom_level, g.pred)) / len(g)),
           "per_level": {k: rep[k] for k in BLOOM_LEVELS},
           "macro_f1": rep["macro avg"]["f1-score"],
           "confusion_matrix": confusion_matrix(g.bloom_level, g.pred, labels=BLOOM_LEVELS).tolist(),
           "errors": g[g.bloom_level != g.pred][["question", "bloom_level", "pred"]].to_dict("records")}
    print(f"[IE] Bloom's extractor ({file}): acc={res['accuracy']:.3f} macro-F1={res['macro_f1']:.3f} "
          f"within-1-level={res['exact_or_adjacent']:.3f}")
    return res


UNIT_GOLD = [
    ("A car travels 120 km in 2 hours.", {"120 km", "2 hours"}),
    ("A 5 kg box is pushed with 20 N of force.", {"5 kg", "20 N"}),
    ("Water boils at 100 °C at sea level.", {"100 °C"}),
    ("The bulb uses 60 W at 220 V.", {"60 W", "220 V"}),
    ("She saved 15% of her 400 dollars.", {"15 %", "400 dollars"}),
    ("The wavelength is 500 nm and frequency 6 Hz.", {"6 Hz"}),
    ("Dissolve 2 mol of salt in 3 liters of water.", {"2 mol", "3 liters"}),
    ("The runner finished 400 m in 50 seconds.", {"400 m", "50 seconds"}),
]


def _norm(s):
    return s.lower().replace(" ", "")


def evaluate_units():
    tp = fp = fn = 0
    for text, gold in UNIT_GOLD:
        pred = {_norm(" ".join(m)) for m in UNIT.findall(text)}
        gold = {_norm(x) for x in gold}
        tp += len(pred & gold); fp += len(pred - gold); fn += len(gold - pred)
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    res = {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9)}
    print(f"[IE] Quantity/unit extraction: P={p:.3f} R={r:.3f} F1={res['f1']:.3f}")
    return res


def main():
    out = {"blooms_explicit_verbs": evaluate_blooms("blooms_gold.csv"),
           "blooms_real_exam_questions": evaluate_blooms("blooms_gold_real.csv"), "quantities": evaluate_units()}
    (REPORTS / "information_extraction_metrics.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
