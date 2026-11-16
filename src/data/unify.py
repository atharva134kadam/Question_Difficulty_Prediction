"""Merge all raw datasets into ONE unified, cleaned, labelled corpus.

Output:
  data/processed/all_questions.parquet  - every question from every source (~150k)
  data/processed/train.parquet / val.parquet / test.parquet - balanced modelling set

Columns: uid, source, group, subject, question_raw, question, options, has_latex,
         score (0-100 target), level (Easy/Medium/Hard), label (0/1/2), split
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ANCHORS, DATA_PROC, DATA_RAW, GROUP_CAP, LEVEL2ID, SEED, score_to_level
from src.nlp.cleaning import clean_text, has_latex


def _read(name: str, pattern: str) -> pd.DataFrame:
    files = sorted((DATA_RAW / name).glob(f"{pattern}.parquet"))
    return pd.concat([pd.read_parquet(f).assign(_file=f.stem) for f in files], ignore_index=True)


def _choices(c) -> list[str]:
    if isinstance(c, dict):
        return [str(x) for x in c.get("text", [])]
    if c is None:
        return []
    return [str(x) for x in list(c)]


def load_arc():
    df = _read("arc", "*")
    df["group"] = np.where(df._file.str.startswith("ARC-Easy"), "arc_easy", "arc_challenge")
    return pd.DataFrame({"source": "arc", "group": df.group, "subject": "science",
                         "question_raw": df.question, "options": df.choices.map(_choices)})


def load_sciq():
    df = _read("sciq", "*")
    opts = df[["correct_answer", "distractor1", "distractor2", "distractor3"]].values.tolist()
    return pd.DataFrame({"source": "sciq", "group": "sciq", "subject": "science",
                         "question_raw": df.question, "options": opts})


def load_obqa():
    df = _read("openbookqa", "*")
    return pd.DataFrame({"source": "openbookqa", "group": "openbookqa", "subject": "science",
                         "question_raw": df.question_stem, "options": df.choices.map(_choices)})


def load_csqa():
    df = _read("commonsense_qa", "*")
    return pd.DataFrame({"source": "commonsense_qa", "group": "commonsense_qa",
                         "subject": "commonsense", "question_raw": df.question,
                         "options": df.choices.map(_choices)})


def load_race():
    df = _read("race", "*")
    df["group"] = np.where(df._file.str.startswith("middle"), "race_middle", "race_high")
    return pd.DataFrame({"source": "race", "group": df.group, "subject": "english_reading",
                         "question_raw": df.question, "options": df.options.map(_choices)})


def load_gsm8k():
    df = _read("gsm8k", "*")
    return pd.DataFrame({"source": "gsm8k", "group": "gsm8k", "subject": "mathematics",
                         "question_raw": df.question, "options": [[] for _ in range(len(df))]})


def load_math():
    df = _read("math", "*")
    lvl = df.level.str.extract(r"(\d)")[0]
    df = df[lvl.notna()].copy()
    df["group"] = "math_l" + lvl[lvl.notna()]
    return pd.DataFrame({"source": "math", "group": df.group,
                         "subject": "mathematics_" + df.type.str.lower().str.replace(" ", "_").str.replace("&", "and"),
                         "question_raw": df.problem, "options": [[] for _ in range(len(df))]})


MMLU_HARD_EXTRA = {"abstract_algebra", "formal_logic", "econometrics", "jurisprudence",
                   "machine_learning", "international_law", "medical_genetics"}


def _mmlu_group(subj: str) -> str | None:
    if subj.startswith("elementary_"):
        return "mmlu_elementary"
    if subj.startswith("high_school_") or subj == "conceptual_physics":
        return "mmlu_highschool"
    if subj.startswith(("college_", "professional_")) or subj in MMLU_HARD_EXTRA:
        return "mmlu_college"
    return None  # level-ambiguous subjects (e.g. 'miscellaneous') are not used


def load_mmlu():
    df = _read("mmlu", "*")
    df["group"] = df.subject.map(_mmlu_group)
    df = df[df.group.notna()]
    return pd.DataFrame({"source": "mmlu", "group": df.group, "subject": df.subject,
                         "question_raw": df.question, "options": df.choices.map(_choices)})


LOADERS = [load_arc, load_sciq, load_obqa, load_csqa, load_race, load_gsm8k, load_math, load_mmlu]


def build_all() -> pd.DataFrame:
    df = pd.concat([f() for f in LOADERS], ignore_index=True)
    n0 = len(df)
    df["question_raw"] = df.question_raw.astype(str)
    df["has_latex"] = df.question_raw.map(has_latex)
    df["question"] = df.question_raw.map(clean_text)
    df = df[df.question.str.len() >= 8]
    # de-duplicate (RACE repeats generic stems like "What's the best title?")
    df["_key"] = df.question.str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    df = df.drop_duplicates("_key").drop(columns="_key")
    df["score"] = df.group.map(ANCHORS).astype(float)
    df["level"] = df.score.map(score_to_level)
    df["label"] = df.level.map(LEVEL2ID)
    df["options"] = df.options.map(lambda o: json.dumps(list(o)))
    df = df.reset_index(drop=True)
    df.insert(0, "uid", [f"q{i:06d}" for i in range(len(df))])
    print(f"[unify] {n0} raw rows -> {len(df)} after cleaning/dedup")
    return df


def make_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Cap each group, then stratified 80/10/10 split by group."""
    parts = []
    for g, d in df.groupby("group"):
        parts.append(d.sample(min(len(d), GROUP_CAP), random_state=SEED))
    m = pd.concat(parts)
    train, rest = train_test_split(m, test_size=0.2, stratify=m.group, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest.group, random_state=SEED)
    return pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])


def main():
    df = build_all()
    df.to_parquet(DATA_PROC / "all_questions.parquet", index=False)
    m = make_splits(df)
    for s in ("train", "val", "test"):
        m[m.split == s].reset_index(drop=True).to_parquet(DATA_PROC / f"{s}.parquet", index=False)
    m.to_csv(DATA_PROC / "modelling_set.csv", index=False)
    print(m.groupby(["split", "level"]).size().unstack())
    print(m.groupby("group").size().sort_values())


if __name__ == "__main__":
    main()
