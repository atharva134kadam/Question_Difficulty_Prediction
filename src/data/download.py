"""Download every public dataset used in the project ONCE and store the full,
unmodified copies under data/raw/<dataset>/ as both Parquet and CSV.

After this script runs, the whole project works offline.

Usage:  python -m src.data.download            (all datasets)
        python -m src.data.download arc sciq   (selected datasets)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

# name -> (hf_repo, [configs], description, license, difficulty signal)
DATASETS = {
    "arc": ("allenai/ai2_arc", ["ARC-Easy", "ARC-Challenge"],
            "AI2 Reasoning Challenge: grade 3-9 science exam questions",
            "CC BY-SA 4.0", "Easy set vs Challenge set"),
    "sciq": ("allenai/sciq", [None],
             "Crowd-sourced science exam questions (physics, chemistry, biology)",
             "CC BY-NC 3.0", "Easy anchor"),
    "openbookqa": ("allenai/openbookqa", ["main"],
                   "Elementary science questions needing multi-step reasoning",
                   "Apache 2.0", "Medium anchor"),
    "commonsense_qa": ("tau/commonsense_qa", [None],
                       "Commonsense reasoning questions built on ConceptNet",
                       "MIT", "Easy anchor"),
    "race": ("ehovy/race", ["middle", "high"],
             "English reading comprehension exams for Chinese students (age 12-18)",
             "Non-commercial research", "Middle school vs High school"),
    "gsm8k": ("openai/gsm8k", ["main"],
              "Grade-school maths word problems",
              "MIT", "Easy maths anchor"),
    "math": ("EleutherAI/hendrycks_math",
             ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
              "number_theory", "prealgebra", "precalculus"],
             "Competition mathematics problems (AMC, AIME, ...)",
             "MIT", "Level 1-5"),
    "mmlu": ("cais/mmlu", ["all"],
             "57 academic subjects from elementary to professional level",
             "MIT", "elementary / high_school / college+professional subject level"),
    "sst2": ("stanfordnlp/sst2", [None],
             "Stanford Sentiment Treebank (binary) - used for CO2 sentiment model",
             "Unknown/research", "N/A (sentiment labels)"),
}


def download(name: str) -> dict:
    repo, configs, desc, lic, signal = DATASETS[name]
    out_dir = RAW / name
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for cfg in configs:
        print(f"[download] {repo} config={cfg}")
        ds = load_dataset(repo, cfg) if cfg else load_dataset(repo)
        for split, d in ds.items():
            if name == "mmlu" and split == "auxiliary_train":
                continue  # not MMLU questions; 100k+ rows from other datasets
            tag = f"{cfg}_{split}" if cfg else split
            df = d.to_pandas()
            df.to_parquet(out_dir / f"{tag}.parquet", index=False)
            df.to_csv(out_dir / f"{tag}.csv", index=False)
            counts[tag] = len(df)
    card = {"name": name, "source": f"https://huggingface.co/datasets/{repo}",
            "description": desc, "license": lic, "difficulty_signal": signal,
            "files": counts, "total_rows": sum(counts.values())}
    (out_dir / "dataset_card.json").write_text(json.dumps(card, indent=2))
    print(f"[done] {name}: {card['total_rows']} rows")
    return card


def main(names: list[str]) -> None:
    names = names or list(DATASETS)
    cards = [download(n) for n in names]
    summary = RAW / "datasets_summary.json"
    existing = json.loads(summary.read_text()) if summary.exists() else {}
    existing.update({c["name"]: c for c in cards})
    summary.write_text(json.dumps(existing, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
