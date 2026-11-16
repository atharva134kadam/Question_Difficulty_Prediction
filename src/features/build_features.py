"""Compute and cache linguistic features for train/val/test."""
import time

import pandas as pd

from src.config import DATA_PROC
from src.features.linguistic import extract_features


def main():
    for split in ("train", "val", "test"):
        t = time.time()
        df = pd.read_parquet(DATA_PROC / f"{split}.parquet")
        feats = extract_features(df.question.tolist(), n_process=2)
        feats.insert(0, "uid", df.uid.values)
        feats.to_parquet(DATA_PROC / f"features_{split}.parquet", index=False)
        print(f"[features] {split}: {feats.shape} in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
