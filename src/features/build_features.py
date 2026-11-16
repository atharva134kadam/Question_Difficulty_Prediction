"""Compute and cache linguistic features for train/val/test (incremental: reuses cached rows)."""
import time

import pandas as pd

from src.config import DATA_PROC
from src.features.linguistic import extract_features


def main():
    for split in ("train", "val", "test"):
        t = time.time()
        df = pd.read_parquet(DATA_PROC / f"{split}.parquet")
        path = DATA_PROC / f"features_{split}.parquet"
        cache = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["uid"])
        cache = cache[cache.uid.isin(df.uid)]
        todo = df[~df.uid.isin(cache.uid)]
        if len(todo):
            new = extract_features(todo.question.tolist(), n_process=2)
            new.insert(0, "uid", todo.uid.values)
            cache = pd.concat([cache, new], ignore_index=True)
        feats = df[["uid"]].merge(cache, on="uid", how="left")   # same row order as the split file
        assert len(feats) == len(df) and feats.drop(columns="uid").notna().all().all()
        feats.to_parquet(path, index=False)
        print(f"[features] {split}: {feats.shape} ({len(todo)} new) in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
