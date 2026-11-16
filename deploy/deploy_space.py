"""Deploy DiffiQ to a Hugging Face Space (Docker SDK).

    hf auth login                          # once, with a WRITE token
    .venv/bin/python deploy/deploy_space.py [space_name]

Stages only what the running app needs (code, frontend, trained models, reports,
datasets as Parquet) into a temp folder and uploads it.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]

INCLUDE_DIRS = ["src", "app", "frontend", "prompts", "samples", "reports", "models_store", "data/gold", "deploy"]
SKIP_SUFFIX = {".pyc"}
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints"}


def stage(dst: Path) -> int:
    n = 0

    def copy(src: Path, rel: Path):
        nonlocal n
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        n += 1

    for d in INCLUDE_DIRS:
        for f in (ROOT / d).rglob("*"):
            if f.is_file() and f.suffix not in SKIP_SUFFIX and not (SKIP_PARTS & set(f.parts)):
                copy(f, f.relative_to(ROOT))
    # raw datasets: Parquet copies + dataset cards (CSV duplicates skipped)
    for f in (ROOT / "data/raw").rglob("*"):
        if f.is_file() and f.suffix in {".parquet", ".json"}:
            copy(f, f.relative_to(ROOT))
    # processed data used at runtime (feature caches not needed)
    for name in ["all_questions.parquet", "train.parquet", "val.parquet", "test.parquet", "modelling_set.csv"]:
        copy(ROOT / "data/processed" / name, Path("data/processed") / name)
    for name in ["requirements.txt", ".gitignore"]:
        copy(ROOT / name, Path(name))
    shutil.copy2(ROOT / "deploy/Dockerfile", dst / "Dockerfile")
    shutil.copy2(ROOT / "deploy/SPACE_README.md", dst / "README.md")
    (dst / ".dockerignore").write_text("**/__pycache__\n**/*.pyc\n")
    return n


def main():
    api = HfApi()
    user = api.whoami()["name"]
    name = sys.argv[1] if len(sys.argv) > 1 else "DiffiQ-Question-Difficulty"
    repo_id = f"{user}/{name}"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        n = stage(Path(tmp))
        size = sum(f.stat().st_size for f in Path(tmp).rglob("*") if f.is_file()) / 1e6
        print(f"[deploy] uploading {n} files ({size:.0f} MB) to spaces/{repo_id} ...", flush=True)
        api.upload_folder(folder_path=tmp, repo_id=repo_id, repo_type="space",
                          commit_message="Deploy DiffiQ question difficulty app")
    print(f"[deploy] done -> https://huggingface.co/spaces/{repo_id}")
    print(f"[deploy] app  -> https://{repo_id.replace('/', '-').lower()}.hf.space")


if __name__ == "__main__":
    main()
