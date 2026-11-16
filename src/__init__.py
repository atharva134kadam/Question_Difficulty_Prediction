"""Question Difficulty Prediction - NLP project package."""
import os

# PyTorch and LightGBM each ship their own OpenMP runtime on macOS. Loading both in one
# process with multi-threaded OpenMP deadlocks / segfaults, so OpenMP is pinned to 1 thread
# (models still use their own explicit thread settings where it matters).
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Load order matters: LightGBM's libomp must be initialised before PyTorch's, otherwise
# using both in one process segfaults on macOS. Importing it here guarantees the order
# for every entry point (CLI, app, notebooks, evaluation scripts).
try:
    import lightgbm  # noqa: F401
except ImportError:
    pass
