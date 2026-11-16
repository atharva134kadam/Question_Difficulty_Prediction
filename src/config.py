"""Central configuration: paths, label scheme and difficulty anchors."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DATA_GOLD = ROOT / "data" / "gold"
MODELS = ROOT / "models_store"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
PROMPTS = ROOT / "prompts"

for _p in (DATA_PROC, DATA_GOLD, MODELS, REPORTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42

LEVELS = ["Easy", "Medium", "Hard"]
LEVEL2ID = {l: i for i, l in enumerate(LEVELS)}

# Score bands (inclusive lower bound). A 0-100 score maps to a level with these.
BANDS = {"Easy": (0, 33.33), "Medium": (33.33, 66.67), "Hard": (66.67, 100)}
LEVEL_CENTERS = [16.7, 50.0, 83.3]


def score_to_level(score: float) -> str:
    if score < 33.33:
        return "Easy"
    if score < 66.67:
        return "Medium"
    return "Hard"


# ----------------------------------------------------------------------------
# Difficulty anchors. Every source/sub-group gets a target score (0-100) chosen
# from the educational level of the source. The level is derived from the score.
# ----------------------------------------------------------------------------
ANCHORS = {
    "arith_basic":     12,   # generated: single-step arithmetic, counting, comparisons (augmentation)
    "prealgebra_basic": 24,  # generated: one-step equations, %, squares, simple fractions (augmentation)
    "sciq":            18,   # crowd-sourced, mostly recall science questions
    "commonsense_qa":  25,   # everyday commonsense
    "arc_easy":        28,   # grade 3-9 science, easy partition
    "race_middle":     28,   # middle-school reading comprehension
    "gsm8k":           30,   # grade-school arithmetic word problems
    "mmlu_elementary": 30,   # elementary mathematics
    "openbookqa":      40,   # elementary science + multi-hop reasoning
    "race_high":       42,   # high-school reading comprehension
    "mmlu_highschool": 52,   # high-school subjects
    "arc_challenge":   52,   # questions that retrieval + co-occurrence methods failed
    "math_l1":         56,   # competition maths level 1
    "math_l2":         63,   # competition maths level 2
    "math_l3":         72,   # competition maths level 3
    "mmlu_college":    78,   # college / professional subjects
    "math_l4":         82,   # competition maths level 4
    "math_l5":         92,   # competition maths level 5 (AIME-style)
}

# Max rows per group used for TRAINING (all raw rows are kept on disk).
# Prevents the 97k RACE rows from dominating the model.
GROUP_CAP = 3000
