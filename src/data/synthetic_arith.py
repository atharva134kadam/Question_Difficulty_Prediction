"""Data augmentation: elementary arithmetic & pre-algebra questions (Easy by construction).

Why: in the public datasets, symbolic maths ("+", "^", "=") comes almost only from the
MATH competition dataset (Medium/Hard), while easy maths (GSM8K) is mostly long word
problems. Models therefore learned "maths symbols => at least Medium", rating "1+2" as
Medium. These generated questions teach that short symbolic arithmetic is Easy.

Two groups (anchors in src/config.py):
  arith_basic       - single operations, counting, comparison, doubles/halves, shape facts
  prealgebra_basic  - one-step equations, % of round numbers, squares/roots, simple fractions,
                      decimals, rounding, order of operations, perimeter/area

Output: data/raw/synthetic_arith/{arith_basic,prealgebra_basic}.parquet (+ csv, dataset card)
Questions in data/gold/easy_math_sanity.csv are excluded so that evaluation stays held-out.
"""
from __future__ import annotations

import json
import random
import re

import pandas as pd

from src.config import DATA_GOLD, DATA_RAW, SEED

OUT = DATA_RAW / "synthetic_arith"
N_PER_GROUP = 1500

OPS = {"+": ("plus", "add", "sum of", lambda a, b: a + b), "-": ("minus", "subtract", "difference between", lambda a, b: a - b),
       "*": ("times", "multiply", "product of", lambda a, b: a * b), "/": ("divided by", "divide", "quotient of", None)}
SYM = {"*": ["*", "×", "x"], "/": ["/", "÷"], "+": ["+"], "-": ["-", "−"]}
SHAPES = {"triangle": 3, "square": 4, "rectangle": 4, "pentagon": 5, "hexagon": 6, "octagon": 8}


def _expr(r: random.Random):
    op = r.choice(list(OPS))
    if op == "/":
        b = r.randint(1, 12); a = b * r.randint(1, 12)
    elif op == "*":
        a, b = r.randint(0, 12), r.randint(0, 12)
    elif op == "-":
        a = r.randint(0, 100); b = r.randint(0, a)
    else:
        a, b = r.randint(0, 100), r.randint(0, 100)
    s = r.choice(SYM[op])
    sp = r.choice(["", " "])
    return op, a, b, f"{a}{sp}{s}{sp}{b}"


def arith_basic(r: random.Random) -> str:
    kind = r.random()
    op, a, b, e = _expr(r)
    word, verb, noun, _ = OPS[op]
    if kind < 0.22:  # bare expression
        return r.choice([e, e + " = ?", e + "?", e + " ="])
    if kind < 0.55:
        return r.choice([f"What is {e}?", f"Calculate {e}.", f"Compute {e}.", f"Evaluate {e}.", f"Find {e}.",
                         f"What is the value of {e}?", f"Work out {e}.", f"How much is {e}?", f"Solve {e}.", f"{e} equals what?"])
    if kind < 0.72:
        if op == "/":
            return r.choice([f"What is {a} divided by {b}?", f"Divide {a} by {b}.", f"How many times does {b} go into {a}?"])
        if op == "-":
            return r.choice([f"What is {a} minus {b}?", f"Subtract {b} from {a}.", f"Take {b} away from {a}. What is left?", f"What is the difference between {a} and {b}?"])
        if op == "*":
            return r.choice([f"What is {a} times {b}?", f"Multiply {a} by {b}.", f"What is the product of {a} and {b}?", f"What is {a} multiplied by {b}?"])
        return r.choice([f"What is {a} plus {b}?", f"Add {a} and {b}.", f"What is the sum of {a} and {b}?", f"What do you get if you add {a} to {b}?"])
    if kind < 0.80:
        x, y = r.randint(1, 99), r.randint(1, 99)
        return r.choice([f"Which is bigger, {x} or {y}?", f"Is {x} greater than {y}?", f"Which number is smaller: {x} or {y}?",
                         f"Is {x} an even number?", f"Is {x} odd or even?"])
    if kind < 0.88:
        x = r.randint(1, 50)
        return r.choice([f"What is double {x}?", f"What is half of {2 * x}?", f"What is {x} doubled?", f"What number comes after {x}?",
                         f"What number comes before {x + 1}?"])
    if kind < 0.94:
        st, d = r.randint(1, 10), r.choice([1, 2, 3, 5, 10])
        seq = ", ".join(str(st + d * i) for i in range(4))
        return r.choice([f"What comes next: {seq}, ...?", f"Continue the pattern: {seq}", f"Count by {d}s: {seq}. What is next?"])
    s = r.choice(list(SHAPES))
    return r.choice([f"How many sides does a {s} have?", f"How many corners does a {s} have?", f"How many days are there in {r.randint(1, 4)} weeks?",
                     f"How many minutes are in {r.randint(1, 5)} hours?"])


def prealgebra_basic(r: random.Random) -> str:
    kind = r.random()
    if kind < 0.20:
        x, a = r.randint(1, 20), r.randint(1, 20)
        form = r.choice(["add", "sub", "mul"])
        if form == "add":
            eq = f"x + {a} = {x + a}"
        elif form == "sub":
            eq = f"x - {a} = {x}"
        else:
            eq = f"{a}x = {a * x}"
        return r.choice([f"Solve {eq}.", f"Solve for x: {eq}", f"If {eq}, what is x?", f"Find x if {eq}.", f"{eq}. Find x."])
    if kind < 0.32:
        p, base = r.choice([10, 20, 25, 50, 75, 100]), r.choice([20, 40, 80, 100, 200, 400, 1000])
        return r.choice([f"What is {p}% of {base}?", f"Find {p} percent of {base}.", f"Calculate {p}% of {base}."])
    if kind < 0.46:
        k = r.randint(1, 15)
        return r.choice([f"What is {k}^2?", f"Compute {k}^2.", f"What is {k} squared?", f"What is the square root of {k * k}?",
                         f"Find sqrt({k * k}).", f"What is {min(k, 5)}^3?", f"Evaluate 2^{r.randint(1, 8)}."])
    if kind < 0.58:
        d = r.choice([2, 3, 4, 5, 6, 8, 10])
        a, b = r.randint(1, d - 1), r.randint(1, d - 1)
        return r.choice([f"What is {a}/{d} + {b}/{d}?", f"Add {a}/{d} and {b}/{d}.", f"Simplify {2 * a}/{2 * d}.", f"What is half of {d}/{d + 1}?" if d < 6 else f"What is {a}/{d} of {d * 4}?"])
    if kind < 0.68:
        a, b = round(r.uniform(0.1, 9.9), 1), round(r.uniform(0.1, 9.9), 1)
        return r.choice([f"What is {a} + {b}?", f"Calculate {a} + {b}.", f"What is {a} x 10?", f"Round {a} to the nearest whole number.",
                         f"Write {r.choice([0.5, 0.25, 0.75, 0.1])} as a fraction."])
    if kind < 0.78:
        n = r.randint(11, 999)
        return r.choice([f"Round {n} to the nearest ten.", f"Round {n} to the nearest hundred.", f"What is the place value of the digit {str(n)[0]} in {n}?",
                         f"Write {n} in words."])
    if kind < 0.90:
        a, b, c = r.randint(1, 12), r.randint(1, 12), r.randint(1, 12)
        e = r.choice([f"({a} + {b}) * {c}", f"{a} + {b} * {c}", f"{a} * {b} - {c}", f"({a} + {b}) / 1", f"{a} * ({b} + {c})"])
        return r.choice([f"Calculate {e}.", f"What is {e}?", f"Evaluate {e}.", f"{e} = ?", e])
    s = r.randint(2, 15); w = r.randint(2, 15)
    return r.choice([f"Find the perimeter of a square with side {s} cm.", f"What is the area of a rectangle {s} m by {w} m?",
                     f"A square has side {s}. What is its area?", f"What is the perimeter of a rectangle with length {s} and width {w}?"])


def generate():
    r = random.Random(SEED)
    held_out = set(re.sub(r"\W", "", q.lower()) for q in pd.read_csv(DATA_GOLD / "easy_math_sanity.csv").question)
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, fn in (("arith_basic", arith_basic), ("prealgebra_basic", prealgebra_basic)):
        seen, rows = set(), []
        while len(rows) < N_PER_GROUP:
            q = fn(r)
            k = re.sub(r"\W", "", q.lower())
            if k in seen or k in held_out:
                continue
            seen.add(k); rows.append(q)
        df = pd.DataFrame({"question": rows, "group": name})
        df.to_parquet(OUT / f"{name}.parquet", index=False); df.to_csv(OUT / f"{name}.csv", index=False)
        counts[name] = len(df)
        print(f"[synthetic] {name}: {len(df)} e.g. {rows[:6]}")
    card = {"name": "synthetic_arith", "source": "generated by src/data/synthetic_arith.py (seed 42)",
            "description": "Elementary arithmetic and pre-algebra questions (bare and worded), added to fix the 'maths symbols => Medium' bias",
            "license": "project-generated", "difficulty_signal": "Easy by construction (elementary school level)",
            "files": counts, "total_rows": sum(counts.values())}
    (OUT / "dataset_card.json").write_text(json.dumps(card, indent=2))
    summ = DATA_RAW / "datasets_summary.json"
    s = json.loads(summ.read_text()); s["synthetic_arith"] = card; summ.write_text(json.dumps(s, indent=2))


if __name__ == "__main__":
    generate()
