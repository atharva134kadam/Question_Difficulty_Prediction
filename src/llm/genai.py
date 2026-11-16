"""CO4 - Generative AI with a LOCAL LLM (Qwen2.5-1.5B-Instruct served by Ollama on
localhost). No external API: the model runs on this machine.

Capabilities
  rate()       - LLM-as-judge difficulty rating with 3 prompt strategies
                 (zero-shot, few-shot, chain-of-thought)
  explain()    - natural-language explanation grounded in the NLP model's measured signals
  rewrite()    - make a question easier / harder, then VERIFY with the trained model
  generate()   - generate new practice questions at a target level, then verify
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

import requests

from src.config import PROMPTS, DATA_PROC, SEED

OLLAMA = "http://localhost:11434"
MODEL = "qwen2.5:1.5b"

LEVEL_DEF = {
    "Easy": "basic recall of a fact or a single simple step, school level",
    "Medium": "understanding or applying a concept with a few steps, high-school level",
    "Hard": "multi-step reasoning or advanced/specialised knowledge, college or competition level",
}
GUIDELINES = {
    "easier": "- reduce the number of steps\n- use simpler, more common words\n- ask for recall or a direct application instead of analysis\n- give helpful information in the question",
    "harder": "- require more reasoning steps or combining concepts\n- ask the student to analyse, justify or evaluate instead of recall\n- add a realistic constraint or an extra condition\n- use precise subject terminology",
}


@lru_cache(maxsize=None)
def prompt(name: str) -> str:
    return (PROMPTS / f"{name}.txt").read_text()


def is_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA}/api/tags", timeout=2)
        return any(m["name"].startswith(MODEL.split(":")[0]) for m in r.json().get("models", []))
    except Exception:
        return False


def chat(user: str, temperature: float = 0.0, json_mode: bool = False, max_tokens: int = 400) -> str:
    body = {"model": MODEL, "stream": False,
            "messages": [{"role": "system", "content": prompt("system")}, {"role": "user", "content": user}],
            "options": {"temperature": temperature, "num_predict": max_tokens, "seed": SEED}}
    if json_mode:
        body["format"] = "json"
    r = requests.post(f"{OLLAMA}/api/chat", json=body, timeout=180)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def _parse_rating(text: str) -> dict:
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        d = json.loads(m.group(0)) if m else {}
    lvl = str(d.get("level", "")).capitalize()
    if lvl not in LEVEL_DEF:
        lvl = next((l for l in LEVEL_DEF if l.lower() in text.lower()), "Medium")
    try:
        score = float(d.get("score"))
    except (TypeError, ValueError):
        score = {"Easy": 20, "Medium": 50, "Hard": 80}[lvl]
    return {"level": lvl, "score": max(0.0, min(100.0, score)), "reasoning": d.get("reasoning", "")}


@lru_cache(maxsize=1)
def few_shot_examples(k_per_level: int = 2) -> str:
    import pandas as pd
    tr = pd.read_parquet(DATA_PROC / "train.parquet")
    tr = tr[tr.question.str.len().between(40, 220)]
    lines = []
    for lvl in ("Easy", "Medium", "Hard"):
        for _, r in tr[tr.level == lvl].sample(k_per_level, random_state=SEED).iterrows():
            lines.append(f'Question: """{r.question}"""\nAnswer: {{"level": "{lvl}", "score": {int(r.score)}}}')
    return "\n\n".join(lines)


def rate(question: str, strategy: str = "zero_shot") -> dict:
    if strategy == "zero_shot":
        p = prompt("rate_zero_shot").format(question=question)
    elif strategy == "few_shot":
        p = prompt("rate_few_shot").format(question=question, examples=few_shot_examples())
    elif strategy == "chain_of_thought":
        p = prompt("rate_chain_of_thought").format(question=question)
    elif strategy == "score_first":
        # v2 prompt: no level labels to copy (small LLMs anchor on the first listed option);
        # calibrated anchors, and the level is derived from the score by our own bands.
        raw = chat(prompt("rate_score_first").format(question=question), json_mode=True, max_tokens=200)
        r = _parse_rating(raw)
        from src.config import score_to_level
        return {**r, "level": score_to_level(r["score"]), "strategy": strategy, "raw": raw}
    else:
        raise ValueError(strategy)
    raw = chat(p, json_mode=True, max_tokens=300)
    return {**_parse_rating(raw), "strategy": strategy, "raw": raw}


def explain(question: str, prediction: dict) -> str:
    f = prediction.get("features", {})
    ie = prediction.get("info", {})
    signals = "\n".join([
        f"- Bloom's taxonomy level: {ie.get('bloom_level', '?')}",
        f"- Readability grade (Flesch-Kincaid): {f.get('fk_grade', 0):.1f}",
        f"- Rare / technical words: {f.get('rare_word_ratio', 0):.0%} of words",
        f"- Words: {int(f.get('n_words', 0))}, maths symbols: {int(f.get('n_math_symbols', 0))}",
        f"- Key concepts: {', '.join(ie.get('keyphrases_yake', [])[:4])}",
    ])
    return chat(prompt("explain").format(question=question, level=prediction["level"],
                                         score=round(prediction["score"]), signals=signals),
                temperature=0.3, max_tokens=300)


def rewrite(question: str, direction: str, predictor=None) -> dict:
    """direction in {'easier', 'harder'}; verified by the trained difficulty model."""
    from src.predict import get_predictor
    predictor = predictor or get_predictor()
    before = predictor.predict(question, explain=False)
    p = prompt("rewrite").format(question=question, direction=direction, level=before["level"],
                                 score=round(before["score"]), guidelines=GUIDELINES[direction])
    new_q = _clean_generation(chat(p, temperature=0.4, max_tokens=250))
    after = predictor.predict(new_q, explain=False)
    delta = after["score"] - before["score"]
    ok = delta > 0 if direction == "harder" else delta < 0
    return {"original": question, "rewritten": new_q, "direction": direction,
            "before": {"level": before["level"], "score": before["score"]},
            "after": {"level": after["level"], "score": after["score"]},
            "score_change": delta, "verified": ok}


def _clean_generation(text: str) -> str:
    """Keep only the rewritten question: drop quotes, labels and trailing commentary."""
    text = text.strip().strip('"').strip()
    text = re.sub(r'^(rewritten question|question)\s*:\s*', "", text, flags=re.I)
    text = re.split(r'"""|\n\s*\n|\n(?:subject|note|explanation|difficulty)\s*:', text, flags=re.I)[0]
    return text.strip().strip('"').strip()


def generate(topic: str, level: str, n: int = 3, predictor=None) -> list[dict]:
    from src.predict import get_predictor
    predictor = predictor or get_predictor()
    raw = chat(prompt("generate").format(n=n, topic=topic, level=level, definition=LEVEL_DEF[level]),
               temperature=0.7, json_mode=True, max_tokens=600)
    try:
        qs = json.loads(raw).get("questions", [])
    except json.JSONDecodeError:
        qs = [l.strip("-• ").strip() for l in raw.splitlines() if l.strip()]
    out = []
    for q in qs[:n]:
        if isinstance(q, dict):
            q = q.get("question") or next(iter(q.values()), "")
        p = predictor.predict(str(q), explain=False)
        out.append({"question": q, "target_level": level, "predicted_level": p["level"],
                    "predicted_score": p["score"], "on_target": p["level"] == level})
    return out
