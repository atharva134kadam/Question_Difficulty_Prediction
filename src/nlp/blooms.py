"""CO2 - Information extraction: Bloom's Taxonomy cognitive level of a question.

Rule-based extractor: find the action verb / question pattern and map it to one of
Bloom's six levels. Higher levels need deeper thinking -> harder questions.
"""
from __future__ import annotations

import re

BLOOM_LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

BLOOM_VERBS = {
    "Remember": ["define", "list", "name", "identify", "recall", "state", "label", "recognize",
                 "memorize", "repeat", "who", "when", "where", "what is", "which of the following is",
                 "match", "select", "locate", "record", "tell", "find the name"],
    "Understand": ["explain", "describe", "summarize", "summarise", "classify", "discuss",
                   "interpret", "paraphrase", "illustrate", "give an example", "why", "what does",
                   "main idea", "mainly about", "best title", "infer", "mean by", "refers to",
                   "most likely", "according to", "distinguish", "translate", "outline", "restate"],
    "Apply": ["calculate", "compute", "solve", "find", "determine", "use", "apply", "demonstrate",
              "how many", "how much", "what is the value", "evaluate the expression", "simplify",
              "convert", "estimate", "show that", "measure", "predict", "sketch", "plot", "graph"],
    "Analyze": ["analyze", "analyse", "compare", "contrast", "differentiate", "examine",
                "break down", "categorize", "deduce", "relationship", "cause", "effect",
                "what would happen", "which best explains", "investigate", "prove", "derive",
                "how does", "how do", "best supports", "is necessary"],
    "Evaluate": ["evaluate", "justify", "assess", "critique", "judge", "argue", "defend",
                 "appraise", "recommend", "decide", "rank", "prioritize", "which is the best",
                 "strongest", "weakest", "most appropriate", "most effective", "validity",
                 "should", "conclude"],
    "Create": ["design", "create", "construct", "formulate", "develop", "compose", "invent",
               "propose", "plan", "devise", "generate", "build", "write a program",
               "write an essay", "hypothesize", "synthesize"],
}

_PATTERNS = {
    lvl: [re.compile(r"\b" + re.escape(v) + r"\b", re.I) for v in verbs]
    for lvl, verbs in BLOOM_VERBS.items()
}


def bloom_scores(text: str) -> dict[str, int]:
    """Number of indicator verbs/phrases found per Bloom level."""
    return {lvl: sum(1 for p in pats if p.search(text)) for lvl, pats in _PATTERNS.items()}


def bloom_level(text: str) -> tuple[str, int, list[str]]:
    """Return (level_name, level_index 1-6, matched_indicators).

    The highest level with at least one hit wins (questions are as hard as their
    most demanding instruction). Defaults to Remember.
    """
    hits = []
    best = 0
    for i, (lvl, pats) in enumerate(_PATTERNS.items()):
        for v, p in zip(BLOOM_VERBS[lvl], pats):
            if p.search(text):
                hits.append(f"{v} ({lvl})")
                best = max(best, i)
    if not hits:
        return "Remember", 1, []
    return BLOOM_LEVELS[best], best + 1, hits
