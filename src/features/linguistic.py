"""Hand-crafted linguistic features (≈45) that describe WHY a question is hard.

Groups
  readability : Flesch, Flesch-Kincaid grade, Gunning Fog, SMOG, Coleman-Liau, ARI, Dale-Chall
  lexical     : length, word length, type-token ratio, rare-word ratio (word frequency),
                long words, digits, maths symbols, LaTeX
  syntactic   : dependency-tree depth, subordinate clauses, POS ratios, noun chunks,
                named entities, sentences (spaCy)
  cognitive   : Bloom's taxonomy level, question word type
  tone        : VADER sentiment, negation / constraint words, hedging (CO2)
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd
import textstat
from wordfreq import zipf_frequency

from src.nlp.blooms import BLOOM_LEVELS, bloom_level, bloom_scores

WORD = re.compile(r"[A-Za-z]+")
MATH_SYM = re.compile(r"[=+\-*/^<>√∑∫π]|\b(sqrt|sin|cos|tan|log|ln|pi|infinity|lim)\b")
NEGATION = re.compile(r"\b(not|no|never|none|except|least|false|incorrect|cannot|isn't|doesn't|NOT|EXCEPT)\b")
CONSTRAINT = re.compile(r"\b(only|all|exactly|at least|at most|must|both|neither|unless|if and only if|distinct|positive integers?)\b", re.I)
HEDGE = re.compile(r"\b(most likely|probably|best|likely|might|may|could|possibly|approximately)\b", re.I)
QWORDS = ["what", "which", "why", "how", "who", "when", "where", "find", "other"]


@lru_cache(maxsize=1)
def _spacy():
    import spacy
    return spacy.load("en_core_web_sm")


@lru_cache(maxsize=1)
def _vader():
    from nltk.sentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def _tree_depth(tok) -> int:
    d = 0
    while tok.head.i != tok.i:
        tok = tok.head
        d += 1
    return d


def _surface_features(text: str) -> dict:
    words = WORD.findall(text)
    n = max(len(words), 1)
    lw = [w.lower() for w in words]
    zipf = [zipf_frequency(w, "en") for w in lw] or [7.0]
    f = {
        "n_chars": len(text),
        "n_words": len(words),
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "type_token_ratio": len(set(lw)) / n,
        "long_word_ratio": sum(len(w) >= 7 for w in words) / n,
        "rare_word_ratio": sum(z < 3.5 for z in zipf) / n,
        "very_rare_word_ratio": sum(z < 2.5 for z in zipf) / n,
        "mean_zipf": float(np.mean(zipf)),
        "min_zipf": float(np.min(zipf)),
        "n_digits": sum(c.isdigit() for c in text),
        "n_numbers": len(re.findall(r"\d+(?:\.\d+)?", text)),
        "n_math_symbols": len(MATH_SYM.findall(text)),
        "n_parentheses": text.count("("),
        "has_blank": int("___" in text),
        "ends_with_question_mark": int(text.strip().endswith("?")),
        # readability
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "fk_grade": textstat.flesch_kincaid_grade(text),
        "gunning_fog": textstat.gunning_fog(text),
        "smog": textstat.smog_index(text),
        "coleman_liau": textstat.coleman_liau_index(text),
        "ari": textstat.automated_readability_index(text),
        "dale_chall": textstat.dale_chall_readability_score(text),
        "difficult_words": textstat.difficult_words(text),
        "syllables_per_word": textstat.syllable_count(text) / n,
        # tone (CO2)
        "n_negations": len(NEGATION.findall(text)),
        "n_constraints": len(CONSTRAINT.findall(text)),
        "n_hedges": len(HEDGE.findall(text)),
    }
    v = _vader().polarity_scores(text)
    f.update({"vader_compound": v["compound"], "vader_pos": v["pos"], "vader_neg": v["neg"]})
    # cognitive
    _, b_idx, _ = bloom_level(text)
    f["bloom_level"] = b_idx
    for lvl, c in bloom_scores(text).items():
        f[f"bloom_{lvl.lower()}"] = c
    first = lw[0] if lw else ""
    for q in QWORDS[:-1]:
        f[f"qw_{q}"] = int(first == q or f" {q} " in f" {' '.join(lw[:4])} ")
    return f


def _syntax_features(doc) -> dict:
    toks = [t for t in doc if not t.is_space]
    n = max(len(toks), 1)
    depths = [_tree_depth(t) for t in toks] or [0]
    pos = [t.pos_ for t in toks]
    return {
        "n_sentences": len(list(doc.sents)),
        "max_tree_depth": max(depths),
        "mean_tree_depth": float(np.mean(depths)),
        "n_subordinate": sum(t.dep_ in {"advcl", "ccomp", "xcomp", "relcl", "acl", "csubj"} for t in toks),
        "n_conjunctions": sum(t.dep_ in {"cc", "conj"} for t in toks),
        "noun_ratio": (pos.count("NOUN") + pos.count("PROPN")) / n,
        "verb_ratio": pos.count("VERB") / n,
        "adj_ratio": pos.count("ADJ") / n,
        "adv_ratio": pos.count("ADV") / n,
        "num_ratio": pos.count("NUM") / n,
        "n_noun_chunks": len(list(doc.noun_chunks)),
        "n_entities": len(doc.ents),
    }


def extract_features(texts: list[str], batch_size: int = 256, n_process: int = 1) -> pd.DataFrame:
    nlp = _spacy()
    rows = []
    for text, doc in zip(texts, nlp.pipe(texts, batch_size=batch_size, n_process=n_process)):
        f = _surface_features(text)
        f.update(_syntax_features(doc))
        rows.append(f)
    return pd.DataFrame(rows).fillna(0.0)


def extract_one(text: str) -> dict:
    return extract_features([text]).iloc[0].to_dict()


FEATURE_GROUPS = {
    "readability": ["flesch_reading_ease", "fk_grade", "gunning_fog", "smog", "coleman_liau",
                    "ari", "dale_chall", "difficult_words", "syllables_per_word"],
    "lexical": ["n_chars", "n_words", "avg_word_len", "type_token_ratio", "long_word_ratio",
                "rare_word_ratio", "very_rare_word_ratio", "mean_zipf", "min_zipf", "n_digits",
                "n_numbers", "n_math_symbols", "n_parentheses", "has_blank", "ends_with_question_mark"],
    "syntactic": ["n_sentences", "max_tree_depth", "mean_tree_depth", "n_subordinate",
                  "n_conjunctions", "noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio",
                  "num_ratio", "n_noun_chunks", "n_entities"],
    "cognitive": ["bloom_level"] + [f"bloom_{l.lower()}" for l in BLOOM_LEVELS] + [f"qw_{q}" for q in QWORDS[:-1]],
    "tone": ["vader_compound", "vader_pos", "vader_neg", "n_negations", "n_constraints", "n_hedges"],
}
