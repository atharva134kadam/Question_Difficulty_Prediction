"""CO2 - Information extraction from questions.

Extracts:
  - Named entities (spaCy statistical NER)
  - Keyphrases: YAKE (statistical), RAKE (co-occurrence), TextRank (graph-based, own implementation)
  - Subject-Verb-Object triples (dependency parsing)
  - Numbers, units and mathematical expressions (regex)
  - Bloom's taxonomy cognitive level (rule-based)
  - Question type (wh-word / imperative / fill-in-the-blank)
"""
from __future__ import annotations

import re
from functools import lru_cache

import networkx as nx

from src.nlp.blooms import bloom_level

UNIT = re.compile(r"(\d+(?:\.\d+)?)\s*(km|m|cm|mm|kg|g|mg|s|sec|seconds|minutes|hours|h|ml|l|liters?|°c|°f|"
                  r"degrees|%|percent|n|newtons?|j|joules?|w|watts?|v|volts?|a|amps?|mol|hz|\$|dollars?|cents?)\b", re.I)
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)?(?:/\d+)?")
EXPR = re.compile(r"[A-Za-z0-9()\s]*[=<>≤≥][A-Za-z0-9()+\-*/^.\s]+")


@lru_cache(maxsize=1)
def _nlp():
    import spacy
    return spacy.load("en_core_web_sm")


def entities(doc) -> list[dict]:
    return [{"text": e.text, "label": e.label_} for e in doc.ents]


def svo_triples(doc) -> list[tuple[str, str, str]]:
    triples = []
    for tok in doc:
        if tok.pos_ in ("VERB", "AUX"):
            subj = [w for w in tok.lefts if w.dep_ in ("nsubj", "nsubjpass")]
            obj = [w for w in tok.rights if w.dep_ in ("dobj", "attr", "pobj", "dative", "oprd")]
            for w in tok.rights:
                if w.dep_ == "prep":
                    obj += [o for o in w.rights if o.dep_ == "pobj"]
            for s in subj:
                for o in obj:
                    triples.append((" ".join(t.text for t in s.subtree),
                                    tok.lemma_,
                                    " ".join(t.text for t in o.subtree)))
    return triples[:5]


def textrank_keyphrases(doc, top_k=5) -> list[str]:
    """TextRank: PageRank over a co-occurrence graph of nouns/adjectives (window=3)."""
    cand = [t for t in doc if t.pos_ in ("NOUN", "PROPN", "ADJ") and not t.is_stop and len(t) > 2]
    g = nx.Graph()
    for i, a in enumerate(cand):
        for b in cand[i + 1:i + 3]:
            if a.lemma_ != b.lemma_:
                g.add_edge(a.lemma_.lower(), b.lemma_.lower())
    if not g:
        return [t.text.lower() for t in cand[:top_k]]
    rank = nx.pagerank(g)
    # score noun chunks by the sum of their word ranks
    phrases = {}
    for nc in doc.noun_chunks:
        content = [t for t in nc if not t.is_stop and not t.is_punct]
        if content:
            text = " ".join(t.text for t in content).lower()
            phrases[text] = sum(rank.get(t.lemma_.lower(), 0) for t in content)
    if not phrases:
        phrases = rank
    return [p for p, _ in sorted(phrases.items(), key=lambda x: -x[1])[:top_k]]


@lru_cache(maxsize=1)
def _yake():
    import yake
    return yake.KeywordExtractor(lan="en", n=3, top=5, dedupLim=0.7)


def yake_keyphrases(text: str) -> list[str]:
    return [k for k, _ in _yake().extract_keywords(text)]


def rake_keyphrases(text: str, top_k=5) -> list[str]:
    from rake_nltk import Rake
    r = Rake()
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()[:top_k]


def question_type(text: str) -> str:
    t = text.strip().lower()
    if "___" in t:
        return "fill-in-the-blank"
    m = re.match(r"(what|which|why|how|who|when|where|whose|whom|is|are|does|do|can)\b", t)
    if m:
        return f"wh-question ({m.group(1)})" if m.group(1) not in ("is", "are", "does", "do", "can") else "yes/no question"
    if re.search(r"\b(what|which|why|how)\b[^.]*\?", t):
        return "embedded wh-question"
    if t.endswith("?"):
        return "other question"
    return "imperative / statement completion"


def extract(text: str) -> dict:
    doc = _nlp()(text)
    level, idx, hits = bloom_level(text)
    return {
        "question_type": question_type(text),
        "bloom_level": level,
        "bloom_index": idx,
        "bloom_indicators": hits,
        "entities": entities(doc),
        "keyphrases_textrank": textrank_keyphrases(doc),
        "keyphrases_yake": yake_keyphrases(text),
        "keyphrases_rake": rake_keyphrases(text),
        "svo_triples": svo_triples(doc),
        "numbers": NUMBER.findall(text),
        "quantities_with_units": [" ".join(m) for m in UNIT.findall(text)],
        "math_expressions": [e.strip() for e in EXPR.findall(text) if len(e.strip()) > 2][:5],
    }


if __name__ == "__main__":
    import json
    for q in ["Evaluate the impact of the Treaty of Versailles on the rise of fascism in Europe.",
              "A car travels 120 km in 2 hours. What is its average speed in km/h?",
              "Which of the following is NOT a function of the mitochondria?"]:
        print(json.dumps(extract(q), indent=1, default=str))
