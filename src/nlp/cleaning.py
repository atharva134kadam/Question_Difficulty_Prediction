"""CO1 - Text cleaning and preprocessing for unstructured question text.

Two layers:
  clean_text()      -> normalised text used by every model (keeps meaning)
  preprocess()      -> tokenised / lemmatised / stop-word-free tokens used by
                       classic models (BoW, TF-IDF, LDA, Word2Vec)
"""
from __future__ import annotations

import html
import re
import unicodedata
from functools import lru_cache

# --------------------------------------------------------------------------
# Regexes
# --------------------------------------------------------------------------
HTML_TAG = re.compile(r"<[^>]+>")
URL = re.compile(r"https?://\S+|www\.\S+")
MARKS = re.compile(r"[\[\(]\s*\d+\s*(marks?|m|pts?|points?)\s*[\]\)]", re.I)
NUMBERING = re.compile(r"^\s*(q(uestion)?\s*\.?\s*\d+[\.\):]?|\d+\s*[\.\)]|\(?[a-h]\)|\(?[ivx]+\))\s+", re.I)
MULTISPACE = re.compile(r"\s+")
BLANK = re.compile(r"_{2,}")
ASY = re.compile(r"\[asy\].*?(\[/asy\]|$)", re.S | re.I)   # Asymptote diagram source code (MATH)

LATEX_REPLACEMENTS = [
    (r"\\left|\\right|\\displaystyle|\\!|\\,|\\;|\\quad|\\qquad", " "),
    (r"\\dfrac|\\tfrac", r"\\frac"),
    (r"\\cdot|\\times", " * "),
    (r"\\div", " / "),
    (r"\\le(q)?\b", " <= "),
    (r"\\ge(q)?\b", " >= "),
    (r"\\neq?\b", " != "),
    (r"\\pi\b", " pi "),
    (r"\\infty\b", " infinity "),
    (r"\\sqrt", " sqrt "),
    (r"\\(sin|cos|tan|log|ln|exp|lim|sum|int|prod|theta|alpha|beta|gamma|angle|triangle|circ|deg)\b", r" \1 "),
    (r"\\text\{([^}]*)\}|\\mathrm\{([^}]*)\}|\\textbf\{([^}]*)\}", r" \1\2\3 "),
    (r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r" (\1)/(\2) "),
    (r"\\begin\{[^}]*\}|\\end\{[^}]*\}", " "),
    (r"\\\[|\\\]|\\\(|\\\)|\$", " "),
    (r"\\[a-zA-Z]+", " "),
    (r"[{}]", " "),
]
_LATEX = [(re.compile(p), r) for p, r in LATEX_REPLACEMENTS]

LATEX_DETECT = re.compile(r"\$|\\[a-zA-Z]+|\^|_\{")


def has_latex(text: str) -> bool:
    return bool(LATEX_DETECT.search(text or ""))


def normalise_latex(text: str) -> str:
    for pat, rep in _LATEX:
        text = pat.sub(rep, text)
    return text


def clean_text(text: str, strip_numbering: bool = True) -> str:
    """Normalise raw question text without destroying its meaning."""
    if text is None:
        return ""
    text = str(text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = ASY.sub(" [diagram] ", text)
    text = HTML_TAG.sub(" ", text)
    text = URL.sub(" URL ", text)
    text = MARKS.sub(" ", text)
    if strip_numbering:
        for _ in range(3):            # "Q2. (a) ..." has stacked labels
            text, n = NUMBERING.subn("", text)
            if not n:
                break
    if has_latex(text):
        text = normalise_latex(text)
    text = BLANK.sub(" ___ ", text)
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = MULTISPACE.sub(" ", text).strip()
    return text


# --------------------------------------------------------------------------
# Token-level preprocessing (NLTK) - for classic NLP models
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _nltk():
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer

    sw = set(stopwords.words("english"))
    # Keep words that matter for difficulty (question words, negations)
    sw -= {"what", "why", "how", "which", "who", "when", "where", "not", "no", "nor",
           "most", "least", "all", "only", "against", "more", "few"}
    return nltk, sw, PorterStemmer(), WordNetLemmatizer()


TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|[+\-*/=<>^]")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def preprocess(text: str, mode: str = "lemma", remove_stopwords: bool = True) -> list[str]:
    """Return processed tokens. mode in {'none', 'stem', 'lemma'}."""
    _, sw, stemmer, lemmatizer = _nltk()
    toks = tokenize(clean_text(text))
    if remove_stopwords:
        toks = [t for t in toks if t not in sw]
    if mode == "stem":
        toks = [stemmer.stem(t) for t in toks]
    elif mode == "lemma":
        toks = [lemmatizer.lemmatize(lemmatizer.lemmatize(t), pos="v") for t in toks]
    return toks


def preprocess_steps(text: str) -> dict:
    """Show every preprocessing stage - used by the notebook and the app."""
    import nltk

    _, sw, stemmer, lemmatizer = _nltk()
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    no_sw = [t for t in tokens if t not in sw]
    try:
        pos = nltk.pos_tag(tokens)
    except LookupError:
        pos = []
    return {
        "raw": text,
        "cleaned": cleaned,
        "tokens": tokens,
        "pos_tags": pos,
        "without_stopwords": no_sw,
        "stemmed": [stemmer.stem(t) for t in no_sw],
        "lemmatized": [lemmatizer.lemmatize(lemmatizer.lemmatize(t), pos="v") for t in no_sw],
    }
