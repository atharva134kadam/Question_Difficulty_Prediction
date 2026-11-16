"""CO3 - Multilingual NLP: questions in Hindi / Marathi (Devanagari) and English.

Pipeline
  1. Language identification: Unicode-script analysis + n-gram language model (langdetect)
     + a Marathi/Hindi disambiguator using function-word lexicons
  2. Indic preprocessing: normalisation (NFC, nukta, danda), tokenisation, Hindi stopwords
  3. Neural machine translation to English with local MarianMT models (Helsinki-NLP OPUS-MT,
     Transformer encoder-decoder) - runs fully offline after the one-time download
  4. Difficulty prediction on the English translation
Evaluation: language-ID accuracy, translation BLEU vs human references, cross-lingual
consistency of predicted difficulty (reports/multilingual_metrics.json)
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")
DANDA = re.compile(r"[।॥]")

# Chosen by BLEU on data/gold/multilingual_eval.csv (see src/multilingual/evaluate.py):
# for Marathi the multilingual OPUS-MT model beats the dedicated mr-en model (0.34 vs 0.08 BLEU)
MT_MODELS = {"hi": "Helsinki-NLP/opus-mt-hi-en", "mr": "Helsinki-NLP/opus-mt-mul-en"}
MT_CANDIDATES = {"hi": ["Helsinki-NLP/opus-mt-hi-en", "Helsinki-NLP/opus-mt-mul-en"],
                 "mr": ["Helsinki-NLP/opus-mt-mr-en", "Helsinki-NLP/opus-mt-mul-en"]}
LANG_NAMES = {"en": "English", "hi": "Hindi", "mr": "Marathi"}

# function words that are distinctive for each language
MARATHI_MARKERS = {"आहे", "आहेत", "काय", "करा", "आणि", "का", "कोणती", "कोणता", "मध्ये", "चे", "ची", "चा",
                   "असतात", "होते", "नाही", "जिथे", "यांमध्ये", "सोडवा", "स्पष्ट"}
HINDI_MARKERS = {"है", "हैं", "क्या", "कीजिए", "और", "की", "का", "के", "में", "से", "को", "कौन", "किस",
                 "कितने", "होते", "नहीं", "जहाँ", "क्यों", "समझाइए"}
HINDI_STOPWORDS = {"है", "हैं", "का", "की", "के", "में", "से", "को", "और", "पर", "यह", "वह", "एक",
                   "था", "थे", "थी", "हो", "भी", "तो", "ही", "कि", "जो", "लिए", "साथ", "कर", "गया"}


def script_profile(text: str) -> dict:
    letters = [c for c in text if c.isalpha()]
    n = max(len(letters), 1)
    return {"devanagari": sum(bool(DEVANAGARI.match(c)) for c in letters) / n,
            "latin": sum(bool(LATIN.match(c)) for c in letters) / n}


def detect_language(text: str) -> dict:
    prof = script_profile(text)
    if prof["devanagari"] > 0.3:
        toks = set(indic_tokenize(text))
        mr, hi = len(toks & MARATHI_MARKERS), len(toks & HINDI_MARKERS)
        try:
            from langdetect import DetectorFactory, detect_langs
            DetectorFactory.seed = 0
            probs = {l.lang: l.prob for l in detect_langs(text)}
        except Exception:
            probs = {}
        # combine lexicon votes with the statistical detector
        score_mr = mr + 2 * probs.get("mr", 0)
        score_hi = hi + 2 * probs.get("hi", 0)
        lang = "mr" if score_mr > score_hi else "hi"
        return {"lang": lang, "name": LANG_NAMES[lang], "script": "Devanagari",
                "evidence": {"marathi_markers": mr, "hindi_markers": hi, "langdetect": probs}}
    return {"lang": "en", "name": "English", "script": "Latin", "evidence": prof}


def normalise_indic(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = DANDA.sub(".", text)
    text = text.replace("‍", "").replace("‌", "")  # zero-width joiners
    return re.sub(r"\s+", " ", text).strip()


def indic_tokenize(text: str) -> list[str]:
    return re.findall(r"[ऀ-ॿ]+|[A-Za-z]+|\d+|[^\s\w]", normalise_indic(text))


def remove_hindi_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in HINDI_STOPWORDS]


@lru_cache(maxsize=2)
def _mt_by_name(name: str):
    from transformers import MarianMTModel, MarianTokenizer
    return MarianTokenizer.from_pretrained(name), MarianMTModel.from_pretrained(name).eval()


def _mt(lang: str):
    return _mt_by_name(MT_MODELS[lang])



def translate_to_english(texts: list[str] | str, lang: str, model_name: str | None = None) -> list[str] | str:
    import torch
    single = isinstance(texts, str)
    texts = [normalise_indic(t) for t in ([texts] if single else texts)]
    tok, model = _mt_by_name(model_name) if model_name else _mt(lang)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            enc = tok(texts[i:i + 16], return_tensors="pt", padding=True, truncation=True, max_length=256)
            gen = model.generate(**enc, num_beams=4, max_new_tokens=128)
            out += tok.batch_decode(gen, skip_special_tokens=True)
    return out[0] if single else out


def to_english(text: str) -> dict:
    """Detect language and translate to English if needed."""
    info = detect_language(text)
    if info["lang"] == "en":
        return {**info, "original": text, "english": text, "translated": False}
    return {**info, "original": text, "english": translate_to_english(text, info["lang"]),
            "translated": True, "tokens": indic_tokenize(text)}
