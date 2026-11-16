"""DiffiQ - FastAPI backend for the Question Difficulty Prediction project.

Serves the single-page frontend (frontend/) and a JSON API. Runs 100% locally.

    .venv/bin/python -m uvicorn app.server:app --port 8000
    open http://localhost:8000
"""
from __future__ import annotations

import json
import math
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src  # noqa: F401  (sets OpenMP env + LightGBM-before-torch load order)
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import DATA_GOLD, DATA_PROC, DATA_RAW, FIGURES, PROMPTS, REPORTS

app = FastAPI(title="DiffiQ - Question Difficulty API", version="1.0")
_lock = threading.Lock()          # models are not thread-safe; serialise inference
_pred = None


def predictor():
    global _pred
    if _pred is None:
        from src.predict import get_predictor
        _pred = get_predictor()
    return _pred


def clean(o):
    """Make numpy / NaN values JSON-safe."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def report(name):
    p = REPORTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text()) if name.endswith(".json") else pd.read_csv(p).to_dict("records")


class Q(BaseModel):
    question: str


class Rewrite(BaseModel):
    question: str
    direction: str = "harder"


class Gen(BaseModel):
    topic: str
    level: str = "Medium"
    n: int = 3


class Judge(BaseModel):
    question: str
    strategy: str = "score_first"


# ------------------------------------------------------------------ core
@app.get("/api/health")
def health():
    from src.llm.genai import MODEL, is_available
    return {"status": "ok", "models_loaded": _pred is not None, "llm": is_available(), "llm_model": MODEL}


@app.post("/api/warmup")
def warmup():
    with _lock:
        predictor().predict("What is the capital of France?", explain=True)
    return {"ok": True}


@app.post("/api/predict")
def predict(q: Q):
    if not q.question.strip():
        raise HTTPException(400, "Empty question")
    with _lock:
        r = predictor().predict(q.question.strip(), explain=True)
    return clean(r)


@app.post("/api/paper")
async def paper(file: UploadFile | None = File(None)):
    from src.data.extract import extract_questions, read_text
    if file is not None and file.filename:
        data, name = await file.read(), file.filename
    else:
        p = ROOT / "samples" / "sample_question_paper.txt"
        data, name = p.read_bytes(), p.name
    try:
        qs = extract_questions(data, name)
    except Exception as e:
        raise HTTPException(400, f"Could not read file: {e}")
    if not qs:
        raise HTTPException(400, "No questions found in the document")
    with _lock:
        df = predictor().predict_batch(qs[:200])
    raw = "" if name.lower().endswith(".csv") else read_text(data, name)[:6000]
    return clean({"filename": name, "questions": df.to_dict("records"), "raw_text": raw})


# ------------------------------------------------------------------ NLP lab
@app.post("/api/nlp/preprocess")
def preprocess(q: Q):
    from src.nlp.cleaning import preprocess_steps
    return clean(preprocess_steps(q.question))


@app.post("/api/nlp/sentiment")
def sentiment(q: Q):
    from src.nlp.sentiment import analyze
    return clean(analyze(q.question))


@app.post("/api/nlp/extract")
def extract(q: Q):
    from src.nlp.info_extraction import extract as ie
    with _lock:
        return clean(ie(q.question))


@app.post("/api/nlp/topic")
def topic(q: Q):
    from src.nlp.topics import topic_of
    return clean(topic_of(q.question))


@app.post("/api/nlp/multilingual")
def multilingual(q: Q):
    from src.multilingual.pipeline import indic_tokenize, remove_hindi_stopwords, to_english
    with _lock:
        info = to_english(q.question)
        r = predictor().predict(q.question, explain=False)
    toks = indic_tokenize(q.question)
    return clean({**info, "tokens": toks, "tokens_no_stopwords": remove_hindi_stopwords(toks),
                  "level": r["level"], "score": r["score"], "confidence": r["confidence"]})


# ------------------------------------------------------------------ GenAI
def _need_llm():
    from src.llm.genai import is_available
    if not is_available():
        raise HTTPException(503, "Local LLM offline - start it with: ollama serve")


@app.post("/api/llm/explain")
def llm_explain(q: Q):
    _need_llm()
    from src.llm.genai import explain
    with _lock:
        r = predictor().predict(q.question, explain=True)
        text = explain(r["english"], r)
    return {"explanation": text, "level": r["level"], "score": r["score"]}


@app.post("/api/llm/rewrite")
def llm_rewrite(b: Rewrite):
    _need_llm()
    from src.llm.genai import rewrite
    with _lock:
        return clean(rewrite(b.question, b.direction, predictor()))


@app.post("/api/llm/generate")
def llm_generate(b: Gen):
    _need_llm()
    from src.llm.genai import generate
    with _lock:
        return clean({"items": generate(b.topic, b.level, max(1, min(b.n, 5)), predictor())})


@app.post("/api/llm/judge")
def llm_judge(b: Judge):
    _need_llm()
    from src.llm.genai import rate
    with _lock:
        o = rate(b.question, b.strategy)
        n = predictor().predict(b.question, explain=False)
    return clean({"llm": {k: o[k] for k in ("level", "score", "reasoning", "strategy")},
                  "model": {"level": n["level"], "score": n["score"]}})


@app.get("/api/prompts")
def prompts():
    return {p.name: p.read_text() for p in sorted(PROMPTS.glob("*.txt"))}


# ------------------------------------------------------------------ reports & data
@app.get("/api/reports")
def reports():
    names = ["metrics.json", "vectorizer_comparison.csv", "sentiment_metrics.json", "sentiment_vs_difficulty.csv",
             "information_extraction_metrics.json", "topics.json", "multilingual_metrics.json",
             "multilingual_predictions.csv", "llm_prompt_eval.json", "llm_rewrite_results.csv",
             "per_source_results.csv", "shap_mean_abs.csv", "model_comparison.csv"]
    out = {n.rsplit(".", 1)[0]: report(n) for n in names}
    out["ensemble"] = json.loads((ROOT / "models_store" / "ensemble.json").read_text())
    out["figures"] = sorted(p.name for p in FIGURES.glob("*.png"))
    return clean(out)


@app.get("/api/datasets")
def datasets():
    summ = json.loads((DATA_RAW / "datasets_summary.json").read_text())
    for k, v in summ.items():
        v["parquet_files"] = sorted(p.name for p in (DATA_RAW / k).glob("*.parquet"))
    m = pd.read_csv(DATA_PROC / "modelling_set.csv", usecols=["group", "level", "split"])
    groups = m.groupby(["group", "level"]).size().reset_index(name="n").to_dict("records")
    splits = m.groupby(["split", "level"]).size().unstack(fill_value=0).to_dict("index")
    gold = {g.name: pd.read_csv(g).to_dict("records") for g in sorted(DATA_GOLD.glob("*.csv"))}
    n_all = len(pd.read_parquet(DATA_PROC / "all_questions.parquet", columns=["uid"]))
    return clean({"datasets": summ, "groups": groups, "splits": splits, "gold": gold,
                  "n_unified": n_all, "n_modelling": len(m)})


@app.get("/api/datasets/{name}/{file}")
def dataset_page(name: str, file: str, page: int = 1, search: str = "", size: int = 50):
    path = (DATA_RAW / name / file).resolve()
    if DATA_RAW.resolve() not in path.parents or not path.exists() or path.suffix != ".parquet":
        raise HTTPException(404, "file not found")
    df = pd.read_parquet(path)
    if search:
        mask = df.astype(str).apply(lambda c: c.str.contains(search, case=False, na=False, regex=False)).any(axis=1)
        df = df[mask]
    total = len(df)
    view = df.iloc[(page - 1) * size: page * size].copy()
    for c in view.columns:
        view[c] = view[c].map(lambda x: str(x)[:400] if not isinstance(x, (int, float)) else x)
    return clean({"columns": list(view.columns), "rows": view.to_dict("records"), "total": total,
                  "page": page, "pages": max(1, math.ceil(total / size))})


@app.get("/api/sample-paper")
def sample_paper():
    return FileResponse(ROOT / "samples" / "sample_question_paper.docx", filename="sample_question_paper.docx")


app.mount("/figures", StaticFiles(directory=FIGURES), name="figures")
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
