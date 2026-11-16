# 🎯 Question Difficulty Prediction using NLP

Predicts the difficulty of any exam question as a **level (Easy / Medium / Hard)** and a **score (0–100)**,
explains *why*, and uses a local generative AI model to rewrite or generate questions at a target difficulty.

**Everything runs on your machine.** It makes no external API calls. Public datasets and pretrained weights
are downloaded once, and the system works fully offline after that.

```
Question : Evaluate the impact of the Treaty of Versailles on the rise of fascism in Europe.
Level    : HARD   (confidence 84%)
Score    : 74.0/100  ██████████████░░░░░░
Bloom    : Evaluate   type: imperative
Keywords : Treaty of Versailles, fascism in Europe, rise of fascism
```

---

## 1. Rubric coverage

| Rubric item | Marks | Where it is implemented |
|---|---|---|
| **CO1** Text extraction, preprocessing, cleaning and vectorisation | 7 | `src/data/extract.py` (PDF/DOCX/TXT/OCR → questions), `src/nlp/cleaning.py` (HTML, LaTeX, marks, numbering, diagram code; tokenisation, stop-words, POS, stemming vs lemmatisation), `src/features/vectorizers.py` (BoW, TF-IDF word/char, Word2Vec, linguistic vectors, with a comparison) · notebook **02** · app tab *CO1* |
| **CO2** Sentiment analysis and information extraction | 8 | `src/nlp/sentiment.py` (VADER vs Naive Bayes vs Logistic Regression on SST-2, plus tone vs difficulty), `src/nlp/info_extraction.py` (NER, YAKE/RAKE/TextRank keyphrases, SVO triples, quantities, Bloom's level), `src/nlp/evaluate_ie.py` (gold-set evaluation) · notebook **03** |
| **CO3** Topic modelling and multilingual NLP | 7 | `src/nlp/topics.py` (LDA with coherence-based k, NMF, difficulty per topic), `src/multilingual/` (Hindi/Marathi language ID, Indic tokenisation, local MarianMT translation, cross-lingual evaluation) · notebook **04** |
| **CO4** Generative AI / LLM, prompt engineering, integration and demo | 8 | `src/llm/genai.py` and `prompts/` (local Qwen2.5 via Ollama: zero-shot vs few-shot vs chain-of-thought judging, grounded explanations, verified easier/harder rewriting, targeted question generation), `src/llm/evaluate_prompts.py` · notebook **06** · Streamlit app |

## 2. Data (all public, stored in full under `data/raw/`)

| Dataset | Rows | Difficulty signal |
|---|---|---|
| ARC (AI2 Reasoning Challenge) | 7,787 | Easy set vs Challenge set |
| SciQ | 13,679 | easy anchor |
| OpenBookQA | 5,957 | medium anchor |
| CommonsenseQA | 12,102 | easy anchor |
| RACE | 97,687 | middle school vs high school |
| GSM8K | 8,792 | grade-school maths |
| MATH (Hendrycks) | 12,500 | levels 1–5 |
| MMLU | 15,858 | elementary / high school / college and professional |
| SST-2 | 70,042 | sentiment model for CO2 |

**Labelling.** Each source sub-group gets an **anchor score** based on its educational level, for example
SciQ = 18, ARC-Challenge = 52, MATH level 5 = 92 (see `src/config.py`). The level comes from the score bands
0–33 Easy, 34–66 Medium, 67–100 Hard. After cleaning and de-duplication there are 144,818 questions. For
training, each group is capped at 3,000 questions so that no source dominates, giving a balanced set of
41,822 questions split 80/10/10.

Hand-labelled gold sets are in `data/gold/`:
- Bloom's levels for 60 explicit-verb questions and 60 real exam questions
- 35 Hindi/Marathi questions with English references

## 3. Models

| Model | Description |
|---|---|
| TF-IDF + LogReg/Ridge | word 1–2-gram and char 3–5-gram baseline |
| LightGBM (linguistic) | 57 interpretable features: readability, lexical, syntactic, Bloom's level, tone |
| LightGBM (stacked) | the linguistic features plus out-of-fold TF-IDF predictions, explained with SHAP |
| BiLSTM + attention | trained from scratch with locally trained Word2Vec embeddings |
| DistilBERT multi-task | fine-tuned transformer with a level head (cross-entropy) and a score head (MSE) |
| **Ensemble** | weighted average tuned on validation; the score is clipped into the predicted level's band so level and score never disagree |

### Results on the held-out test set (4,183 questions)

| Model | Accuracy | Macro-F1 | QWK | Score MAE | Spearman |
|---|---|---|---|---|---|
| LightGBM (linguistic features only) | 0.731 | 0.733 | 0.741 | 9.37 | 0.825 |
| BiLSTM + attention (from scratch) | 0.742 | 0.745 | 0.766 | 9.09 | 0.837 |
| TF-IDF + LogReg/Ridge (baseline) | 0.762 | 0.765 | 0.789 | 9.05 | 0.838 |
| LightGBM (linguistic + stacked TF-IDF) | 0.783 | 0.786 | 0.810 | 7.36 | 0.877 |
| DistilBERT multi-task | 0.799 | 0.800 | 0.831 | **6.57** | **0.893** |
| **Ensemble (final)** | **0.804** | **0.805** | **0.835** | 6.66 | 0.892 |

QWK is quadratic weighted kappa (agreement on ordered levels). Score MAE is the average error in points on
the 0–100 scale. The ensemble weights are LightGBM 0.4, DistilBERT 0.5 and BiLSTM 0.1, tuned on the
validation set.

### Other component results
| Component | Result |
|---|---|
| CO1 best vectoriser (same classifier) | TF-IDF word+char: F1 0.765, compared with BoW 0.705 and Word2Vec 0.672 |
| CO2 sentiment (SST-2) | TF-IDF + Naive Bayes 0.822 accuracy, LogReg 0.820, VADER 0.625 |
| CO2 Bloom's extractor | 100% on explicit-verb questions, 42% exact and 82% within one level on real exam questions |
| CO2 quantity extraction | F1 0.96 |
| CO3 topics | LDA k=10, C_v 0.51; NMF C_v 0.56. Maths and geometry topics are hardest (about 70), everyday arithmetic and biology easiest (about 38) |
| CO3 multilingual | language ID 100%. Hindi: BLEU 0.49, scores within 2.4 points of the English originals, 92% level agreement. Marathi: BLEU 0.34 |
| CO4 local LLM | see `reports/llm_prompt_eval.json` and notebook 06 |

## 4. Quick start

```bash
# 1. environment (Python 3.11)
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m spacy download en_core_web_sm
# 2. local LLM for CO4 (one-time download, then offline)
ollama pull qwen2.5:1.5b && ollama serve &
# 3. rebuild everything (or skip: trained models are already in models_store/)
./run_pipeline.sh
# 4. demo: web app at http://localhost:8000
./start.sh
#    (alternative quick UI: .venv/bin/streamlit run app/streamlit_app.py)
.venv/bin/python -m src.predict "Prove that there are infinitely many primes."
.venv/bin/python -m src.predict --file samples/sample_question_paper.docx --out results.csv
```

## 5. Project structure

```
data/raw/            all 9 public datasets, full copies (parquet + csv) + dataset cards
data/processed/      unified corpus, train/val/test, cached features
data/gold/           hand-labelled evaluation sets (Bloom's, multilingual)
src/data/            download.py · unify.py · extract.py (PDF/DOCX/TXT/OCR → questions)
src/nlp/             cleaning · blooms · sentiment · info_extraction · topics · evaluate_ie
src/features/        linguistic.py (57 features) · vectorizers.py (CO1 comparison)
src/models/          gbm · bilstm · transformer (DistilBERT) · ensemble
src/multilingual/    language ID, Indic tokenisation, MarianMT translation, evaluation
src/llm/             local LLM integration + prompt-strategy evaluation
src/predict.py       DifficultyPredictor (CLI + API used by the app)
prompts/             all prompt templates (prompt engineering)
app/server.py        FastAPI backend (JSON API, serves the frontend)
frontend/            DiffiQ web app (HTML/CSS/JS, Chart.js vendored: works offline)
app/streamlit_app.py alternative Streamlit demo
notebooks/           01 data/EDA · 02 CO1 · 03 CO2 · 04 CO3 · 05 models · 06 CO4 (executed, with outputs)
reports/             metrics.json, comparison CSVs, figures/
samples/             sample question paper (.txt, .docx)
```

## 6. Limitations and honest notes

- **Labels are source-level anchors, not per-question human ratings.** Two questions from the same group share the same target score. The model learns what separates educational levels, but part of that signal is subject and style, not pure difficulty (see the per-source analysis in notebook 05).
- **RACE questions are judged without their reading passage,** so middle-school versus high-school is hard to tell apart from the question stem alone.
- **Rule-based Bloom's tagging is far weaker on real questions** (where the thinking skill is implied) than on explicit-verb questions. This is reported, not hidden.
- **Marathi translation with the compact offline OPUS-MT model is weak,** and the BLEU scores reflect this.
- **The 1.5B-parameter local LLM is a weak difficulty judge.** It is used for generation and explanation, while scoring is left to the trained model.
