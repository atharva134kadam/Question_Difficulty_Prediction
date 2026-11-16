"""Build the presentation notebooks (one per rubric component) and execute them.

    python -m src.build_notebooks            # build + execute all
"""
from __future__ import annotations

import sys

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

from src.config import ROOT

NB_DIR = ROOT / "notebooks"
SETUP = """import sys, json, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown
pd.set_option('display.max_colwidth', 120)
REPORTS = ROOT / 'reports'; FIG = REPORTS / 'figures'
def show(name): display(Image(filename=str(FIG / name)))
def rj(name): return json.loads((REPORTS / name).read_text())"""


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


NOTEBOOKS = {
# ----------------------------------------------------------------------------------------------
"01_data_collection_and_eda.ipynb": [
    md("""# 01 · Data collection & exploratory analysis
**Goal:** build a labelled corpus of exam questions with difficulty labels from public datasets.

No single public dataset labels question difficulty across subjects, so we combine **9 datasets**
whose structure *already encodes difficulty* (school grade, exam partition, competition level).
All datasets are downloaded **once** (`python -m src.data.download`) and stored in full under `data/raw/`."""),
    code(SETUP),
    code("""summ = json.loads((ROOT/'data/raw/datasets_summary.json').read_text())
cards = pd.DataFrame([{'dataset':k,'rows':v['total_rows'],'difficulty_signal':v['difficulty_signal'],
                       'license':v['license'],'description':v['description']} for k,v in summ.items()])
print('Total raw rows:', cards.rows.sum()); cards"""),
    md("## Raw samples from every dataset"),
    code("""for name in summ:
    f = sorted((ROOT/'data/raw'/name).glob('*.parquet'))[0]
    d = pd.read_parquet(f)
    col = next(c for c in ['question','question_stem','problem','sentence'] if c in d.columns)
    print(f'--- {name} ({f.name}) ---')
    for t in d[col].head(2): print('  •', str(t)[:160].replace('\\n',' '))"""),
    md("""## Difficulty labelling scheme
Each source / sub-group gets a **difficulty anchor score (0-100)** based on its educational level.
The level is derived from the score: Easy < 33.3 ≤ Medium < 66.7 ≤ Hard."""),
    code("""from src.config import ANCHORS, score_to_level
pd.DataFrame([{'group':g,'anchor_score':s,'level':score_to_level(s)} for g,s in sorted(ANCHORS.items(), key=lambda x:x[1])])"""),
    md("## Unified corpus & balanced modelling set"),
    code("""allq = pd.read_parquet(ROOT/'data/processed/all_questions.parquet')
m = pd.read_csv(ROOT/'data/processed/modelling_set.csv')
print('All cleaned & de-duplicated questions:', len(allq))
print('Balanced modelling set (≤3000 per group):', len(m))
pd.crosstab(m.split, m.level, margins=True)"""),
    code("show('data_distribution.png')"),
    code("""m['n_words'] = m.question.str.split().str.len()
m.groupby('level').n_words.describe().round(1)"""),
    code("show('readability_by_level.png')"),
    md("""**Observations**
* Hard questions are clearly longer and syntactically deeper; readability grade increases with difficulty.
* Overlap between levels is large, so no single surface feature is enough → we need learned models."""),
],
# ----------------------------------------------------------------------------------------------
"02_CO1_extraction_preprocessing_vectorization.ipynb": [
    md("""# 02 · CO1: Text extraction, preprocessing, cleaning & vectorisation (7 marks)
1. **Extraction** of questions from unstructured documents (PDF / DOCX / TXT / images via OCR)
2. **Cleaning**: HTML, LaTeX, marks, numbering, unicode, diagram code
3. **Preprocessing**: tokenisation, stop-words, POS tagging, stemming vs lemmatisation
4. **Vectorisation**: BoW, TF-IDF (word & char n-grams), Word2Vec, linguistic feature vectors"""),
    code(SETUP),
    md("## 1 · Text extraction from an unstructured question paper"),
    code("""from src.data.extract import read_text, extract_questions
raw = read_text(ROOT/'samples/sample_question_paper.txt')
print(raw[:700])"""),
    code("""qs = extract_questions(ROOT/'samples/sample_question_paper.txt')
print(f'{len(qs)} questions extracted (headers, MCQ options and marks removed, sub-questions split):')
for i,q in enumerate(qs,1): print(f'{i:2d}.', q)"""),
    code("""# the same works on DOCX and PDF files
qs_docx = extract_questions(ROOT/'samples/sample_question_paper.docx')
print(len(qs_docx), 'questions from DOCX')"""),
    md("## 2 · Cleaning noisy text"),
    code("""from src.nlp.cleaning import clean_text
noisy = ['Q3. Find the value of $\\\\frac{x^2}{4}$ when $x = 6$ <b>[2 marks]</b>',
         '(a) Compute $\\\\sqrt{45} - 2\\\\sqrt{5}$. [asy] draw((0,0)--(1,1)); [/asy]',
         'Which of the following   is NOT a ___ gas?&nbsp;',
         '1) If $a \\\\le b$ and $b \\\\neq 0$, find $\\\\dfrac{a}{b}$.']
pd.DataFrame({'raw': noisy, 'cleaned': [clean_text(t) for t in noisy]})"""),
    md("## 3 · Preprocessing pipeline step by step"),
    code("""from src.nlp.cleaning import preprocess_steps
s = preprocess_steps('Q5. The students were running experiments to determine how quickly the enzymes break down proteins.')
for k,v in s.items(): print(f'{k:18s}', v)"""),
    code("""from src.nlp.cleaning import preprocess
words = ['studies','studying','better','running','analysis','analyses','caring','cared']
pd.DataFrame({'word':words,'porter_stem':[preprocess(w,'stem',False)[0] for w in words],
              'wordnet_lemma':[preprocess(w,'lemma',False)[0] for w in words]})"""),
    md("## 4 · Vectorisation"),
    code("""from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
docs = ['What is the capital of France?', 'Prove that the square root of two is irrational.',
        'What is the square of seven?']
for V in (CountVectorizer(), TfidfVectorizer()):
    X = V.fit_transform(docs)
    display(Markdown(f'**{V.__class__.__name__}**'))
    display(pd.DataFrame(X.toarray().round(2), columns=V.get_feature_names_out()))"""),
    code("""from gensim.models import Word2Vec
w2v = Word2Vec.load(str(ROOT/'models_store/word2vec.model'))
print('Vocabulary:', len(w2v.wv), 'words × 100 dims (trained locally on our questions)')
for w in ['photosynthesis','integer','triangle','passage','force']:
    if w in w2v.wv: print(f'{w:15s} →', [x for x,_ in w2v.wv.most_similar(w, topn=6)])"""),
    code("""from src.features.linguistic import extract_one, FEATURE_GROUPS
f = extract_one('Evaluate the impact of the Treaty of Versailles on the rise of fascism in Europe.')
pd.DataFrame([(g, k, round(f[k],3)) for g,ks in FEATURE_GROUPS.items() for k in ks], columns=['group','feature','value']).head(30)"""),
    md("## 5 · Which representation works best? (same Logistic Regression / Ridge, held-out test set)"),
    code("""vc = pd.read_csv(REPORTS/'vectorizer_comparison.csv'); vc"""),
    code("""ax = vc.set_index('vectorizer')[['accuracy','macro_f1','qwk']].plot.barh(figsize=(8,4)); ax.set_xlim(0,1); plt.title('Vectoriser comparison'); plt.show()"""),
    code("show('embedding_tsne.png')"),
    md("""**Conclusion:** TF-IDF with word + character n-grams is the strongest classic representation
(char n-grams capture maths notation and morphology). Hand-crafted features are weaker alone but
**interpretable** and complementary, so they are combined with TF-IDF in the LightGBM model (notebook 05)."""),
],
# ----------------------------------------------------------------------------------------------
"03_CO2_sentiment_and_information_extraction.ipynb": [
    md("""# 03 · CO2: Sentiment analysis & information extraction (8 marks)
**Sentiment**: VADER (rule/lexicon) vs TF-IDF + Naive Bayes vs TF-IDF + Logistic Regression on SST-2,
then applied to exam questions (tone, negation, hedging as difficulty signals).

**Information extraction**: named entities, keyphrases (YAKE, RAKE, TextRank), subject-verb-object triples,
numbers / units / formulas, Bloom's-taxonomy cognitive level, with evaluation on hand-labelled gold data."""),
    code(SETUP),
    md("## Sentiment analysis models"),
    code("""sm = rj('sentiment_metrics.json')
pd.DataFrame({k:v for k,v in sm.items() if isinstance(v,dict) and 'macro_f1' in v}).T.round(4)"""),
    code("""pd.DataFrame(sm['classification_report_lr']).T.round(3)"""),
    code("""from src.nlp.sentiment import analyze
for t in ['This is a wonderful and easy question!', 'Unfortunately, which of the following is NOT a cause of the terrible famine?',
          'What is the boiling point of water?']:
    print(analyze(t), '|', t)"""),
    md("### Does question tone relate to difficulty?"),
    code("""pd.read_csv(REPORTS/'sentiment_vs_difficulty.csv').round(3)"""),
    code("""import seaborn as sns
from src.config import DATA_PROC
f = pd.read_parquet(DATA_PROC/'features_train.parquet'); tr = pd.read_parquet(DATA_PROC/'train.parquet'); f['level']=tr.level.values
display(f.groupby('level')[['vader_compound','n_negations','n_constraints','n_hedges']].mean().round(3).loc[['Easy','Medium','Hard']])"""),
    md("""Questions are mostly **neutral** in tone. Hard questions carry more constraint words ("exactly", "at least",
"distinct") and negations ("NOT", "EXCEPT"), which is why tone features are part of the difficulty model."""),
    md("## Information extraction"),
    code("""from src.nlp.info_extraction import extract
for q in ['Calculate the force needed to accelerate a 1200 kg car from rest to 25 m/s in 8 seconds on the Mumbai-Pune expressway.',
          'Evaluate the impact of the Treaty of Versailles on the rise of fascism in Europe.',
          'Which of the following is NOT a function of the mitochondria?']:
    print('\\nQ:', q); r = extract(q)
    for k in ['question_type','bloom_level','bloom_indicators','entities','keyphrases_yake','keyphrases_textrank','keyphrases_rake','svo_triples','numbers','quantities_with_units']:
        print(f'   {k:22s}', r[k])"""),
    md("## Evaluation on gold data"),
    code("""ie = rj('information_extraction_metrics.json')
pd.DataFrame({'Bloom (explicit-verb set, 60 q)': {k: ie['blooms_explicit_verbs'][k] for k in ['accuracy','macro_f1','exact_or_adjacent']},
              'Bloom (real exam questions, 60 q)': {k: ie['blooms_real_exam_questions'][k] for k in ['accuracy','macro_f1','exact_or_adjacent']},
              'Quantity extraction': {'accuracy': None, 'macro_f1': ie['quantities']['f1'], 'exact_or_adjacent': None}}).round(3)"""),
    code("""import seaborn as sns
from src.nlp.blooms import BLOOM_LEVELS
cm = np.array(ie['blooms_real_exam_questions']['confusion_matrix'])
sns.heatmap(cm, annot=True, fmt='d', xticklabels=BLOOM_LEVELS, yticklabels=BLOOM_LEVELS, cmap='Purples')
plt.xlabel('predicted'); plt.ylabel('gold'); plt.title("Bloom's extractor on real exam questions"); plt.show()"""),
    code("""pd.DataFrame(ie['blooms_real_exam_questions']['errors']).head(12)"""),
    md("""**Insight:** the rule-based extractor is perfect when the command verb is explicit, but real exam questions
often *imply* the cognitive level ("A 10 kg box slides ... The force is"). This motivates learning
difficulty from data instead of relying on rules alone."""),
    code("show('bloom_vs_difficulty.png')"),
],
# ----------------------------------------------------------------------------------------------
"04_CO3_topic_modeling_and_multilingual.ipynb": [
    md("""# 04 · CO3: Topic modelling & multilingual NLP (7 marks)
* **Topic modelling:** LDA (gensim) with the number of topics chosen by **C_v coherence**, compared with NMF,
  and linked to difficulty (which topics are hardest?).
* **Multilingual NLP:** Hindi & Marathi questions → language identification → Indic normalisation and
  tokenisation → **local neural machine translation** (MarianMT, OPUS-MT) → difficulty prediction."""),
    code(SETUP),
    md("## Topic modelling"),
    code("""tp = rj('topics.json')
ck = pd.Series(tp['lda_coherence_by_k']); ck.index = ck.index.astype(int)
ck.plot(marker='o', title='LDA coherence (C_v) vs number of topics'); plt.xlabel('k'); plt.show()
print('best k =', tp['best_k'], '| LDA C_v =', round(tp['lda_coherence'],3), '| NMF C_v =', round(tp['nmf_coherence'],3))"""),
    code("""lt = pd.DataFrame(tp['lda_topics']); lt['top_words'] = lt.top_words.map(lambda w: ', '.join(w[:8]))
lt.sort_values('mean_score')[['topic','top_words','n_questions','mean_score','pct_hard','main_sources']]"""),
    code("""lt2 = lt.sort_values('mean_score')
plt.figure(figsize=(8,4.5)); plt.barh(lt2.top_words.str.slice(0,45), lt2.mean_score, color=plt.cm.RdYlGn_r(lt2.mean_score/100))
plt.xlabel('average difficulty score'); plt.title('Difficulty per LDA topic'); plt.show()"""),
    code("""pd.DataFrame(tp['nmf_topics']).assign(top_words=lambda d: d.top_words.map(', '.join))"""),
    code("""from src.nlp.topics import topic_of
for q in ['What is the probability of rolling two sixes?', 'Why do plants need sunlight?', 'The passage mainly tells us that']:
    print(topic_of(q), '|', q)"""),
    md("## Multilingual NLP"),
    code("""from src.multilingual.pipeline import detect_language, indic_tokenize, remove_hindi_stopwords, to_english
for q in ['न्यूटन के गति के तीसरे नियम को एक उदाहरण सहित समझाइए।', 'पृथ्वीवर ऋतू का बदलतात?', 'Why do seasons change on Earth?']:
    d = detect_language(q); print(d['name'], '|', d['evidence'], '|', q)"""),
    code("""q = 'न्यूटन के गति के तीसरे नियम को एक उदाहरण सहित समझाइए।'
toks = indic_tokenize(q)
print('tokens          :', toks); print('no stop-words   :', remove_hindi_stopwords(toks))
print('MT → English    :', to_english(q)['english'])"""),
    code("""mm = rj('multilingual_metrics.json')
print('Language-ID accuracy:', mm['language_id_accuracy'])
pd.DataFrame({k:v for k,v in mm.items() if isinstance(v,dict)}).round(3)"""),
    code("""mp = pd.read_csv(REPORTS/'multilingual_predictions.csv')
mp[['lang','question','english_mt','expected_level','level_mt','score_mt','level_ref','score_ref','bleu']].round(2)"""),
    md("""**Observations:** language identification is reliable; Hindi→English MT preserves meaning well enough
for consistent difficulty predictions, while the small Marathi model is weaker. This is the main limitation of
fully-offline multilingual NLP with compact models (see the BLEU column)."""),
],
# ----------------------------------------------------------------------------------------------
"05_difficulty_models_training_evaluation.ipynb": [
    md("""# 05 · Difficulty models: training, evaluation & explainability
| Model | Idea |
|---|---|
| TF-IDF + LogReg / Ridge | strong linear baseline |
| LightGBM (linguistic features) | interpretable features only |
| LightGBM (linguistic + stacked TF-IDF) | gradient boosting, explained with SHAP |
| BiLSTM + attention | deep learning from scratch with our Word2Vec embeddings |
| DistilBERT multi-task | fine-tuned transformer: level head + score head |
| **Ensemble** | weighted combination, tuned on validation |

Metrics: accuracy, macro-F1, quadratic weighted kappa (ordinal agreement) for the **level**;
MAE, RMSE, Spearman for the **0-100 score**."""),
    code(SETUP),
    code("""mc = pd.read_csv(REPORTS/'model_comparison.csv'); mc.round(4)"""),
    code("show('model_comparison.png')"),
    code("""h = json.loads((ROOT/'models_store/distilbert_difficulty/history.json').read_text()); pd.DataFrame(h).round(4)"""),
    code("""print('Ensemble configuration:', json.loads((ROOT/'models_store/ensemble.json').read_text()))
show('confusion_matrix_ensemble.png')"""),
    code("show('score_distribution.png')"),
    md("## Per-source analysis: does the score follow the educational level?"),
    code("""ps = pd.read_csv(REPORTS/'per_source_results.csv'); ps"""),
    code("""plt.figure(figsize=(6,6)); plt.scatter(ps.target_score, ps.mean_pred_score, s=ps.n/3)
for _,r in ps.iterrows(): plt.annotate(r.group, (r.target_score, r.mean_pred_score), fontsize=8)
plt.plot([0,100],[0,100],'--',c='grey'); plt.xlabel('anchor score'); plt.ylabel('mean predicted score'); plt.title('Calibration per source'); plt.show()"""),
    md("## Explainability"),
    code("show('shap_summary.png')"),
    code("""pd.read_csv(REPORTS/'shap_mean_abs.csv').head(15)"""),
    md("## Error analysis"),
    code("""tp = pd.read_csv(REPORTS/'test_predictions.csv')
from src.config import LEVELS
tp['pred_level'] = tp.pred_label.map(dict(enumerate(LEVELS)))
err = tp[tp.level != tp.pred_level]
print(f'{len(err)} errors of {len(tp)}'); display(pd.crosstab(err.group, err.pred_level))
err.sample(12, random_state=0)[['group','question','level','pred_level','pred_score']]"""),
    md("## Using the predictor"),
    code("""from src.predict import get_predictor
p = get_predictor()
tests = ['What is the capital of France?', 'Explain why the sky appears blue during the day.',
         'A car accelerates uniformly from 10 m/s to 30 m/s in 5 seconds. Calculate its acceleration.',
         'Evaluate the impact of the Treaty of Versailles on the rise of fascism in Europe.',
         'Find all positive integers n such that n^2 + 1 is divisible by n + 1.']
p.predict_batch(tests)[['question','level','score','confidence','p_easy','p_medium','p_hard']]"""),
    code("""r = p.predict('Prove that there are infinitely many prime numbers.')
print(r['level'], r['score'], r['probabilities']); print('SHAP:', r['shap'][:5]); print('Attention:', r.get('attention'))"""),
],
# ----------------------------------------------------------------------------------------------
"06_CO4_generative_ai_local_llm.ipynb": [
    md("""# 06 · CO4: Generative AI / LLM application & prompt engineering (8 marks)
A **local LLM** (Qwen2.5-1.5B-Instruct, served by Ollama on `localhost`, no external API) is integrated with the
trained NLP model:
1. **LLM-as-judge:** four prompt strategies (zero-shot, few-shot, chain-of-thought, and an improved v2 "score-first" prompt) compared with the trained model
2. **Grounded explanation:** the LLM explains a prediction using the NLP model's measured signals
3. **Controlled rewriting:** make a question easier/harder, then **verify** with the trained model
4. **Question generation** at a target difficulty, verified the same way"""),
    code(SETUP),
    code("""from src.llm import genai
print('Local LLM available:', genai.is_available(), '| model:', genai.MODEL)
for p in sorted((ROOT/'prompts').glob('*.txt')): print(f'\\n===== {p.name} =====\\n' + p.read_text())"""),
    md("## 1 · Prompt-strategy evaluation (90 balanced test questions: 30 Easy / 30 Medium / 30 Hard)"),
    code("""le = rj('llm_prompt_eval.json')
pd.DataFrame({k:v for k,v in le.items() if k!='rewrite'}).T[['accuracy','macro_f1','qwk','mae','spearman','accuracy_level_from_score','seconds_per_question','predicted_level_distribution']]"""),
    code("""lp = pd.read_csv(REPORTS/'llm_prompt_predictions.csv')
lp[['question','level','nlp_level','llm_zero_shot_level','llm_few_shot_level','llm_chain_of_thought_level','llm_score_first_level']].head(12)"""),
    md("""### Prompt-engineering findings
1. **Label anchoring (v1 prompts):** with the zero-shot and chain-of-thought templates the 1.5B model answered
   `"Easy"` for **all 90 questions**, even while giving scores like 67. It copies the first option of the
   `"Easy|Medium|Hard"` JSON template. Accuracy collapses to chance (33%), QWK = 0.
2. **Few-shot examples** break the anchoring: the model now uses all three levels (best macro-F1 / QWK of the LLM strategies).
3. **v2 "score-first" prompt (our fix):** remove the level words entirely, give calibrated anchor questions
   (10 to 95), ask for a short reasoning plus a score, and derive the level from the score in code. This gives the
   best LLM accuracy and no label anchoring.
4. Even the best prompt is far below the **task-specific trained ensemble**. So in the final system the LLM is used
   for what it is good at, *generation and explanation*, while the trained NLP model does the scoring and
   **verifies** every LLM output."""),
    md("## 2 · Live demo: rating with each strategy"),
    code("""q = 'Prove that the square root of 2 is irrational.'
for s in ['zero_shot','few_shot','chain_of_thought','score_first']:
    r = genai.rate(q, s); print(f"{s:17s} → {r['level']:6s} {r['score']:5.0f}  {r.get('reasoning','')[:150]}")"""),
    md("## 3 · Grounded explanation"),
    code("""from src.predict import get_predictor
pred = get_predictor()
r = pred.predict('Evaluate the impact of the Treaty of Versailles on the rise of fascism in Europe.')
print(r['level'], r['score']); print(genai.explain(r['english'], r))"""),
    md("## 4 · Controlled rewriting (verified by the trained model)"),
    code("""print('Success rate:', le['rewrite']); pd.read_csv(REPORTS/'llm_rewrite_results.csv').head(10)"""),
    code("""o = genai.rewrite('What is the chemical formula of water?', 'harder', pred)
print(json.dumps({k:o[k] for k in ['original','rewritten','before','after','score_change','verified']}, indent=1))"""),
    md("## 5 · Difficulty-targeted question generation"),
    code("""pd.DataFrame(genai.generate('Newton\\'s laws of motion', 'Hard', 3, pred))"""),
],
}


def build(execute: bool = True, only: list[str] | None = None):
    NB_DIR.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        if only and not any(o in name for o in only):
            continue
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        if execute:
            print(f"[nb] executing {name} ...", flush=True)
            ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
        nbf.write(nb, NB_DIR / name)
        print(f"[nb] wrote {name}")


if __name__ == "__main__":
    build(only=sys.argv[1:] or None)
