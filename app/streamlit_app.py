"""Question Difficulty Predictor - interactive demo (runs 100% locally).

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import DATA_GOLD, DATA_RAW, FIGURES, PROMPTS, REPORTS

st.set_page_config(page_title="Question Difficulty Predictor", page_icon="🎯", layout="wide")

LEVEL_COLORS = {"Easy": "#2e9d5b", "Medium": "#e0a100", "Hard": "#d6453d"}
EXAMPLES = [
    "What is the capital of France?",
    "Explain why the sky appears blue during the day.",
    "A car accelerates uniformly from 10 m/s to 30 m/s in 5 seconds. Calculate its acceleration.",
    "Evaluate the impact of the Treaty of Versailles on the rise of fascism in Europe.",
    "Find all positive integers n such that n^2 + 1 is divisible by n + 1.",
    "सिद्ध कीजिए कि 2 का वर्गमूल एक अपरिमेय संख्या है।",
]


# ----------------------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Loading NLP models (first time only)...")
def predictor():
    from src.predict import get_predictor
    return get_predictor()


def read_json(name):
    p = REPORTS / name
    return json.loads(p.read_text()) if p.exists() else None


def read_csv(name):
    p = REPORTS / name
    return pd.read_csv(p) if p.exists() else None


def gauge(score, level):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"suffix": "/100", "font": {"size": 40}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": LEVEL_COLORS[level], "thickness": 0.3},
               "steps": [{"range": [0, 33.3], "color": "rgba(46,157,91,0.18)"},
                         {"range": [33.3, 66.7], "color": "rgba(224,161,0,0.18)"},
                         {"range": [66.7, 100], "color": "rgba(214,69,61,0.18)"}]}))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=10, b=0))
    return fig


def badge(level):
    return (f"<span style='background:{LEVEL_COLORS[level]};color:white;padding:6px 18px;"
            f"border-radius:20px;font-size:1.4rem;font-weight:700'>{level.upper()}</span>")


def llm_ok():
    from src.llm.genai import is_available
    return is_available()


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🎯 Question Difficulty")
    st.caption("NLP project: predicts **Easy / Medium / Hard** and a **0-100 difficulty score** "
               "for any exam question. Everything runs locally: no external APIs.")
    st.markdown("**Rubric coverage**\n"
                "- CO1 Extraction, cleaning, vectorisation\n"
                "- CO2 Sentiment + information extraction\n"
                "- CO3 Topic modelling + multilingual (Hindi/Marathi)\n"
                "- CO4 Local generative AI + prompt engineering")
    st.markdown("**Local LLM:** " + ("🟢 Ollama qwen2.5:1.5b online" if llm_ok() else "🔴 offline (run `ollama serve`)"))

tabs = st.tabs(["🎯 Predict", "📄 Paper Analyzer", "🔬 CO1 · Pipeline", "💬 CO2 · Sentiment & IE",
                "🗂️ CO3 · Topics", "🌐 CO3 · Multilingual", "🤖 CO4 · GenAI Studio",
                "📊 Model Report", "📚 Datasets"])

# ============================================================================ 1. PREDICT
with tabs[0]:
    st.header("Predict question difficulty")
    ex = st.selectbox("Try an example (or type your own below)", ["—"] + EXAMPLES)
    q = st.text_area("Question (English, Hindi or Marathi)", value="" if ex == "—" else ex, height=100)
    if st.button("Predict difficulty", type="primary") and q.strip():
        with st.spinner("Analysing..."):
            r = predictor().predict(q)
        st.session_state["last"] = r
    r = st.session_state.get("last")
    if r:
        if r["language"]["translated"]:
            st.info(f"Detected **{r['language']['name']}** → translated: *{r['english']}*")
        c1, c2, c3 = st.columns([1.1, 1, 1.2])
        with c1:
            st.markdown(badge(r["level"]), unsafe_allow_html=True)
            st.plotly_chart(gauge(r["score"], r["level"]), width="stretch")
        with c2:
            st.metric("Confidence", f"{r['confidence']:.0%}")
            pr = pd.DataFrame({"level": list(r["probabilities"]), "p": list(r["probabilities"].values())})
            st.plotly_chart(px.bar(pr, x="p", y="level", orientation="h", color="level", range_x=[0, 1],
                                   color_discrete_map=LEVEL_COLORS, height=200).update_layout(
                showlegend=False, margin=dict(l=0, r=0, t=0, b=0), xaxis_title="probability", yaxis_title=None),
                width="stretch")
            st.caption("Ensemble members: " + ", ".join(f"{k}: {r['member_levels'][k]} ({v:.0f})"
                                                        for k, v in r["member_scores"].items()))
        with c3:
            i, f = r["info"], r["features"]
            st.markdown(f"**Bloom's level:** {i['bloom_level']}  \n"
                        f"**Question type:** {i['question_type']}  \n"
                        f"**Readability (FK grade):** {f['fk_grade']:.1f}  \n"
                        f"**Rare words:** {f['rare_word_ratio']:.0%} · **Words:** {int(f['n_words'])}  \n"
                        f"**Syntax depth:** {int(f['max_tree_depth'])} · **Clauses:** {int(f['n_subordinate'])}  \n"
                        f"**Tone (VADER):** {r['sentiment']['vader_label']} ({r['sentiment']['vader_compound']:+.2f})")
            if r.get("topic") and r["topic"]["topic"] is not None:
                st.markdown(f"**Topic:** T{r['topic']['topic']} · {', '.join(r['topic']['top_words'][:5])}")

        st.subheader("Why? Explainability")
        e1, e2 = st.columns(2)
        with e1:
            sh = pd.DataFrame(r["shap"], columns=["feature", "impact", "value"])
            sh["direction"] = sh.impact.map(lambda v: "harder" if v > 0 else "easier")
            st.plotly_chart(px.bar(sh[::-1], x="impact", y="feature", orientation="h", color="direction",
                                   color_discrete_map={"harder": "#d6453d", "easier": "#2e9d5b"},
                                   title="SHAP: feature contributions to the score (LightGBM)", height=330),
                            width="stretch")
        with e2:
            if r.get("attention"):
                att = r["attention"]
                mx = max(w for _, w in att) or 1
                html = " ".join(f"<span style='background:rgba(214,69,61,{w / mx:.2f});padding:2px 4px;"
                                f"border-radius:4px;margin:1px;display:inline-block'>{t}</span>" for t, w in att)
                st.markdown("**BiLSTM attention: words the neural model focused on**")
                st.markdown(html, unsafe_allow_html=True)
            st.markdown("**Extracted information (CO2)**")
            st.json({"entities": i["entities"], "keyphrases (YAKE)": i["keyphrases_yake"],
                     "keyphrases (TextRank)": i["keyphrases_textrank"], "SVO triples": i["svo_triples"],
                     "numbers": i["numbers"], "quantities": i["quantities_with_units"],
                     "bloom indicators": i["bloom_indicators"]}, expanded=False)
        if llm_ok() and st.button("🤖 Explain with local AI"):
            from src.llm.genai import explain
            with st.spinner("Local LLM is writing an explanation..."):
                st.markdown(explain(r["english"], r))

# ============================================================================ 2. PAPER
with tabs[1]:
    st.header("Question paper analyzer")
    st.caption("Upload a question paper (PDF, DOCX, TXT or CSV). Questions are extracted, cleaned and scored.")
    up = st.file_uploader("Question paper", type=["pdf", "docx", "txt", "csv", "md"])
    use_sample = st.checkbox("Use the bundled sample paper", value=up is None)
    if st.button("Analyze paper", type="primary"):
        from src.data.extract import extract_questions, read_text
        if up is not None and not use_sample:
            data, name = up.getvalue(), up.name
        else:
            p = ROOT / "samples" / "sample_question_paper.txt"
            data, name = p.read_bytes(), p.name
        qs = extract_questions(data, name)
        with st.spinner(f"Scoring {len(qs)} questions..."):
            df = predictor().predict_batch(qs)
        st.session_state["paper"] = df
        st.session_state["paper_raw"] = read_text(data, name) if not name.endswith(".csv") else ""
    df = st.session_state.get("paper")
    if df is not None:
        a, b, c, d = st.columns(4)
        a.metric("Questions", len(df))
        b.metric("Average score", f"{df.score.mean():.1f}")
        vc = df.level.value_counts()
        c.metric("Easy / Medium / Hard", f"{vc.get('Easy', 0)} / {vc.get('Medium', 0)} / {vc.get('Hard', 0)}")
        overall = "Easy" if df.score.mean() < 33.3 else "Medium" if df.score.mean() < 66.7 else "Hard"
        d.markdown("**Overall paper**<br>" + badge(overall), unsafe_allow_html=True)
        g1, g2 = st.columns(2)
        g1.plotly_chart(px.pie(df, names="level", color="level", color_discrete_map=LEVEL_COLORS, hole=0.45,
                               title="Difficulty mix"), width="stretch")
        g2.plotly_chart(px.bar(df.reset_index(), x="index", y="score", color="level", color_discrete_map=LEVEL_COLORS,
                               title="Score per question", labels={"index": "question #"}), width="stretch")
        ideal = {"Easy": 0.3, "Medium": 0.5, "Hard": 0.2}
        share = (vc / len(df)).reindex(ideal).fillna(0)
        st.caption("Balance vs a typical 30/50/20 blueprint: " +
                   " · ".join(f"{k} {share[k]:.0%} ({'+' if share[k] >= v else ''}{share[k] - v:+.0%})" for k, v in ideal.items()))
        st.dataframe(df[["question", "english", "level", "score", "confidence", "fk_grade", "bloom_level"]],
                     width="stretch", hide_index=True)
        st.download_button("Download results CSV", df.to_csv(index=False), "paper_difficulty.csv")
        with st.expander("Raw extracted text"):
            st.text(st.session_state.get("paper_raw", "")[:5000])

# ============================================================================ 3. CO1
with tabs[2]:
    st.header("CO1 · Text extraction, preprocessing, cleaning & vectorisation")
    from src.nlp.cleaning import preprocess_steps
    t = st.text_input("Type any raw question to see each preprocessing step",
                      "Q3. Find the value of $\\frac{x^2}{4}$ when x = 6 <b>[2 marks]</b> — show   your working!")
    steps = preprocess_steps(t)
    s1, s2 = st.columns(2)
    s1.markdown("**1 · Raw input**"); s1.code(steps["raw"])
    s2.markdown("**2 · Cleaned** (HTML, marks, numbering, LaTeX, unicode)"); s2.code(steps["cleaned"])
    s1.markdown("**3 · Tokens**"); s1.write(steps["tokens"])
    s2.markdown("**4 · POS tags**"); s2.write(steps["pos_tags"])
    s1.markdown("**5 · Stop-words removed**"); s1.write(steps["without_stopwords"])
    s2.markdown("**6 · Stemmed (Porter)** vs **Lemmatized (WordNet)**")
    s2.write(pd.DataFrame({"stem": steps["stemmed"], "lemma": steps["lemmatized"]}))
    st.subheader("Vectorisation comparison (same classifier, test set)")
    vc = read_csv("vectorizer_comparison.csv")
    if vc is not None:
        st.dataframe(vc, hide_index=True, width="stretch")
        st.plotly_chart(px.bar(vc, x="vectorizer", y=["accuracy", "macro_f1", "qwk"], barmode="group",
                               title="Representation vs performance"), width="stretch")
    for fig in ("embedding_tsne.png", "readability_by_level.png"):
        if (FIGURES / fig).exists():
            st.image(str(FIGURES / fig))

# ============================================================================ 4. CO2
with tabs[3]:
    st.header("CO2 · Sentiment analysis & information extraction")
    sm = read_json("sentiment_metrics.json")
    if sm:
        st.subheader("Sentiment models (SST-2 validation, 872 sentences)")
        st.dataframe(pd.DataFrame({k: v for k, v in sm.items() if isinstance(v, dict) and "macro_f1" in v}).T.round(4),
                     width="stretch")
    sd = read_csv("sentiment_vs_difficulty.csv")
    if sd is not None:
        st.subheader("Does question tone relate to difficulty?")
        st.dataframe(sd.round(3), hide_index=True)
    from src.nlp.sentiment import analyze
    stx = st.text_input("Analyse sentiment of any text", "Unfortunately, which of the following is NOT a cause of the terrible famine?")
    st.json(analyze(stx))
    st.subheader("Information extraction")
    from src.nlp.info_extraction import extract
    iet = st.text_input("Extract information from a question",
                        "Calculate the force needed to accelerate a 1200 kg car from rest to 25 m/s in 8 seconds on the Mumbai-Pune expressway.")
    st.json(extract(iet))
    ie = read_json("information_extraction_metrics.json")
    if ie:
        st.subheader("Evaluation against hand-labelled gold data")
        c1, c2, c3 = st.columns(3)
        c1.metric("Bloom's - explicit verb questions", f"{ie['blooms_explicit_verbs']['accuracy']:.0%}")
        c2.metric("Bloom's - real exam questions", f"{ie['blooms_real_exam_questions']['accuracy']:.0%}",
                  help="Within-1-level: %.0f%%" % (100 * ie['blooms_real_exam_questions']['exact_or_adjacent']))
        c3.metric("Quantity extraction F1", f"{ie['quantities']['f1']:.2f}")
        st.caption("Rule-based Bloom's tagging is perfect when the command verb is explicit, but real exam "
                   "questions often hide the cognitive level, which is why the difficulty model is learned from data.")

# ============================================================================ 5. CO3 topics
with tabs[4]:
    st.header("CO3 · Topic modelling (LDA & NMF)")
    tp = read_json("topics.json")
    if tp:
        c1, c2, c3 = st.columns(3)
        c1.metric("Topics (best k)", tp["best_k"]); c2.metric("LDA coherence C_v", f"{tp['lda_coherence']:.3f}")
        c3.metric("NMF coherence C_v", f"{tp['nmf_coherence']:.3f}")
        ck = pd.DataFrame({"k": list(map(int, tp["lda_coherence_by_k"])), "C_v": list(tp["lda_coherence_by_k"].values())})
        st.plotly_chart(px.line(ck, x="k", y="C_v", markers=True, title="Choosing k by coherence"), width="stretch")
        lt = pd.DataFrame(tp["lda_topics"])
        lt["words"] = lt.top_words.map(lambda w: ", ".join(w[:8]))
        lt = lt.sort_values("mean_score")
        st.plotly_chart(px.bar(lt, x="mean_score", y="words", orientation="h", color="mean_score",
                               color_continuous_scale=["#2e9d5b", "#e0a100", "#d6453d"], range_color=[20, 80],
                               title="Average difficulty per LDA topic", height=450), width="stretch")
        st.dataframe(lt[["topic", "words", "n_questions", "mean_score", "pct_hard", "main_sources"]], hide_index=True,
                     width="stretch")
        with st.expander("NMF topics"):
            st.dataframe(pd.DataFrame(tp["nmf_topics"]).assign(top_words=lambda d: d.top_words.map(", ".join)), hide_index=True)
    from src.nlp.topics import topic_of
    tq = st.text_input("Find the topic of a question", "What is the probability of rolling two sixes with two dice?")
    st.json(topic_of(tq))

# ============================================================================ 6. CO3 multilingual
with tabs[5]:
    st.header("CO3 · Multilingual NLP (Hindi · Marathi · English)")
    st.caption("Language ID (Unicode script + n-gram model + function-word lexicon) → Indic normalisation & "
               "tokenisation → local neural machine translation (MarianMT / OPUS-MT) → difficulty model.")
    mq = st.text_input("Question in Hindi or Marathi", "न्यूटन के गति के तीसरे नियम को एक उदाहरण सहित समझाइए।")
    if st.button("Analyse multilingual question", type="primary"):
        from src.multilingual.pipeline import detect_language, indic_tokenize, remove_hindi_stopwords, to_english
        with st.spinner("Translating locally..."):
            info = to_english(mq)
            r = predictor().predict(mq, explain=False)
        c1, c2 = st.columns(2)
        c1.markdown(f"**Detected language:** {info['name']} ({info['script']})")
        c1.json(info["evidence"], expanded=False)
        toks = indic_tokenize(mq)
        c1.markdown("**Tokens:** " + " | ".join(toks))
        c1.markdown("**Without stop-words:** " + " | ".join(remove_hindi_stopwords(toks)))
        c2.markdown(f"**English translation:** {info['english']}")
        c2.markdown(badge(r["level"]), unsafe_allow_html=True)
        c2.plotly_chart(gauge(r["score"], r["level"]), width="stretch")
    mm = read_json("multilingual_metrics.json")
    if mm:
        st.subheader("Evaluation on hand-written parallel set")
        st.dataframe(pd.DataFrame({k: v for k, v in mm.items() if isinstance(v, dict)}).round(3), width="stretch")
        st.caption(f"Language identification accuracy: {mm['language_id_accuracy']:.0%}")
    mp = read_csv("multilingual_predictions.csv")
    if mp is not None:
        with st.expander("Per-question results"):
            st.dataframe(mp[["lang", "question", "english_mt", "english_reference", "expected_level", "level_mt", "score_mt",
                             "score_ref", "bleu"]].round(2), hide_index=True)

# ============================================================================ 7. CO4
with tabs[6]:
    st.header("CO4 · Generative AI studio (local LLM, no API)")
    if not llm_ok():
        st.warning("Local LLM is offline. Start it with `ollama serve` (model: qwen2.5:1.5b).")
    mode = st.radio("Task", ["Rewrite easier / harder (verified)", "Generate practice questions", "LLM-as-judge (prompt strategies)"],
                    horizontal=True)
    from src.llm import genai
    if mode.startswith("Rewrite"):
        rq = st.text_input("Question", "Explain the role of chlorophyll in photosynthesis.")
        direction = st.radio("Direction", ["harder", "easier"], horizontal=True)
        if st.button("Rewrite", type="primary", disabled=not llm_ok()):
            with st.spinner("Generating and verifying..."):
                o = genai.rewrite(rq, direction, predictor())
            c1, c2 = st.columns(2)
            c1.markdown(f"**Original**: {o['original']}"); c1.markdown(badge(o["before"]["level"]) + f" &nbsp; {o['before']['score']:.0f}/100", unsafe_allow_html=True)
            c2.markdown(f"**Rewritten**: {o['rewritten']}"); c2.markdown(badge(o["after"]["level"]) + f" &nbsp; {o['after']['score']:.0f}/100", unsafe_allow_html=True)
            (st.success if o["verified"] else st.error)(
                f"Trained model verification: score changed by {o['score_change']:+.1f} → "
                f"{'✅ moved in the requested direction' if o['verified'] else '❌ did not move as requested'}")
    elif mode.startswith("Generate"):
        c1, c2, c3 = st.columns([2, 1, 1])
        topic = c1.text_input("Topic", "Photosynthesis")
        lvl = c2.selectbox("Target level", ["Easy", "Medium", "Hard"], index=1)
        n = c3.number_input("How many", 1, 5, 3)
        if st.button("Generate", type="primary", disabled=not llm_ok()):
            with st.spinner("Generating and scoring..."):
                gen = genai.generate(topic, lvl, int(n), predictor())
            st.dataframe(pd.DataFrame(gen), hide_index=True, width="stretch")
    else:
        jq = st.text_input("Question", "Prove that the square root of 2 is irrational.")
        strat = st.selectbox("Prompt strategy", ["score_first", "few_shot", "zero_shot", "chain_of_thought"])
        if st.button("Ask the LLM", type="primary", disabled=not llm_ok()):
            with st.spinner("Thinking..."):
                o = genai.rate(jq, strat)
                n = predictor().predict(jq, explain=False)
            c1, c2 = st.columns(2)
            c1.markdown("**LLM judge**<br>" + badge(o["level"]) + f" &nbsp; {o['score']:.0f}/100", unsafe_allow_html=True)
            if o.get("reasoning"):
                c1.caption(o["reasoning"])
            c2.markdown("**Trained NLP model**<br>" + badge(n["level"]) + f" &nbsp; {n['score']:.0f}/100", unsafe_allow_html=True)
    st.subheader("Prompt engineering results")
    le = read_json("llm_prompt_eval.json")
    if le:
        tbl = pd.DataFrame({k: v for k, v in le.items() if k != "rewrite"}).T
        st.dataframe(tbl[["accuracy", "macro_f1", "qwk", "mae", "spearman"]].astype(float).round(3), width="stretch")
        st.caption(f"Controlled rewriting success (verified by the trained model): {le['rewrite']['success_rate']:.0%} "
                   f"of {le['rewrite']['n']} rewrites")
    with st.expander("📝 Prompt templates (prompts/ folder)"):
        for p in sorted(PROMPTS.glob("*.txt")):
            st.markdown(f"**{p.name}**"); st.code(p.read_text(), language="text")

# ============================================================================ 8. REPORT
with tabs[7]:
    st.header("Model comparison (held-out test set)")
    mt = read_json("metrics.json")
    if mt:
        rows = [{"model": k.split("::", 1)[1], **{m: v for m, v in d.items() if m != "confusion_matrix"}}
                for k, d in mt.items() if k.startswith("model::")]
        md = pd.DataFrame(rows).sort_values("macro_f1")
        st.dataframe(md.round(4), hide_index=True, width="stretch")
        st.plotly_chart(px.bar(md, x="macro_f1", y="model", orientation="h", text_auto=".3f", title="Macro-F1 by model"),
                        width="stretch")
        if "model::Ensemble (final)" in mt:
            cm = mt["model::Ensemble (final)"]["confusion_matrix"]
            st.plotly_chart(px.imshow(cm, x=["Easy", "Medium", "Hard"], y=["Easy", "Medium", "Hard"], text_auto=True,
                                      labels=dict(x="Predicted", y="True"), color_continuous_scale="Blues",
                                      title="Confusion matrix: final ensemble"), width="stretch")
    ps = read_csv("per_source_results.csv")
    if ps is not None:
        st.subheader("Per-source analysis")
        st.plotly_chart(px.scatter(ps, x="target_score", y="mean_pred_score", text="group", size="n",
                                   title="Target vs mean predicted score per source").add_shape(
            type="line", x0=0, y0=0, x1=100, y1=100, line=dict(dash="dot")), width="stretch")
        st.dataframe(ps, hide_index=True, width="stretch")
    for fig in sorted(FIGURES.glob("*.png")):
        with st.expander(fig.stem.replace("_", " ")):
            st.image(str(fig))

# ============================================================================ 9. DATASETS
with tabs[8]:
    st.header("Datasets (full copies stored locally in data/raw)")
    summ = json.loads((DATA_RAW / "datasets_summary.json").read_text())
    st.dataframe(pd.DataFrame([{"dataset": k, "rows": v["total_rows"], "difficulty signal": v["difficulty_signal"],
                                "license": v["license"], "source": v["source"], "description": v["description"]}
                               for k, v in summ.items()]), hide_index=True, width="stretch")
    st.metric("Total raw rows", f"{sum(v['total_rows'] for v in summ.values()):,}")
    c1, c2 = st.columns(2)
    ds = c1.selectbox("Dataset", list(summ))
    files = sorted(p.name for p in (DATA_RAW / ds).glob("*.parquet"))
    fn = c2.selectbox("File / split", files)
    df = pd.read_parquet(DATA_RAW / ds / fn)
    search = st.text_input("Search text in this file")
    if search:
        mask = df.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
        df = df[mask]
    st.caption(f"{len(df):,} rows")
    page = st.number_input("Page (100 rows per page)", 1, max(1, (len(df) - 1) // 100 + 1), 1)
    view = df.iloc[(page - 1) * 100: page * 100].copy()
    for col in view.columns:
        view[col] = view[col].astype(str).str.slice(0, 300)
    st.dataframe(view, width="stretch")
    st.subheader("Unified modelling set")
    proc = ROOT / "data" / "processed" / "modelling_set.csv"
    if proc.exists():
        m = pd.read_csv(proc, usecols=["source", "group", "level", "score", "split"])
        st.plotly_chart(px.histogram(m, x="group", color="level", color_discrete_map=LEVEL_COLORS,
                                     category_orders={"level": ["Easy", "Medium", "Hard"]},
                                     title="Questions per source group and level").update_xaxes(categoryorder="total descending"),
                        width="stretch")
    gold = sorted(DATA_GOLD.glob("*.csv"))
    with st.expander("Hand-labelled gold sets (data/gold)"):
        for g in gold:
            st.markdown(f"**{g.name}**"); st.dataframe(pd.read_csv(g), hide_index=True)
