/* DiffiQ frontend - vanilla JS single-page app talking to the local FastAPI backend. */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const COLORS = { Easy: "#16a34a", Medium: "#d97706", Hard: "#dc2626" };
const LEVELS = ["Easy", "Medium", "Hard"];
const PALETTE = ["#4f46e5", "#0ea5e9", "#f59e0b", "#10b981", "#ec4899", "#8b5cf6"];
const charts = {};
let REPORTS = null;
let DATASETS = null;
const inited = new Set();

Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
Chart.defaults.font.size = 12;
Chart.defaults.color = "#6b7489";
Chart.defaults.plugins.legend.labels.boxWidth = 12;

/* ---------------------------------------------------------------- utils */
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 3) => (v === null || v === undefined || v === "" ? "–" : typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(d)) : esc(v));
const pct = (v, d = 0) => (v === null || v === undefined ? "–" : (v * 100).toFixed(d) + "%");
const badge = (lvl, cls = "") => `<span class="badge ${lvl} ${cls}"><span class="lvl-dot" style="background:${COLORS[lvl]}"></span>${lvl.toUpperCase()}</span>`;
const FEATURE_NAMES = {
  tfidf_score: "Text model score (TF-IDF)", tfidf_p_hard: "Text model P(hard)", tfidf_p_easy: "Text model P(easy)",
  tfidf_p_medium: "Text model P(medium)", ends_with_question_mark: "Ends with '?'", n_chars: "Characters", n_words: "Words",
  n_math_symbols: "Maths symbols", n_sentences: "Sentences", avg_word_len: "Avg word length", rare_word_ratio: "Rare-word ratio",
  fk_grade: "FK grade level", bloom_level: "Bloom's level", n_entities: "Named entities", adj_ratio: "Adjective ratio",
  noun_ratio: "Noun ratio", verb_ratio: "Verb ratio", max_tree_depth: "Parse-tree depth", difficult_words: "Difficult words",
  n_numbers: "Numbers", n_digits: "Digits", mean_zipf: "Word frequency (Zipf)", type_token_ratio: "Type-token ratio",
  n_subordinate: "Subordinate clauses", n_negations: "Negations", vader_compound: "Sentiment (VADER)",
};
const featName = (n) => FEATURE_NAMES[n] || n.replace(/_/g, " ");
const levelOf = (s) => (s < 33.33 ? "Easy" : s < 66.67 ? "Medium" : "Hard");

function toast(msg, err = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast show" + (err ? " err" : "");
  clearTimeout(t._h);
  t._h = setTimeout(() => (t.className = "toast"), 4200);
}

async function api(path, body, isForm = false) {
  const opt = body === undefined ? {} : { method: "POST", body: isForm ? body : JSON.stringify(body), headers: isForm ? {} : { "Content-Type": "application/json" } };
  const r = await fetch(path, opt);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return r.json();
}

async function busy(btn, fn) {
  const html = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span class="loader"></span> Working…`;
  try { return await fn(); }
  catch (e) { toast(e.message, true); }
  finally { btn.disabled = false; btn.innerHTML = html; }
}

function chart(id, cfg) {
  if (charts[id]) charts[id].destroy();
  const el = document.getElementById(id);
  if (!el) return;
  cfg.options = Object.assign({ responsive: true, maintainAspectRatio: false }, cfg.options || {});
  charts[id] = new Chart(el, cfg);
}

function table(el, cols, rows, opts = {}) {
  const head = `<tr>${cols.map((c) => `<th${c.num ? ' style="text-align:right"' : ""}>${esc(c.label)}</th>`).join("")}</tr>`;
  const body = rows.map((r, i) => `<tr class="${opts.rowClass ? opts.rowClass(r, i) : ""}">${cols.map((c) => {
    const v = c.render ? c.render(r[c.key], r) : fmt(r[c.key], c.d ?? 3);
    return `<td class="${c.num ? "num" : ""}">${v}</td>`;
  }).join("")}</tr>`).join("");
  (typeof el === "string" ? $(el) : el).innerHTML = `<thead>${head}</thead><tbody>${body || `<tr><td colspan="${cols.length}" class="muted">No data</td></tr>`}</tbody>`;
}

function gaugeSVG(score, level) {
  const r = 95, cx = 120, cy = 118, start = Math.PI, ang = start - (Math.PI * Math.max(0, Math.min(100, score))) / 100;
  const p = (a) => [cx + r * Math.cos(a), cy - r * Math.sin(a)];
  const arc = (a0, a1, color, w = 18) => { const [x0, y0] = p(a0), [x1, y1] = p(a1); return `<path d="M${x0} ${y0} A${r} ${r} 0 0 1 ${x1} ${y1}" stroke="${color}" stroke-width="${w}" fill="none" stroke-linecap="butt"/>`; };
  const seg = (f0, f1) => [Math.PI - Math.PI * f0, Math.PI - Math.PI * f1];
  const [nx, ny] = p(ang);
  return `<svg viewBox="0 0 240 150">
    ${arc(...seg(0, 0.333), "#dcf3e4")}${arc(...seg(0.333, 0.667), "#fbecd0")}${arc(...seg(0.667, 1), "#f9d9d9")}
    ${arc(Math.PI, ang + 0.0001, COLORS[level], 18)}
    <circle cx="${nx}" cy="${ny}" r="9" fill="#fff" stroke="${COLORS[level]}" stroke-width="4"/>
    <text x="22" y="146" font-size="10" fill="#9aa1b6">0</text><text x="208" y="146" font-size="10" fill="#9aa1b6">100</text></svg>`;
}

function probBars(probs) {
  return LEVELS.map((l) => `<div class="prob-row"><span>${l}</span><div class="bar"><span style="width:${(probs[l] * 100).toFixed(1)}%;background:${COLORS[l]}"></span></div><span style="text-align:right">${pct(probs[l])}</span></div>`).join("");
}

function miniMarkdown(t) {
  return esc(t)
    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
    .replace(/(^|\s)\*(\S.+?)\*/g, "$1<i>$2</i>")
    .replace(/^#{1,4}\s*(.+)$/gm, "<b>$1</b>")
    .replace(/^\s*[-•]\s+/gm, "• ");
}

/* ---------------------------------------------------------------- router */
const PAGES = { analyze: initAnalyze, pipeline: initPipeline, paper: initPaper, lab: initLab, genai: initGenai, report: initReport, datasets: initDatasets };

function route() {
  const [page, sub] = (location.hash.slice(1) || "analyze").split("/");
  const name = PAGES[page] ? page : "analyze";
  $$(".page").forEach((p) => p.classList.toggle("active", p.id === "page-" + name));
  $$(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === name && (!a.dataset.sub || a.dataset.sub === (sub || "co1"))));
  if (!inited.has(name)) { inited.add(name); PAGES[name](); }
  if (name === "lab") showSub("lab", sub || "co1");
  window.scrollTo(0, 0);
}

function showSub(group, sub) {
  const tabs = group === "lab" ? "#lab-tabs" : "#gen-tabs";
  const prefix = group === "lab" ? "sub-" : "gen-";
  $$(tabs + " button").forEach((b) => b.classList.toggle("active", b.dataset.sub === sub));
  $$(`[id^="${prefix}"]`).forEach((p) => p.classList.toggle("active", p.id === prefix + sub));
  if (group === "lab") initLabSub(sub);
}

$$(".nav a").forEach((a) => (a.href = "#" + a.dataset.page + (a.dataset.sub ? "/" + a.dataset.sub : "")));
$$("#lab-tabs button").forEach((b) => b.addEventListener("click", () => (location.hash = "lab/" + b.dataset.sub)));
$$("#gen-tabs button").forEach((b) => b.addEventListener("click", () => { showSub("gen", b.dataset.sub); if (b.dataset.sub === "prompts") initPrompts(); }));
window.addEventListener("hashchange", route);

async function getReports() { return REPORTS || (REPORTS = await api("/api/reports")); }

/* ---------------------------------------------------------------- status */
async function status() {
  try {
    const h = await api("/api/health");
    $("#dot-llm").className = "status-dot " + (h.llm ? "on" : "off");
    $("#st-llm").textContent = h.llm ? `Local LLM: ${h.llm_model}` : "Local LLM: offline";
    $("#dot-models").className = "status-dot " + (h.models_loaded ? "on" : "wait");
    $("#st-models").textContent = h.models_loaded ? "Models: ready" : "Models: loading…";
    return h;
  } catch (_) {
    $("#dot-models").className = "status-dot off";
    $("#st-models").textContent = "Server offline";
  }
}
status().then((h) => {
  if (h && !h.models_loaded) api("/api/warmup", {}).then(status).catch(() => {});
});
setInterval(status, 15000);

/* ================================================================ ANALYZE */
const EXAMPLES = [
  "What is the capital of France?",
  "Explain why the sky appears blue during the day.",
  "A car accelerates uniformly from 10 m/s to 30 m/s in 5 seconds. Calculate its acceleration and the distance travelled.",
  "Evaluate the impact of the Treaty of Versailles on the rise of fascism in Europe.",
  "Find all positive integers n such that n^2 + 1 is divisible by n + 1.",
  "सिद्ध कीजिए कि 2 का वर्गमूल एक अपरिमेय संख्या है।",
];
let LAST = null;

function initAnalyze() {
  $("#examples").innerHTML = EXAMPLES.map((e) => `<span class="ex" title="${esc(e)}">${esc(e.length > 46 ? e.slice(0, 44) + "…" : e)}</span>`).join("");
  $$("#examples .ex").forEach((el, i) => el.addEventListener("click", () => { $("#q-input").value = EXAMPLES[i]; analyze(); }));
  $("#q-go").addEventListener("click", analyze);
  $("#q-clear").addEventListener("click", () => { $("#q-input").value = ""; $("#result").classList.add("hidden"); $("#q-input").focus(); });
  $("#q-input").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) analyze(); });
  $("#r-explain").addEventListener("click", explain);
  const q = new URLSearchParams(location.search).get("q");   // e.g. /?q=What+is+gravity
  if (q) { $("#q-input").value = q; analyze(); }
}

async function analyze() {
  const q = $("#q-input").value.trim();
  if (!q) return toast("Type a question first");
  await busy($("#q-go"), async () => {
    const r = await api("/api/predict", { question: q });
    LAST = r;
    renderResult(r);
    status();
  });
}

function renderResult(r) {
  $("#result").classList.remove("hidden");
  $("#r-ai").classList.add("hidden");
  const tr = $("#r-translated");
  if (r.language.translated) { tr.classList.remove("hidden"); tr.innerHTML = `🌐 Detected <b>${esc(r.language.name)}</b> → translated locally: <i>${esc(r.english)}</i>`; }
  else tr.classList.add("hidden");

  $("#r-gauge").innerHTML = gaugeSVG(r.score, r.level);
  $("#r-score").innerHTML = `${r.score.toFixed(1)}<small>/100</small>`;
  $("#r-badge").innerHTML = badge(r.level, "lg");
  $("#r-conf").textContent = `Confidence ${pct(r.confidence)}`;
  $("#r-probs").innerHTML = probBars(r.probabilities);

  const names = { gbm: "LightGBM", bert: "DistilBERT", bilstm: "BiLSTM (explainer)" };
  $("#r-members").innerHTML = Object.keys(r.member_scores).map((k) =>
    `<div class="row" style="justify-content:space-between;margin:7px 0"><span style="font-weight:600">${names[k] || k}</span><span class="row" style="gap:8px"><span class="muted small">${r.member_scores[k].toFixed(0)}/100</span>${badge(r.member_levels[k])}</span></div>`).join("");

  const f = r.features, i = r.info || {}, s = r.sentiment || {};
  const prof = [
    ["Bloom's level", i.bloom_level],
    ["Question type", i.question_type],
    ["Readability (FK grade)", f.fk_grade.toFixed(1)],
    ["Words / sentences", `${f.n_words} / ${f.n_sentences}`],
    ["Rare words", pct(f.rare_word_ratio)],
    ["Parse-tree depth", f.max_tree_depth],
    ["Subordinate clauses", f.n_subordinate],
    ["Maths symbols", f.n_math_symbols],
    ["Tone (VADER)", `${s.vader_label || "–"} (${(s.vader_compound ?? 0).toFixed(2)})`],
    ["Topic", r.topic && r.topic.topic !== null ? r.topic.top_words.slice(0, 3).join(", ") : "–"],
  ];
  $("#r-profile").innerHTML = prof.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("");

  const shap = r.shap || [];
  const mx = Math.max(...shap.map((x) => Math.abs(x[1])), 0.01);
  $("#r-shap").innerHTML = shap.map(([n, v, val]) => {
    const w = (Math.abs(v) / mx) * 50, left = v >= 0 ? 50 : 50 - w;
    return `<div class="shap-row"><span title="${esc(n)} = ${val}">${esc(featName(n))}</span>
      <div class="shap-track"><div class="mid"></div><div class="fill" style="left:${left}%;width:${w}%;background:${v >= 0 ? "#ef4444" : "#22c55e"}"></div></div>
      <span style="text-align:right;font-weight:650;color:${v >= 0 ? "#dc2626" : "#16a34a"}">${v >= 0 ? "+" : ""}${v.toFixed(1)}</span></div>`;
  }).join("");

  const att = r.attention || [];
  const am = Math.max(...att.map((a) => a[1]), 0.001);
  $("#r-attn").innerHTML = att.length ? att.map(([t, w]) => `<span style="background:rgba(79,70,229,${(0.08 + 0.85 * w / am).toFixed(2)});color:${w / am > 0.55 ? "#fff" : "inherit"}">${esc(t)}</span>`).join("") : '<span class="muted">–</span>';

  const pills = (arr, cls) => (arr && arr.length ? arr.map((x) => `<span class="pill ${cls}">${esc(x)}</span>`).join("") : '<span class="muted small">none</span>');
  $("#r-ie").innerHTML = `
    <div class="small muted" style="margin:4px 0">Entities</div>${pills((i.entities || []).map((e) => `${e.text} · ${e.label}`), "ent")}
    <div class="small muted" style="margin:10px 0 4px">Keyphrases (YAKE)</div>${pills(i.keyphrases_yake, "key")}
    <div class="small muted" style="margin:10px 0 4px">Numbers &amp; quantities</div>${pills([...(i.quantities_with_units || []), ...(i.numbers || [])].slice(0, 8), "num")}
    <div class="small muted" style="margin:10px 0 4px">Bloom indicators</div>${pills(i.bloom_indicators, "")}`;
  $("#result").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function explain() {
  if (!LAST) return;
  await busy($("#r-explain"), async () => {
    const r = await api("/api/llm/explain", { question: LAST.question });
    const box = $("#r-ai");
    box.classList.remove("hidden");
    box.innerHTML = miniMarkdown(r.explanation);
  });
}

/* ================================================================ PAPER */
let PAPER = null;
function initPaper() {
  const drop = $("#drop"), input = $("#paper-file");
  input.addEventListener("change", () => input.files[0] && uploadPaper(input.files[0]));
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
  drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && uploadPaper(e.dataTransfer.files[0]));
  $("#paper-sample").addEventListener("click", () => busy($("#paper-sample"), async () => renderPaper(await api("/api/paper", new FormData(), true))));
  $("#paper-csv").addEventListener("click", downloadCSV);
}

async function uploadPaper(file) {
  $("#drop-text").innerHTML = `<span class="loader dark"></span> Analysing <b>${esc(file.name)}</b>…`;
  const fd = new FormData();
  fd.append("file", file);
  try { renderPaper(await api("/api/paper", fd, true)); }
  catch (e) { toast(e.message, true); }
  finally { $("#drop-text").textContent = "Drop a PDF, DOCX, TXT or CSV here, or click to browse"; }
}

function renderPaper(d) {
  PAPER = d;
  const qs = d.questions;
  $("#paper-out").classList.remove("hidden");
  const cnt = Object.fromEntries(LEVELS.map((l) => [l, qs.filter((q) => q.level === l).length]));
  const mean = qs.reduce((a, q) => a + q.score, 0) / qs.length;
  $("#paper-stats").innerHTML = [
    ["Questions extracted", qs.length, d.filename],
    ["Average difficulty", mean.toFixed(1) + " /100", ""],
    ["Overall paper", badge(levelOf(mean)), ""],
    ["Easy · Medium · Hard", `${cnt.Easy} · ${cnt.Medium} · ${cnt.Hard}`, ""],
  ].map(([l, v, s]) => `<div class="card stat"><div class="l">${l}</div><div class="v">${v}</div><div class="muted small">${esc(s)}</div></div>`).join("");
  const ideal = { Easy: 0.3, Medium: 0.5, Hard: 0.2 };
  $("#paper-balance").innerHTML = "vs a typical 30/50/20 blueprint: " + LEVELS.map((l) => {
    const sh = cnt[l] / qs.length, diff = sh - ideal[l];
    return `${l} ${pct(sh)} <span style="color:${Math.abs(diff) < 0.1 ? "#16a34a" : "#d97706"}">(${diff >= 0 ? "+" : ""}${(diff * 100).toFixed(0)})</span>`;
  }).join(" · ");
  chart("ch-paper-pie", { type: "doughnut", data: { labels: LEVELS, datasets: [{ data: LEVELS.map((l) => cnt[l]), backgroundColor: LEVELS.map((l) => COLORS[l]), borderWidth: 2 }] }, options: { cutout: "62%", plugins: { legend: { position: "right" } } } });
  chart("ch-paper-bar", { type: "bar", data: { labels: qs.map((_, i) => "Q" + (i + 1)), datasets: [{ data: qs.map((q) => q.score), backgroundColor: qs.map((q) => COLORS[q.level]), borderRadius: 5 }] }, options: { plugins: { legend: { display: false }, tooltip: { callbacks: { title: (it) => qs[it[0].dataIndex].question.slice(0, 80) } } }, scales: { y: { min: 0, max: 100 } } } });
  table("#paper-table", [
    { key: "i", label: "#", render: (_, r) => qs.indexOf(r) + 1 },
    { key: "question", label: "Question", render: (v, r) => esc(v) + (r.english !== v ? `<div class="muted small">→ ${esc(r.english)}</div>` : "") },
    { key: "level", label: "Level", render: (v) => badge(v) },
    { key: "score", label: "Score", num: true, d: 1 },
    { key: "confidence", label: "Conf.", num: true, render: (v) => pct(v) },
    { key: "fk_grade", label: "FK grade", num: true, d: 1 },
    { key: "bloom_level", label: "Bloom", num: true, render: (v) => ["", "Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"][v] || v },
  ], qs);
  $("#paper-out").scrollIntoView({ behavior: "smooth" });
}

function downloadCSV() {
  if (!PAPER) return;
  const cols = ["question", "english", "level", "score", "confidence", "p_easy", "p_medium", "p_hard", "fk_grade", "bloom_level"];
  const csv = [cols.join(","), ...PAPER.questions.map((q) => cols.map((c) => `"${String(q[c] ?? "").replace(/"/g, '""')}"`).join(","))].join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob(["﻿" + csv], { type: "text/csv" }));
  a.download = "paper_difficulty.csv";
  a.click();
}

/* ================================================================ NLP LAB */
function initLab() {
  $("#pp-go").addEventListener("click", runPreprocess);
  $("#se-go").addEventListener("click", runSentiment);
  $("#ie-go").addEventListener("click", runIE);
  $("#tp-go").addEventListener("click", runTopic);
  $("#ml-go").addEventListener("click", runMulti);
  const mex = ["न्यूटन के गति के तीसरे नियम को एक उदाहरण सहित समझाइए।", "भारत की राजधानी क्या है?", "मौद्रिक नीति का मुद्रास्फीति और बेरोजगारी पर प्रभाव का मूल्यांकन कीजिए।", "पृथ्वीवर ऋतू का बदलतात?", "सिद्ध करा की 2 चे वर्गमूळ ही अपरिमेय संख्या आहे."];
  $("#ml-examples").innerHTML = mex.map((e) => `<span class="ex">${esc(e)}</span>`).join("");
  $$("#ml-examples .ex").forEach((el, i) => el.addEventListener("click", () => { $("#ml-input").value = mex[i]; runMulti(); }));
}

const labInited = new Set();
async function initLabSub(sub) {
  if (labInited.has(sub)) return;
  labInited.add(sub);
  const R = await getReports();
  if (sub === "co1") { runPreprocess(); renderVec(R); }
  if (sub === "co2") { renderCO2(R); runSentiment(); runIE(); }
  if (sub === "topics") { renderTopics(R); runTopic(); }
  if (sub === "multi") { renderMulti(R); runMulti(); }
}

async function runPreprocess() {
  const s = await api("/api/nlp/preprocess", { question: $("#pp-input").value }).catch((e) => toast(e.message, true));
  if (!s) return;
  const join = (a) => a.map((t) => (Array.isArray(t) ? `${t[0]}/<b>${t[1]}</b>` : esc(t))).join("  ·  ");
  const steps = [["Raw input", esc(s.raw)], ["Cleaned (HTML, marks, numbering, LaTeX, unicode)", esc(s.cleaned)], ["Tokens", join(s.tokens)], ["POS tags", join(s.pos_tags)],
    ["Stop-words removed", join(s.without_stopwords)], ["Porter stems", join(s.stemmed)], ["WordNet lemmas", join(s.lemmatized)]];
  $("#pp-steps").innerHTML = steps.map(([t, b], i) => `<div class="step"><div class="n">${i + 1}</div><div><div class="t">${t}</div><div class="box">${b || "–"}</div></div></div>`).join("");
}

function renderVec(R) {
  const v = R.vectorizer_comparison || [];
  chart("ch-vec", { type: "bar", data: { labels: v.map((x) => x.vectorizer), datasets: ["accuracy", "macro_f1", "qwk"].map((m, i) => ({ label: m, data: v.map((x) => x[m]), backgroundColor: PALETTE[i], borderRadius: 4 })) }, options: { scales: { y: { min: 0.4, max: 0.85 } } } });
  const best = Math.max(...v.map((x) => x.macro_f1));
  table("#vec-table", [{ key: "vectorizer", label: "Representation" }, { key: "accuracy", label: "Acc", num: true }, { key: "macro_f1", label: "F1", num: true }, { key: "qwk", label: "QWK", num: true }, { key: "mae", label: "MAE", num: true, d: 2 }, { key: "dims", label: "Dims", num: true }], v, { rowClass: (r) => (r.macro_f1 === best ? "best" : "") });
}

function renderCO2(R) {
  const sm = R.sentiment_metrics || {};
  const names = Object.keys(sm).filter((k) => sm[k] && sm[k].macro_f1 !== undefined);
  chart("ch-sent", { type: "bar", data: { labels: names, datasets: [{ label: "accuracy", data: names.map((n) => sm[n].accuracy), backgroundColor: PALETTE[0], borderRadius: 4 }, { label: "macro-F1", data: names.map((n) => sm[n].macro_f1), backgroundColor: PALETTE[1], borderRadius: 4 }] }, options: { indexAxis: "y", scales: { x: { min: 0.5, max: 0.9 } } } });
  const ie = R.information_extraction_metrics;
  if (ie) {
    $("#ie-metrics").innerHTML = [
      ["Bloom's tagger: explicit-verb questions", pct(ie.blooms_explicit_verbs.accuracy), "60 hand-labelled questions"],
      ["Bloom's tagger: real exam questions", pct(ie.blooms_real_exam_questions.accuracy), `${pct(ie.blooms_real_exam_questions.exact_or_adjacent)} within one level. Rules miss implicit verbs.`],
      ["Quantity / unit extraction F1", ie.quantities.f1.toFixed(2), `P ${ie.quantities.precision.toFixed(2)} · R ${ie.quantities.recall.toFixed(2)}`],
    ].map(([l, v, s]) => `<div class="card stat"><div class="l">${l}</div><div class="v">${v}</div><div class="muted small">${s}</div></div>`).join("");
  }
  const sd = (R.sentiment_vs_difficulty || []).sort((a, b) => LEVELS.indexOf(a.level) - LEVELS.indexOf(b.level));
  table("#sent-diff", [{ key: "level", label: "Level", render: (v) => badge(v) }, { key: "n", label: "Questions", num: true }, { key: "vader_mean", label: "Mean VADER", num: true }, { key: "ml_pos_mean", label: "P(positive) LR", num: true }, { key: "pct_negative", label: "% negative tone", num: true, render: (v) => pct(v, 1) }, { key: "pct_neutral", label: "% neutral", num: true, render: (v) => pct(v, 1) }], sd);
}

async function runSentiment() {
  const s = await api("/api/nlp/sentiment", { question: $("#se-input").value }).catch((e) => toast(e.message, true));
  if (!s) return;
  const lc = { positive: "#16a34a", negative: "#dc2626", neutral: "#6b7489" }[s.vader_label];
  $("#se-out").innerHTML = `<div class="grid g2">
    <div><div class="muted small">VADER (lexicon)</div><div style="font-size:22px;font-weight:750;color:${lc}">${s.vader_label} ${s.vader_compound.toFixed(2)}</div>
      <div class="small muted">pos ${s.vader_pos} · neu ${s.vader_neu} · neg ${s.vader_neg}</div></div>
    <div><div class="muted small">TF-IDF + Logistic Regression</div><div style="font-size:22px;font-weight:750">${pct(s.ml_positive_prob)} positive</div>
      <div class="bar" style="margin-top:8px"><span style="width:${s.ml_positive_prob * 100}%;background:linear-gradient(90deg,#dc2626,#16a34a)"></span></div></div></div>`;
}

async function runIE() {
  const i = await api("/api/nlp/extract", { question: $("#ie-input").value }).catch((e) => toast(e.message, true));
  if (!i) return;
  const pills = (arr, cls) => (arr && arr.length ? arr.map((x) => `<span class="pill ${cls}">${esc(x)}</span>`).join("") : '<span class="muted small">none</span>');
  const blk = (t, h) => `<div><div class="muted small" style="margin-bottom:5px;font-weight:600">${t}</div>${h}</div>`;
  $("#ie-out").innerHTML = `<div class="grid g3">
    ${blk("Question type · Bloom's level", `<span class="pill">${esc(i.question_type)}</span><span class="pill key">${esc(i.bloom_level)}</span>`)}
    ${blk("Named entities (spaCy)", pills(i.entities.map((e) => `${e.text} · ${e.label}`), "ent"))}
    ${blk("Quantities & numbers", pills([...i.quantities_with_units, ...i.numbers], "num"))}
    ${blk("YAKE keyphrases", pills(i.keyphrases_yake, "key"))}
    ${blk("RAKE keyphrases", pills(i.keyphrases_rake, "key"))}
    ${blk("TextRank keyphrases", pills(i.keyphrases_textrank, "key"))}
    ${blk("Subject–verb–object triples", pills(i.svo_triples.map((t) => t.join(" → ")), ""))}
    ${blk("Maths expressions", pills(i.math_expressions, "num"))}
    ${blk("Bloom indicators", pills(i.bloom_indicators, ""))}</div>`;
}

function renderTopics(R) {
  const t = R.topics;
  if (!t) return;
  $("#topic-stats").innerHTML = [["Topics (best k)", t.best_k], ["LDA coherence C_v", t.lda_coherence.toFixed(3)], ["NMF coherence C_v", t.nmf_coherence.toFixed(3)]]
    .map(([l, v]) => `<div class="card stat"><div class="l">${l}</div><div class="v">${v}</div></div>`).join("");
  const ks = Object.keys(t.lda_coherence_by_k);
  chart("ch-coh", { type: "line", data: { labels: ks, datasets: [{ label: "C_v", data: ks.map((k) => t.lda_coherence_by_k[k]), borderColor: PALETTE[0], backgroundColor: "rgba(79,70,229,.12)", fill: true, tension: 0.3, pointRadius: 5 }] }, options: { plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: "number of topics k" } } } } });
  const lt = [...t.lda_topics].sort((a, b) => a.mean_score - b.mean_score);
  chart("ch-topics", { type: "bar", data: { labels: lt.map((x) => `T${x.topic}: ${x.top_words.slice(0, 6).join(", ")}`), datasets: [{ data: lt.map((x) => x.mean_score), backgroundColor: lt.map((x) => COLORS[levelOf(x.mean_score)]), borderRadius: 4 }] },
    options: { indexAxis: "y", plugins: { legend: { display: false }, tooltip: { callbacks: { afterLabel: (c) => `${lt[c.dataIndex].n_questions} questions · ${pct(lt[c.dataIndex].pct_hard)} hard` } } }, scales: { x: { min: 0, max: 100, title: { display: true, text: "mean difficulty score" } } } } });
  table("#nmf-table", [{ key: "topic", label: "#", num: true }, { key: "top_words", label: "Top words", render: (v) => esc(v.join(", ")) }, { key: "n_questions", label: "Questions", num: true }, { key: "mean_score", label: "Mean score", num: true, d: 1 }], [...t.nmf_topics].sort((a, b) => a.mean_score - b.mean_score));
}

async function runTopic() {
  const t = await api("/api/nlp/topic", { question: $("#tp-input").value }).catch((e) => toast(e.message, true));
  if (!t) return;
  $("#tp-out").innerHTML = t.topic === null ? '<span class="muted">No dominant topic</span>' : `
    <div class="row"><span style="font-size:22px;font-weight:780">Topic ${t.topic}</span><span class="muted">p = ${t.probability}</span>${t.topic_mean_score ? badge(levelOf(t.topic_mean_score)) : ""}</div>
    <div class="mt">${t.top_words.map((w) => `<span class="pill key">${esc(w)}</span>`).join("")}</div>
    <div class="muted small mt">Average difficulty of this topic: <b>${t.topic_mean_score}</b> · distribution: ${t.distribution.map(([a, b]) => `T${a} ${pct(b)}`).join(", ")}</div>`;
}

function renderMulti(R) {
  const m = R.multilingual_metrics;
  if (!m) return;
  const rows = ["hi", "mr"].map((l) => ({ lang: l === "hi" ? "Hindi" : "Marathi", ...m[l] }));
  table("#ml-metrics", [{ key: "lang", label: "Language" }, { key: "n", label: "N", num: true }, { key: "language_id_accuracy", label: "Lang-ID", num: true, render: (v) => pct(v) }, { key: "corpus_bleu", label: "BLEU", num: true }, { key: "score_mae_vs_english", label: "Score Δ vs EN", num: true, d: 1 }, { key: "level_agreement_with_english", label: "Level = EN", num: true, render: (v) => pct(v) }, { key: "accuracy_vs_expected_level", label: "Accuracy", num: true, render: (v) => pct(v) }], rows);
  const sel = m.mt_model_selection || {};
  const labels = Object.keys(sel);
  const isSel = (k) => { const [lang, model] = k.split(" | "); return (m.selected_models[lang] || "").endsWith(model); };
  chart("ch-mt", { type: "bar", data: { labels, datasets: [{ data: labels.map((k) => sel[k]), backgroundColor: labels.map((k) => (isSel(k) ? PALETTE[0] : "#c7cbe0")), borderRadius: 4 }] }, options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { min: 0, max: 0.6, title: { display: true, text: "BLEU (selected models in indigo)" } } } } });
}

async function runMulti() {
  const out = $("#ml-out");
  out.innerHTML = '<span class="loader dark"></span> <span class="muted">Translating locally…</span>';
  const r = await api("/api/nlp/multilingual", { question: $("#ml-input").value }).catch((e) => { toast(e.message, true); out.innerHTML = ""; });
  if (!r) return;
  out.innerHTML = `<div class="grid g3">
    <div><div class="muted small">Detected language</div><div style="font-size:22px;font-weight:780">${esc(r.name)}</div><div class="muted small">${esc(r.script)} script</div>
      ${r.evidence && r.evidence.hindi_markers !== undefined ? `<div class="small mt">Hindi markers: <b>${r.evidence.hindi_markers}</b> · Marathi markers: <b>${r.evidence.marathi_markers}</b></div>` : ""}</div>
    <div><div class="muted small">Indic tokens (stop-words removed)</div><div class="mt">${(r.tokens_no_stopwords || r.tokens || []).map((t) => `<span class="pill">${esc(t)}</span>`).join("")}</div></div>
    <div><div class="muted small">Prediction</div><div class="row mt">${badge(r.level, "lg")}<span style="font-size:24px;font-weight:780">${r.score.toFixed(1)}</span></div></div></div>
    <div class="translated mt" style="margin-bottom:0">English (local MarianMT): <b>${esc(r.english)}</b></div>`;
}

/* ================================================================ GENAI */
function initGenai() {
  $("#rw-go").addEventListener("click", () => busy($("#rw-go"), async () => {
    const o = await api("/api/llm/rewrite", { question: $("#rw-input").value, direction: $("#rw-dir").value });
    const card = (t, q, s) => `<div class="card" style="box-shadow:none"><div class="muted small" style="font-weight:600">${t}</div><div style="margin:8px 0 12px;font-size:15px">${esc(q)}</div><div class="row">${badge(s.level)}<b>${s.score.toFixed(1)}</b>/100</div></div>`;
    $("#rw-out").innerHTML = `<div class="compare">${card("Original", o.original, o.before)}<div class="arrow">→</div>${card("Rewritten by local LLM", o.rewritten, o.after)}</div>
      <div class="verdict ${o.verified ? "ok" : "bad"}">${o.verified ? "✅" : "❌"} Trained-model verification: score changed by ${o.score_change >= 0 ? "+" : ""}${o.score_change.toFixed(1)} points, ${o.verified ? "moved in the requested direction" : "did not move as requested"}.</div>`;
  }));
  $("#gn-go").addEventListener("click", () => busy($("#gn-go"), async () => {
    const r = await api("/api/llm/generate", { topic: $("#gn-topic").value, level: $("#gn-level").value, n: +$("#gn-n").value });
    const hit = r.items.filter((x) => x.on_target).length;
    $("#gn-out").innerHTML = `<div class="muted small" style="margin-bottom:8px">${hit}/${r.items.length} generated questions verified at the target level by the trained model</div><div class="table-wrap"><table class="tbl" id="gn-table"></table></div>`;
    table("#gn-table", [{ key: "question", label: "Generated question" }, { key: "target_level", label: "Target", render: (v) => badge(v) }, { key: "predicted_level", label: "Verified as", render: (v) => badge(v) }, { key: "predicted_score", label: "Score", num: true, d: 1 }, { key: "on_target", label: "", render: (v) => (v ? "✅" : "⚠️") }], r.items);
  }));
  $("#jd-go").addEventListener("click", () => busy($("#jd-go"), async () => {
    const r = await api("/api/llm/judge", { question: $("#jd-input").value, strategy: $("#jd-strat").value });
    $("#jd-out").innerHTML = `<div class="grid g2">
      <div class="card" style="box-shadow:none"><div class="muted small" style="font-weight:600">Local LLM (${esc(r.llm.strategy)})</div><div class="row mt">${badge(r.llm.level, "lg")}<span style="font-size:24px;font-weight:780">${r.llm.score.toFixed(0)}</span></div>${r.llm.reasoning ? `<div class="muted small mt">${esc(r.llm.reasoning)}</div>` : ""}</div>
      <div class="card" style="box-shadow:none"><div class="muted small" style="font-weight:600">Trained NLP ensemble</div><div class="row mt">${badge(r.model.level, "lg")}<span style="font-size:24px;font-weight:780">${r.model.score.toFixed(1)}</span></div></div></div>`;
  }));
}

let promptsInited = false;
async function initPrompts() {
  if (promptsInited) return;
  promptsInited = true;
  const R = await getReports();
  const le = R.llm_prompt_eval || {};
  const names = Object.keys(le).filter((k) => k !== "rewrite");
  const label = (n) => n.replace("LLM ", "LLM · ").replace(/_/g, " ");
  chart("ch-llm", { type: "bar", data: { labels: names.map(label), datasets: ["accuracy", "macro_f1", "qwk"].map((m, i) => ({ label: m, data: names.map((n) => le[n][m]), backgroundColor: PALETTE[i], borderRadius: 4 })) }, options: { scales: { y: { min: 0, max: 0.85 } } } });
  if (le.rewrite) {
    const bd = le.rewrite.by_direction;
    $("#rw-stats").innerHTML = [["Rewrite success (verified)", pct(le.rewrite.success_rate), `${le.rewrite.n} rewrites`], ["Harder / easier success", `${pct(bd.harder.success)} / ${pct(bd.easier.success)}`, `mean Δ +${bd.harder.mean_change.toFixed(1)} / ${bd.easier.mean_change.toFixed(1)}`]]
      .map(([l, v, s]) => `<div class="card stat" style="box-shadow:none"><div class="l">${l}</div><div class="v">${v}</div><div class="muted small">${s}</div></div>`).join("");
  }
  const P = await api("/api/prompts");
  $("#prompt-list").innerHTML = Object.entries(P).map(([n, t]) => `<div><div class="small" style="font-weight:650;margin-bottom:6px">${esc(n)}</div><div class="prompt-box">${esc(t)}</div></div>`).join("");
}

/* ================================================================ REPORT */
async function initReport() {
  const R = await getReports();
  const m = R.metrics || {};
  // headline = original 4,183-question test set ("orig::"); falls back to "model::" if absent
  const pref = Object.keys(m).some((k) => k.startsWith("orig::")) ? "orig::" : "model::";
  const models = Object.entries(m).filter(([k]) => k.startsWith(pref)).map(([k, v]) => ({ model: k.split("::")[1], ...v })).sort((a, b) => a.macro_f1 - b.macro_f1);
  const fin = models.find((x) => x.model.startsWith("Ensemble")) || models[models.length - 1];
  $("#rep-stats").innerHTML = [["Accuracy", pct(fin.accuracy, 1)], ["Macro-F1", fin.macro_f1.toFixed(3)], ["Quadratic weighted kappa", fin.qwk.toFixed(3)], ["Score MAE (points)", "±" + fin.mae.toFixed(2)]]
    .map(([l, v]) => `<div class="card stat"><div class="l">Final ensemble · ${l}</div><div class="v">${v}</div></div>`).join("");
  const W = R.ensemble.weights;
  $("#rep-table").parentElement.previousElementSibling.textContent = `Final ensemble weights (tuned on validation): ${Object.entries(W).map(([k, v]) => `${k} ${v}`).join(" · ")}`;
  table("#rep-table", [{ key: "model", label: "Model" }, { key: "accuracy", label: "Accuracy", num: true }, { key: "macro_f1", label: "Macro-F1", num: true }, { key: "qwk", label: "QWK", num: true }, { key: "mae", label: "MAE", num: true, d: 2 }, { key: "rmse", label: "RMSE", num: true, d: 2 }, { key: "spearman", label: "Spearman", num: true }, { key: "band_accuracy", label: "Band acc.", num: true }],
    [...models].reverse(), { rowClass: (r) => (r === fin ? "best" : "") });
  chart("ch-models", { type: "bar", data: { labels: models.map((x) => x.model), datasets: [{ data: models.map((x) => x.macro_f1), backgroundColor: models.map((x) => (x === fin ? "#4f46e5" : "#a5b4fc")), borderRadius: 4 }] }, options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { min: 0.6, max: 0.85 } } } });
  const ps = R.per_source_results || [];
  chart("ch-calib", { type: "scatter", data: { datasets: [{ label: "source group", data: ps.map((p) => ({ x: p.target_score, y: p.mean_pred_score, g: p.group })), backgroundColor: ps.map((p) => COLORS[levelOf(p.target_score)]), pointRadius: 7 }, { type: "line", label: "perfect", data: [{ x: 0, y: 0 }, { x: 100, y: 100 }], borderColor: "#c5c9d8", borderDash: [5, 5], pointRadius: 0 }] },
    options: { plugins: { tooltip: { callbacks: { label: (c) => `${c.raw.g || ""}: anchor ${c.raw.x}, predicted ${c.raw.y.toFixed(1)}` } } }, scales: { x: { min: 0, max: 100, title: { display: true, text: "anchor score" } }, y: { min: 0, max: 100, title: { display: true, text: "mean predicted" } } } } });
  const cm = fin.confusion_matrix;
  const mx = Math.max(...cm.flat());
  $("#rep-cm").innerHTML = `<table class="tbl" style="text-align:center"><tr><th></th>${LEVELS.map((l) => `<th style="text-align:center">pred ${l}</th>`).join("")}</tr>${cm.map((row, i) => `<tr><th>true ${LEVELS[i]}</th>${row.map((v, j) => `<td style="text-align:center;font-weight:700;font-size:16px;background:rgba(79,70,229,${(0.06 + 0.8 * v / mx).toFixed(2)});color:${v / mx > 0.5 ? "#fff" : "inherit"}">${v}<div style="font-size:11px;font-weight:500;opacity:.8">${pct(v / row.reduce((a, b) => a + b, 0))}</div></td>`).join("")}</tr>`).join("")}</table>`;
  table("#rep-src", [{ key: "group", label: "Source group" }, { key: "n", label: "N", num: true }, { key: "target_score", label: "Anchor", num: true, d: 0 }, { key: "mean_pred_score", label: "Predicted", num: true, d: 1 }, { key: "level_accuracy", label: "Accuracy", num: true, render: (v) => pct(v, 1) }], ps);
  $("#rep-figs").innerHTML = R.figures.map((f) => `<figure class="card"><img class="fig" loading="lazy" src="/figures/${f}" alt=""><figcaption>${f.replace(".png", "").replace(/_/g, " ")}</figcaption></figure>`).join("");
}

/* ================================================================ DATASETS */
const DS = { name: null, file: null, page: 1, search: "" };
async function initDatasets() {
  DATASETS = await api("/api/datasets");
  const d = DATASETS.datasets;
  const total = Object.values(d).reduce((a, x) => a + x.total_rows, 0);
  $("#ds-stats").innerHTML = [["Public datasets", Object.keys(d).length], ["Raw rows stored", total.toLocaleString()], ["Unified cleaned questions", DATASETS.n_unified.toLocaleString()], ["Balanced modelling set", DATASETS.n_modelling.toLocaleString()]]
    .map(([l, v]) => `<div class="card stat"><div class="l">${l}</div><div class="v">${v}</div></div>`).join("");
  $("#ds-cards").innerHTML = Object.entries(d).map(([k, v]) => `<div class="card ds-card" data-k="${k}"><div class="row" style="justify-content:space-between"><h3>${esc(k)}</h3><span class="pill">${esc(v.license)}</span></div>
    <div class="rows">${v.total_rows.toLocaleString()} <span class="muted small" style="font-weight:500">rows</span></div><div class="muted small">${esc(v.description)}</div><div class="small" style="margin-top:6px"><b>Signal:</b> ${esc(v.difficulty_signal)}</div></div>`).join("");
  $$(".ds-card").forEach((c) => c.addEventListener("click", () => selectDataset(c.dataset.k)));
  $("#ds-file").addEventListener("change", () => { DS.file = $("#ds-file").value; DS.page = 1; loadPage(); });
  let t;
  $("#ds-search").addEventListener("input", () => { clearTimeout(t); t = setTimeout(() => { DS.search = $("#ds-search").value; DS.page = 1; loadPage(); }, 350); });
  $("#ds-prev").addEventListener("click", () => { if (DS.page > 1) { DS.page--; loadPage(); } });
  $("#ds-next").addEventListener("click", () => { DS.page++; loadPage(); });
  selectDataset(Object.keys(d)[0], false);

  const groups = [...new Set(DATASETS.groups.map((g) => g.group))];
  chart("ch-groups", { type: "bar", data: { labels: groups, datasets: LEVELS.map((l) => ({ label: l, data: groups.map((g) => (DATASETS.groups.find((x) => x.group === g && x.level === l) || {}).n || 0), backgroundColor: COLORS[l], borderRadius: 3 })) }, options: { scales: { x: { stacked: true }, y: { stacked: true } } } });

  const gold = DATASETS.gold;
  $("#gold-btns").innerHTML = Object.keys(gold).map((g) => `<button class="btn btn-ghost" data-g="${g}">${esc(g)} (${gold[g].length})</button>`).join("");
  $$("#gold-btns button").forEach((b) => b.addEventListener("click", () => {
    const rows = gold[b.dataset.g];
    table("#gold-table", Object.keys(rows[0]).map((k) => ({ key: k, label: k })), rows);
  }));
  $("#gold-btns button")?.click();
}

function selectDataset(k, scroll = true) {
  DS.name = k;
  DS.page = 1;
  DS.search = "";
  $("#ds-search").value = "";
  $$(".ds-card").forEach((c) => c.classList.toggle("sel", c.dataset.k === k));
  const files = DATASETS.datasets[k].parquet_files;
  $("#ds-file").innerHTML = files.map((f) => `<option>${f}</option>`).join("");
  DS.file = files[0];
  $("#ds-title").textContent = `Browse · ${k}`;
  loadPage();
  if (scroll) $("#ds-browser").scrollIntoView({ behavior: "smooth" });
}

async function loadPage() {
  $("#ds-table").innerHTML = `<tr><td><span class="loader dark"></span></td></tr>`;
  const r = await api(`/api/datasets/${DS.name}/${DS.file}?page=${DS.page}&search=${encodeURIComponent(DS.search)}`).catch((e) => toast(e.message, true));
  if (!r) return;
  DS.page = Math.min(DS.page, r.pages);
  table("#ds-table", r.columns.map((c) => ({ key: c, label: c, render: (v) => esc(v) })), r.rows);
  $("#ds-page").textContent = `Page ${r.page} of ${r.pages.toLocaleString()} · ${r.total.toLocaleString()} rows`;
  $("#ds-prev").disabled = r.page <= 1;
  $("#ds-next").disabled = r.page >= r.pages;
}

/* ---------------------------------------------------------------- boot */

/* ================================================================ PIPELINE VIEW */
const PL_EXAMPLES = [
  "1+2",
  "Q2. (a) Explain the process of photosynthesis in green plants. [5 marks]",
  "Prove that the square root of 2 is irrational.",
  "Evaluate the impact of the Treaty of Versailles on the rise of fascism in Europe.",
  "पृथ्वी सूर्य के चारों ओर क्यों घूमती है?",
];
const BLOOMS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"];
const PL_WAIT = '<div class="pl-wait">Run a question above to see this step\'s real output.</div>';

const plBars = (pairs, color = "#4f46e5", label = (x) => x) => {
  const mx = Math.max(...pairs.map((p) => p[1]), 1e-6);
  return pairs.map(([k, v]) => `<div class="pl-bar"><span class="mono">${esc(label(k))}</span><div class="bar"><span style="width:${(100 * v / mx).toFixed(1)}%;background:${color}"></span></div><span class="num">${v.toFixed(3)}</span></div>`).join("");
};
const plOut = (m) => `<div class="pl-mini">${probBars(m.probabilities)}<div class="row" style="justify-content:space-between;margin-top:8px"><span class="muted small">Score <b style="color:var(--text)">${m.score.toFixed(1)}</b>/100</span>${badge(m.level || LEVELS.reduce((a, l) => (m.probabilities[l] > m.probabilities[a] ? l : a), "Easy"))}</div></div>`;
const kvList = (rows) => `<dl class="kv">${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>`;

const PL_STAGES = [
  { id: "input", title: "Input & language check", co: "CO3", tech: ["langdetect", "Devanagari script check", "MarianMT (opus-mt-hi-en / mul-en)"],
    simple: "First we check which language the question is written in. The models learned from English questions, so a Hindi or Marathi question is translated to English by a small translation model that runs on this computer.",
    live: (t) => `<div class="pl-box">${esc(t.input)}</div>` + (t.language.translated
      ? `<div class="pl-arrow">↓ detected <b>${esc(t.language.name)}</b> → translated locally</div><div class="pl-box">${esc(t.english)}</div>`
      : `<div class="pl-note">✓ Detected <b>${esc(t.language.name)}</b>, so no translation is needed.</div>`) },

  { id: "clean", title: "Text cleaning", co: "CO1", tech: ["Regular expressions", "HTML / URL removal", "LaTeX normalisation", "Numbering & marks removal"],
    simple: "Real question papers contain noise such as “Q3.”, “[5 marks]”, web links or maths code like \\frac{1}{2}. We strip or normalise this so that only the actual question is left.",
    live: (t) => `<div class="pl-cmp"><div><div class="lbl">Before</div><div class="pl-box">${esc(t.english)}</div></div><div class="pl-arrow-h">→</div><div><div class="lbl">After</div><div class="pl-box">${esc(t.cleaned)}</div></div></div>` +
      (t.english.trim() === t.cleaned.trim() ? '<div class="pl-note">The text was already clean, so nothing needed removing.</div>' : "") },

  { id: "tokens", title: "Tokenisation & normalisation", co: "CO1", tech: ["NLTK word tokenizer", "Stop-word removal", "Porter stemmer", "WordNet lemmatizer", "POS tagging"],
    simple: "The sentence is split into words (tokens). Very common words such as “the” and “is” are marked as stop words, and the remaining words are reduced to their root form (“appears” → “appear”). These clean tokens feed the topic model and the Word2Vec embeddings.",
    live: (t) => {
      const p = t.preprocess, sw = new Set(p.without_stopwords);
      return `<div class="lbl">Tokens <span class="muted">(stop words struck out)</span></div><div class="pl-toks">${p.tokens.map((x) => `<span class="tok ${sw.has(x) ? "" : "sw"}">${esc(x)}</span>`).join("")}</div>
        <div class="table-wrap mt" style="max-height:none"><table class="tbl"><tbody>
        <tr><td class="muted">POS tags</td><td>${p.pos_tags.map(([w, g]) => `${esc(w)}<sub class="muted">${esc(g)}</sub>`).join(" ")}</td></tr>
        <tr><td class="muted">Stems</td><td class="mono">${esc(p.stemmed.join("  "))}</td></tr>
        <tr><td class="muted">Lemmas</td><td class="mono">${esc(p.lemmatized.join("  "))}</td></tr></tbody></table></div>`;
    } },

  { id: "vector", title: "Vectorisation (text → numbers)", co: "CO1", tech: ["TF-IDF word 1–2 grams", "TF-IDF character 3–5 grams", "Sparse vectors"],
    simple: "Models only understand numbers. TF-IDF gives every word or word pair a weight: high if it is rare in general but present in this question. Character pieces such as “sqr” also catch maths and spelling patterns. The result is a very long list of numbers in which almost every value is zero.",
    live: (t) => {
      const v = t.tfidf, dim = v.word_dim + v.char_dim, nz = v.word_nonzero + v.char_nonzero;
      return `<div class="pl-note">Vector size <b>${dim.toLocaleString()}</b>, of which only <b>${nz}</b> values are non-zero (${(100 * nz / dim).toFixed(3)}%).</div>
        <div class="grid g2 mt"><div><div class="lbl">Top word n-grams (${v.word_nonzero} non-zero)</div>${plBars(v.top_word)}</div>
        <div><div class="lbl">Top character n-grams (${v.char_nonzero} non-zero)</div>${plBars(v.top_char, "#0ea5e9", (k) => k.replace(/ /g, "␣"))}</div></div>`;
    } },

  { id: "extract", title: "Understanding the question", co: "CO2 · CO3", tech: ["spaCy NER", "YAKE / TextRank keyphrases", "Bloom's taxonomy rules", "VADER sentiment", "LDA topic model"],
    simple: "Next we pull out what the question is about and what it asks you to do. Named entities (dates, places, numbers), key phrases, the thinking level from Bloom's taxonomy (remember → create) and the tone. The Bloom level and tone become features in the next step. The topic is shown only as extra insight.",
    live: (t) => {
      const i = t.info || {}, s = t.sentiment || {}, tp = t.topic;
      return kvList([
        ["Bloom's level", `<b>${esc(i.bloom_level)}</b> <span class="muted small">${esc((i.bloom_indicators || []).join(", "))}</span>`],
        ["Question type", esc(i.question_type)],
        ["Entities", (i.entities || []).map((e) => `<span class="pill ent">${esc(e.text)} · ${esc(e.label)}</span>`).join("") || "–"],
        ["Key phrases", (i.keyphrases_yake || []).slice(0, 5).map((k) => `<span class="pill key">${esc(k)}</span>`).join("") || "–"],
        ["Tone (VADER)", `${esc(s.vader_label)} (${(s.vader_compound ?? 0).toFixed(2)})`],
        ["Topic (LDA)", tp && tp.top_words ? esc(tp.top_words.slice(0, 5).join(", ")) : "–"],
      ]);
    } },

  { id: "features", title: "Linguistic features (57 numbers)", co: "CO1 · CO2", tech: ["textstat readability", "wordfreq rarity", "spaCy dependency parse", "Bloom's level", "VADER tone"],
    simple: "We measure the question the way a teacher might judge it. How long is it? Are the words rare? How deeply nested are the sentences? What thinking level does it ask for? These 57 measurements form a profile of the question.",
    live: (t) => {
      const f = Object.assign({}, ...Object.values(t.features));
      const pick = {
        "Readability": [["FK grade level", f.fk_grade], ["Flesch reading ease", f.flesch_reading_ease], ["Gunning fog", f.gunning_fog], ["Difficult words", f.difficult_words]],
        "Vocabulary": [["Words", f.n_words], ["Rare-word ratio", f.rare_word_ratio], ["Avg. word frequency (Zipf)", f.mean_zipf], ["Maths symbols", f.n_math_symbols]],
        "Sentence structure": [["Sentences", f.n_sentences], ["Parse-tree depth", f.max_tree_depth], ["Subordinate clauses", f.n_subordinate], ["Named entities", f.n_entities]],
        "Thinking & tone": [["Bloom's level", BLOOMS[(f.bloom_level || 1) - 1] || f.bloom_level], ["VADER compound", f.vader_compound], ["Negations", f.n_negations], ["Constraints (exactly, only…)", f.n_constraints]],
      };
      const all = Object.entries(t.features).map(([g, d]) => `<div><div class="lbl">${esc(g)}</div>${kvList(Object.entries(d).map(([k, v]) => [esc(featName(k)), fmt(v, 2)]))}</div>`).join("");
      return `<div class="grid g4">${Object.entries(pick).map(([g, rows]) => `<div class="pl-fgroup"><div class="lbl">${g}</div>${kvList(rows.map(([k, v]) => [k, typeof v === "number" ? fmt(v, 2) : esc(v)]))}</div>`).join("")}</div>
        <details class="mt"><summary class="small" style="cursor:pointer;color:var(--primary);font-weight:600">Show all ${t.n_features} features</summary><div class="grid g3 mt">${all}</div></details>`;
    } },

  { id: "models", title: "Three models predict in parallel", co: "CO1 · CO4", tech: ["LightGBM", "DistilBERT (transformer)", "BiLSTM + attention", "SHAP"], lanes: true,
    simple: "Three different kinds of model look at the question independently. Each one gives a probability for Easy / Medium / Hard and its own 0–100 score. Classic machine learning, a transformer and a recurrent network each notice different things, so combining them is more reliable than trusting just one. All three were trained on the same 35,681 training questions, drawn from 9 public datasets plus a generated arithmetic set." },

  { id: "ensemble", title: "Weighted ensemble", co: "", tech: ["Weighted soft voting", "Weights tuned on validation set", "Band consistency check"],
    simple: "The models' answers are combined using weights chosen on a validation set that was held out from training. DistilBERT gets 60% of the vote and LightGBM 40%. The BiLSTM gets 0% because adding it did not improve accuracy. Finally the score is kept inside the band of the chosen level, so the level and the score never disagree.",
    live: (t) => {
      const e = t.ensemble, m = t.members, names = { gbm: "LightGBM", bert: "DistilBERT", bilstm: "BiLSTM" };
      const used = Object.entries(e.weights).filter(([, w]) => w > 0);
      const probRows = LEVELS.map((l) => `<tr><td><b>${l}</b></td><td class="mono">${used.map(([k, w]) => `${w.toFixed(1)} × ${m[k].probabilities[l].toFixed(3)}`).join(" + ")}</td><td class="num">${pct(e.probabilities[l], 1)}</td></tr>`).join("");
      const sc = used.map(([k, w]) => `${w.toFixed(1)} × ${m[k].score.toFixed(1)}`).join(" + ");
      const [lo, hi] = e.band;
      return `<div class="pl-weights">${Object.entries(e.weights).map(([k, w]) => `<div class="pl-w ${w > 0 ? "" : "off"}"><span>${names[k]}</span><b>${Math.round(w * 100)}%</b></div>`).join("")}</div>
        <div class="lbl mt">① Level probabilities = weighted average</div>
        <div class="table-wrap" style="max-height:none"><table class="tbl"><tbody>${probRows}</tbody></table></div>
        <div class="lbl mt">② Difficulty score = weighted average of the models' scores</div>
        <div class="pl-box mono">${sc}${e.alpha < 1 ? `  (blended ${e.alpha} : ${(1 - e.alpha).toFixed(2)} with expected score ${e.expected_score})` : ""} = <b>${e.raw_score.toFixed(2)}</b></div>
        <div class="lbl mt">③ Level = highest probability → ${badge(e.level)}</div>
        <div class="lbl mt">④ Consistency check: ${e.level} band is ${lo}–${hi}</div>
        <div class="pl-note ${e.clipped ? "warn" : ""}">${e.clipped ? `The raw score ${e.raw_score.toFixed(2)} falls outside the ${e.level} band, so it is moved to the nearest edge: <b>${e.final_score.toFixed(1)}</b>.` : `${e.raw_score.toFixed(2)} is inside the band, so it is kept: <b>${e.final_score.toFixed(1)}</b>.`}</div>`;
    } },

  { id: "result", title: "Final prediction", co: "", tech: ["Level: Easy / Medium / Hard", "Score: 0–100", "Confidence"],
    simple: "The output is a difficulty level, a 0–100 score and a confidence, which is how sure the ensemble is about the level. The score bands are Easy 0–33, Medium 33–67 and Hard 67–100.",
    live: (t) => {
      const r = t.result;
      return `<div class="pl-final"><div class="gauge-wrap"><div>${gaugeSVG(r.score, r.level)}</div><div class="gauge-score">${r.score.toFixed(1)}<small>/100</small></div></div>
        <div style="flex:1;min-width:220px">${badge(r.level, "lg")}<div class="muted small" style="margin:8px 0">Confidence ${pct(r.confidence)}</div>${probBars(r.probabilities)}
        <a class="btn btn-ghost mt" href="?q=${encodeURIComponent(t.input)}#analyze">Open full analysis →</a></div></div>`;
    } },
];

const PL_LANES = [
  { k: "gbm", name: "LightGBM", kind: "Classic machine learning", tech: ["TF-IDF → LogReg + Ridge", "57 features + 4 = 61 inputs", "Gradient-boosted trees", "SHAP"],
    simple: "First a quick text model reads the TF-IDF vector and gives 4 numbers. These are added to the 57 features, and a forest of decision trees learns which combinations mean “hard”. SHAP shows which inputs pushed the score up or down.",
    live: (t) => {
      const s1 = t.stage1, shap = (t.shap || []).slice(0, 6), mx = Math.max(...shap.map((x) => Math.abs(x[1])), 0.01);
      return `<div class="lbl">Step A: TF-IDF text model</div><div class="small mono">P = ${LEVELS.map((l) => `${l[0]} ${s1.probabilities[l].toFixed(2)}`).join(" · ")} · score ${s1.score.toFixed(1)}</div>
        <div class="lbl mt">Step B: LightGBM output</div>${plOut(t.members.gbm)}
        <div class="lbl mt">Why? Top SHAP contributions</div>${shap.map(([n, v]) => {
          const w = (Math.abs(v) / mx) * 50, left = v >= 0 ? 50 : 50 - w;
          return `<div class="shap-row" style="grid-template-columns:110px 1fr 40px"><span title="${esc(n)}">${esc(featName(n))}</span><div class="shap-track"><div class="mid"></div><div class="fill" style="left:${left}%;width:${w}%;background:${v >= 0 ? "#ef4444" : "#22c55e"}"></div></div><span style="text-align:right;color:${v >= 0 ? "#dc2626" : "#16a34a"}">${v >= 0 ? "+" : ""}${v.toFixed(1)}</span></div>`;
        }).join("")}`;
    } },
  { k: "bert", name: "DistilBERT", kind: "Transformer · deep learning", tech: ["WordPiece tokenizer", "6 transformer layers, 66 M parameters", "Classification + regression heads"],
    simple: "A pre-trained language model that already understands English, fine-tuned on our questions. It splits words into sub-word pieces, reads the whole sentence at once with self-attention, and has two output heads: one picks the level and one gives the score.",
    live: (t) => `<div class="lbl">Sub-word tokens (${t.bert.n_tokens} of max ${t.bert.max_len})</div><div class="pl-toks">${t.bert.tokens.map((x) => `<span class="tok ${x.startsWith("[") ? "sp" : x.startsWith("##") ? "sub" : ""}">${esc(x)}</span>`).join("")}</div>
      <div class="lbl mt">Output</div>${plOut(t.members.bert)}` },
  { k: "bilstm", name: "BiLSTM + attention", kind: "Recurrent network · explainer", tech: ["Word2Vec embeddings", "Bidirectional LSTM", "Attention layer"],
    simple: "Reads the words left-to-right and right-to-left, and an attention layer learns which words matter most. Its vote weight is 0% because it did not improve the ensemble, but its attention is a nice visual explanation.",
    live: (t) => {
      const att = t.attention || [], am = Math.max(...att.map((a) => a[1]), 0.001);
      return `<div class="lbl">Attention (darker = more important)</div><div class="attn">${att.map(([w, a]) => `<span style="background:rgba(79,70,229,${(0.08 + 0.85 * a / am).toFixed(2)});color:${a / am > 0.55 ? "#fff" : "inherit"}">${esc(w)}</span>`).join("")}</div>
        <div class="lbl mt">Output</div>${plOut(t.members.bilstm)}`;
    } },
];

function plRender(t) {
  const w = t ? t.ensemble.weights : { gbm: 0.4, bert: 0.6, bilstm: 0 };
  $("#pl-flow").innerHTML = PL_STAGES.map((s, i) => `<a class="pl-node" href="javascript:void 0" data-id="${s.id}"><span class="n">${i + 1}</span>${esc(s.title.split(" (")[0])}</a>`).join('<span class="pl-chev">›</span>');
  $$("#pl-flow .pl-node").forEach((a) => a.addEventListener("click", () => document.getElementById("pl-" + a.dataset.id).scrollIntoView({ behavior: "smooth", block: "start" })));
  $("#pl-stages").innerHTML = PL_STAGES.map((s, i) => {
    const head = `<div class="pl-head"><h3>${esc(s.title)}</h3>${s.co ? `<span class="chip-co">${s.co}</span>` : ""}</div>
      <div class="pl-tech">${s.tech.map((x) => `<span>${esc(x)}</span>`).join("")}</div>
      <div class="pl-simple"><b>In simple words:</b> ${esc(s.simple)}</div>`;
    let body;
    if (s.lanes) {
      body = `<div class="pl-lanes">${PL_LANES.map((l) => `<div class="pl-lane ${w[l.k] > 0 ? "" : "off"}">
          <div class="row" style="justify-content:space-between"><div><div style="font-weight:700">${l.name}</div><div class="muted small">${l.kind}</div></div><span class="pl-wchip">vote ${Math.round((w[l.k] || 0) * 100)}%</span></div>
          <div class="pl-tech">${l.tech.map((x) => `<span>${esc(x)}</span>`).join("")}</div>
          <div class="pl-simple small">${esc(l.simple)}</div>
          <div class="pl-live">${t ? l.live(t) : PL_WAIT}</div></div>`).join("")}</div>`;
    } else body = `<div class="pl-live">${t ? s.live(t) : PL_WAIT}</div>`;
    return `<div class="pl-stage" id="pl-${s.id}"><div class="pl-rail"><span class="pl-num">${i + 1}</span></div><div class="card">${head}${body}</div></div>`;
  }).join("");
}

function initPipeline() {
  $("#pl-examples").innerHTML = PL_EXAMPLES.map((e) => `<span class="ex" title="${esc(e)}">${esc(e.length > 40 ? e.slice(0, 38) + "…" : e)}</span>`).join("");
  $$("#pl-examples .ex").forEach((el, i) => el.addEventListener("click", () => { $("#pl-input").value = PL_EXAMPLES[i]; runPipeline(); }));
  $("#pl-go").addEventListener("click", runPipeline);
  $("#pl-input").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) runPipeline(); });
  plRender(null);
  const q = new URLSearchParams(location.search).get("q");   // e.g. /?q=1%2B2#pipeline
  if (q) { $("#pl-input").value = q; runPipeline(); }
}

async function runPipeline() {
  const q = $("#pl-input").value.trim();
  if (!q) return toast("Type a question first");
  await busy($("#pl-go"), async () => {
    const t = await api("/api/pipeline", { question: q });
    plRender(t);
    $$(".pl-stage").forEach((el, i) => { el.style.animationDelay = `${i * 70}ms`; el.classList.add("pl-in"); });
  });
}

route();   // keep last: every page constant above must be initialised first
