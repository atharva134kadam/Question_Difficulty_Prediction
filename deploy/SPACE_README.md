---
title: DiffiQ Question Difficulty Predictor
emoji: 🎯
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Predict exam question difficulty (Easy/Medium/Hard + 0-100)
---

# DiffiQ · Question Difficulty Prediction (NLP project)

Predicts the difficulty of any exam question as **Easy / Medium / Hard** plus a **0–100 score**, and explains why.
It has an NLP lab (preprocessing, sentiment, information extraction, topics, Hindi/Marathi) and a GenAI studio
backed by a local LLM (Qwen2.5-1.5B via Ollama, running inside this Space). No external APIs are called.

Ensemble of DistilBERT, LightGBM and a BiLSTM: **80.4% accuracy, QWK 0.835, ±6.7-point score error** on 4,183 held-out questions.

Source code: https://github.com/atharva134kadam/Question_Difficulty_Prediction

The Space runs on CPU. The first request after it wakes loads the models (about 30–60 s). LLM features take about 10–40 s.
