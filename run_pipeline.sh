#!/usr/bin/env bash
# Reproduce the whole project end-to-end (≈2 h on an 8 GB Apple-Silicon Mac).
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "== 1. Download all public datasets (one time) ==";      $PY -m src.data.download
echo "== 2. Unify, clean, label, split ==";                     $PY -m src.data.unify
echo "== 3. Linguistic features ==";                            $PY -m src.features.build_features
echo "== 4. CO1 vectoriser comparison + TF-IDF baseline ==";    $PY -m src.features.vectorizers
echo "== 5. CO2 sentiment models ==";                           $PY -m src.nlp.sentiment
echo "== 6. CO2 information-extraction evaluation ==";          $PY -m src.nlp.evaluate_ie
echo "== 7. LightGBM + SHAP ==";                                $PY -m src.models.gbm
echo "== 8. DistilBERT fine-tuning ==";                         $PY -u -m src.models.transformer --epochs 2
echo "== 9. BiLSTM + attention ==";                             $PY -m src.models.bilstm
echo "== 10. Ensemble ==";                                      $PY -m src.models.ensemble
echo "== 11. CO3 topic models ==";                              $PY -m src.nlp.topics
echo "== 12. CO3 multilingual evaluation ==";                   $PY -m src.multilingual.evaluate
echo "== 13. CO4 LLM prompt evaluation (needs: ollama serve) =="; $PY -m src.llm.evaluate_prompts
echo "== 14. Figures ==";                                       $PY -m src.figures
echo "== 15. Notebooks ==";                                     $PY -m src.build_notebooks
echo "Done. Launch the app:  .venv/bin/streamlit run app/streamlit_app.py"
