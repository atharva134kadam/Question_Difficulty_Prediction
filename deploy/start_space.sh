#!/usr/bin/env bash
# Start the local LLM in the background, then the API + frontend on port 7860.
(ollama serve > /tmp/ollama.log 2>&1 &)
exec python -m uvicorn app.server:app --host 0.0.0.0 --port 7860
