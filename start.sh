#!/usr/bin/env bash
# Start DiffiQ: local LLM (Ollama) + FastAPI backend + web frontend at http://localhost:8000
cd "$(dirname "$0")"
curl -s localhost:11434/api/tags >/dev/null 2>&1 || { echo "Starting Ollama..."; (ollama serve >/dev/null 2>&1 &); sleep 3; }
( sleep 4; open http://localhost:8000 2>/dev/null || xdg-open http://localhost:8000 2>/dev/null ) &
echo "DiffiQ running at http://localhost:8000  (Ctrl+C to stop)"
exec .venv/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
