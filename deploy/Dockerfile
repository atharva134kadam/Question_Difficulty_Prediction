# DiffiQ - Question Difficulty Prediction (Hugging Face Space, Docker SDK)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates zstd libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# Local LLM runtime (Ollama) - the model runs inside this container, no external API
RUN curl -fsSL https://ollama.com/install.sh | sh

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface OLLAMA_MODELS=/home/user/.ollama/models \
    OLLAMA_HOST=127.0.0.1:11434 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user deploy/requirements-space.txt requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt

# one-time model/resource downloads baked into the image (offline at runtime)
RUN python -m spacy download en_core_web_sm && \
    python -c "import nltk; [nltk.download(p, quiet=True) for p in ['punkt','punkt_tab','stopwords','wordnet','omw-1.4','vader_lexicon','averaged_perceptron_tagger','averaged_perceptron_tagger_eng']]" && \
    python -c "from transformers import MarianMTModel, MarianTokenizer as T; [ (T.from_pretrained(m), MarianMTModel.from_pretrained(m)) for m in ['Helsinki-NLP/opus-mt-hi-en','Helsinki-NLP/opus-mt-mul-en'] ]"
RUN (ollama serve > /tmp/ollama.log 2>&1 &) && sleep 8 && ollama pull qwen2.5:1.5b

COPY --chown=user . .
EXPOSE 7860
CMD ["bash", "deploy/start_space.sh"]
