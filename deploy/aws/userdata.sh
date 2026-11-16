#!/bin/bash
# DiffiQ - one-shot bootstrap for an Ubuntu 24.04 server (AWS Lightsail / EC2).
# Installs everything, then serves the app on port 80. Progress: /var/log/diffiq-setup.log
set -euxo pipefail
exec > >(tee -a /var/log/diffiq-setup.log) 2>&1
echo "=== DiffiQ setup started $(date) ==="
export HOME=/root   # cloud-init runs without $HOME; ollama's CLI panics if it is unset

REPO=https://github.com/atharva134kadam/Question_Difficulty_Prediction.git
APP_USER=ubuntu
APP=/opt/diffiq

# 1. swap (safety margin for model loading)
if [ ! -f /swapfile ]; then fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile && echo '/swapfile none swap sw 0 0' >> /etc/fstab; fi

# 2. system packages
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y git git-lfs curl ca-certificates zstd nginx build-essential libgomp1
git lfs install --system

# 3. code (+ LFS files: DistilBERT weights, RACE csv)
if [ ! -d $APP/.git ]; then git clone $REPO $APP; fi
cd $APP && git lfs pull && chown -R $APP_USER:$APP_USER $APP

cat > /tmp/requirements-lock.txt <<'EOF_LOCK'
fastapi==0.163.1
gensim==4.4.0
huggingface-hub==2.4.0
joblib==1.6.0
langdetect==1.0.9
lightgbm==4.7.0
matplotlib==3.11.2
networkx==3.6.1
nltk==3.10.4
numpy==2.2.6
pandas==3.1.1
pyarrow==26.0.1
pypdf==7.1.0
pyphen==0.18.1
python-docx==1.2.0
python-multipart==0.0.34
rake-nltk==1.0.6
requests==2.34.2
sacremoses==0.2.0
safetensors==0.8.0
scikit-learn==1.9.1
scipy==1.17.1
sentencepiece==0.2.3
shap==0.51.0
spacy==3.8.18
textstat==0.7.13
tokenizers==0.23.3
transformers==5.23.0
uvicorn==0.54.0
wordfreq==3.1.1
yake==0.7.3
EOF_LOCK
chmod 644 /tmp/requirements-lock.txt

# 4. Python 3.11 env with the exact versions the models were trained with
sudo -u $APP_USER -H bash -lc "
  set -eux
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH=\$HOME/.local/bin:\$PATH
  cd $APP
  uv venv --python 3.11 .venv
  uv pip install --python .venv/bin/python torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python .venv/bin/python -r /tmp/requirements-lock.txt 'uvicorn[standard]==0.54.0'
  .venv/bin/python -m spacy download en_core_web_sm || uv pip install --python .venv/bin/python en_core_web_sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
  .venv/bin/python -c \"import nltk; [nltk.download(p, quiet=True) for p in ['punkt','punkt_tab','stopwords','wordnet','omw-1.4','vader_lexicon','averaged_perceptron_tagger','averaged_perceptron_tagger_eng']]\"
  .venv/bin/python -c \"from transformers import MarianMTModel, MarianTokenizer as T; [(T.from_pretrained(m), MarianMTModel.from_pretrained(m)) for m in ['Helsinki-NLP/opus-mt-hi-en','Helsinki-NLP/opus-mt-mul-en']]\"
"

# 5. local LLM (Ollama runs on this server; no external API)
curl -fsSL https://ollama.com/install.sh | sh
systemctl enable --now ollama
for i in $(seq 1 30); do curl -s localhost:11434/api/tags >/dev/null && break; sleep 2; done
ollama pull qwen2.5:1.5b

# 6. app service
cat > /etc/systemd/system/diffiq.service <<EOF
[Unit]
Description=DiffiQ question difficulty web app
After=network.target ollama.service
[Service]
User=$APP_USER
WorkingDirectory=$APP
Environment=OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
ExecStart=$APP/.venv/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now diffiq

# 7. nginx reverse proxy on :80
cat > /etc/nginx/sites-available/diffiq <<'EOF'
server {
    listen 80 default_server;
    server_name _;
    client_max_body_size 25M;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
EOF
ln -sf /etc/nginx/sites-available/diffiq /etc/nginx/sites-enabled/diffiq
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl restart nginx

# load the models once so the first visitor doesn't wait
for i in $(seq 1 60); do curl -s -X POST localhost:8000/api/warmup >/dev/null && break; sleep 5; done

echo "=== DiffiQ setup finished $(date) ==="
touch /var/log/diffiq-setup.done
