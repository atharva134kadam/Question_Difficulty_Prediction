"""BiLSTM + additive attention, trained from scratch (deep learning without pre-training).

Embeddings initialised from our locally trained Word2Vec. Two heads like DistilBERT:
level (cross-entropy) + score (MSE). Attention weights show which words drove the
prediction (visualised in the app).
"""
from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gensim.models import Word2Vec

from src.config import DATA_PROC, MODELS, SEED
from src.evaluation import full_report, save_metrics
from src.nlp.cleaning import tokenize

OUT = MODELS / "bilstm"
MAX_LEN = 80


class Vocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    @classmethod
    def build(cls, texts, min_freq=2, max_size=30000):
        c = Counter(t for s in texts for t in tokenize(s))
        words = [w for w, n in c.most_common(max_size) if n >= min_freq]
        return cls(["<pad>", "<unk>"] + words)

    def encode(self, text):
        ids = [self.stoi.get(t, 1) for t in tokenize(text)][:MAX_LEN]
        return ids or [1]


class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, emb_dim=100, hidden=128, n_classes=3, emb_weights=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if emb_weights is not None:
            self.emb.weight.data.copy_(torch.tensor(emb_weights))
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True, num_layers=1)
        self.att = nn.Sequential(nn.Linear(2 * hidden, 64), nn.Tanh(), nn.Linear(64, 1))
        self.drop = nn.Dropout(0.3)
        self.cls = nn.Linear(2 * hidden, n_classes)
        self.reg = nn.Linear(2 * hidden, 1)

    def forward(self, x):
        mask = x != 0
        h, _ = self.lstm(self.emb(x))
        a = self.att(h).squeeze(-1).masked_fill(~mask, -1e9)
        w = torch.softmax(a, -1)
        ctx = self.drop((h * w.unsqueeze(-1)).sum(1))
        return self.cls(ctx), torch.sigmoid(self.reg(ctx)).squeeze(-1), w


def pad(batch):
    L = max(len(b) for b in batch)
    return torch.tensor([b + [0] * (L - len(b)) for b in batch])


@torch.no_grad()
def predict(model, vocab, texts, bs=256):
    model.eval()
    P, S = [], []
    for i in range(0, len(texts), bs):
        x = pad([vocab.encode(t) for t in texts[i:i + bs]])
        lo, s, _ = model(x)
        P.append(torch.softmax(lo, -1).numpy()); S.append(s.numpy() * 100)
    return np.vstack(P), np.concatenate(S)


@torch.no_grad()
def attention(model, vocab, text):
    model.eval()
    toks = tokenize(text)[:MAX_LEN]
    _, _, w = model(pad([vocab.encode(text)]))
    return list(zip(toks, w[0, :len(toks)].numpy().round(3).tolist()))


def load_trained():
    vocab = Vocab(json.loads((OUT / "vocab.json").read_text()))
    m = BiLSTMAttention(len(vocab.itos))
    m.load_state_dict(torch.load(OUT / "model.pt", map_location="cpu"))
    return m.eval(), vocab


def train(epochs=6, bs=64, lr=2e-3):
    torch.manual_seed(SEED)
    tr, va, te = (pd.read_parquet(DATA_PROC / f"{s}.parquet") for s in ("train", "val", "test"))
    vocab = Vocab.build(tr.question)
    w2v = Word2Vec.load(str(MODELS / "word2vec.model")).wv
    W = np.random.normal(0, 0.1, (len(vocab.itos), 100)).astype(np.float32)
    W[0] = 0
    hit = 0
    for w, i in vocab.stoi.items():
        if w in w2v:
            W[i] = w2v[w]; hit += 1
    print(f"[bilstm] vocab={len(vocab.itos)} word2vec coverage={hit / len(vocab.itos):.2%}")
    model = BiLSTMAttention(len(vocab.itos), emb_weights=W)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
    X = [vocab.encode(t) for t in tr.question]
    y, s = tr.label.values, tr.score.values / 100
    best = -1
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "vocab.json").write_text(json.dumps(vocab.itos))
    for ep in range(epochs):
        model.train()
        idx = np.random.RandomState(ep).permutation(len(X))
        t0, tot = time.time(), 0
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            lo, sp, _ = model(pad([X[j] for j in b]))
            loss = ce(lo, torch.tensor(y[b])) + 2 * mse(sp, torch.tensor(s[b], dtype=torch.float32))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * len(b)
        p, sc = predict(model, vocab, va.question.tolist())
        m = full_report(va.label, p.argmax(1), va.score, sc)
        print(f"[bilstm] ep{ep + 1} loss={tot / len(X):.4f} val acc={m['accuracy']:.4f} f1={m['macro_f1']:.4f} "
              f"mae={m['mae']:.2f} ({time.time() - t0:.0f}s)", flush=True)
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            torch.save(model.state_dict(), OUT / "model.pt")
    model, vocab = load_trained()
    for split, df in (("val", va), ("test", te)):
        p, sc = predict(model, vocab, df.question.tolist())
        np.save(OUT / f"{split}_probs.npy", p); np.save(OUT / f"{split}_scores.npy", sc)
    p, sc = np.load(OUT / "test_probs.npy"), np.load(OUT / "test_scores.npy")
    m = full_report(te.label, p.argmax(1), te.score, sc)
    save_metrics("model::BiLSTM + attention (from scratch)", m)
    print("[bilstm] TEST", {k: v for k, v in m.items() if k != "confusion_matrix"})


if __name__ == "__main__":
    pass  # thread count is fixed via OMP_NUM_THREADS in src/__init__.py (OpenMP clash on macOS)
    train()
