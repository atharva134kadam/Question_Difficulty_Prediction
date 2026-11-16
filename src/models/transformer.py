"""Fine-tuned DistilBERT with two heads (multi-task learning):
   - classification head -> Easy / Medium / Hard   (cross-entropy)
   - regression head     -> difficulty score 0-100 (MSE on score/100)

Runs on Apple Silicon GPU (MPS) / CUDA / CPU. Weights are stored locally.
Usage: python -m src.models.transformer [--epochs 3]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from src.config import DATA_PROC, MODELS, SEED
from src.evaluation import full_report, save_metrics

BASE = "distilbert-base-uncased"
OUT = MODELS / "distilbert_difficulty"


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DifficultyTransformer(nn.Module):
    def __init__(self, base=BASE, n_classes=3, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(h, n_classes)
        self.reg_head = nn.Sequential(nn.Linear(h, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, input_ids, attention_mask):
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # mean pooling over real tokens
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
        pooled = self.dropout(pooled)
        return self.cls_head(pooled), torch.sigmoid(self.reg_head(pooled)).squeeze(-1)


class QDataset(Dataset):
    def __init__(self, df):
        self.t = df.question.tolist()
        self.y = df.label.values if "label" in df else np.zeros(len(df), int)
        self.s = (df.score.values / 100.0) if "score" in df else np.zeros(len(df))

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        return self.t[i], self.y[i], self.s[i]


def make_collate(tok, max_len=128):
    def collate(batch):
        texts, y, s = zip(*batch)
        enc = tok(list(texts), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"], torch.tensor(y), torch.tensor(s, dtype=torch.float32)
    return collate


@torch.no_grad()
def predict(model, tok, texts, batch_size=32, dev=None, max_len=96):
    dev = dev or next(model.parameters()).device
    model.eval()
    probs, scores = [], []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        logits, s = model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev))
        probs.append(torch.softmax(logits, -1).cpu().numpy())
        scores.append((s * 100).cpu().numpy())
    return np.vstack(probs), np.concatenate(scores)


def load_trained(dev=None):
    dev = dev or device()
    tok = AutoTokenizer.from_pretrained(OUT)
    model = DifficultyTransformer(base=str(OUT / "encoder"))
    state = torch.load(OUT / "heads.pt", map_location="cpu")
    model.cls_head.load_state_dict(state["cls"])
    model.reg_head.load_state_dict(state["reg"])
    return model.to(dev).eval(), tok


def train(epochs=2, lr=4e-5, batch_size=16, reg_weight=2.0, max_len=96, resume=False, rehearsal=0):
    torch.manual_seed(SEED)
    dev = device()
    tr, va, te = (pd.read_parquet(DATA_PROC / f"{s}.parquet") for s in ("train", "val", "test"))
    if rehearsal:  # continued fine-tuning: all new-source rows + a random sample of the original rows
        new = tr[tr.source == "synthetic_arith"]
        old = tr[tr.source != "synthetic_arith"].sample(min(rehearsal, (tr.source != "synthetic_arith").sum()), random_state=SEED)
        tr = pd.concat([new, old]).sample(frac=1, random_state=SEED)
        print(f"[bert] rehearsal set: {len(new)} new + {len(old)} original rows", flush=True)
    if resume:  # continue fine-tuning the saved model (e.g. after adding new training data)
        model, tok = load_trained(dev)
        model.train()
        print("[bert] resuming from", OUT, flush=True)
    else:
        tok = AutoTokenizer.from_pretrained(BASE)
        model = DifficultyTransformer().to(dev)
    dl = DataLoader(QDataset(tr), batch_size=batch_size, shuffle=True, collate_fn=make_collate(tok, max_len))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * len(dl)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    ce, mse = nn.CrossEntropyLoss(label_smoothing=0.05), nn.MSELoss()
    best, history = -1, []
    print(f"[bert] device={dev} steps/epoch={len(dl)}", flush=True)
    for ep in range(epochs):
        model.train()
        t0, run = time.time(), 0.0
        for step, (ids, am, y, s) in enumerate(dl):
            ids, am, y, s = ids.to(dev), am.to(dev), y.to(dev), s.to(dev)
            logits, sp = model(ids, am)
            loss = ce(logits, y) + reg_weight * mse(sp, s)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            run += loss.item()
            if step % 100 == 0:
                print(f"  ep{ep + 1} step {step}/{len(dl)} loss={run / (step + 1):.4f} "
                      f"{time.time() - t0:.0f}s", flush=True)
        p, sc = predict(model, tok, va.question.tolist(), dev=dev)
        m = full_report(va.label, p.argmax(1), va.score, sc)
        history.append({"epoch": ep + 1, "train_loss": run / len(dl), **{k: v for k, v in m.items() if k != "confusion_matrix"}})
        print(f"[bert] epoch {ep + 1} val acc={m['accuracy']:.4f} f1={m['macro_f1']:.4f} mae={m['mae']:.2f}", flush=True)
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            OUT.mkdir(parents=True, exist_ok=True)
            model.encoder.save_pretrained(OUT / "encoder")
            tok.save_pretrained(OUT)
            torch.save({"cls": model.cls_head.state_dict(), "reg": model.reg_head.state_dict()}, OUT / "heads.pt")
    (OUT / "history.json").write_text(json.dumps(history, indent=2))

    model, tok = load_trained(dev)
    for split, df in (("val", va), ("test", te)):
        p, sc = predict(model, tok, df.question.tolist(), dev=dev)
        np.save(OUT / f"{split}_probs.npy", p)
        np.save(OUT / f"{split}_scores.npy", sc)
    m = full_report(te.label, np.load(OUT / "test_probs.npy").argmax(1), te.score, np.load(OUT / "test_scores.npy"))
    save_metrics("model::DistilBERT (multi-task)", m)
    print("[bert] TEST", json.dumps({k: v for k, v in m.items() if k != "confusion_matrix"}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--rehearsal", type=int, default=0)
    a = ap.parse_args()
    train(epochs=a.epochs, lr=a.lr, resume=a.resume, rehearsal=a.rehearsal)
