"""CO1 - Text extraction from unstructured documents (question papers).

Supports PDF (pypdf), DOCX (python-docx), TXT/MD, CSV, and images (OCR with Tesseract
if it is installed - optional). Raw text is then segmented into individual questions
using numbering patterns (Q1. / 1) / (a) / i.) and cleaned of marks, headers and options.
"""
from __future__ import annotations

import io
import re
import shutil
from pathlib import Path

Q_START = re.compile(
    r"(?im)^\s*(?:q(?:uestion)?\s*\.?\s*\d+\s*[\.\):\-]?|\d{1,3}\s*[\.\)]|\(\d{1,3}\))\s+")
SUB_Q = re.compile(r"(?im)^\s*\(?[a-h]\)\s+|^\s*\(?(?:i|ii|iii|iv|v|vi|vii|viii)\)\s+")
OPTION = re.compile(r"(?im)^\s*\(?[A-Ea-e][\)\.]\s+.{0,45}$")
HEADER = re.compile(r"(?im)^\s*(time\s*:|max(imum)?\.?\s*marks|total marks|page \d+|section [a-z]\b|"
                    r"instructions?|roll no|name\s*:|date\s*:|general instructions).*$")
MARKS = re.compile(r"[\[\(]\s*\d+\s*(marks?|m)\s*[\]\)]", re.I)


def read_text(path_or_bytes, filename: str | None = None) -> str:
    """Return the raw text of a document. Accepts a path or (bytes + filename)."""
    if isinstance(path_or_bytes, (str, Path)):
        path = Path(path_or_bytes)
        data, name = path.read_bytes(), path.name
    else:
        data, name = path_or_bytes, filename or "upload.txt"
    ext = name.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    if ext == "docx":
        import docx
        return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    if ext in ("png", "jpg", "jpeg", "tif", "tiff", "bmp"):
        if not shutil.which("tesseract"):
            raise RuntimeError("OCR needs Tesseract: brew install tesseract && pip install pytesseract")
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(io.BytesIO(data)), lang="eng+hin")
    return data.decode("utf-8", errors="ignore")


def segment_questions(text: str, min_len: int = 12) -> list[str]:
    """Split raw paper text into individual questions."""
    text = text.replace("\r", "\n")
    text = HEADER.sub("", text)
    starts = [m.start() for m in Q_START.finditer(text)]
    if len(starts) >= 2:
        chunks = [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]
    else:  # fall back: one question per non-empty line / sentence ending with '?'
        chunks = [l for l in re.split(r"\n\s*\n|\n(?=[A-Z])", text) if l.strip()]
    out = []
    for c in chunks:
        c = Q_START.sub("", c, count=1)
        c = OPTION.sub("", c)              # drop MCQ options
        c = MARKS.sub("", c)
        c = re.sub(r"\s+", " ", c).strip(" -:")
        # split sub-questions (a), (b) ... into separate items if present
        parts = [p.strip() for p in re.split(r"\s\(?[a-h]\)\s", " " + c) if p.strip()]
        for p in (parts if len(parts) > 1 else [c]):
            if len(p) >= min_len:
                out.append(p)
    return out


def extract_questions(path_or_bytes, filename: str | None = None) -> list[str]:
    name = filename or str(path_or_bytes)
    if name.lower().endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(path_or_bytes if isinstance(path_or_bytes, (str, Path)) else io.BytesIO(path_or_bytes))
        col = next((c for c in df.columns if c.lower() in ("question", "questions", "text")), df.columns[0])
        return df[col].dropna().astype(str).tolist()
    return segment_questions(read_text(path_or_bytes, filename))
