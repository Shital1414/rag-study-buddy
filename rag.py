"""
rag.py - Core Retrieval-Augmented Generation (RAG) engine for RAG Study Buddy.

Pipeline
    1. Load      : read .txt / .md / .pdf files
    2. Chunk     : split into overlapping word windows
    3. Index     : TF-IDF vectors (unigrams + bigrams)
    4. Retrieve  : cosine-similarity top-k search
    5. Generate  : LLM answer grounded in retrieved chunks (Gemini or Groq),
                   or an offline extractive answer when no API key is set
"""
from __future__ import annotations

import io
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------- #
# 1. Loading
# --------------------------------------------------------------------------- #

def load_text(name: str, data: bytes) -> str:
    """Return plain text from an uploaded file (txt, md or pdf)."""
    lower = name.lower()
    if lower.endswith(".pdf"):
        from pypdf import PdfReader  # imported lazily so txt-only use needs no pypdf

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return data.decode("utf-8", errors="ignore")


# --------------------------------------------------------------------------- #
# 2. Chunking
# --------------------------------------------------------------------------- #

@dataclass
class Chunk:
    id: int
    source: str
    text: str


def chunk_text(text: str, source: str, chunk_words: int = 110, overlap: int = 25,
               start_id: int = 0) -> list[Chunk]:
    """Split text into overlapping chunks, preferring paragraph boundaries.

    Paragraphs are packed together until chunk_words is reached. A long
    paragraph is split with a sliding window that overlaps by `overlap` words.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if buf:
            chunks.append(Chunk(start_id + len(chunks), source, " ".join(buf)))
        buf, buf_len = [], 0

    for para in paragraphs:
        words = para.split()
        if len(words) > chunk_words:
            flush()
            step = max(1, chunk_words - overlap)
            for i in range(0, len(words), step):
                piece = words[i:i + chunk_words]
                chunks.append(Chunk(start_id + len(chunks), source, " ".join(piece)))
                if i + chunk_words >= len(words):
                    break
            continue
        if buf_len + len(words) > chunk_words:
            flush()
        buf.append(para)
        buf_len += len(words)
    flush()
    return chunks


# --------------------------------------------------------------------------- #
# 3 + 4. Index and retrieve
# --------------------------------------------------------------------------- #

@dataclass
class Hit:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("No text found to index.")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2),
            sublinear_tf=True, min_df=1,
        )
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def search(self, query: str, k: int = 3) -> list[Hit]:
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        order = np.argsort(-sims)[:k]
        return [Hit(self.chunks[i], float(sims[i])) for i in order]


# --------------------------------------------------------------------------- #
# 5. Generation
# --------------------------------------------------------------------------- #

NOT_FOUND = "I couldn't find that in the uploaded documents. Try rephrasing, or upload a document that covers this topic."
MIN_SCORE = 0.08  # below this the question is treated as out-of-scope

PROMPT = """You are a study assistant. Answer the question using ONLY the context below.
If the context does not contain the answer, reply exactly: "{not_found}"
Keep the answer clear and concise (max 5 sentences). Cite sources as [1], [2] matching the context numbers.

Context:
{context}

Question: {question}
Answer:"""


def build_prompt(question: str, hits: list[Hit]) -> str:
    context = "\n\n".join(f"[{i}] ({h.chunk.source}) {h.chunk.text}" for i, h in enumerate(hits, 1))
    return PROMPT.format(not_found=NOT_FOUND, context=context, question=question)


def _post_json(url: str, payload: dict, headers: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=40) as resp:
        return json.loads(resp.read().decode())


def call_gemini(prompt: str, api_key: str) -> str:
    model = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    out = _post_json(
        url,
        {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}},
        {"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    return out["candidates"][0]["content"]["parts"][0]["text"].strip()


def call_groq(prompt: str, api_key: str) -> str:
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    out = _post_json(
        "https://api.groq.com/openai/v1/chat/completions",
        {"model": model, "temperature": 0.2, "messages": [{"role": "user", "content": prompt}]},
        {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    return out["choices"][0]["message"]["content"].strip()


def _sentences(text: str) -> list[str]:
    # keep real sentences only (headings have no closing punctuation)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
            if len(s.strip()) > 20 and s.strip()[-1] in ".!?\""]


def extractive_answer(question: str, hits: list[Hit], n_sentences: int = 3) -> str:
    """Offline fallback: return the most question-relevant sentences from the top chunks."""
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    cands = [(i, s) for i, h in enumerate(hits, 1) for s in _sentences(h.chunk.text)]
    if not cands:
        return hits[0].chunk.text
    try:
        mat = vec.fit_transform([s for _, s in cands] + [question])
    except ValueError:
        return hits[0].chunk.text
    sims = cosine_similarity(mat[-1], mat[:-1]).ravel()
    top = np.argsort(-sims)[:n_sentences]
    cutoff = 0.5 * sims[top[0]]  # drop sentences much less relevant than the best one
    best = sorted(j for j in top if sims[j] >= cutoff and sims[j] > 0)
    if not best:
        return hits[0].chunk.text
    return " ".join(f"{cands[j][1]} [{cands[j][0]}]" for j in best)


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    mode: str          # "gemini", "groq", "extractive" or "not_found"
    top_score: float = 0.0
    error: str = field(default="")


class StudyBuddy:
    def __init__(self, chunks: list[Chunk]):
        self.retriever = Retriever(chunks)

    @classmethod
    def from_texts(cls, docs: dict[str, str], chunk_words: int = 110, overlap: int = 25):
        chunks: list[Chunk] = []
        for name, text in docs.items():
            chunks += chunk_text(text, name, chunk_words, overlap, start_id=len(chunks))
        return cls(chunks)

    def ask(self, question: str, k: int = 3, use_llm: bool = True) -> Answer:
        hits = self.retriever.search(question, k)
        top = hits[0].score if hits else 0.0
        if not hits or top < MIN_SCORE:
            return Answer(NOT_FOUND, [], "not_found", top)

        gemini_key = os.getenv("GEMINI_API_KEY", "")
        groq_key = os.getenv("GROQ_API_KEY", "")
        if use_llm and (gemini_key or groq_key):
            prompt = build_prompt(question, hits)
            try:
                if gemini_key:
                    return Answer(call_gemini(prompt, gemini_key), hits, "gemini", top)
                return Answer(call_groq(prompt, groq_key), hits, "groq", top)
            except (urllib.error.URLError, KeyError, IndexError, TimeoutError, json.JSONDecodeError) as e:
                # Never crash the app: fall back to the offline answer and report why.
                return Answer(extractive_answer(question, hits), hits, "extractive", top,
                              error=f"LLM call failed ({type(e).__name__}); showing extractive answer.")
        return Answer(extractive_answer(question, hits), hits, "extractive", top)
