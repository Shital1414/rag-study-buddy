# 📚 RAG Study Buddy

A Retrieval-Augmented Generation (RAG) chatbot that answers questions **only from your own documents** (PDF / TXT / MD) and shows the exact passages it used as sources.

Built as the capstone project for *From Tokens to Transformers: A Course on GenAI x LLM Mastery* (Consulting & Analytics Club, IIT Guwahati).

GitHub: https://github.com/Shital1414/rag-study-buddy

## Features
- Upload PDFs, text or markdown notes (or use the built-in sample notes on GenAI/LLMs)
- Paragraph-aware chunking with adjustable chunk size and overlap
- TF-IDF (unigram + bigram) retrieval with cosine similarity and top-k search
- Source citations `[1] [2]` with similarity scores for every answer
- **Out-of-scope detection**: refuses to answer when the document doesn't cover the question (reduces hallucination)
- Works **offline with no API key** (extractive answers); optionally plug in a free **Gemini** or **Groq** key for fluent LLM-written answers
- Built-in evaluation script (Hit@k, MRR, out-of-scope refusal rate, chunk-size ablation)

## Architecture
```
Documents -> load -> chunk -> TF-IDF index
                                  |
Question -> TF-IDF vector -> cosine top-k -> score check (refuse if low)
                                  |
                  prompt = context + question -> LLM (Gemini/Groq)
                                  |            or extractive fallback
                          Answer + cited sources
```

## Quick start
```bash
git clone https://github.com/Shital1414/rag-study-buddy.git
cd rag-study-buddy
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501. To use an LLM, choose *Gemini* or *Groq* in the sidebar and paste a free API key
(or set `GEMINI_API_KEY` / `GROQ_API_KEY` environment variables).

## Evaluate retrieval
```bash
python evaluate.py
```
Runs 24 in-scope and 6 out-of-scope questions against the sample notes for four chunk sizes and writes `results.json`.

## Project structure
| File | Purpose |
|---|---|
| `app.py` | Streamlit chat interface |
| `rag.py` | Loading, chunking, retrieval, generation |
| `evaluate.py` | Retrieval evaluation and chunk-size ablation |
| `data/llm_notes.txt` | Sample study notes (NLP, Transformers, LLMs, RAG, agents) |
| `results.json` | Saved evaluation output |

## Limitations
- TF-IDF matches words, not meaning, so heavy paraphrasing can reduce recall. Swapping in dense embeddings (e.g. sentence-transformers + FAISS) is the natural next step.
- Scanned PDFs without a text layer are not supported (no OCR).
- The evaluation set is small and written by the author; the LLM (Gemini/Groq) path was not part of the automated evaluation.
