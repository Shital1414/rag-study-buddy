"""
app.py - Streamlit UI for RAG Study Buddy.
Run with:  streamlit run app.py
"""
import os
from pathlib import Path

import streamlit as st

from rag import StudyBuddy, load_text

st.set_page_config(page_title="RAG Study Buddy", page_icon="📚", layout="wide")
SAMPLE = Path(__file__).parent / "data" / "llm_notes.txt"

st.title("📚 RAG Study Buddy")
st.caption("Ask questions about your notes. Answers are grounded in your documents and show their sources.")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("1. Documents")
    files = st.file_uploader("Upload PDF / TXT / MD", type=["pdf", "txt", "md"], accept_multiple_files=True)
    use_sample = st.checkbox("Use built-in sample notes (GenAI & LLMs)", value=not files)

    st.header("2. Settings")
    k = st.slider("Passages to retrieve (top-k)", 1, 6, 3)
    chunk_words = st.slider("Chunk size (words)", 50, 400, 110, step=10)
    overlap = st.slider("Chunk overlap (words)", 0, 100, 25, step=5)

    st.header("3. LLM (optional)")
    provider = st.selectbox("Provider", ["None (offline extractive)", "Gemini", "Groq"])
    key = st.text_input("API key", type="password", help="Free keys: Google AI Studio (Gemini) or console.groq.com (Groq)")
    os.environ.pop("GEMINI_API_KEY", None)
    os.environ.pop("GROQ_API_KEY", None)
    if key and provider == "Gemini":
        os.environ["GEMINI_API_KEY"] = key
    elif key and provider == "Groq":
        os.environ["GROQ_API_KEY"] = key

# ---------------- Build index ----------------
docs: dict[str, str] = {}
for f in files or []:
    try:
        docs[f.name] = load_text(f.name, f.getvalue())
    except Exception as e:  # corrupted / scanned PDF etc.
        st.sidebar.error(f"Could not read {f.name}: {e}")
if use_sample and SAMPLE.exists():
    docs[SAMPLE.name] = SAMPLE.read_text(encoding="utf-8")
docs = {n: t for n, t in docs.items() if t.strip()}

if not docs:
    st.info("Upload a document or tick the sample notes in the sidebar to begin.")
    st.stop()


@st.cache_resource(show_spinner="Indexing documents...")
def build(doc_items: tuple, cw: int, ov: int) -> StudyBuddy:
    return StudyBuddy.from_texts(dict(doc_items), cw, ov)


try:
    buddy = build(tuple(sorted(docs.items())), chunk_words, overlap)
except ValueError as e:
    st.error(str(e))
    st.stop()

st.sidebar.success(f"Indexed {len(buddy.retriever.chunks)} chunks from {len(docs)} file(s)")

# ---------------- Chat ----------------
if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["a"])
        if turn["sources"]:
            with st.expander("Sources"):
                for i, (src, score, text) in enumerate(turn["sources"], 1):
                    st.markdown(f"**[{i}] {src}** (similarity {score:.2f})")
                    st.caption(text)

q = st.chat_input("Ask a question about your documents...")
if q:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ans = buddy.ask(q, k=k)
        st.write(ans.text)
        mode_label = {"gemini": "Gemini", "groq": "Groq", "extractive": "Offline extractive", "not_found": "Not found"}
        st.caption(f"Mode: {mode_label[ans.mode]} · top similarity {ans.top_score:.2f}")
        if ans.error:
            st.warning(ans.error)
        sources = [(h.chunk.source, h.score, h.chunk.text) for h in ans.hits]
        if sources:
            with st.expander("Sources"):
                for i, (src, score, text) in enumerate(sources, 1):
                    st.markdown(f"**[{i}] {src}** (similarity {score:.2f})")
                    st.caption(text)
    st.session_state.history.append({"q": q, "a": ans.text, "sources": sources})
