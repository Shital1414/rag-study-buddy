"""
make_report.py - Generates the capstone project report PDF from live results.

Usage:  python make_report.py "Your Name" "Your Roll/Unstop ID (optional)"
Requires: pip install reportlab   (run evaluate.py first to create results.json)
"""
import json
import sys
from pathlib import Path

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from rag import StudyBuddy

ROOT = Path(__file__).parent
NAME = sys.argv[1] if len(sys.argv) > 1 else "[Your Name]"
ID = sys.argv[2] if len(sys.argv) > 2 else ""
GITHUB = sys.argv[3] if len(sys.argv) > 3 else ""
OUT = ROOT / "RAG_Study_Buddy_Capstone_Report.pdf"

results = json.loads((ROOT / "results.json").read_text())
default = next(r for r in results if r["chunk_words"] == 110)

# ------------------------------ styles ------------------------------ #
INK, ACCENT, MUTED = colors.HexColor("#1f2937"), colors.HexColor("#1d4ed8"), colors.HexColor("#6b7280")
ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=10.5, leading=15.5,
                      textColor=INK, spaceAfter=7)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=20,
                    textColor=ACCENT, spaceBefore=14, spaceAfter=8, keepWithNext=1)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=16,
                    textColor=INK, spaceBefore=10, spaceAfter=5, keepWithNext=1)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=2, spaceAfter=3)
small = ParagraphStyle("small", parent=body, fontSize=9, leading=12, textColor=MUTED)
cell = ParagraphStyle("cell", parent=body, fontSize=9.5, leading=12.5, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
title = ParagraphStyle("title", parent=h1, fontSize=26, leading=32, alignment=TA_CENTER, textColor=INK)
sub = ParagraphStyle("sub", parent=body, fontSize=13, leading=18, alignment=TA_CENTER, textColor=MUTED)
code = ParagraphStyle("code", parent=ss["Code"], fontName="Courier", fontSize=8.5, leading=11,
                      backColor=colors.HexColor("#f3f4f6"), borderPadding=6, leftIndent=6)

P = lambda t, s=body: Paragraph(t, s)
B = lambda t: Paragraph(t, bullet, bulletText="•")


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def table(rows, widths, header=True, align_center_from=None):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), ACCENT), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    if align_center_from is not None:
        st.append(("ALIGN", (align_center_from, 0), (-1, -1), "CENTER"))
    t.setStyle(TableStyle(st))
    return t


# ------------------------------ figures ------------------------------ #
def architecture() -> Drawing:
    d = Drawing(16.5 * cm, 6.2 * cm)

    def box(x, y, w, h, text, fill):
        d.add(Rect(x, y, w, h, rx=6, ry=6, fillColor=fill, strokeColor=ACCENT, strokeWidth=0.8))
        lines = text.split("\n")
        for i, ln in enumerate(lines):
            d.add(String(x + w / 2, y + h / 2 + (len(lines) - 1) * 5 - i * 11 - 3, ln,
                         textAnchor="middle", fontName="Helvetica", fontSize=8.5, fillColor=INK))

    def arrow(x1, y1, x2, y2):
        d.add(Line(x1, y1, x2, y2, strokeColor=MUTED, strokeWidth=1.1))
        if abs(y2 - y1) < 1:  # horizontal
            d.add(Polygon([x2, y2, x2 - 5, y2 + 3, x2 - 5, y2 - 3], fillColor=MUTED, strokeColor=MUTED))
        else:  # vertical (downwards)
            d.add(Polygon([x2, y2, x2 - 3, y2 + 5, x2 + 3, y2 + 5], fillColor=MUTED, strokeColor=MUTED))

    light, mid = colors.HexColor("#eff6ff"), colors.HexColor("#dbeafe")
    d.add(String(0, 5.6 * cm, "INDEXING (once per document set)", fontName="Helvetica-Bold", fontSize=8, fillColor=MUTED))
    w, h, y = 2.6 * cm, 1.3 * cm, 4.0 * cm
    xs = [0, 3.5 * cm, 7.0 * cm, 10.5 * cm, 14.0 * cm]
    labels = ["Documents\nPDF / TXT / MD", "Load &\nextract text", "Chunk\n(size + overlap)", "TF-IDF\nvectorizer", "Vector index\n(sparse matrix)"]
    for x, lb in zip(xs, labels):
        box(x, y, w, h, lb, light)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(a + w, y + h / 2, b, y + h / 2)

    d.add(String(0, 2.75 * cm, "QUERY (every question)", fontName="Helvetica-Bold", fontSize=8, fillColor=MUTED))
    y2 = 1.2 * cm
    labels2 = ["User\nquestion", "Cosine\ntop-k search", "Score check\n(refuse if low)", "Prompt: context\n+ question", "LLM or extractive\n-> cited answer"]
    for x, lb in zip(xs, labels2):
        box(x, y2, w, h, lb, mid)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(a + w, y2 + h / 2, b, y2 + h / 2)
    # index feeds search
    LINE_Y = 3.3 * cm
    d.add(Line(xs[4] + w / 2, y, xs[4] + w / 2, LINE_Y, strokeColor=MUTED, strokeWidth=0.8, strokeDashArray=[3, 2]))
    d.add(Line(xs[4] + w / 2, LINE_Y, xs[1] + w / 2, LINE_Y, strokeColor=MUTED, strokeWidth=0.8, strokeDashArray=[3, 2]))
    arrow(xs[1] + w / 2, LINE_Y, xs[1] + w / 2, y2 + h)
    return d


def metrics_chart() -> Drawing:
    d = Drawing(16 * cm, 6.5 * cm)
    ch = VerticalBarChart()
    ch.x, ch.y, ch.width, ch.height = 35, 30, 330, 140
    ch.data = [[r["hit@1"] for r in results], [r["hit@3"] for r in results], [r["mrr"] for r in results]]
    ch.categoryAxis.categoryNames = [f"{r['chunk_words']} words\n({r['n_chunks']} chunks)" for r in results]
    ch.categoryAxis.labels.fontSize = 8
    ch.valueAxis.valueMin, ch.valueAxis.valueMax, ch.valueAxis.valueStep = 0, 1, 0.25
    ch.valueAxis.labels.fontSize = 8
    ch.valueAxis.visibleGrid, ch.valueAxis.gridStrokeColor = True, colors.HexColor("#e5e7eb")
    cols = [colors.HexColor("#1d4ed8"), colors.HexColor("#0891b2"), colors.HexColor("#f59e0b")]
    for i, c in enumerate(cols):
        ch.bars[i].fillColor = c
        ch.bars[i].strokeColor = None
    ch.barWidth, ch.groupSpacing = 12, 14
    d.add(ch)
    lg = Legend()
    lg.x, lg.y, lg.fontSize = 380, 150, 8.5
    lg.alignment, lg.columnMaximum = "right", 3
    lg.colorNamePairs = [(c, n) for c, n in zip(cols, ["Hit@1", "Hit@3", "MRR"])]
    d.add(lg)
    return d


# ------------------------------ live sample interactions ------------------------------ #
sb = StudyBuddy.from_texts({"llm_notes.txt": (ROOT / "data" / "llm_notes.txt").read_text(encoding="utf-8")})
samples = []
for q in ["What is Retrieval-Augmented Generation?", "How does self-attention work?",
          "What is the difference between fine-tuning and RAG?", "Who won the 2011 cricket world cup?"]:
    a = sb.ask(q, use_llm=False)
    samples.append((q, a))

# ------------------------------ document ------------------------------ #
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, "RAG Study Buddy - Capstone Project Report")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=2 * cm, bottomMargin=2 * cm,
                        title="RAG Study Buddy - Capstone Project Report", author=NAME)
W = A4[0] - 4 * cm
s = []

# --- cover ---
s += [Spacer(1, 4.2 * cm), P("RAG Study Buddy", title),
      P("A Retrieval-Augmented Generation chatbot that answers questions from your own study documents, with source citations", sub),
      Spacer(1, 1.2 * cm), P("Capstone Project Report", sub),
      P("From Tokens to Transformers: A Course on GenAI x LLM Mastery", sub),
      P("Consulting &amp; Analytics Club, IIT Guwahati", sub), Spacer(1, 1.6 * cm),
      P(f"<b>Submitted by:</b> {esc(NAME)}" + (f" &nbsp;|&nbsp; <b>ID:</b> {esc(ID)}" if ID else ""), sub),
      P("<b>Date:</b> 19 October 2026", sub)]
if GITHUB:
    s.append(P(f"<b>GitHub:</b> {esc(GITHUB)}", sub))
s.append(PageBreak())

# --- 1 abstract ---
s += [P("1. Abstract", h1),
      P("Large Language Models (LLMs) write fluent text but can state wrong facts with confidence, and they know nothing about "
        "a student's own notes. This project builds <b>RAG Study Buddy</b>, a Retrieval-Augmented Generation system that answers "
        "questions using only the documents the user uploads (PDF, TXT or Markdown). Documents are split into overlapping chunks, "
        "indexed with TF-IDF (unigrams and bigrams) and searched with cosine similarity. The best passages are given to an LLM "
        "(Gemini or Groq, optional) or to an offline extractive answerer, and every answer lists the passages it came from. "
        "If no passage is similar enough, the system refuses instead of guessing."),
      P(f"On a 24-question test set built from the sample notes, the default configuration (110-word chunks) achieved "
        f"<b>Hit@1 = {default['hit@1']:.2f}</b>, <b>Hit@3 = {default['hit@3']:.2f}</b> and <b>MRR = {default['mrr']:.2f}</b>, and it "
        f"correctly refused <b>{round(default['out_of_scope_refused'] * default['n_out_of_scope'])} of {default['n_out_of_scope']}</b> off-topic questions. "
        "The complete system runs with no API key and no GPU, and comes with a Streamlit web interface and a reproducible evaluation script.")]

# --- 2 problem ---
s += [P("2. Problem Statement and Motivation", h1),
      P("Students collect lecture notes, slides and papers, but finding a specific answer inside them is slow. A general chatbot "
        "does not know these private documents and may hallucinate. The goal is a study assistant that is:"),
      B("<b>Grounded</b> - answers come from the user's documents, not from the model's memory."),
      B("<b>Verifiable</b> - every answer shows the source passages and similarity scores."),
      B("<b>Honest</b> - it says it cannot find the answer when the documents do not cover the question."),
      B("<b>Accessible</b> - it runs on a laptop for free, with an optional free LLM key for better wording.")]

# --- 3 background ---
s += [P("3. Background: Concepts from the Course Used Here", h1),
      table([[P("Concept", cellb), P("How it appears in this project", cellb)],
             [P("Tokenization", cell), P("TF-IDF tokenizes text into words and word pairs (bigrams); chunk size is measured in words. LLMs use sub-word tokens.", cell)],
             [P("Vector representations", cell), P("Each chunk and question becomes a sparse TF-IDF vector. Similarity is the cosine of the angle between vectors, the same measure used for dense embeddings.", cell)],
             [P("Retrieval-Augmented Generation", cell), P("Retrieve the most relevant chunks first, then generate an answer grounded in them (Lewis et al., 2020).", cell)],
             [P("Prompt engineering", cell), P("The prompt restricts the LLM to the given context, demands a fixed refusal sentence, limits length and asks for [1], [2] citations.", cell)],
             [P("Hallucination control", cell), P("Grounding, low temperature (0.2), a similarity threshold that triggers refusal, and visible sources.", cell)],
             [P("Evaluation", cell), P("Hit@k, Mean Reciprocal Rank and out-of-scope refusal rate, plus a chunk-size ablation.", cell)]],
            [4 * cm, W - 4 * cm])]

# --- 4 architecture ---
s += [KeepTogether([P("4. System Design", h1), P("The system has an indexing phase and a query phase, as in a standard RAG pipeline."),
      architecture(), P("Figure 1. RAG Study Buddy architecture (dashed line: the index is used by the search step).", small)])]

s += [P("4.1 Modules", h2),
      table([[P("File", cellb), P("Responsibility", cellb)],
             [P("rag.py", cell), P("Loading (txt, md, pdf), chunking, TF-IDF retrieval, prompt building, Gemini/Groq calls, extractive fallback, refusal logic.", cell)],
             [P("app.py", cell), P("Streamlit interface: file upload, settings sliders, chat history, expandable sources, optional API key.", cell)],
             [P("evaluate.py", cell), P("Retrieval metrics and chunk-size ablation; writes results.json.", cell)],
             [P("make_report.py", cell), P("Builds this PDF from the live system and the saved results.", cell)],
             [P("data/llm_notes.txt", cell), P("Sample study notes (about 1,500 words) on NLP, Transformers, LLMs, RAG, agents and evaluation.", cell)]],
            [3.6 * cm, W - 3.6 * cm])]

# --- 5 implementation ---
s += [P("5. Implementation Details", h1),
      P("5.1 Chunking", h2),
      P("Text is split on blank lines into paragraphs. Paragraphs are packed into a chunk until the word limit is reached, so a chunk "
        "usually keeps one idea together. A paragraph longer than the limit is cut with a sliding window that overlaps the previous piece "
        "by a set number of words, so a sentence at a boundary still appears whole in one chunk. Defaults: 110 words, 25 overlap."),
      P("5.2 Retrieval", h2),
      P("A scikit-learn <font face='Courier'>TfidfVectorizer</font> (English stop words removed, unigrams plus bigrams, sub-linear term frequency) "
        "indexes all chunks. A question is transformed with the same vectorizer and compared to every chunk by cosine similarity; the top-k chunks "
        "(default 3) are returned with their scores. TF-IDF was chosen because it needs no model download, is fast, and is easy to explain."),
      P("5.3 Refusal to answer (hallucination control)", h2),
      P("If the best similarity score is below 0.08, the system returns a fixed message that the answer is not in the documents. This threshold was "
        "chosen by inspecting scores of in-scope versus off-topic questions on the sample notes; it is a tunable constant (<font face='Courier'>MIN_SCORE</font>)."),
      P("5.4 Answer generation", h2),
      B("<b>LLM mode</b> (Gemini or Groq key provided): the prompt contains the numbered passages and question, tells the model to use only that context, "
        "to reply with the refusal sentence if the answer is missing, and to cite [1], [2]. Temperature is 0.2. If the API call fails, the app falls back to extractive mode and shows a warning."),
      B("<b>Extractive mode</b> (no key): the top chunks are split into sentences, sentences are ranked by TF-IDF similarity to the question, and the best ones "
        "(at most three, dropping those far less relevant than the best) are returned in reading order with their source numbers."),
      P("5.5 Web interface", h2),
      P("The Streamlit app lets the user upload several files, adjust top-k, chunk size and overlap, choose a provider, and chat. Each answer has a "
        "Sources expander showing the passages and similarity scores. The index is cached so it is rebuilt only when documents or chunk settings change.")]

# --- 6 evaluation ---
s += [PageBreak(), P("6. Evaluation", h1),
      P("6.1 Setup", h2),
      P(f"The test set has <b>{default['n_questions']} in-scope questions</b>, each paired with a phrase that must appear in a correct passage, and "
        f"<b>{default['n_out_of_scope']} out-of-scope questions</b> (for example, cricket and recipes) that the system should refuse. A retrieved chunk counts as correct "
        "if it contains the evidence phrase. Metrics:"),
      B("<b>Hit@1 / Hit@3</b> - fraction of questions where a correct chunk is ranked first / within the top 3."),
      B("<b>MRR</b> - mean of 1/rank of the first correct chunk (higher is better, maximum 1)."),
      B("<b>Refusal rate</b> - fraction of off-topic questions for which the system answers \"not found\"."),
      P("6.2 Results and chunk-size ablation", h2)]

rows = [[P(x, cellb) for x in ["Chunk size", "Overlap", "# chunks", "Hit@1", "Hit@3", "MRR", "Refusal"]]]
for r in results:
    mark = " (default)" if r["chunk_words"] == 110 else ""
    rows.append([P(f"{r['chunk_words']} words{mark}", cell), P(str(r["overlap"]), cell), P(str(r["n_chunks"]), cell),
                 P(f"{r['hit@1']:.2f}", cell), P(f"{r['hit@3']:.2f}", cell), P(f"{r['mrr']:.2f}", cell),
                 P(f"{r['out_of_scope_refused']:.2f}", cell)])
s += [table(rows, [3.8 * cm, 1.9 * cm, 2.0 * cm, 1.9 * cm, 1.9 * cm, 1.9 * cm, 2.0 * cm], align_center_from=1),
      P("Table 1. Retrieval results on the sample notes (k = 3).", small), metrics_chart(),
      P("Figure 2. Retrieval metrics by chunk size.", small),
      P("6.3 Analysis", h2),
      B("<b>Very small chunks (50 words) hurt.</b> Hit@1 falls to 0.50 because an answer is often split across chunks; two questions "
        "(\"Why do LLMs hallucinate?\" and \"Which vector databases are mentioned?\") were missed even at rank 3."),
      B(f"<b>110 words is a good balance.</b> It finds the right passage for all {default['n_questions']} questions within the top 3 and ranks it first for "
        f"{round(default['hit@1'] * default['n_questions'])} of them, while keeping passages short enough to read."),
      B("<b>Bigger chunks score higher here, but be careful.</b> With only 5 chunks in total, a top-3 search covers 60% of the whole document, so Hit@3 = 1.00 is "
        "almost guaranteed and says little. Large chunks also carry more irrelevant text into the LLM prompt. For this reason 110 words was kept as the default."),
      B("<b>All off-topic questions were refused at every chunk size</b>, showing the similarity threshold works on this document."),
      P("6.4 Limitations of this evaluation", h2),
      P("The test questions were written by the project author using the same notes, so scores are likely higher than for real users. The notes are short (about 1,500 words), "
        "and the set is small, so differences of one question move Hit@1 by about 4 percentage points. Answer quality (faithfulness and relevance) was checked "
        "manually on a few examples only, and the LLM mode (Gemini/Groq) was not part of the automated evaluation because it needs an external API key.")]

# --- 7 sample interactions ---
s += [P("7. Sample Interactions", h1),
      P("Real output of the offline (extractive) mode on the sample notes, using the default settings:")]
for q, a in samples:
    block = [P(f"<b>Q:</b> {esc(q)}", body),
             P(f"<b>A:</b> {esc(a.text)}", body)]
    if a.mode == "not_found":
        block.append(P("<i>Refused: top similarity below the threshold, so no answer was invented.</i>", small))
    else:
        block.append(P(f"<i>Mode: extractive | top similarity {a.top_score:.2f} | sources: "
                       + ", ".join(f"[{i}] chunk {h.chunk.id}" for i, h in enumerate(a.hits, 1)) + "</i>", small))
    s.append(KeepTogether(block + [Spacer(1, 4)]))
s.append(P("Note: in extractive mode, the last cited sentence is sometimes only loosely related to the question. An LLM answers more fluently and filters such sentences, "
           "which is why LLM mode is offered as an option.", small))

# --- 8 limitations / future ---
s += [P("8. Limitations and Future Work", h1),
      B("<b>Keyword matching, not meaning.</b> TF-IDF cannot match a question that uses different words from the document (for example \"car\" versus \"automobile\"). "
        "Next step: dense embeddings (sentence-transformers) stored in FAISS or Chroma, and hybrid search combining both."),
      B("<b>No OCR.</b> Scanned PDFs without a text layer cannot be read."),
      B("<b>Single-step retrieval.</b> Complex questions that need several documents could use a multi-agent design: a planner agent that splits the question, a "
        "retriever agent, and a writer agent that checks citations (the Multi-Agent Systems topic from the course)."),
      B("<b>Evaluation.</b> A larger and independently written question set, plus faithfulness scoring with an LLM judge, would give more reliable numbers."),
      B("<b>Study features.</b> Quiz and flashcard generation from the retrieved passages, and conversation memory for follow-up questions.")]

# --- 9 conclusion ---
s += [P("9. Conclusion", h1),
      P("RAG Study Buddy shows the full RAG workflow end to end: load, chunk, index, retrieve, generate and cite. Its main design choices (grounded prompts, "
        "visible sources and a refusal threshold) target the main weakness of plain LLMs, which is confident but unsupported answers. The chunk-size experiment showed that "
        "chunking has a measurable effect on retrieval quality, and the project runs fully offline so anyone can reproduce the results. "
        "The main lesson is that the quality of a RAG system depends as much on retrieval and evaluation as on the language model.")]

# --- 10 reproduce ---
s += [P("10. How to Run", h1),
      Preformatted("pip install -r requirements.txt\nstreamlit run app.py      # web app at http://localhost:8501\npython evaluate.py        # reproduce Table 1", code),
      Spacer(1, 6), P("Optional: choose Gemini or Groq in the sidebar and paste a free API key, or set GEMINI_API_KEY / GROQ_API_KEY.")]

# --- refs ---
s += [P("References", h1),
      B("Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020."),
      B("Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS 2017."),
      B("Salton, G. and Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. Information Processing &amp; Management."),
      B("Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12."),
      B("Streamlit documentation, https://docs.streamlit.io")]

doc.build(s, onFirstPage=lambda c, d: None, onLaterPages=footer)
print("Wrote", OUT)
