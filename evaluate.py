"""
evaluate.py - Measures retrieval quality of RAG Study Buddy (no API key needed).

Metrics
    Hit@1 / Hit@3 : correct passage is the top result / in the top 3
    MRR           : mean reciprocal rank of the first correct passage
    Out-of-scope  : % of unrelated questions correctly refused

A retrieved chunk counts as correct if it contains the `evidence` phrase.
Also runs a chunk-size ablation. Results are saved to results.json.
"""
import json
import sys
from pathlib import Path

from rag import StudyBuddy

DOC = Path(__file__).parent / "data" / "llm_notes.txt"

# (question, evidence phrase that must appear in a correct chunk)
IN_SCOPE = [
    ("What is tokenization?", "splitting raw text into smaller units"),
    ("Which tokenization methods do modern language models use?", "Byte Pair Encoding"),
    ("How is similarity between embeddings measured?", "cosine similarity"),
    ("Which paper introduced the Transformer?", "Attention Is All You Need"),
    ("Why did Transformers replace RNNs?", "processes all tokens in parallel"),
    ("What are query, key and value in self-attention?", "query, a key and a value"),
    ("What is multi-head attention?", "Multi-head attention runs several attention operations"),
    ("What are the stages of LLM training?", "pre-training"),
    ("What is RLHF?", "Reinforcement Learning from Human Feedback"),
    ("What is chain-of-thought prompting?", "chain-of-thought prompting"),
    ("What does temperature do when generating text?", "Temperature controls randomness"),
    ("What is top-p sampling?", "nucleus"),
    ("Why do LLMs hallucinate?", "predict plausible text"),
    ("How can hallucinations be reduced?", "grounding the model in external documents"),
    ("What is Retrieval-Augmented Generation?", "combines a retrieval system with an LLM"),
    ("What are the phases of a RAG pipeline?", "indexing phase"),
    ("Which vector databases are mentioned?", "FAISS is a library from Meta"),
    ("What is TF-IDF or BM25?", "TF-IDF or BM25"),
    ("What is the ReAct pattern for agents?", "ReAct"),
    ("What are the drawbacks of multi-agent systems?", "adds cost, latency"),
    ("What is hit rate in retrieval evaluation?", "hit rate"),
    ("What does faithfulness mean for LLM answers?", "supported by the retrieved context"),
    ("When should I use fine-tuning instead of RAG?", "teaching a style, format"),
    ("Which frameworks help build multi-agent workflows?", "LangGraph, CrewAI"),
]

OUT_OF_SCOPE = [
    "Who won the cricket world cup in 2011?",
    "What is the recipe for butter chicken?",
    "How do I renew my passport in India?",
    "What is the capital of Australia?",
    "Explain the plot of Interstellar.",
    "What is the boiling point of mercury?",
]


def run(chunk_words: int, overlap: int, k: int = 3, verbose: bool = False) -> dict:
    sb = StudyBuddy.from_texts({DOC.name: DOC.read_text(encoding="utf-8")}, chunk_words, overlap)
    hit1 = hit3 = 0
    rr = 0.0
    misses = []
    for q, ev in IN_SCOPE:
        hits = sb.retriever.search(q, k)
        rank = next((i for i, h in enumerate(hits, 1) if ev.lower() in h.chunk.text.lower()), None)
        if rank == 1:
            hit1 += 1
        if rank:
            hit3 += 1
            rr += 1 / rank
        else:
            misses.append(q)
    refused = sum(1 for q in OUT_OF_SCOPE if sb.ask(q, use_llm=False).mode == "not_found")
    n = len(IN_SCOPE)
    res = {
        "chunk_words": chunk_words, "overlap": overlap, "n_chunks": len(sb.retriever.chunks),
        "hit@1": round(hit1 / n, 3), "hit@3": round(hit3 / n, 3), "mrr": round(rr / n, 3),
        "out_of_scope_refused": round(refused / len(OUT_OF_SCOPE), 3),
        "n_questions": n, "n_out_of_scope": len(OUT_OF_SCOPE), "misses": misses,
    }
    if verbose:
        print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    configs = [(50, 10), (110, 25), (200, 40), (350, 60)]
    results = [run(cw, ov) for cw, ov in configs]
    print(f"{'chunk':>6} {'overlap':>8} {'#chunks':>8} {'Hit@1':>7} {'Hit@3':>7} {'MRR':>6} {'Refused':>8}")
    for r in results:
        print(f"{r['chunk_words']:>6} {r['overlap']:>8} {r['n_chunks']:>8} {r['hit@1']:>7.2f} "
              f"{r['hit@3']:>7.2f} {r['mrr']:>6.2f} {r['out_of_scope_refused']:>8.2f}")
    Path(__file__).with_name("results.json").write_text(json.dumps(results, indent=2))
    print("\nSaved results.json")
    if "--misses" in sys.argv:
        for r in results:
            print(r["chunk_words"], r["misses"])
