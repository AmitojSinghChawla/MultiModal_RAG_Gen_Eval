"""
06_evaluate_gpt4o.py
--------------------
Unified evaluation using GPT-4O Mini for answer generation.

For every question in ground_truth.json:
  1. Run all 4 retrieval methods
  2. Compute retrieval metrics (Recall@k, Precision@k, MRR, nDCG@k)
  3. Generate answer with GPT-4O Mini (DIRECT multimodal):
       - Send text, tables, and images DIRECTLY to GPT-4O Mini in one call
       - No separate vision model needed
  4. Score answers with RAGAS

This matches the actual chatbot behavior (single multimodal LLM call).

Usage:
    export OPENAI_API_KEY=your_key_here
    python 06_evaluate_gpt4o.py
"""

import json
import math
import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from retrieve import load_indexes, retrieve
from dotenv import load_dotenv
load_dotenv(verbose=True)

QUESTIONS_FILE = "ground_truth.json"
RESULTS_FILE   = "evaluation_results_gpt4o.json"

METHODS = ["bm25", "dense", "hybrid", "hybrid_reranker"]
TOP_K   = 5
MODEL   = "gpt-4o-mini"  # Used for answer generation
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# ─────────────────────────────────────────
# Utility
# ─────────────────────────────────────────

def safe(val) -> float | None:
    """Convert a RAGAS metric value to float, returning None if NaN."""
    try:
        f = float(val)
        return round(f, 4) if not math.isnan(f) else None
    except (TypeError, ValueError):
        return None


def load_questions() -> list:
    """Load evaluation questions from ground_truth.json"""
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────
# Retrieval metrics
# ─────────────────────────────────────────

def recall_at_k(retrieved: list, relevant_ids: list, k: int) -> float:
    top_k_ids = {r["chunk_id"] for r in retrieved[:k]}
    hits = len(top_k_ids & set(relevant_ids))
    return hits / len(relevant_ids) if relevant_ids else 0.0


def precision_at_k(retrieved: list, relevant_ids: list, k: int) -> float:
    top_k_ids = {r["chunk_id"] for r in retrieved[:k]}
    hits = len(top_k_ids & set(relevant_ids))
    return hits / k


def mrr(retrieved: list, relevant_ids: list) -> float:
    for result in retrieved:
        if result["chunk_id"] in relevant_ids:
            return 1.0 / result["rank"]
    return 0.0


def ndcg_at_k(retrieved: list, relevant_ids: list, k: int) -> float:
    top_k_ids = [r["chunk_id"] for r in retrieved[:k]]
    dcg  = sum(
        1.0 / math.log2(i + 2)
        for i, cid in enumerate(top_k_ids)
        if cid in relevant_ids
    )
    idcg = sum(
        1.0 / math.log2(i + 2)
        for i in range(min(len(relevant_ids), k))
    )
    return dcg / idcg if idcg > 0 else 0.0


def compute_retrieval_metrics(results: list, relevant_ids: list, k: int = TOP_K) -> dict:
    return {
        f"recall@{k}"   : round(recall_at_k(results, relevant_ids, k), 4),
        f"precision@{k}": round(precision_at_k(results, relevant_ids, k), 4),
        "mrr"           : round(mrr(results, relevant_ids), 4),
        f"ndcg@{k}"     : round(ndcg_at_k(results, relevant_ids, k), 4),
    }


# ─────────────────────────────────────────
# Answer generation with GPT-4O Mini
# ─────────────────────────────────────────

def generate_answer(question: str, retrieved_chunks: list) -> tuple[str, list[str]]:
    """
    Generate answer by sending everything directly to GPT-4O Mini.

    Images are sent as base64, text/tables as text - all in ONE call.

    Returns
    -------
    answer : str
        Generated answer
    context_texts : list[str]
        Text-only version of context for RAGAS
        (RAGAS requires text, so images are represented by their
        retrieval_text descriptions)
    """

    llm = ChatOpenAI(model=MODEL, temperature=0.0,api_key=OPENAI_API_KEY)

    # Build multimodal content for GPT-4O Mini
    content = []
    context_texts = []  # For RAGAS - text only

    for chunk in retrieved_chunks:
        source = chunk["source_pdf"]
        modality = chunk["modality"]

        if modality == "image" and chunk.get("image_b64"):
            # ── IMAGE: Send to GPT-4O Mini directly ──
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{chunk['image_b64']}",
                    "detail": "high"
                }
            })
            content.append({
                "type": "text",
                "text": f"[Image from {source}]"
            })
            # For RAGAS: use the pre-generated text description
            context_texts.append(chunk["retrieval_text"])

        elif modality == "table":
            text = chunk.get("raw_text") or chunk["retrieval_text"]
            content.append({
                "type": "text",
                "text": f"[TABLE from {source}]\n{text}\n"
            })
            context_texts.append(text)

        else:
            # text chunk
            text = chunk.get("raw_text") or chunk["retrieval_text"]
            content.append({
                "type": "text",
                "text": f"[TEXT from {source}]\n{text}\n"
            })
            context_texts.append(text)

    # Add instruction and question
    content.append({
        "type": "text",
        "text": (
            "\n─────────────────────────────────────\n\n"
            "Answer the question based only on the context above. "
            "Be concise and accurate. "
            "If the context does not contain the answer, say 'Not found in context.'\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
    })

    messages = [{"role": "user", "content": content}]

    response = llm.invoke(messages)
    answer = StrOutputParser().invoke(response)

    return answer, context_texts


# ─────────────────────────────────────────
# RAGAS evaluation
# ─────────────────────────────────────────

def run_ragas(questions, answers, contexts, ground_truths):
    """Run RAGAS evaluation on the generated answers"""
    data = {
        "question"    : questions,
        "answer"      : answers,
        "contexts"    : contexts,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data)
    return evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )


# ─────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────

def main():

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("=" * 60)
        print("ERROR: OPENAI_API_KEY not set")
        print("=" * 60)
        print("\nSet it with:")
        print("  export OPENAI_API_KEY=your_key_here")
        print("\nBoth answer generation and RAGAS require OpenAI API access.")
        return

    questions = load_questions()
    print(f"Loaded {len(questions)} gold questions\n")

    indexes = load_indexes()
    all_results = {}

    for method in METHODS:
        print(f"\n{'='*60}")
        print(f"Evaluating: {method.upper()}")
        print(f"{'='*60}")

        method_results      = []
        ragas_questions     = []
        ragas_answers       = []
        ragas_contexts      = []
        ragas_ground_truths = []

        for i, q in enumerate(questions, 1):
            question     = q["question"]
            relevant_ids = q["relevant_chunk_ids"]
            ground_truth = q["ground_truth"]

            print(f"\n[{i}/{len(questions)}] {q['question_id']}: {question[:60]}...")

            # ── Retrieve ─────────────────────────────────────────────────
            retrieved = retrieve(
                question,
                method=method,
                indexes=indexes,
                top_k=TOP_K
            )

            # ── Retrieval metrics ─────────────────────────────────────────
            retrieval_metrics = compute_retrieval_metrics(retrieved, relevant_ids, k=TOP_K)
            print(f"  Retrieval: {retrieval_metrics}")

            # ── Count modalities ──────────────────────────────────────────
            n_images = sum(1 for c in retrieved if c.get("image_b64"))
            n_text   = sum(1 for c in retrieved if c["modality"] in ("text", "table"))

            if n_images > 0:
                print(f"  Context  : {n_images} image(s) + {n_text} text chunk(s)")

            # ── Generate answer ───────────────────────────────────────────
            print(f"  Generating answer with {MODEL}...", end=" ", flush=True)
            answer, context_texts = generate_answer(question, retrieved)
            print("done")
            print(f"  Answer   : {answer[:100]}...")

            # ── Collect for RAGAS ─────────────────────────────────────────
            ragas_questions.append(question)
            ragas_answers.append(answer)
            ragas_contexts.append(context_texts)
            ragas_ground_truths.append(ground_truth)

            method_results.append({
                "question_id"      : q["question_id"],
                "question"         : question,
                "modality"         : q["modality"],
                "retrieved_ids"    : [r["chunk_id"] for r in retrieved],
                "retrieval_metrics": retrieval_metrics,
                "generated_answer" : answer,
                "ground_truth"     : ground_truth,
                "n_image_chunks"   : n_images,
                "n_text_chunks"    : n_text,
            })

        # ── RAGAS ─────────────────────────────────────────────────────────
        print(f"\n  Running RAGAS evaluation...")
        try:
            ragas_scores = run_ragas(
                ragas_questions,
                ragas_answers,
                ragas_contexts,
                ragas_ground_truths
            )
            ragas_df = ragas_scores.to_pandas()

            for i, row in ragas_df.iterrows():
                method_results[i]["ragas_metrics"] = {
                    "faithfulness"     : safe(row.get("faithfulness")),
                    "answer_relevancy" : safe(row.get("answer_relevancy")),
                    "context_precision": safe(row.get("context_precision")),
                    "context_recall"   : safe(row.get("context_recall")),
                }

            print("  RAGAS evaluation complete ✓")

        except Exception as e:
            print(f"  WARNING: RAGAS failed — {e}")
            print("  Retrieval metrics are still saved.")

        all_results[method] = method_results

    # ── Save ──────────────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*60}")
    print(f"Results saved → {RESULTS_FILE}")
    print(f"{'='*60}")

    print_summary(all_results)


# ─────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────

def print_summary(all_results: dict) -> None:
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")

    for method, results in all_results.items():
        n = len(results)
        if n == 0:
            continue

        avg_recall    = sum(r["retrieval_metrics"][f"recall@{TOP_K}"]    for r in results) / n
        avg_precision = sum(r["retrieval_metrics"][f"precision@{TOP_K}"] for r in results) / n
        avg_mrr       = sum(r["retrieval_metrics"]["mrr"]                 for r in results) / n
        avg_ndcg      = sum(r["retrieval_metrics"][f"ndcg@{TOP_K}"]      for r in results) / n

        ragas_results = [r for r in results if "ragas_metrics" in r]

        def avg_metric(key: str) -> float:
            vals = [
                r["ragas_metrics"][key]
                for r in ragas_results
                if r["ragas_metrics"].get(key) is not None
            ]
            return sum(vals) / len(vals) if vals else 0.0

        print(f"\nMethod: {method.upper()}")
        print(f"  Retrieval Metrics")
        print(f"    Recall@{TOP_K}         : {avg_recall:.4f}")
        print(f"    Precision@{TOP_K}      : {avg_precision:.4f}")
        print(f"    MRR               : {avg_mrr:.4f}")
        print(f"    nDCG@{TOP_K}          : {avg_ndcg:.4f}")

        if ragas_results:
            print(f"  Generation Metrics (RAGAS)")
            print(f"    Faithfulness      : {avg_metric('faithfulness'):.4f}")
            print(f"    Answer Relevancy  : {avg_metric('answer_relevancy'):.4f}")
            print(f"    Context Precision : {avg_metric('context_precision'):.4f}")
            print(f"    Context Recall    : {avg_metric('context_recall'):.4f}")
        else:
            print(f"  Generation Metrics: Not available")

    print(f"\n{'='*70}")
    print(f"Model used: {MODEL}")
    print(f"Architecture: Direct multimodal (one LLM call per question)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()