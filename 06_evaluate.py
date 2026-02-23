"""
06_evaluate.py
--------------
Unified evaluation — retrieval metrics + RAGAS generation metrics.

For every question in gold_questions.json:
  1. Run all 4 retrieval methods
  2. Compute Recall@k, Precision@k, MRR, nDCG@k from ranked chunk_ids
  3. Generate answer multimodally:
       - image chunks  → send real image to llava (vision LLM) for a
                         question-focused description, then use that text
       - text / table  → use raw_text directly
  4. Score the answer with RAGAS (Faithfulness, Answer Relevancy, etc.)

Results saved to evaluation_results.json and printed as a summary table.

Usage:
    python 06_evaluate.py
"""

import json
import math
import os
from langchain_ollama import ChatOllama
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


QUESTIONS_FILE = "gold_questions.json"
RESULTS_FILE   = "evaluation_results.json"

METHODS      = ["bm25", "dense", "hybrid", "hybrid_reranker"]
TOP_K        = 5
TEXT_MODEL   = "gemma:2b"
VISION_MODEL = "llava"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


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
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────
# Retrieval metrics
# ─────────────────────────────────────────

def recall_at_k(retrieved: list, relevant_ids: list, k: int) -> float:
    top_k_ids = {r["chunk_id"] for r in retrieved[:k]}
    hits = len(top_k_ids & set(relevant_ids))
    return hits / len(relevant_ids)


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
    dcg  = sum(1.0 / math.log2(i + 2) for i, cid in enumerate(top_k_ids) if cid in relevant_ids)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg / idcg if idcg > 0 else 0.0


def compute_retrieval_metrics(results: list, relevant_ids: list, k: int = TOP_K) -> dict:
    return {
        f"recall@{k}"   : round(recall_at_k(results, relevant_ids, k), 4),
        f"precision@{k}": round(precision_at_k(results, relevant_ids, k), 4),
        "mrr"           : round(mrr(results, relevant_ids), 4),
        f"ndcg@{k}"     : round(ndcg_at_k(results, relevant_ids, k), 4),
    }


# ─────────────────────────────────────────
# Multimodal answer generation
# ─────────────────────────────────────────

def describe_image_for_question(question: str, image_b64: str, source_pdf: str) -> str:
    """
    Send the actual retrieved image to the vision LLM and ask it to describe
    what it sees in the context of the user's question.

    This is called during evaluation answer generation — the same approach
    used in the chatbot. Keeping both in sync ensures evaluation measures
    the actual system behaviour.

    Parameters
    ----------
    question   : gold question string (provides context for the description)
    image_b64  : base64 image string from the retrieved chunk
    source_pdf : PDF filename for traceability in the returned description
    """
    vision_llm = ChatOllama(model=VISION_MODEL, temperature=0.0)
    data_url   = f"data:image/jpeg;base64,{image_b64}"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
                {
                    "type": "text",
                    "text": (
                        f"The user asked: '{question}'\n\n"
                        f"This image is from the document: {source_pdf}\n\n"
                        "Describe the image in detail, focusing on information relevant "
                        "to the question. If it is a chart or graph, state the exact "
                        "values and trends. If it is a diagram, explain the structure."
                    ),
                },
            ],
        }
    ]

    response = vision_llm.invoke(messages)
    return StrOutputParser().invoke(response)


def generate_answer(question: str, retrieved_chunks: list) -> tuple[str, list[str]]:
    """
    Generate a multimodal answer and return both the answer string and the
    list of context strings used (for RAGAS).

    For image chunks:  call vision LLM on the actual image → get description
    For text/table:    use raw_text directly

    Returns
    -------
    answer         : generated answer string
    context_texts  : list of context strings fed to the LLM (for RAGAS context metrics)
    """
    context_parts  = []
    context_texts  = []     # collected for RAGAS — must be strings

    for chunk in retrieved_chunks:
        source   = chunk["source_pdf"]
        modality = chunk["modality"]

        if modality == "image" and chunk.get("image_b64"):
            # True multimodal: ask vision LLM about the real image
            vision_desc = describe_image_for_question(
                question, chunk["image_b64"], source
            )
            block = f"[IMAGE from {source}]\nVisual analysis: {vision_desc}"
            context_parts.append(block)
            context_texts.append(vision_desc)       # RAGAS gets the vision text

        elif modality == "image":
            # Fallback if image_b64 is missing
            fallback = chunk["retrieval_text"]
            context_parts.append(f"[IMAGE description from {source}]\n{fallback}")
            context_texts.append(fallback)

        elif modality == "table":
            text = chunk.get("raw_text") or chunk["retrieval_text"]
            context_parts.append(f"[TABLE from {source}]\n{text}")
            context_texts.append(text)

        else:
            text = chunk.get("raw_text") or chunk["retrieval_text"]
            context_parts.append(f"[TEXT from {source}]\n{text}")
            context_texts.append(text)

    context = "\n\n".join(context_parts)

    text_llm = ChatOllama(model=TEXT_MODEL, temperature=0.0)

    prompt = (
        "Answer the question based only on the context below. "
        "Be concise. If the context does not contain the answer, "
        "say 'Not found in context.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    response = text_llm.invoke(prompt)
    answer   = StrOutputParser().invoke(response)

    return answer, context_texts


# ─────────────────────────────────────────
# RAGAS evaluation
# ─────────────────────────────────────────

def run_ragas(questions, answers, contexts, ground_truths):
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

    if not OPENAI_API_KEY:
        print("=" * 60)
        print("WARNING: OPENAI_API_KEY not set — RAGAS will fail.")
        print("Retrieval metrics will still be computed and saved.")
        print("=" * 60)
        print()

    questions = load_questions()
    print(f"Loaded {len(questions)} gold questions\n")

    indexes     = load_indexes()
    all_results = {}

    for method in METHODS:
        print(f"\n{'='*55}")
        print(f"Evaluating: {method.upper()}")
        print(f"{'='*55}")

        method_results      = []
        ragas_questions     = []
        ragas_answers       = []
        ragas_contexts      = []
        ragas_ground_truths = []

        for q in questions:
            question     = q["question"]
            relevant_ids = q["relevant_chunk_ids"]
            ground_truth = q["ground_truth"]

            # ── Retrieve ─────────────────────────────────────────────────
            retrieved = retrieve(question, method=method, indexes=indexes, top_k=TOP_K)

            # ── Retrieval metrics ─────────────────────────────────────────
            retrieval_metrics = compute_retrieval_metrics(retrieved, relevant_ids, k=TOP_K)
            print(f"\n  Q [{q['question_id']}]: {question}")
            print(f"  Retrieval : {retrieval_metrics}")

            n_images = sum(1 for c in retrieved if c.get("image_b64"))
            if n_images:
                print(f"  Images    : {n_images} image chunk(s) — sending to {VISION_MODEL}")

            # ── Generate answer (multimodal) ──────────────────────────────
            answer, context_texts = generate_answer(question, retrieved)
            print(f"  Answer    : {answer[:120]}...")

            # ── Collect for RAGAS ─────────────────────────────────────────
            ragas_questions.append(question)
            ragas_answers.append(answer)
            ragas_contexts.append(context_texts)    # list of strings per question
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
            })

        # ── RAGAS ─────────────────────────────────────────────────────────
        print(f"\n  Running RAGAS for {method}...")
        try:
            ragas_scores = run_ragas(
                ragas_questions, ragas_answers, ragas_contexts, ragas_ground_truths
            )
            ragas_df = ragas_scores.to_pandas()

            for i, row in ragas_df.iterrows():
                method_results[i]["ragas_metrics"] = {
                    "faithfulness"     : safe(row.get("faithfulness")),
                    "answer_relevancy" : safe(row.get("answer_relevancy")),
                    "context_precision": safe(row.get("context_precision")),
                    "context_recall"   : safe(row.get("context_recall")),
                }

        except Exception as e:
            print(f"  WARNING: RAGAS failed — {e}")
            print("  Retrieval metrics are still saved.")

        all_results[method] = method_results

    # ── Save ──────────────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved → {RESULTS_FILE}")

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
            vals = [r["ragas_metrics"][key] for r in ragas_results if r["ragas_metrics"].get(key) is not None]
            return sum(vals) / len(vals) if vals else 0.0

        print(f"\nMethod: {method.upper()}")
        print(f"  Retrieval")
        print(f"    Recall@{TOP_K}         : {avg_recall:.4f}")
        print(f"    Precision@{TOP_K}      : {avg_precision:.4f}")
        print(f"    MRR               : {avg_mrr:.4f}")
        print(f"    nDCG@{TOP_K}          : {avg_ndcg:.4f}")

        if ragas_results:
            print(f"  Generation (RAGAS)")
            print(f"    Faithfulness      : {avg_metric('faithfulness'):.4f}")
            print(f"    Answer Relevancy  : {avg_metric('answer_relevancy'):.4f}")
            print(f"    Context Precision : {avg_metric('context_precision'):.4f}")
            print(f"    Context Recall    : {avg_metric('context_recall'):.4f}")
        else:
            print(f"  Generation (RAGAS) : not available (set OPENAI_API_KEY)")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
