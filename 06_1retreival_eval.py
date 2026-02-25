"""
06a_retrieval_eval.py
---------------------
Phase 1: Retrieval evaluation ONLY.

Computes:
  - Recall@k
  - Precision@k
  - MRR
  - nDCG@k

Saves:
  retrieval_results.json

NO OpenAI calls.
"""

import json
import math
from retrieve import load_indexes, retrieve

QUESTIONS_FILE = "ground_truth.json"
RESULTS_FILE   = "retrieval_results.json"

METHODS = ["bm25", "dense", "hybrid", "hybrid_reranker"]
TOP_K   = 5


# ─────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────

def recall_at_k(retrieved, relevant_ids, k):
    top_k_ids = {r["chunk_id"] for r in retrieved[:k]}
    hits = len(top_k_ids & set(relevant_ids))
    return hits / len(relevant_ids) if relevant_ids else 0.0


def precision_at_k(retrieved, relevant_ids, k):
    top_k_ids = {r["chunk_id"] for r in retrieved[:k]}
    hits = len(top_k_ids & set(relevant_ids))
    return hits / k


def mrr(retrieved, relevant_ids):
    for r in retrieved:
        if r["chunk_id"] in relevant_ids:
            return 1.0 / r["rank"]
    return 0.0


def ndcg_at_k(retrieved, relevant_ids, k):
    top_k_ids = [r["chunk_id"] for r in retrieved[:k]]

    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, cid in enumerate(top_k_ids)
        if cid in relevant_ids
    )

    idcg = sum(
        1.0 / math.log2(i + 2)
        for i in range(min(len(relevant_ids), k))
    )

    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(results, relevant_ids):
    return {
        f"recall@{TOP_K}": round(recall_at_k(results, relevant_ids, TOP_K), 4),
        f"precision@{TOP_K}": round(precision_at_k(results, relevant_ids, TOP_K), 4),
        "mrr": round(mrr(results, relevant_ids), 4),
        f"ndcg@{TOP_K}": round(ndcg_at_k(results, relevant_ids, TOP_K), 4),
    }


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main():

    print("Loading questions...")
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print("Loading indexes...")
    indexes = load_indexes()

    all_results = {}

    for method in METHODS:
        print(f"\nEvaluating retrieval: {method.upper()}")

        method_results = []

        for q in questions:

            retrieved = retrieve(
                q["question"],
                method=method,
                indexes=indexes,
                top_k=TOP_K
            )

            metrics = compute_metrics(
                retrieved,
                q["relevant_chunk_ids"]
            )

            method_results.append({
                "question_id": q["question_id"],
                "question": q["question"],
                "ground_truth": q["ground_truth"],
                "retrieved_chunks": retrieved,  # FULL chunks saved
                "retrieval_metrics": metrics,
            })

        all_results[method] = method_results

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\nRetrieval evaluation complete.")
    print(f"Saved → {RESULTS_FILE}")


if __name__ == "__main__":
    main()