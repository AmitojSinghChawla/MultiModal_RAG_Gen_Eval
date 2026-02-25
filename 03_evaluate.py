"""
03_evaluate.py
--------------
End-to-end evaluation of multimodal RAG retrieval strategies.

Compares 4 retrieval methods on answer quality ONLY:
  - bm25
  - dense
  - hybrid
  - hybrid_reranker

Metrics computed per method:
  - Exact Match (EM)
  - Token-level F1
  - ROUGE-L
  - RAGAS: answer_correctness, faithfulness, context_precision, context_recall

No manual chunk relevance annotation required.
Predictions are generated dynamically during evaluation.

Dataset format (gold_questions.json):
  [{"question": "...", "ground_truth": "..."}, ...]

Usage:
    export OPENAI_API_KEY=your_key
    python 03_evaluate.py
"""

import json
import math
import os
import re
import string
import statistics
from collections import Counter

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    faithfulness,
    context_precision,
    context_recall,
)
from rouge_score import rouge_scorer

from retrieve import load_indexes, retrieve
from dotenv import load_dotenv

load_dotenv(verbose=True)

# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────

QUESTIONS_FILE = "gold_questions.json"
RESULTS_FILE   = "evaluation_results.json"

METHODS = ["bm25", "dense", "hybrid", "hybrid_reranker"]
TOP_K   = 5
MODEL   = "gpt-4o-mini"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# ─────────────────────────────────────────
# Text normalisation helpers
# ─────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_for_f1(text: str) -> list[str]:
    """Simple whitespace tokenizer after normalization."""
    return normalize_text(text).split()


# ─────────────────────────────────────────
# Metric functions
# ─────────────────────────────────────────

def exact_match(prediction: str, ground_truth: str) -> float:
    """1.0 if normalized strings match exactly, else 0.0."""
    return float(normalize_text(prediction) == normalize_text(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1 between prediction and ground truth.

    Counts token overlap, computes precision and recall,
    returns their harmonic mean.
    """
    pred_tokens = tokenize_for_f1(prediction)
    gt_tokens   = tokenize_for_f1(ground_truth)

    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gt_counter   = Counter(gt_tokens)

    common = sum((pred_counter & gt_counter).values())

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall    = common / len(gt_tokens)
    f1        = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def rouge_l(prediction: str, ground_truth: str) -> float:
    """ROUGE-L F1 score."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(ground_truth, prediction)
    return round(scores["rougeL"].fmeasure, 4)


def compute_metrics(prediction: str, ground_truth: str) -> dict:
    """Compute all non-RAGAS metrics for a single (prediction, ground_truth) pair."""
    return {
        "exact_match": exact_match(prediction, ground_truth),
        "token_f1"   : token_f1(prediction, ground_truth),
        "rouge_l"    : rouge_l(prediction, ground_truth),
    }


# ─────────────────────────────────────────
# Answer generation
# ─────────────────────────────────────────

def generate_answer(question: str, retrieved_chunks: list) -> tuple[str, list[str]]:
    """
    Generate an answer by sending the question + retrieved context to GPT-4O Mini.

    Images are sent as base64; text and tables as plain text — all in one call.

    Returns
    -------
    answer        : str  — generated answer
    context_texts : list[str] — text-only context for RAGAS
    """
    llm = ChatOpenAI(model=MODEL, temperature=0.0, api_key=OPENAI_API_KEY)

    content       = []
    context_texts = []

    for chunk in retrieved_chunks:
        source   = chunk["source_pdf"]
        modality = chunk["modality"]

        if modality == "image" and chunk.get("image_b64"):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url"   : f"data:image/jpeg;base64,{chunk['image_b64']}",
                    "detail": "high",
                },
            })
            content.append({"type": "text", "text": f"[Image from {source}]"})
            context_texts.append(chunk["retrieval_text"])

        elif modality == "table":
            text = chunk.get("raw_text") or chunk["retrieval_text"]
            content.append({"type": "text", "text": f"[TABLE from {source}]\n{text}\n"})
            context_texts.append(text)

        else:
            text = chunk.get("raw_text") or chunk["retrieval_text"]
            content.append({"type": "text", "text": f"[TEXT from {source}]\n{text}\n"})
            context_texts.append(text)

    content.append({
        "type": "text",
        "text": (
            "\n─────────────────────────────────────\n\n"
            "Answer the question based only on the context above. "
            "Be concise and accurate. "
            "If the context does not contain the answer, say 'Not found in context.'\n\n"
            f"Question: {question}\n\nAnswer:"
        ),
    })

    messages = [{"role": "user", "content": content}]
    response = llm.invoke(messages)
    answer   = StrOutputParser().invoke(response)
    return answer, context_texts


# ─────────────────────────────────────────
# RAGAS evaluation
# ─────────────────────────────────────────

def safe_float(val) -> float | None:
    """Convert RAGAS metric value to float, returning None on NaN."""
    try:
        f = float(val)
        return round(f, 4) if not math.isnan(f) else None
    except (TypeError, ValueError):
        return None


def run_ragas(
    questions     : list[str],
    answers       : list[str],
    contexts      : list[list[str]],
    ground_truths : list[str],
) -> list[dict]:
    """
    Run RAGAS on a batch of (question, answer, contexts, ground_truth) tuples.

    Returns a list of per-question metric dicts.
    answer_correctness replaces the old context-based metrics as the primary
    generation quality signal; no chunk relevance labels are needed.
    """
    data = {
        "question"    : questions,
        "answer"      : answers,
        "contexts"    : contexts,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data)
    scores  = evaluate(
        dataset,
        metrics=[answer_correctness, faithfulness, context_precision, context_recall],
    )
    df = scores.to_pandas()

    results = []
    for _, row in df.iterrows():
        results.append({
            "answer_correctness": safe_float(row.get("answer_correctness")),
            "faithfulness"      : safe_float(row.get("faithfulness")),
            "context_precision" : safe_float(row.get("context_precision")),
            "context_recall"    : safe_float(row.get("context_recall")),
        })
    return results


# ─────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────

def mean_std(values: list[float | None]) -> tuple[float, float]:
    """Return (mean, std) ignoring None entries."""
    clean = [v for v in values if v is not None]
    if not clean:
        return 0.0, 0.0
    m = statistics.mean(clean)
    s = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return round(m, 4), round(s, 4)


# ─────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────

def main():

    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set.\n  export OPENAI_API_KEY=your_key")
        return

    # Load dataset
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions from {QUESTIONS_FILE}\n")

    # Load indexes once
    indexes = load_indexes()

    all_results   : dict[str, list] = {}
    method_summary: dict[str, dict] = {}

    for method in METHODS:
        print(f"\n{'='*60}")
        print(f"Evaluating method: {method.upper()}")
        print(f"{'='*60}")

        method_results  : list[dict] = []
        ragas_questions : list[str]  = []
        ragas_answers   : list[str]  = []
        ragas_contexts  : list       = []
        ragas_gts       : list[str]  = []

        for i, q in enumerate(questions, 1):
            question     = q["question"]
            ground_truth = q["ground_truth"]

            print(f"\n  [{i}/{len(questions)}] {question[:70]}...")

            # ── Retrieve ─────────────────────────────────────────────
            retrieved = retrieve(question, method=method, indexes=indexes, top_k=TOP_K)

            # ── Generate answer ───────────────────────────────────────
            print(f"  Generating answer with {MODEL}...", end=" ", flush=True)
            answer, context_texts = generate_answer(question, retrieved)
            print("done")
            print(f"  Prediction : {answer[:100]}...")
            print(f"  Ground truth: {ground_truth[:100]}...")

            # ── String-based metrics ──────────────────────────────────
            metrics = compute_metrics(answer, ground_truth)
            print(
                f"  EM={metrics['exact_match']:.2f}  "
                f"F1={metrics['token_f1']:.4f}  "
                f"ROUGE-L={metrics['rouge_l']:.4f}"
            )

            # Collect for RAGAS
            ragas_questions.append(question)
            ragas_answers.append(answer)
            ragas_contexts.append(context_texts)
            ragas_gts.append(ground_truth)

            method_results.append({
                "question_id"    : q.get("question_id", f"q{i}"),
                "question"       : question,
                "ground_truth"   : ground_truth,
                "prediction"     : answer,
                "retrieved_ids"  : [r["chunk_id"] for r in retrieved],
                "string_metrics" : metrics,
                "ragas_metrics"  : None,   # filled in after batch RAGAS run
            })

        # ── RAGAS batch evaluation ────────────────────────────────────
        print(f"\n  Running RAGAS evaluation for {method}...")
        try:
            ragas_scores = run_ragas(
                ragas_questions, ragas_answers, ragas_contexts, ragas_gts
            )
            for entry, rs in zip(method_results, ragas_scores):
                entry["ragas_metrics"] = rs
            print("  RAGAS complete ✓")
        except Exception as e:
            print(f"  WARNING: RAGAS failed — {e}")
            for entry in method_results:
                entry["ragas_metrics"] = {
                    "answer_correctness": None,
                    "faithfulness"      : None,
                    "context_precision" : None,
                    "context_recall"    : None,
                    "error"             : str(e),
                }

        all_results[method] = method_results

        # ── Aggregate ─────────────────────────────────────────────────
        em_m,     em_s     = mean_std([r["string_metrics"]["exact_match"] for r in method_results])
        f1_m,     f1_s     = mean_std([r["string_metrics"]["token_f1"]    for r in method_results])
        rl_m,     rl_s     = mean_std([r["string_metrics"]["rouge_l"]     for r in method_results])
        ac_m,     ac_s     = mean_std([r["ragas_metrics"].get("answer_correctness") for r in method_results])
        faith_m,  faith_s  = mean_std([r["ragas_metrics"].get("faithfulness")       for r in method_results])
        cp_m,     cp_s     = mean_std([r["ragas_metrics"].get("context_precision")  for r in method_results])
        cr_m,     cr_s     = mean_std([r["ragas_metrics"].get("context_recall")     for r in method_results])

        method_summary[method] = {
            "exact_match"       : {"mean": em_m,    "std": em_s},
            "token_f1"          : {"mean": f1_m,    "std": f1_s},
            "rouge_l"           : {"mean": rl_m,    "std": rl_s},
            "answer_correctness": {"mean": ac_m,    "std": ac_s},
            "faithfulness"      : {"mean": faith_m, "std": faith_s},
            "context_precision" : {"mean": cp_m,    "std": cp_s},
            "context_recall"    : {"mean": cr_m,    "std": cr_s},
        }

    # ── Save full results ─────────────────────────────────────────────
    output = {"summary": method_summary, "per_question": all_results}
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {RESULTS_FILE}")

    print_summary(method_summary)


# ─────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────

def print_summary(summary: dict) -> None:
    """Print a comparison table of all methods and metrics."""

    col_w = 14
    metrics_order = [
        "exact_match",
        "token_f1",
        "rouge_l",
        "answer_correctness",
        "faithfulness",
        "context_precision",
        "context_recall",
    ]
    headers = [
        "Method",
        "EM",
        "F1",
        "ROUGE-L",
        "AnswerCorr",
        "Faithfulness",
        "CtxPrec",
        "CtxRecall",
    ]

    sep = "─" * (10 + col_w * len(metrics_order))
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY  (mean ± std)")
    print(f"{'='*70}")
    print(f"{'Method':<18}", end="")
    for h in headers[1:]:
        print(f"{h:>{col_w}}", end="")
    print()
    print(sep)

    for method, s in summary.items():
        print(f"{method:<18}", end="")
        for key in metrics_order:
            m = s[key]["mean"]
            d = s[key]["std"]
            cell = f"{m:.3f}±{d:.3f}"
            print(f"{cell:>{col_w}}", end="")
        print()

    print(sep)
    print(f"\nModel: {MODEL}  |  Top-K: {TOP_K}")
    print(
        "Evaluation: end-to-end answer quality — no manual chunk "
        "relevance annotation required."
    )
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()