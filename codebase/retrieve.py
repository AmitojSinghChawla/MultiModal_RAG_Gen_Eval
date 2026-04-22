"""
parse_results.py
----------------
Parses evaluation_results.json produced by 03_evaluate.py and prints
three formatted tables using PrettyTable:

  Table 1 — Summary       : mean ± std per method per metric
  Table 2 — Significance  : paired t-test between every method pair
  Table 3 — Best method   : winner per metric with margin over 2nd place

Paired t-test is appropriate because all 50 questions are answered by
every method — scores are paired by question_id, not independent samples.

Usage:
    python parse_results.py

Requirements:
    pip install prettytable scipy numpy
"""

import json
import itertools
import numpy as np
from scipy import stats
from prettytable import PrettyTable

RESULTS_FILE = r"C:\Users\amito\PycharmProjects\MultiModal_RAG_Gen_Eval\Result\evaluation_results.json"

METHODS        = ["bm25", "dense", "hybrid", "hybrid_reranker"]
STRING_METRICS = ["exact_match", "token_f1", "rouge_l"]
RAGAS_METRICS  = ["answer_correctness", "faithfulness", "context_precision", "context_recall"]
ALL_METRICS    = STRING_METRICS + RAGAS_METRICS

METHOD_LABELS = {
    "bm25":            "BM25",
    "dense":           "Dense",
    "hybrid":          "Hybrid",
    "hybrid_reranker": "Hybrid Reranker",
}

METRIC_LABELS = {
    "exact_match":        "Exact Match",
    "token_f1":           "Token F1",
    "rouge_l":            "ROUGE-L",
    "answer_correctness": "Answer Correctness",
    "faithfulness":       "Faithfulness",
    "context_precision":  "Context Precision",
    "context_recall":     "Context Recall",
}


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_scores(per_question):
    scores = {m: {k: [] for k in ALL_METRICS} for m in METHODS}
    for method in METHODS:
        for entry in per_question.get(method, []):
            sm = entry.get("string_metrics", {})
            for k in STRING_METRICS:
                v = sm.get(k)
                if isinstance(v, float):
                    scores[method][k].append(v)
            rm = entry.get("ragas_metrics") or {}
            for k in RAGAS_METRICS:
                v = rm.get(k)
                if isinstance(v, float):
                    scores[method][k].append(v)
    return scores


def table_summary(scores):
    t = PrettyTable()
    t.field_names = ["Metric"] + [METHOD_LABELS[m] for m in METHODS]
    t.align = "l"
    for col in t.field_names[1:]:
        t.align[col] = "c"

    for metric in ALL_METRICS:
        means    = {m: np.mean(scores[m][metric]) for m in METHODS if scores[m][metric]}
        best_val = max(means.values()) if means else None
        row      = [METRIC_LABELS[metric]]
        for method in METHODS:
            v = scores[method][metric]
            if not v:
                row.append("N/A")
                continue
            mean, std = np.mean(v), np.std(v)
            cell      = f"{mean:.3f} ± {std:.3f}"
            if mean == best_val:
                cell = "★ " + cell
            row.append(cell)
        t.add_row(row)

    print("\n" + "─" * 10 + " TABLE 1: SUMMARY (mean ± std, n=50) " + "─" * 10)
    print(t)
    print("  ★ = best score for that metric\n")


def table_significance(scores):
    pairs       = list(itertools.combinations(METHODS, 2))
    pair_labels = [f"{METHOD_LABELS[a][:6]} vs {METHOD_LABELS[b][:6]}" for a, b in pairs]

    t = PrettyTable()
    t.field_names = ["Metric"] + pair_labels
    t.align = "l"
    for col in t.field_names[1:]:
        t.align[col] = "c"

    for metric in ALL_METRICS:
        row = [METRIC_LABELS[metric]]
        for a, b in pairs:
            va, vb = scores[a][metric], scores[b][metric]
            n      = min(len(va), len(vb))
            if n < 2:
                row.append("n/a")
                continue
            _, p   = stats.ttest_rel(va[:n], vb[:n])
            winner = METHOD_LABELS[a][:6] if np.mean(va[:n]) > np.mean(vb[:n]) else METHOD_LABELS[b][:6]
            sig    = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
            row.append(f"{winner} {sig} (p={p:.3f})")
        t.add_row(row)

    print("─" * 10 + " TABLE 2: STATISTICAL SIGNIFICANCE (Paired t-test) " + "─" * 10)
    print(t)
    print("  ** p<0.01   * p<0.05   ns = not significant\n")


def table_best(scores):
    t = PrettyTable()
    t.field_names = ["Metric", "Best Method", "Mean", "2nd Place", "2nd Mean", "Margin"]
    t.align = "l"
    for col in ["Mean", "2nd Mean", "Margin"]:
        t.align[col] = "r"

    for metric in ALL_METRICS:
        ranked = sorted(
            [(m, np.mean(scores[m][metric])) for m in METHODS if scores[m][metric]],
            key=lambda x: x[1], reverse=True,
        )
        if len(ranked) < 2:
            continue
        best_m, best_v     = ranked[0]
        second_m, second_v = ranked[1]
        t.add_row([
            METRIC_LABELS[metric],
            METHOD_LABELS[best_m],
            f"{best_v:.3f}",
            METHOD_LABELS[second_m],
            f"{second_v:.3f}",
            f"+{best_v - second_v:.3f}",
        ])

    print("─" * 10 + " TABLE 3: BEST METHOD PER METRIC " + "─" * 10)
    print(t)


if __name__ == "__main__":
    data   = load(RESULTS_FILE)
    scores = extract_scores(data["per_question"])
    table_summary(scores)
    table_significance(scores)
    table_best(scores)