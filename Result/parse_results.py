"""
parse_results.py
----------------
Parses evaluation_results.json and prints one summary table
(mean ± std, best method marked with ★) plus one significance
sentence for the three metrics that matter.

Also writes two CSVs:
  - summary_table.csv        — the table as a CSV
  - per_question_scores.csv  — every individual question score

Usage:
    python parse_results.py
    python parse_results.py --input Result/evaluation_results.json
"""

import json
import csv
import math
import argparse
import statistics
import itertools
from pathlib import Path

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from prettytable import PrettyTable
    HAS_PRETTY = True
except ImportError:
    HAS_PRETTY = False


# ── Config ────────────────────────────────────────────────────────────────────

METHODS = ["bm25", "dense", "hybrid", "hybrid_reranker"]

METHOD_LABELS = {
    "bm25":            "BM25",
    "dense":           "Dense",
    "hybrid":          "Hybrid",
    "hybrid_reranker": "Hybrid+Reranker",
}

# Only the 3 metrics that actually matter for the thesis
KEY_METRICS = ["answer_correctness", "context_precision", "context_recall"]

# All 7 metrics shown in the table
ALL_METRICS = [
    "exact_match", "token_f1", "rouge_l",
    "answer_correctness", "faithfulness",
    "context_precision", "context_recall",
]

METRIC_LABELS = {
    "exact_match":        "Exact Match",
    "token_f1":           "Token F1",
    "rouge_l":            "ROUGE-L",
    "answer_correctness": "Answer Correctness",
    "faithfulness":       "Faithfulness",
    "context_precision":  "Context Precision",
    "context_recall":     "Context Recall",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_float(val):
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def mean_std(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None, 0
    m = statistics.mean(clean)
    s = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return m, s, len(clean)


# ── Load & compute ────────────────────────────────────────────────────────────

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_scores(data):
    """{ method: { qid: { metric: float } } }"""
    result = {}
    for method in METHODS:
        result[method] = {}
        for entry in data.get("per_question", {}).get(method, []):
            qid = entry["question_id"]
            scores = {}
            for m in ["exact_match", "token_f1", "rouge_l"]:
                scores[m] = safe_float((entry.get("string_metrics") or {}).get(m))
            for m in ["answer_correctness", "faithfulness",
                      "context_precision", "context_recall"]:
                scores[m] = safe_float((entry.get("ragas_metrics") or {}).get(m))
            scores["question"]     = entry.get("question", "")
            scores["ground_truth"] = entry.get("ground_truth", "")
            scores["prediction"]   = entry.get("prediction", "")
            result[method][qid] = scores
    return result


def compute_summary(scores):
    """{ method: { metric: { mean, std, n, values } } }"""
    summary = {}
    for method, questions in scores.items():
        summary[method] = {}
        for metric in ALL_METRICS:
            vals = [q[metric] for q in questions.values()
                    if q.get(metric) is not None]
            m, s, n = mean_std(vals)
            summary[method][metric] = {"mean": m, "std": s, "n": n, "values": vals}
    return summary


# ── Significance (one sentence) ───────────────────────────────────────────────

def significance_sentence(scores, alpha=0.05):
    """
    Runs Wilcoxon between Hybrid+Reranker and every other method
    for the 3 key metrics. Returns a ready-to-use sentence.
    """
    if not HAS_SCIPY:
        return ("  [scipy not installed — run: pip install scipy]\n"
                "  Significance sentence skipped.")

    all_qids = sorted(
        {q for m in METHODS for q in scores[m]},
        key=lambda x: int(x[1:]) if x[1:].isdigit() else 0
    )

    sig_pairs = []
    ns_pairs  = []

    for metric in KEY_METRICS:
        for other in ["bm25", "dense", "hybrid"]:
            pairs = [
                (scores["hybrid_reranker"][q][metric],
                 scores[other][q][metric])
                for q in all_qids
                if (q in scores["hybrid_reranker"] and q in scores[other]
                    and scores["hybrid_reranker"][q].get(metric) is not None
                    and scores[other][q].get(metric) is not None)
            ]
            if len(pairs) < 5:
                continue
            x = [p[0] for p in pairs]
            y = [p[1] for p in pairs]
            diffs = [a - b for a, b in zip(x, y)]
            if all(d == 0 for d in diffs):
                ns_pairs.append(f"{METRIC_LABELS[metric]} vs {METHOD_LABELS[other]}")
                continue
            try:
                _, p = scipy_stats.wilcoxon(x, y, alternative="two-sided",
                                            zero_method="wilcox")
                label = f"{METRIC_LABELS[metric]} vs {METHOD_LABELS[other]}"
                if p < alpha:
                    sig_pairs.append((label, p))
                else:
                    ns_pairs.append((label, p))
            except ValueError:
                pass

    if not sig_pairs:
        return "  No significant differences found at α=0.05."

    sig_str = ", ".join(f"{l} (p={p:.3f})" for l, p in sig_pairs)
    return (f"  Hybrid+Reranker improvements are statistically significant for:\n"
            f"  {sig_str}\n"
            f"  (Wilcoxon signed-rank, two-sided, α={alpha})")


# ── Print the one table ───────────────────────────────────────────────────────

def print_table(summary):
    col_names = [METHOD_LABELS[m] for m in METHODS]

    if HAS_PRETTY:
        t = PrettyTable()
        t.field_names = ["Metric"] + col_names
        t.align = "l"

        for metric in ALL_METRICS:
            means = {m: summary[m][metric]["mean"] for m in METHODS}
            valid = {m: v for m, v in means.items() if v is not None}
            best  = max(valid, key=valid.__getitem__) if valid else None

            row = [METRIC_LABELS[metric]]
            for method in METHODS:
                s   = summary[method][metric]
                m_v = s["mean"]
                std = s["std"]
                if m_v is None:
                    row.append("N/A")
                else:
                    star = "★ " if method == best else "  "
                    row.append(f"{star}{m_v:.3f} ± {std:.3f}")
            t.add_row(row)

        print(t)
        print("  ★ = best score for that metric")

    else:
        # Plain text fallback
        w = 20
        print(f"{'Metric':<22}" + "".join(f"{c:>{w}}" for c in col_names))
        print("-" * (22 + w * len(METHODS)))
        for metric in ALL_METRICS:
            means = {m: summary[m][metric]["mean"] for m in METHODS}
            valid = {m: v for m, v in means.items() if v is not None}
            best  = max(valid, key=valid.__getitem__) if valid else None
            row   = f"{METRIC_LABELS[metric]:<22}"
            for method in METHODS:
                s   = summary[method][metric]
                m_v = s["mean"]
                std = s["std"]
                if m_v is None:
                    row += f"{'N/A':>{w}}"
                else:
                    star = "* " if method == best else "  "
                    row += f"{star+f'{m_v:.3f}±{std:.3f}':>{w}}"
            print(row)
        print("  * = best score for that metric")


# ── CSV writers ───────────────────────────────────────────────────────────────

def write_summary_csv(summary, path):
    """The table as a flat CSV — one row per method × metric."""
    rows = []
    for method in METHODS:
        for metric in ALL_METRICS:
            s     = summary[method][metric]
            total = sum(s["values"]) if s["values"] else None
            rows.append({
                "Method":      METHOD_LABELS[method],
                "Metric":      METRIC_LABELS[metric],
                "Mean":        f"{s['mean']:.4f}" if s["mean"] is not None else "N/A",
                "Std":         f"{s['std']:.4f}"  if s["std"]  is not None else "N/A",
                "N":           s["n"],
                "Sum":         f"{total:.4f}" if total is not None else "N/A",
                "Calculation": (f"{total:.4f} / {s['n']} = {s['mean']:.4f}"
                                if total is not None else "N/A"),
            })

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "Method", "Metric", "Mean", "Std", "N", "Sum", "Calculation"
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"  ✓ summary_table.csv       ({len(rows)} rows) → {path}")


def write_per_question_csv(scores, path):
    """Every question × method score in one CSV."""
    all_qids = sorted(
        {q for m in METHODS for q in scores[m]},
        key=lambda x: int(x[1:]) if x[1:].isdigit() else 0
    )
    rows = []
    for method in METHODS:
        for qid in all_qids:
            q = scores[method].get(qid, {})
            rows.append({
                "Method":             METHOD_LABELS[method],
                "Question_ID":        qid,
                "Question":           q.get("question", ""),
                "Ground_Truth":       q.get("ground_truth", ""),
                "Prediction":         q.get("prediction", ""),
                "Exact_Match":        f"{q['exact_match']:.3f}"        if q.get("exact_match")        is not None else "N/A",
                "Token_F1":           f"{q['token_f1']:.4f}"           if q.get("token_f1")           is not None else "N/A",
                "ROUGE_L":            f"{q['rouge_l']:.4f}"            if q.get("rouge_l")            is not None else "N/A",
                "Answer_Correctness": f"{q['answer_correctness']:.4f}" if q.get("answer_correctness") is not None else "N/A",
                "Faithfulness":       f"{q['faithfulness']:.4f}"       if q.get("faithfulness")       is not None else "N/A",
                "Context_Precision":  f"{q['context_precision']:.4f}"  if q.get("context_precision")  is not None else "N/A",
                "Context_Recall":     f"{q['context_recall']:.4f}"     if q.get("context_recall")     is not None else "N/A",
            })

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "Method", "Question_ID", "Question", "Ground_Truth", "Prediction",
            "Exact_Match", "Token_F1", "ROUGE_L",
            "Answer_Correctness", "Faithfulness",
            "Context_Precision", "Context_Recall",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"  ✓ per_question_scores.csv ({len(rows)} rows) → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",      "-i", default="evaluation_results.json")
    p.add_argument("--output_dir", "-o", default=None)
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found.")
        return

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    data    = load(input_path)
    scores  = extract_scores(data)
    summary = compute_summary(scores)

    # ── The one table ─────────────────────────────────────────────────────────
    print("\n  RESULTS SUMMARY  (mean ± std across 50 questions, ★ = best)\n")
    print_table(summary)

    # ── One significance sentence ─────────────────────────────────────────────
    print("\n  SIGNIFICANCE (Hybrid+Reranker vs others):")
    print(significance_sentence(scores))

    # ── Two CSVs ──────────────────────────────────────────────────────────────
    print(f"\n  Writing CSVs to {output_dir}/")
    write_summary_csv(     summary, output_dir / "summary_table.csv")
    write_per_question_csv(scores,  output_dir / "per_question_scores.csv")
    print()


if __name__ == "__main__":
    main()