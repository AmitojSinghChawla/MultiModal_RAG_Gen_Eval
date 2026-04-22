"""
parse_results.py
----------------
Parses evaluation_results.json and produces:

  Console output:
    - TABLE 1: Summary (mean ± std, best marked with ★)
    - TABLE 2: Pairwise statistical significance (Wilcoxon signed-rank test)
    - TABLE 3: Delta table (absolute + relative % vs Hybrid+Reranker)

  CSV files:
    - summary_metrics.csv          — mean ± std per method × metric
    - per_question_metrics.csv     — every individual question score
    - method_comparison.csv        — deltas vs best method
    - clean_table_with_calcs.csv   — full sum / n = mean workings

Usage:
    python parse_results.py
    python parse_results.py --input /path/to/evaluation_results.json
    python parse_results.py --input results.json --output_dir ./csv_output
"""

import json
import csv
import math
import argparse
import os
import statistics
import itertools
from pathlib import Path

# scipy for Wilcoxon test — soft dependency
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("  [INFO] scipy not installed — skipping significance tests.")
    print("         Install with: pip install scipy\n")

# prettytable for console display — soft dependency
try:
    from prettytable import PrettyTable
    HAS_PRETTY = True
except ImportError:
    HAS_PRETTY = False
    print("  [INFO] prettytable not installed — using plain text tables.")
    print("         Install with: pip install prettytable\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Parse evaluation_results.json → CSV + console tables")
    p.add_argument("--input",  "-i", default="evaluation_results.json",
                   help="Path to evaluation_results.json (default: ./evaluation_results.json)")
    p.add_argument("--output_dir", "-o", default=None,
                   help="Directory to write CSVs (default: same directory as input file)")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance level for Wilcoxon test (default: 0.05)")
    return p.parse_args()


# ── Constants ─────────────────────────────────────────────────────────────────

METHODS = ["bm25", "dense", "hybrid", "hybrid_reranker"]

METHOD_LABELS = {
    "bm25":            "BM25",
    "dense":           "Dense",
    "hybrid":          "Hybrid",
    "hybrid_reranker": "Hybrid+Reranker",
}

STRING_METRICS = ["exact_match", "token_f1", "rouge_l"]
RAGAS_METRICS  = ["answer_correctness", "faithfulness", "context_precision", "context_recall"]
ALL_METRICS    = STRING_METRICS + RAGAS_METRICS

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


def fmt(val, decimals=4):
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def mean_std(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None, 0
    m = statistics.mean(clean)
    s = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return m, s, len(clean)


def pct_change(new_val, base_val):
    if base_val is None or new_val is None or base_val == 0:
        return None
    return (new_val - base_val) / abs(base_val) * 100


# ── Data extraction ───────────────────────────────────────────────────────────

def extract_per_question(data):
    """
    Returns { method: { qid: { metric: float_or_None, 'question': str, ... } } }
    """
    result = {}
    per_q = data.get("per_question", {})

    for method in METHODS:
        result[method] = {}
        for entry in per_q.get(method, []):
            qid    = entry["question_id"]
            scores = {}

            sm = entry.get("string_metrics", {}) or {}
            for m in STRING_METRICS:
                scores[m] = safe_float(sm.get(m))

            rm = entry.get("ragas_metrics", {}) or {}
            for m in RAGAS_METRICS:
                scores[m] = safe_float(rm.get(m))

            scores["question"]     = entry.get("question", "")
            scores["ground_truth"] = entry.get("ground_truth", "")
            scores["prediction"]   = entry.get("prediction", "")

            result[method][qid] = scores

    return result


def compute_summary(per_question):
    """
    { method: { metric: { mean, std, n, values } } }
    Re-derives stats from raw scores — used to verify JSON summary values.
    """
    summary = {}
    for method, questions in per_question.items():
        summary[method] = {}
        for metric in ALL_METRICS:
            values = [q[metric] for q in questions.values()
                      if q.get(metric) is not None]
            m, s, n = mean_std(values)
            summary[method][metric] = {
                "mean":   m,
                "std":    s,
                "n":      n,
                "values": values,
            }
    return summary


# ── TABLE 1: Summary ──────────────────────────────────────────────────────────

def print_summary_table(summary):
    """
    Console table: Method columns, metric rows, best marked with ★.
    mean ± std format.
    """
    print("\n" + "─" * 70)
    print("  TABLE 1: SUMMARY (mean ± std, n=50)")
    print("─" * 70)

    method_cols = [METHOD_LABELS[m] for m in METHODS]

    if HAS_PRETTY:
        t = PrettyTable()
        # Ensure unique field names — METHOD_LABELS guarantees uniqueness
        t.field_names = ["Metric"] + method_cols
        t.align = "l"
        t.align["Metric"] = "l"

        for metric in ALL_METRICS:
            means = {m: summary[m][metric]["mean"] for m in METHODS}
            valid_means = {m: v for m, v in means.items() if v is not None}
            best_method = max(valid_means, key=valid_means.__getitem__) if valid_means else None

            row = [METRIC_LABELS[metric]]
            for method in METHODS:
                s   = summary[method][metric]
                m   = s["mean"]
                std = s["std"]
                if m is None:
                    cell = "N/A"
                else:
                    star = "★ " if method == best_method else "  "
                    cell = f"{star}{m:.3f} ± {std:.3f}"
                row.append(cell)
            t.add_row(row)

        print(t)
        print("  ★ = best score for that metric")
    else:
        # Plain-text fallback
        col_w = 20
        header = f"{'Metric':<22}" + "".join(f"{c:>{col_w}}" for c in method_cols)
        print(header)
        print("-" * (22 + col_w * len(METHODS)))
        for metric in ALL_METRICS:
            means = {m: summary[m][metric]["mean"] for m in METHODS}
            valid  = {m: v for m, v in means.items() if v is not None}
            best   = max(valid, key=valid.__getitem__) if valid else None
            row    = f"{METRIC_LABELS[metric]:<22}"
            for method in METHODS:
                s   = summary[method][metric]
                m_v = s["mean"]
                std = s["std"]
                if m_v is None:
                    cell = "N/A"
                else:
                    star = "* " if method == best else "  "
                    cell = f"{star}{m_v:.3f}±{std:.3f}"
                row += f"{cell:>{col_w}}"
            print(row)


# ── TABLE 2: Pairwise significance ────────────────────────────────────────────

def compute_significance(per_question, alpha=0.05):
    """
    For every metric × every ordered pair of methods:
      run a two-sided Wilcoxon signed-rank test on the paired per-question scores.
    Returns { metric: { (m1, m2): { stat, p, sig, direction } } }
    """
    if not HAS_SCIPY:
        return {}

    # Collect all question IDs present in all methods
    all_qids = set()
    for method in METHODS:
        all_qids.update(per_question[method].keys())
    qids = sorted(all_qids, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)

    results = {}
    for metric in ALL_METRICS:
        results[metric] = {}
        for m1, m2 in itertools.combinations(METHODS, 2):
            # Only include questions where BOTH methods have a score
            pairs = [
                (per_question[m1][q][metric], per_question[m2][q][metric])
                for q in qids
                if q in per_question[m1] and q in per_question[m2]
                and per_question[m1][q].get(metric) is not None
                and per_question[m2][q].get(metric) is not None
            ]
            if len(pairs) < 5:
                results[metric][(m1, m2)] = {"stat": None, "p": None,
                                              "sig": False, "direction": "N/A"}
                continue

            x = [p[0] for p in pairs]
            y = [p[1] for p in pairs]

            # If all differences are zero, Wilcoxon is undefined
            diffs = [a - b for a, b in zip(x, y)]
            if all(d == 0 for d in diffs):
                results[metric][(m1, m2)] = {"stat": 0.0, "p": 1.0,
                                              "sig": False, "direction": "tie"}
                continue

            try:
                stat, p = scipy_stats.wilcoxon(x, y, alternative="two-sided",
                                               zero_method="wilcox")
            except ValueError:
                results[metric][(m1, m2)] = {"stat": None, "p": None,
                                              "sig": False, "direction": "N/A"}
                continue

            mean1 = statistics.mean(x)
            mean2 = statistics.mean(y)
            direction = f"{METHOD_LABELS[m1]}>{METHOD_LABELS[m2]}" \
                        if mean1 > mean2 else f"{METHOD_LABELS[m2]}>{METHOD_LABELS[m1]}"

            results[metric][(m1, m2)] = {
                "stat":      round(stat, 2),
                "p":         round(p, 4),
                "sig":       p < alpha,
                "direction": direction,
            }

    return results


def print_significance_table(sig_results, alpha=0.05):
    """
    Console table: rows = metrics, columns = method pairs.
    Cell = p-value (* if significant).
    Duplicate column names are avoided by using short unique pair codes.
    """
    if not HAS_SCIPY or not sig_results:
        print("\n  [TABLE 2 skipped — scipy not available]\n")
        return

    # Build unique pair labels — short codes avoid PrettyTable duplicate error
    pairs = list(itertools.combinations(METHODS, 2))
    # Use short abbreviations: B=BM25, D=Dense, H=Hybrid, R=Reranker
    short = {"bm25": "B", "dense": "D", "hybrid": "H", "hybrid_reranker": "R"}
    pair_labels = [f"{short[m1]}vs{short[m2]}" for m1, m2 in pairs]

    # Verify uniqueness before handing to PrettyTable
    assert len(pair_labels) == len(set(pair_labels)), \
        f"Duplicate pair labels: {pair_labels}"

    print("\n" + "─" * 70)
    print(f"  TABLE 2: PAIRWISE SIGNIFICANCE (Wilcoxon signed-rank, α={alpha})")
    print(f"  Pair codes: B=BM25  D=Dense  H=Hybrid  R=Hybrid+Reranker")
    print(f"  * = p < {alpha}  (statistically significant difference)")
    print("─" * 70)

    if HAS_PRETTY:
        t = PrettyTable()
        t.field_names = ["Metric"] + pair_labels   # guaranteed unique
        t.align = "l"

        for metric in ALL_METRICS:
            row = [METRIC_LABELS[metric]]
            for (m1, m2) in pairs:
                r = sig_results.get(metric, {}).get((m1, m2), {})
                p = r.get("p")
                if p is None:
                    cell = "N/A"
                else:
                    star = "*" if r.get("sig") else " "
                    cell = f"p={p:.3f}{star}"
                row.append(cell)
            t.add_row(row)
        print(t)
    else:
        col_w = 12
        header = f"{'Metric':<22}" + "".join(f"{c:>{col_w}}" for c in pair_labels)
        print(header)
        print("-" * (22 + col_w * len(pairs)))
        for metric in ALL_METRICS:
            row = f"{METRIC_LABELS[metric]:<22}"
            for (m1, m2) in pairs:
                r = sig_results.get(metric, {}).get((m1, m2), {})
                p = r.get("p")
                if p is None:
                    cell = "N/A"
                else:
                    star = "*" if r.get("sig") else " "
                    cell = f"p={p:.3f}{star}"
                row += f"{cell:>{col_w}}"
            print(row)

    # Print legend of full pair names
    print("\n  Full pair names:")
    for (m1, m2), code in zip(pairs, pair_labels):
        r_sample = sig_results.get("answer_correctness", {}).get((m1, m2), {})
        direction = r_sample.get("direction", "?")
        print(f"    {code}: {METHOD_LABELS[m1]} vs {METHOD_LABELS[m2]}  "
              f"(higher: {direction})")


# ── TABLE 3: Delta table ──────────────────────────────────────────────────────

def print_delta_table(summary):
    """
    Absolute and relative % difference of every method vs Hybrid+Reranker.
    """
    print("\n" + "─" * 70)
    print("  TABLE 3: DELTAS vs Hybrid+Reranker  (abs | rel%)")
    print("─" * 70)

    comparators = [m for m in METHODS if m != "hybrid_reranker"]
    col_labels  = [f"vs {METHOD_LABELS[m]}" for m in comparators]

    if HAS_PRETTY:
        t = PrettyTable()
        # Interleave abs and rel% columns — guaranteed unique because method names differ
        field_names = ["Metric"]
        for m in comparators:
            field_names.append(f"Δabs vs {METHOD_LABELS[m]}")
            field_names.append(f"Δrel% vs {METHOD_LABELS[m]}")
        t.field_names = field_names
        t.align = "l"

        for metric in ALL_METRICS:
            reranker_mean = summary["hybrid_reranker"][metric]["mean"]
            row = [METRIC_LABELS[metric]]
            for m in comparators:
                base = summary[m][metric]["mean"]
                if base is None or reranker_mean is None:
                    row += ["N/A", "N/A"]
                else:
                    abs_d = reranker_mean - base
                    rel_d = pct_change(reranker_mean, base)
                    row.append(f"{abs_d:+.4f}")
                    row.append(f"{rel_d:+.2f}%" if rel_d is not None else "N/A")
            t.add_row(row)
        print(t)
    else:
        col_w = 16
        header = f"{'Metric':<22}"
        for m in comparators:
            header += f"{'abs vs '+METHOD_LABELS[m]:>{col_w}}{'rel% vs '+METHOD_LABELS[m]:>{col_w}}"
        print(header)
        print("-" * (22 + col_w * 2 * len(comparators)))
        for metric in ALL_METRICS:
            reranker_mean = summary["hybrid_reranker"][metric]["mean"]
            row = f"{METRIC_LABELS[metric]:<22}"
            for m in comparators:
                base = summary[m][metric]["mean"]
                if base is None or reranker_mean is None:
                    row += f"{'N/A':>{col_w}}{'N/A':>{col_w}}"
                else:
                    abs_d = reranker_mean - base
                    rel_d = pct_change(reranker_mean, base)
                    row += f"{abs_d:>+{col_w}.4f}"
                    row += f"{str(f'{rel_d:+.2f}%') if rel_d is not None else 'N/A':>{col_w}}"
            print(row)


# ── CSV writers ───────────────────────────────────────────────────────────────

def write_summary_csv(summary, json_summary, path):
    rows = []
    for method in METHODS:
        for metric in ALL_METRICS:
            s  = summary[method][metric]
            js = (json_summary.get(method, {}).get(metric, {}) or {})
            json_mean = safe_float(js.get("mean"))
            json_std  = safe_float(js.get("std"))

            match = "OK"
            if json_mean is not None and s["mean"] is not None:
                if abs(json_mean - s["mean"]) > 0.001:
                    match = f"MISMATCH (JSON={json_mean:.4f} computed={s['mean']:.4f})"

            rows.append({
                "Method":             METHOD_LABELS[method],
                "Method_key":         method,
                "Metric":             METRIC_LABELS[metric],
                "Metric_key":         metric,
                "Mean_computed":      fmt(s["mean"]),
                "Std_computed":       fmt(s["std"]),
                "N_scored":           s["n"],
                "Sum_computed":       fmt(sum(s["values"]), 4) if s["values"] else "N/A",
                "Calculation":        (f"{sum(s['values']):.4f} / {s['n']} = {s['mean']:.4f}"
                                       if s["values"] else "N/A"),
                "Mean_JSON":          fmt(json_mean),
                "Std_JSON":           fmt(json_std),
                "Verification":       match,
            })

    fields = ["Method", "Method_key", "Metric", "Metric_key",
              "Mean_computed", "Std_computed", "N_scored",
              "Sum_computed", "Calculation",
              "Mean_JSON", "Std_JSON", "Verification"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  ✓ summary_metrics.csv          ({len(rows)} rows) → {path}")


def write_per_question_csv(per_question, path):
    all_qids = sorted(
        {q for m in METHODS for q in per_question[m]},
        key=lambda x: int(x[1:]) if x[1:].isdigit() else 0
    )
    rows = []
    for method in METHODS:
        for qid in all_qids:
            q = per_question[method].get(qid, {})
            rows.append({
                "Method":             METHOD_LABELS[method],
                "Method_key":         method,
                "Question_ID":        qid,
                "Question":           q.get("question", ""),
                "Ground_Truth":       q.get("ground_truth", ""),
                "Prediction":         q.get("prediction", ""),
                "Exact_Match":        fmt(q.get("exact_match"), 3),
                "Token_F1":           fmt(q.get("token_f1"), 4),
                "ROUGE_L":            fmt(q.get("rouge_l"), 4),
                "Answer_Correctness": fmt(q.get("answer_correctness"), 4),
                "Faithfulness":       fmt(q.get("faithfulness"), 4),
                "Context_Precision":  fmt(q.get("context_precision"), 4),
                "Context_Recall":     fmt(q.get("context_recall"), 4),
            })

    fields = ["Method", "Method_key", "Question_ID", "Question",
              "Ground_Truth", "Prediction",
              "Exact_Match", "Token_F1", "ROUGE_L",
              "Answer_Correctness", "Faithfulness",
              "Context_Precision", "Context_Recall"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  ✓ per_question_metrics.csv     ({len(rows)} rows) → {path}")


def write_comparison_csv(summary, sig_results, path):
    pairs  = list(itertools.combinations(METHODS, 2))
    rows   = []

    for metric in ALL_METRICS:
        row = {"Metric": METRIC_LABELS[metric], "Metric_key": metric}

        means = {}
        for method in METHODS:
            m = summary[method][metric]["mean"]
            means[method] = m
            row[f"{METHOD_LABELS[method]}_mean"] = fmt(m, 4)
            row[f"{METHOD_LABELS[method]}_std"]  = fmt(summary[method][metric]["std"], 4)

        valid = {k: v for k, v in means.items() if v is not None}
        best  = max(valid, key=valid.__getitem__) if valid else None
        row["Best_Method"] = METHOD_LABELS.get(best, "N/A")

        # Deltas vs hybrid_reranker
        reranker = means.get("hybrid_reranker")
        for method in [m for m in METHODS if m != "hybrid_reranker"]:
            base  = means[method]
            label = METHOD_LABELS[method].replace(" ", "_").replace("+", "plus")
            abs_d = None if (base is None or reranker is None) else round(reranker - base, 4)
            rel_d = pct_change(reranker, base)
            row[f"AbsDelta_vs_{label}"]  = fmt(abs_d, 4)
            row[f"RelDelta%_vs_{label}"] = fmt(rel_d, 2) if rel_d is not None else "N/A"

        # Significance p-values
        for (m1, m2) in pairs:
            r = (sig_results or {}).get(metric, {}).get((m1, m2), {})
            p = r.get("p")
            l1 = METHOD_LABELS[m1].replace(" ", "_").replace("+", "plus")
            l2 = METHOD_LABELS[m2].replace(" ", "_").replace("+", "plus")
            row[f"p_{l1}_vs_{l2}"]   = fmt(p, 4) if p is not None else "N/A"
            row[f"sig_{l1}_vs_{l2}"] = str(r.get("sig", "N/A"))

        rows.append(row)

    # Build fieldnames
    base_cols = (["Metric", "Metric_key"] +
                 [f"{METHOD_LABELS[m]}_mean" for m in METHODS] +
                 [f"{METHOD_LABELS[m]}_std"  for m in METHODS] +
                 ["Best_Method"])
    delta_cols = []
    for method in [m for m in METHODS if m != "hybrid_reranker"]:
        label = METHOD_LABELS[method].replace(" ", "_").replace("+", "plus")
        delta_cols += [f"AbsDelta_vs_{label}", f"RelDelta%_vs_{label}"]
    sig_cols = []
    for (m1, m2) in pairs:
        l1 = METHOD_LABELS[m1].replace(" ", "_").replace("+", "plus")
        l2 = METHOD_LABELS[m2].replace(" ", "_").replace("+", "plus")
        sig_cols += [f"p_{l1}_vs_{l2}", f"sig_{l1}_vs_{l2}"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=base_cols + delta_cols + sig_cols)
        w.writeheader()
        w.writerows(rows)
    print(f"  ✓ method_comparison.csv        ({len(rows)} rows) → {path}")


def write_clean_table_csv(summary, path):
    """The doc table — one row per method, with sum/n workings."""
    rows = []
    for method in METHODS:
        row = {"Method": METHOD_LABELS[method]}
        for metric in ALL_METRICS:
            s     = summary[method][metric]
            total = sum(s["values"]) if s["values"] else None
            label = METRIC_LABELS[metric]
            row[label]                         = fmt(s["mean"], 3)
            row[f"{label} (std)"]              = fmt(s["std"], 3)
            row[f"{label} sum"]                = fmt(total, 4) if total is not None else "N/A"
            row[f"{label} n"]                  = s["n"]
            row[f"{label} calculation"]        = (
                f"{total:.4f} / {s['n']} = {s['mean']:.4f}"
                if total is not None else "N/A"
            )
        rows.append(row)

    fields = ["Method"]
    for metric in ALL_METRICS:
        label = METRIC_LABELS[metric]
        fields += [label, f"{label} (std)", f"{label} sum",
                   f"{label} n", f"{label} calculation"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  ✓ clean_table_with_calcs.csv   ({len(rows)} rows) → {path}")


def write_significance_csv(sig_results, path):
    """All pairwise Wilcoxon results in one flat CSV."""
    if not sig_results:
        return

    rows = []
    for metric in ALL_METRICS:
        for (m1, m2), r in sig_results.get(metric, {}).items():
            rows.append({
                "Metric":          METRIC_LABELS[metric],
                "Metric_key":      metric,
                "Method_A":        METHOD_LABELS[m1],
                "Method_B":        METHOD_LABELS[m2],
                "Wilcoxon_stat":   fmt(r.get("stat"), 4) if r.get("stat") is not None else "N/A",
                "p_value":         fmt(r.get("p"), 4) if r.get("p") is not None else "N/A",
                "Significant":     str(r.get("sig", "N/A")),
                "Direction":       r.get("direction", "N/A"),
            })

    fields = ["Metric", "Metric_key", "Method_A", "Method_B",
              "Wilcoxon_stat", "p_value", "Significant", "Direction"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  ✓ significance_tests.csv       ({len(rows)} rows) → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    json_summary = data.get("summary", {})

    print("Extracting per-question scores ...")
    per_question = extract_per_question(data)

    print("Computing summary statistics ...")
    summary = compute_summary(per_question)

    # ── Console tables ────────────────────────────────────────────────────────
    print_summary_table(summary)

    print("\nRunning pairwise significance tests ...")
    sig_results = compute_significance(per_question, alpha=args.alpha)
    print_significance_table(sig_results, alpha=args.alpha)

    print_delta_table(summary)

    # ── CSVs ──────────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"Writing CSVs to {output_dir}/")
    print(f"{'─'*50}")

    write_summary_csv(     summary, json_summary,    output_dir / "summary_metrics.csv")
    write_per_question_csv(per_question,             output_dir / "per_question_metrics.csv")
    write_comparison_csv(  summary, sig_results,     output_dir / "method_comparison.csv")
    write_clean_table_csv( summary,                  output_dir / "clean_table_with_calcs.csv")
    write_significance_csv(sig_results,              output_dir / "significance_tests.csv")

    print(f"\nDone. 5 CSV files written to: {output_dir.resolve()}\n")


if __name__ == "__main__":
    main()