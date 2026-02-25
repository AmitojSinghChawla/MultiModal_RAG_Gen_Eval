"""
metrics.py
----------
Modular, standalone metric functions for end-to-end RAG evaluation.

All metrics compare a generated prediction against a ground-truth answer.
No chunk-level relevance annotations are required.

Metrics
-------
  exact_match(prediction, ground_truth) -> float
  token_f1(prediction, ground_truth)    -> float
  rouge_l(prediction, ground_truth)     -> float
  compute_all(prediction, ground_truth) -> dict
  aggregate(per_question_scores)        -> dict

Usage example:
    from metrics import compute_all, aggregate

    scores = [compute_all(pred, gt) for pred, gt in pairs]
    summary = aggregate(scores)
"""

import re
import string
import statistics
from collections import Counter

from rouge_score import rouge_scorer as _rouge_scorer


# ─────────────────────────────────────────
# Text normalisation
# ─────────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    Prepare text for string comparison.

    Steps: lowercase → strip punctuation → collapse whitespace.
    Matches the normalisation used in SQuAD-style EM/F1 evaluation.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    """Whitespace-split after normalisation."""
    return normalize_text(text).split()


# ─────────────────────────────────────────
# Individual metrics
# ─────────────────────────────────────────

def exact_match(prediction: str, ground_truth: str) -> float:
    """
    Exact Match (EM).

    Returns 1.0 if the normalised prediction equals the normalised
    ground truth, else 0.0.
    """
    return float(normalize_text(prediction) == normalize_text(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1.

    Counts the token overlap between prediction and ground truth,
    then computes the harmonic mean of precision and recall.

    This is the standard metric used in SQuAD extractive QA evaluation
    and handles paraphrasing better than EM.
    """
    pred_tokens = _tokenize(prediction)
    gt_tokens   = _tokenize(ground_truth)

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = sum((Counter(pred_tokens) & Counter(gt_tokens)).values())

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall    = common / len(gt_tokens)
    f1        = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def rouge_l(prediction: str, ground_truth: str) -> float:
    """
    ROUGE-L F1.

    Measures the longest common subsequence (LCS) between prediction
    and ground truth, normalised as an F1 score.
    Stemming is applied for robustness.
    """
    scorer = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    result = scorer.score(ground_truth, prediction)
    return round(result["rougeL"].fmeasure, 4)


# ─────────────────────────────────────────
# Batch helpers
# ─────────────────────────────────────────

def compute_all(prediction: str, ground_truth: str) -> dict:
    """
    Compute all string-based metrics for a single (prediction, ground_truth) pair.

    Returns
    -------
    dict with keys: exact_match, token_f1, rouge_l
    """
    return {
        "exact_match": exact_match(prediction, ground_truth),
        "token_f1"   : token_f1(prediction, ground_truth),
        "rouge_l"    : rouge_l(prediction, ground_truth),
    }


def _mean_std(values: list) -> tuple[float, float]:
    """Return (mean, std) over a list, ignoring None entries."""
    clean = [v for v in values if v is not None]
    if not clean:
        return 0.0, 0.0
    m = statistics.mean(clean)
    s = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return round(m, 4), round(s, 4)


def aggregate(per_question_scores: list[dict]) -> dict:
    """
    Aggregate a list of per-question metric dicts.

    Accepts dicts that may contain both string-based metrics
    (exact_match, token_f1, rouge_l) and RAGAS metrics
    (answer_correctness, faithfulness, context_precision, context_recall).

    Returns a summary dict where each key maps to {"mean": ..., "std": ...}.
    """
    all_keys = set()
    for entry in per_question_scores:
        all_keys.update(entry.keys())

    # Remove non-numeric keys
    all_keys -= {"error"}

    summary = {}
    for key in sorted(all_keys):
        values = [entry.get(key) for entry in per_question_scores]
        m, s   = _mean_std(values)
        summary[key] = {"mean": m, "std": s}

    return summary