"""
03_evaluate.py — Multimodal RAG evaluation
Methods: bm25 | dense | hybrid | hybrid_reranker
Metrics: exact_match, token_f1, rouge_l + RAGAS (answer_correctness,
         faithfulness, context_precision, context_recall)
"""

import asyncio

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import json, math, os, re, string, time, statistics
from collections import Counter

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from rouge_score import rouge_scorer as rouge_lib
from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    answer_correctness,
    faithfulness,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from retrieve import load_indexes, retrieve

load_dotenv()

# ── Config ────────────────────────────────────────────────────
QUESTIONS_FILE = "gold_questions.json"
RESULTS_FILE = "evaluation_results.json"
METHODS = ["bm25", "dense", "hybrid", "hybrid_reranker"]
TOP_K = 5
MODEL = "gpt-4o-mini"
API_KEY = os.environ.get("OPENAI_API_KEY")
RETRY_DELAYS = [5, 15, 30]  # seconds to wait between retries


# ── Rate-limit-safe API caller ────────────────────────────────
def call_with_retry(fn, *args, **kwargs):
    for i, delay in enumerate(RETRY_DELAYS + [None]):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if delay is None or "rate" not in str(e).lower():
                raise
            print(f"  Rate limit hit, retrying in {delay}s...")
            time.sleep(delay)


# ── Disk I/O ──────────────────────────────────────────────────
def load_results():
    if not os.path.exists(RESULTS_FILE):
        return {"summary": {}, "per_question": {}}
    return json.load(open(RESULTS_FILE, encoding="utf-8"))


def save_results(per_q, summary):
    json.dump(
        {"summary": summary, "per_question": per_q},
        open(RESULTS_FILE, "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )


# ── Resume helpers ────────────────────────────────────────────
def answer_done(results, method, qid):
    return any(e["question_id"] == qid for e in results.get(method, []))


def ragas_done(entry):
    rm = entry.get("ragas_metrics")
    return isinstance(rm, dict) and isinstance(rm.get("faithfulness"), float)


# ── String metrics ────────────────────────────────────────────
def normalise(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def exact_match(pred, gt):
    return float(normalise(pred) == normalise(gt))


def token_f1(pred, gt):
    p, g = normalise(pred).split(), normalise(gt).split()
    if not p or not g:
        return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    if not common:
        return 0.0
    return round(
        2
        * (common / len(p))
        * (common / len(g))
        / ((common / len(p)) + (common / len(g))),
        4,
    )


def rouge_l(pred, gt):
    return round(
        rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
        .score(gt, pred)["rougeL"]
        .fmeasure,
        4,
    )


def string_metrics(pred, gt):
    return {
        "exact_match": exact_match(pred, gt),
        "token_f1": token_f1(pred, gt),
        "rouge_l": rouge_l(pred, gt),
    }


# ── Answer generation ─────────────────────────────────────────
def generate_answer(question, chunks):
    llm, content, ctx_texts = (
        ChatOpenAI(model=MODEL, temperature=0, api_key=API_KEY),
        [],
        [],
    )

    for chunk in chunks:
        src, mod = chunk["source_pdf"], chunk["modality"]
        if mod == "image" and chunk.get("image_b64"):
            content += [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{chunk['image_b64']}",
                        "detail": "high",
                    },
                },
                {"type": "text", "text": f"[Image from {src}]"},
            ]
            ctx_texts.append(chunk["retrieval_text"])
        else:
            text = chunk.get("raw_text") or chunk["retrieval_text"]
            content.append(
                {"type": "text", "text": f"[{mod.upper()} from {src}]\n{text}\n"}
            )
            ctx_texts.append(text)

    content.append(
        {
            "type": "text",
            "text": (
                "\nAnswer the question based only on the context above. "
                "Be concise. If the answer isn't in the context, say 'Not found in context.'\n\n"
                f"Question: {question}\n\nAnswer:"
            ),
        }
    )

    response = call_with_retry(llm.invoke, [{"role": "user", "content": content}])
    return StrOutputParser().invoke(response), ctx_texts


# ── RAGAS scoring ─────────────────────────────────────────────
def safe_float(val):
    try:
        f = float(val)
        return round(f, 4) if not math.isnan(f) else None
    except (TypeError, ValueError):
        return None


def score_ragas(question, answer, contexts, ground_truth):
    llm = LangchainLLMWrapper(ChatOpenAI(model=MODEL, temperature=0, api_key=API_KEY))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=API_KEY))

    metrics = []
    for m in [answer_correctness, faithfulness, context_precision, context_recall]:
        inst = m.__class__()
        inst.llm, inst.embeddings = llm, emb
        metrics.append(inst)

    dataset = Dataset.from_dict(
        {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts or ["No context available."]],
            "ground_truth": [ground_truth],
        }
    )

    def _run():
        return ragas_evaluate(dataset, metrics=metrics, llm=llm, embeddings=emb)

    scores = call_with_retry(_run)
    row = scores.to_pandas().iloc[0]
    return {
        k: safe_float(row.get(k))
        for k in [
            "answer_correctness",
            "faithfulness",
            "context_precision",
            "context_recall",
        ]
    }


# ── Aggregation ───────────────────────────────────────────────
def mean_std(vals):
    clean = [v for v in vals if isinstance(v, float)]
    if not clean:
        return 0.0, 0.0
    return round(statistics.mean(clean), 4), round(
        statistics.stdev(clean) if len(clean) > 1 else 0.0, 4
    )


def compute_summary(entries):
    scored = [e for e in entries if ragas_done(e)]
    return {
        k: {"mean": m, "std": s}
        for k, (m, s) in {
            "exact_match": mean_std(
                [e["string_metrics"]["exact_match"] for e in entries]
            ),
            "token_f1": mean_std([e["string_metrics"]["token_f1"] for e in entries]),
            "rouge_l": mean_std([e["string_metrics"]["rouge_l"] for e in entries]),
            "answer_correctness": mean_std(
                [e["ragas_metrics"]["answer_correctness"] for e in scored]
            ),
            "faithfulness": mean_std(
                [e["ragas_metrics"]["faithfulness"] for e in scored]
            ),
            "context_precision": mean_std(
                [e["ragas_metrics"]["context_precision"] for e in scored]
            ),
            "context_recall": mean_std(
                [e["ragas_metrics"]["context_recall"] for e in scored]
            ),
        }.items()
    }


# ── Summary table ─────────────────────────────────────────────
def print_summary(summary):
    keys = [
        "exact_match",
        "token_f1",
        "rouge_l",
        "answer_correctness",
        "faithfulness",
        "context_precision",
        "context_recall",
    ]
    heads = [
        "Method",
        "EM",
        "F1",
        "ROUGE-L",
        "AnsCorr",
        "Faithful",
        "CtxPrec",
        "CtxRecall",
    ]
    w = 13
    print(f"\n{'='*70}\nEVALUATION SUMMARY (mean ± std)\n{'='*70}")
    print(f"{'Method':<18}" + "".join(f"{h:>{w}}" for h in heads[1:]))
    print("─" * (18 + w * len(keys)))
    for method, s in summary.items():
        row = f"{method:<18}"
        for k in keys:
            m = (s.get(k) or {}).get("mean", 0.0) or 0.0
            d = (s.get(k) or {}).get("std", 0.0) or 0.0
            row += f"{f'{m:.3f}±{d:.3f}':>{w}}"
        print(row)
    print(f"─" * (18 + w * len(keys)))
    print(f"\nModel: {MODEL}  |  Top-K: {TOP_K}\n{'='*70}\n")


# ── Main ──────────────────────────────────────────────────────
def main():
    if not API_KEY:
        print("ERROR: OPENAI_API_KEY not set.")
        return

    questions = json.load(open(QUESTIONS_FILE, encoding="utf-8"))
    print(f"Loaded {len(questions)} questions")

    data = load_results()
    per_q = data["per_question"]
    summary = data["summary"]
    indexes = load_indexes()

    for method in METHODS:
        print(f"\n{'='*60}\nMETHOD: {method.upper()}\n{'='*60}")
        per_q.setdefault(method, [])

        # Phase 1 — Generate answers
        for i, q in enumerate(questions, 1):
            qid, quest, gt = (
                q.get("question_id", f"q{i}"),
                q["question"],
                q["ground_truth"],
            )

            if answer_done(per_q, method, qid):
                print(f"  [{i:02d}] {qid} — skipping (done)")
                continue

            print(f"  [{i:02d}] {qid}: {quest[:65]}...")
            chunks = retrieve(quest, method=method, indexes=indexes, top_k=TOP_K)

            try:
                answer, ctx_texts = generate_answer(quest, chunks)
            except Exception as e:
                print(f"  Generation failed: {e}")
                answer, ctx_texts = f"ERROR: {e}", []

            sm = string_metrics(answer, gt)
            print(
                f"       EM={sm['exact_match']:.2f}  F1={sm['token_f1']:.4f}  ROUGE-L={sm['rouge_l']:.4f}"
            )

            per_q[method].append(
                {
                    "question_id": qid,
                    "question": quest,
                    "ground_truth": gt,
                    "prediction": answer,
                    "context_texts": ctx_texts,
                    "retrieved_ids": [c["chunk_id"] for c in chunks],
                    "string_metrics": sm,
                    "ragas_metrics": None,
                }
            )
            summary[method] = compute_summary(per_q[method])
            save_results(per_q, summary)

        # Phase 2 — RAGAS
        pending = [e for e in per_q[method] if not ragas_done(e)]
        print(
            f"\n  RAGAS: {len(pending)} to score, {len(per_q[method]) - len(pending)} already done"
        )

        for i, entry in enumerate(pending, 1):
            print(
                f"  [{i:02d}/{len(pending)}] {entry['question_id']}...",
                end=" ",
                flush=True,
            )
            try:
                rs = score_ragas(
                    entry["question"],
                    entry["prediction"],
                    entry.get("context_texts", []),
                    entry["ground_truth"],
                )
                entry["ragas_metrics"] = rs
                print(
                    f"AC={rs['answer_correctness']}  F={rs['faithfulness']}  "
                    f"CP={rs['context_precision']}  CR={rs['context_recall']}"
                )
            except Exception as e:
                print(f"FAILED: {e}")
                entry["ragas_metrics"] = {
                    k: None
                    for k in [
                        "answer_correctness",
                        "faithfulness",
                        "context_precision",
                        "context_recall",
                    ]
                }
                entry["ragas_metrics"]["error"] = str(e)

            summary[method] = compute_summary(per_q[method])
            save_results(per_q, summary)

        # Strip context_texts from completed entries
        for entry in per_q[method]:
            if ragas_done(entry):
                entry.pop("context_texts", None)

        summary[method] = compute_summary(per_q[method])
        save_results(per_q, summary)
        s = summary[method]
        print(
            f"\n  {method}: EM={s['exact_match']['mean']:.3f}  F1={s['token_f1']['mean']:.3f}  "
            f"AnsCorr={s['answer_correctness']['mean']:.3f}  Faithful={s['faithfulness']['mean']:.3f}"
        )

    print_summary(summary)


if __name__ == "__main__":
    main()
