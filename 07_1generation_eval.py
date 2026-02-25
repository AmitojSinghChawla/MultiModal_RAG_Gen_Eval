"""
07_1generation_eval.py - FIXED VERSION
---------------------------------------
Phase 2: Generation + RAGAS evaluation.

Loads:
  retrieval_results.json

Saves:
  generation_results.json

FIXES APPLIED:
- Added comprehensive error handling
- Added file existence checks
- Added progress indicators
- Improved RAGAS compatibility
- Better error messages
"""

import json
import os
from dotenv import load_dotenv
from tqdm import tqdm

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

# RAGAS version compatibility
try:
    from ragas.llms import LangchainLLM
except ImportError:
    try:
        from ragas.llms import LangchainLLMWrapper as LangchainLLM
    except ImportError:
        raise ImportError(
            "Cannot import RAGAS LLM wrapper. "
            "Check your RAGAS version: pip install ragas>=0.1.0"
        )

load_dotenv()

INPUT_FILE  = "retrieval_results.json"
OUTPUT_FILE = "generation_results.json"
MODEL       = "gpt-4o-mini"


# ─────────────────────────────────────────
# Answer Generation (with error handling)
# ─────────────────────────────────────────

def generate_answer(question, retrieved_chunks):
    """
    Generate answer with comprehensive error handling.

    Returns:
        tuple: (answer_text, context_texts) or (error_message, [])
    """

    try:
        llm = ChatOpenAI(model=MODEL, temperature=0.0)

        content = []
        context_texts = []

        for chunk in retrieved_chunks:
            try:
                source = chunk.get("source_pdf", "unknown")
                modality = chunk.get("modality", "unknown")

                if modality == "image":
                    # Check if image data exists
                    if not chunk.get("image_b64"):
                        print(f"  ⚠️  Warning: Image chunk {chunk.get('chunk_id', 'unknown')[:10]} missing base64 data - skipping")
                        continue

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
                    context_texts.append(chunk.get("retrieval_text", ""))

                else:
                    # Text or table chunk
                    text = chunk.get("raw_text") or chunk.get("retrieval_text", "")
                    if not text:
                        print(f"  ⚠️  Warning: Chunk {chunk.get('chunk_id', 'unknown')[:10]} has no text - skipping")
                        continue

                    content.append({
                        "type": "text",
                        "text": f"[{modality.upper()} from {source}]\n{text}\n"
                    })
                    context_texts.append(text)

            except Exception as chunk_error:
                print(f"  ⚠️  Error processing chunk {chunk.get('chunk_id', 'unknown')[:10]}: {chunk_error}")
                continue

        # Check if we have any content
        if not content:
            return "ERROR: No valid chunks to process", []

        # Add instruction and question
        content.append({
            "type": "text",
            "text": (
                "Answer strictly based on the provided context.\n\n"
                f"Question: {question}\n\nAnswer:"
            )
        })

        messages = [{"role": "user", "content": content}]

        # Call LLM with timeout handling
        try:
            response = llm.invoke(messages)
            answer = StrOutputParser().invoke(response)
            return answer, context_texts
        except Exception as llm_error:
            return f"ERROR: LLM call failed - {str(llm_error)}", context_texts

    except Exception as e:
        print(f"  ❌ Fatal error in generate_answer: {e}")
        return f"ERROR: {str(e)}", []


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main():

    # Check environment
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY not set.\n"
            "Set it in .env file or: export OPENAI_API_KEY=your_key"
        )

    # Check input file exists
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"{INPUT_FILE} not found.\n"
            f"Run 06_1retrieval_eval.py first to generate retrieval results."
        )

    print("Loading retrieval results...")
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            retrieval_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"{INPUT_FILE} is corrupted: {e}")

    print(f"Loaded {len(retrieval_data)} retrieval methods")

    all_results = {}

    # Create RAGAS LLM wrapper
    try:
        ragas_llm = LangchainLLM(
            ChatOpenAI(model=MODEL, temperature=0.0)
        )
        print(f"Initialized RAGAS with {MODEL}")
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize RAGAS LLM: {e}")
        print("Continuing without RAGAS evaluation...")
        ragas_llm = None

    # Process each retrieval method
    for method, entries in retrieval_data.items():

        print(f"\n{'='*60}")
        print(f"Processing: {method.upper()}")
        print(f"{'='*60}")

        ragas_questions = []
        ragas_answers = []
        ragas_contexts = []
        ragas_ground_truths = []

        method_results = []

        # Generate answers with progress bar
        for entry in tqdm(entries, desc=f"Generating answers ({method})"):

            answer, context_texts = generate_answer(
                entry["question"],
                entry["retrieved_chunks"]
            )

            # Store for RAGAS
            ragas_questions.append(entry["question"])
            ragas_answers.append(answer)
            ragas_contexts.append(context_texts)
            ragas_ground_truths.append(entry["ground_truth"])

            # Store results
            method_results.append({
                "question_id": entry["question_id"],
                "question": entry["question"],
                "generated_answer": answer,
                "retrieval_metrics": entry.get("retrieval_metrics", {}),
            })

        # Run RAGAS evaluation if available
        if ragas_llm:
            print("\nRunning RAGAS evaluation...")

            try:
                dataset = Dataset.from_dict({
                    "question": ragas_questions,
                    "answer": ragas_answers,
                    "contexts": ragas_contexts,
                    "ground_truth": ragas_ground_truths,
                })

                ragas_scores = evaluate(
                    dataset,
                    metrics=[
                        faithfulness,
                        answer_relevancy,
                        context_precision,
                        context_recall
                    ],
                    llm=ragas_llm
                )

                ragas_df = ragas_scores.to_pandas()

                # Add RAGAS scores to results
                for i, row in ragas_df.iterrows():
                    method_results[i]["ragas_metrics"] = {
                        "faithfulness": float(row.get("faithfulness", 0)),
                        "answer_relevancy": float(row.get("answer_relevancy", 0)),
                        "context_precision": float(row.get("context_precision", 0)),
                        "context_recall": float(row.get("context_recall", 0)),
                    }

                print("✅ RAGAS evaluation complete")

            except Exception as ragas_error:
                print(f"⚠️  RAGAS evaluation failed: {ragas_error}")
                print("Continuing without RAGAS metrics...")

                # Add empty RAGAS metrics
                for result in method_results:
                    result["ragas_metrics"] = {
                        "faithfulness": None,
                        "answer_relevancy": None,
                        "context_precision": None,
                        "context_recall": None,
                        "error": str(ragas_error)
                    }
        else:
            # No RAGAS available
            for result in method_results:
                result["ragas_metrics"] = {
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "context_precision": None,
                    "context_recall": None,
                    "error": "RAGAS not initialized"
                }

        all_results[method] = method_results

    # Save results
    print(f"\n{'='*60}")
    print("Saving results...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"✅ Generation evaluation complete.")
    print(f"📄 Saved → {OUTPUT_FILE}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for method, results in all_results.items():
        n_questions = len(results)
        n_errors = sum(1 for r in results if r["generated_answer"].startswith("ERROR:"))
        print(f"  {method:20s}: {n_questions} questions, {n_errors} errors")


if __name__ == "__main__":
    main()