"""
05_annotate.py
--------------
Helps you build your gold_questions.json file for evaluation.

Browse chunks to find chunk_ids, then add gold questions interactively.
You can also edit gold_questions.json directly by hand — that is fine too.

gold_questions.json format:
[
  {
    "question_id"       : "q1",
    "question"          : "What was the revenue in Q3?",
    "relevant_chunk_ids": ["abc-123"],
    "ground_truth"      : "Revenue in Q3 was $5.2 million.",
    "modality"          : "table"
  },
  ...
]

Fix applied vs original:
  - Entry point now correctly reads from sys.argv instead of input().
    The original used input() so "--browse" / "--add" / "--show" flags
    passed on the command line were completely ignored.

Usage:
    python 05_annotate.py --browse        # browse chunks to find chunk_ids
    python 05_annotate.py --add           # add a new question interactively
    python 05_annotate.py --show          # show all questions added so far
"""

import json
import sys
import os


CHUNKS_FILE    = "chunks.json"
QUESTIONS_FILE = "gold_questions.json"


# ===============================
# Load helpers
# ===============================

def load_chunks() -> list[dict]:
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_questions() -> list[dict]:
    if not os.path.exists(QUESTIONS_FILE):
        return []
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_questions(questions: list[dict]) -> None:
    with open(QUESTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)


# ===============================
# Browse chunks
# ===============================

def browse_chunks(filter_modality: str = None) -> None:
    """
    Print chunks to the terminal so you can find the right chunk_id
    for each gold question you want to write.
    """
    chunks = load_chunks()

    if filter_modality:
        chunks = [c for c in chunks if c["modality"] == filter_modality]
        print(f"\nShowing {len(chunks)} chunks with modality='{filter_modality}'\n")
    else:
        print(f"\nShowing all {len(chunks)} chunks\n")

    for i, chunk in enumerate(chunks):
        print(f"{'─'*60}")
        print(f"Index        : {i}")
        print(f"chunk_id     : {chunk['chunk_id']}")
        print(f"modality     : {chunk['modality']}")
        print(f"source_pdf   : {chunk['source_pdf']}")
        print(f"page_number  : {chunk['page_number']}")
        print(f"retrieval_text (first 300 chars):")
        print(f"  {chunk['retrieval_text'][:300]}")
        print()

        # Pause every 5 chunks so the terminal doesn't flood
        if (i + 1) % 5 == 0:
            cont = input("Press Enter to see more, or q to quit: ")
            if cont.strip().lower() == "q":
                break


# ===============================
# Add a question interactively
# ===============================

def add_question() -> None:
    """Prompt the user to enter a gold question and save it."""
    questions = load_questions()

    print("\nAdd a new gold question")
    print("(Browse chunks first to find the correct chunk_id)\n")

    question_id = f"q{len(questions) + 1}"

    question = input("Question: ").strip()
    if not question:
        print("Question cannot be empty.")
        return

    chunk_ids_input    = input("Relevant chunk_id(s) — comma separated: ").strip()
    relevant_chunk_ids = [c.strip() for c in chunk_ids_input.split(",") if c.strip()]
    if not relevant_chunk_ids:
        print("Must provide at least one chunk_id.")
        return

    ground_truth = input("Ground truth answer: ").strip()
    if not ground_truth:
        print("Ground truth cannot be empty — RAGAS needs this.")
        return

    modality = input("Modality of the answer chunk (text / table / image): ").strip()
    if modality not in ("text", "table", "image"):
        print(f"Warning: '{modality}' is not a standard modality. Saving anyway.")

    record = {
        "question_id"       : question_id,
        "question"          : question,
        "relevant_chunk_ids": relevant_chunk_ids,
        "ground_truth"      : ground_truth,
        "modality"          : modality,
    }

    questions.append(record)
    save_questions(questions)

    print(f"\nSaved question '{question_id}' to {QUESTIONS_FILE}")
    print(json.dumps(record, indent=2))


# ===============================
# Show all questions
# ===============================

def show_questions() -> None:
    questions = load_questions()

    if not questions:
        print(f"\nNo questions yet in {QUESTIONS_FILE}")
        print("Run:  python 05_annotate.py --add")
        return

    print(f"\n{len(questions)} gold question(s) in {QUESTIONS_FILE}\n")

    for q in questions:
        print(f"{'─'*60}")
        print(f"ID           : {q['question_id']}")
        print(f"Question     : {q['question']}")
        print(f"Chunk IDs    : {q['relevant_chunk_ids']}")
        print(f"Ground Truth : {q['ground_truth']}")
        print(f"Modality     : {q['modality']}")
    print()


# ===============================
# Entry point
# ===============================

if __name__ == "__main__":

    # FIX: read flag from sys.argv, not from input().
    # Original code used input() so command-line flags were never read.
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python 05_annotate.py --browse")
        print("  python 05_annotate.py --add")
        print("  python 05_annotate.py --show")
        sys.exit(1)

    flag = sys.argv[1].strip()

    if flag == "--browse":
        modality = input(
            "Filter by modality (text / table / image) or press Enter for all: "
        ).strip()
        modality = modality if modality in ("text", "table", "image") else None
        browse_chunks(filter_modality=modality)

    elif flag == "--add":
        add_question()

    elif flag == "--show":
        show_questions()

    else:
        print(f"Unknown option: '{flag}'. Use --browse, --add, or --show.")
        sys.exit(1)
