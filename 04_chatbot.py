"""
07_chatbot_gpt4o.py
-------------------
Multimodal RAG chatbot using GPT-4O Mini.

DIRECT MULTIMODAL: Retrieved chunks (text, tables, images) are sent
directly to GPT-4O Mini in ONE call. Images are sent as base64, text
as text. No separate vision model needed.

This is the optimal architecture:
  retrieve → send everything to GPT-4O Mini → answer

Usage:
    export OPENAI_API_KEY=your_key_here
    python 07_chatbot_gpt4o.py
    python 07_chatbot_gpt4o.py --method hybrid
    python 07_chatbot_gpt4o.py --top_k 3
"""

import argparse
import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from retrieve import load_indexes, retrieve
from dotenv import load_dotenv
load_dotenv()

DEFAULT_METHOD = "hybrid_reranker"
DEFAULT_TOP_K  = 5
MODEL          = "gpt-4o-mini"  # Fast, cheap, multimodal
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def generate_answer(question: str, retrieved_chunks: list) -> str:
    """
    Generate answer by sending question + all chunks directly to GPT-4O Mini.

    For image chunks: send the actual base64 image
    For text/table chunks: send the raw text

    ONE LLM call handles everything - no separate vision model needed.

    Parameters
    ----------
    question : str
        User's question
    retrieved_chunks : list
        Retrieved chunks from any retrieval method

    Returns
    -------
    str
        Generated answer
    """

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        return "ERROR: OPENAI_API_KEY environment variable not set"

    llm = ChatOpenAI(model=MODEL, temperature=0.1,api_key=OPENAI_API_KEY)

    # Build multimodal message content
    content = []

    # Add retrieved chunks
    for chunk in retrieved_chunks:
        source = chunk["source_pdf"]
        page = chunk.get("page_number", "?")
        modality = chunk["modality"]

        if modality == "image" and chunk.get("image_b64"):
            # ── IMAGE: Send directly to GPT-4O Mini ──
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{chunk['image_b64']}",
                    "detail": "high"  # Use high detail for better understanding
                }
            })
            content.append({
                "type": "text",
                "text": f"[Image from {source}, page {page}]"
            })

        elif modality == "table":
            # ── TABLE: Send as text ──
            text = chunk.get("raw_text") or chunk["retrieval_text"]
            content.append({
                "type": "text",
                "text": f"[TABLE from {source}, page {page}]\n{text}\n"
            })

        else:
            # ── TEXT: Send as text ──
            text = chunk.get("raw_text") or chunk["retrieval_text"]
            content.append({
                "type": "text",
                "text": f"[TEXT from {source}, page {page}]\n{text}\n"
            })

    # Add instruction and question
    content.append({
        "type": "text",
        "text": (
            "\n─────────────────────────────────────\n\n"
            "Based on the context above (text, tables, and images), "
            "answer the following question.\n\n"
            "Instructions:\n"
            "- Use only information from the provided context\n"
            "- Be concise and accurate\n"
            "- For images: describe what you see and how it relates to the question\n"
            "- If the context doesn't contain the answer, say so clearly\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
    })

    # Create message
    messages = [{"role": "user", "content": content}]

    # Generate answer
    response = llm.invoke(messages)
    return StrOutputParser().invoke(response)


def show_sources(retrieved_chunks: list) -> None:
    """Display retrieved sources with icons and metadata"""
    print("\n  Sources retrieved:")
    for chunk in retrieved_chunks:
        icon = "🖼" if chunk.get("image_b64") else "  "
        print(
            f"  {icon} [{chunk['rank']}] {chunk['modality']:6s} | "
            f"{chunk['source_pdf']:30s} | "
            f"score={chunk['score']:.4f} | "
            f"id={chunk['chunk_id'][:10]}..."
        )


def main(method: str, top_k: int) -> None:
    """Main chat loop"""

    # Check API key before loading indexes
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("\nSet it with:")
        print("  export OPENAI_API_KEY=your_key_here")
        return

    print("Loading indexes...")
    indexes = load_indexes()

    print(f"\n{'='*60}")
    print("Multimodal RAG Chatbot - GPT-4O Mini")
    print(f"{'='*60}")
    print(f"  Model     : {MODEL}")
    print(f"  Retrieval : {method.upper()}, top_k={top_k}")
    print(f"  Modality  : Text + Tables + Images (all in one call)")
    print(f"\nType your question and press Enter.")
    print(f"Type 'quit' or 'exit' to quit.\n")

    while True:
        print("─" * 60)
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Retrieve chunks
        retrieved = retrieve(question, method=method, indexes=indexes, top_k=top_k)

        if not retrieved:
            print("Bot: No relevant chunks found.\n")
            continue

        # Show what was retrieved
        show_sources(retrieved)

        # Count images
        n_images = sum(1 for c in retrieved if c.get("image_b64"))
        n_text = sum(1 for c in retrieved if c["modality"] in ("text", "table"))

        if n_images > 0:
            print(f"\n  📸 Sending {n_images} image(s) + {n_text} text chunk(s) to {MODEL}...")
        else:
            print(f"\n  📄 Sending {n_text} text chunk(s) to {MODEL}...")

        # Generate answer
        print(f"\nBot: ", end="", flush=True)
        answer = generate_answer(question, retrieved)
        print(answer)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multimodal RAG chatbot with GPT-4O Mini"
    )
    parser.add_argument(
        "--method",
        type=str,
        default=DEFAULT_METHOD,
        choices=["bm25", "dense", "hybrid", "hybrid_reranker"],
        help="Retrieval method (default: hybrid_reranker)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve (default: 5)",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top_k must be at least 1")

    main(args.method, args.top_k)