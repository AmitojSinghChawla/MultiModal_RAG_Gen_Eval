"""
07_chatbot.py
-------------
Terminal chatbot over your ingested PDFs.

Uses hybrid + reranker (best retrieval method) to find relevant chunks,
then passes them to Ollama to generate an answer.

MULTIMODAL: When retrieved chunks include image chunks, the actual
base64 image is sent directly to the vision LLM (llava) so it can *see*
the image. Text and table chunks are sent as plain text to gemma:2b.
If a query returns a mix of image and text chunks, we make two calls:
  1. Ask llava to describe what it sees in each image in the context of
     the question, producing a rich text description.
  2. Feed that description + all text context to gemma:2b for the final answer.

Usage:
    python 07_chatbot.py
    python 07_chatbot.py --method hybrid
    python 07_chatbot.py --top_k 3
"""

import argparse
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from retrieve import load_indexes, retrieve


DEFAULT_METHOD   = "hybrid_reranker"
DEFAULT_TOP_K    = 5
TEXT_MODEL       = "gemma:2b"       # handles text + table chunks
VISION_MODEL     = "llava"          # handles image chunks (vision LLM)


# ─────────────────────────────────────────
# Step 1: Ask vision LLM about each image
# ─────────────────────────────────────────

def describe_image_for_question(question: str, image_b64: str, source_pdf: str) -> str:
    """
    Send the actual image to llava along with the user's question.

    Instead of relying on the pre-generated LLaVA description stored in
    retrieval_text (which is generic), we now ask the vision model to look
    at the image *in the context of this specific question*. This produces
    a much more targeted and useful description.

    Parameters
    ----------
    question   : the user's question (used as context for the description)
    image_b64  : raw base64 image string from the retrieved chunk
    source_pdf : PDF filename — included in the description for traceability

    Returns
    -------
    A focused text description of what the image contains, relevant to the question.
    """
    vision_llm = ChatOllama(model=VISION_MODEL, temperature=0.0)

    # Build a multimodal message: image + instruction text
    # LangChain's ChatOllama accepts image_url with a base64 data URI
    data_url = f"data:image/jpeg;base64,{image_b64}"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
                {
                    "type": "text",
                    "text": (
                        f"The user asked: '{question}'\n\n"
                        f"This image is from the document: {source_pdf}\n\n"
                        "Please describe what you see in this image in detail, "
                        "focusing on information that is relevant to the user's question. "
                        "If it is a chart or graph, state the values and trends precisely. "
                        "If it is a diagram, explain the structure and what it represents."
                    ),
                },
            ],
        }
    ]

    response = vision_llm.invoke(messages)
    return StrOutputParser().invoke(response)


# ─────────────────────────────────────────
# Step 2: Build full context & generate answer
# ─────────────────────────────────────────

def generate_answer(question: str, retrieved_chunks: list) -> str:
    """
    Build a multimodal context and generate a final answer.

    For each retrieved chunk:
      - text / table chunks → use raw_text directly as context
      - image chunks        → call the vision LLM to get a question-focused
                              description of the actual image, then use that
                              as context for the text LLM

    All context (text + vision-generated image descriptions) is then passed
    to the text LLM (gemma:2b) for the final answer.

    This is true multimodal RAG:
      retrieve (text proxy) → see (vision LLM on real image) → answer (text LLM)
    """
    context_parts = []

    for chunk in retrieved_chunks:
        source = chunk["source_pdf"]
        modality = chunk["modality"]

        if modality == "image" and chunk.get("image_b64"):
            # ── MULTIMODAL PATH ──────────────────────────────────────────
            # We have the real image — ask llava to describe it in context
            print(f"   👁  Sending image from {source} to {VISION_MODEL}...", end=" ", flush=True)
            vision_description = describe_image_for_question(
                question, chunk["image_b64"], source
            )
            print("done")

            context_parts.append(
                f"[IMAGE from {source}]\n"
                f"Visual analysis: {vision_description}"
            )

        elif modality == "image" and not chunk.get("image_b64"):
            # Fallback: image_b64 missing (shouldn't happen, but be safe)
            context_parts.append(
                f"[IMAGE from {source} — pre-generated description]\n"
                f"{chunk['retrieval_text']}"
            )

        elif modality == "table":
            context_parts.append(
                f"[TABLE from {source}]\n"
                f"{chunk.get('raw_text') or chunk['retrieval_text']}"
            )

        else:
            # text chunk
            context_parts.append(
                f"[TEXT from {source}]\n"
                f"{chunk.get('raw_text') or chunk['retrieval_text']}"
            )

    context = "\n\n".join(context_parts)

    # Final answer generation via text LLM
    text_llm = ChatOllama(model=TEXT_MODEL, temperature=0.1)

    prompt_text = (
        "You are a helpful assistant answering questions based on document excerpts.\n"
        "The context below may include text passages, table data, and descriptions of images.\n"
        "Answer using only the information provided. "
        "If the context does not contain the answer, say so clearly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    response = text_llm.invoke(prompt_text)
    return StrOutputParser().invoke(response)


# ─────────────────────────────────────────
# Show sources
# ─────────────────────────────────────────

def show_sources(retrieved_chunks: list) -> None:
    print("\n  Sources retrieved:")
    for chunk in retrieved_chunks:
        has_image = "🖼 " if chunk.get("image_b64") else "   "
        print(
            f"  {has_image}[{chunk['rank']}] {chunk['modality']:6s} | "
            f"{chunk['source_pdf']} | "
            f"score={chunk['score']:.4f} | "
            f"id={chunk['chunk_id'][:8]}..."
        )


# ─────────────────────────────────────────
# Main chat loop
# ─────────────────────────────────────────

def main(method: str, top_k: int) -> None:

    print("Loading indexes...")
    indexes = load_indexes()

    print(f"\nChatbot ready")
    print(f"  Retrieval : {method.upper()}, top_k={top_k}")
    print(f"  Text LLM  : {TEXT_MODEL}")
    print(f"  Vision LLM: {VISION_MODEL} (used for image chunks)")
    print("Type your question and press Enter.  Type 'quit' to exit.\n")

    while True:
        print("─" * 55)
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        retrieved = retrieve(question, method=method, indexes=indexes, top_k=top_k)

        if not retrieved:
            print("Bot: No relevant chunks found.")
            continue

        show_sources(retrieved)

        n_images = sum(1 for c in retrieved if c.get("image_b64"))
        if n_images:
            print(f"\n  Sending {n_images} image(s) to {VISION_MODEL} for visual analysis...")

        print(f"\nBot:", end=" ", flush=True)
        answer = generate_answer(question, retrieved)
        print(answer)
        print()


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal RAG chatbot")
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
