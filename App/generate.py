import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


CHAT_HISTORY_WINDOW=3
# How many past conversation to include in each API CALL
# 1 turn = 1 user message + 1 llm response so --> 2 Messages in total
# chat history window = 3 means that we include last 6 messages as context to keep it small as gpt charges per token

Model = "gpt-40-mini"


#BUILDING MULTIMODAL CONTEXT FROM CHUNKS

def _build_context(retrieved_chunks : list)-> tuple[list, list]:

    content=[]
    ctx_texts=[]


    for chunk in retrieved_chunks:
        source= chunk["source_pdf"]
        page= chunk["page_number"]
        modality= chunk["modality"]

        # ── IMAGE CHUNK ──────────────────────────────────────────────
        # OpenAI vision API expects images as data URIs inside image_url blocks.
        # A data URI looks like: "data:image/jpeg;base64,<base64string>"
        # "detail": "high" tells GPT-4o mini to use its high-res image
        # processing mode — important for charts with small text or fine detail.

        if modality == "image" and chunk.get("image_b64"):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{chunk['image_b64']}",
                    "detail": "high",
                },
            })


            content.append({
                "type": "text",
                "text": f"Image from {source} page {page}",
            })

            ctx_texts.append(chunk["retrieval_text"])


        else:
            # ── TEXT or TABLE CHUNK ───────────────────────────────────────
            # raw_text is the original extracted text from the PDF.
            # retrieval_text is the same for text/table chunks (no summarization).
            # We prefer raw_text here because for tables it's the pipe-delimited
            # version which is more readable than the HTML original.
            # .get("raw_text") returns None if the key is missing — the "or"
            # fallback to retrieval_text handles that edge case.
            text= chunk.get("raw_text") or chunk.get("retrieval_text")

            # Label the block so the LLM knows whether it's a table or plain text.
            # This helps it format the answer appropriately (e.g. "according to
            # the table..." vs "according to the text...").
            label = "TABLE" if modality == "table" else "TEXT"
            content.append({
                "type": "text",
                "text": f"[{label} from {source}, page {page}]\n{text}\n",
            })
            ctx_texts.append(text)

    return content, ctx_texts


def _build_messages(question : str, context_content: list, chat_history: list) -> list: