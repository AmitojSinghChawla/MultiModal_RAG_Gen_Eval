"""
generate.py
───────────
Converts retrieved chunks and a user question into a final answer using GPT-4o mini.

Pipeline position
─────────────────
  Called by  : streamlit.py → generate_answer()
               (called once per user message, after retrieve() returns chunks)
  Calls out to: ChatOpenAI (OpenAI API via LangChain) — the answering LLM

Input  (to generate_answer())
──────
  question         : the user's current plain-text question
                     ← from the chat input widget in streamlit.py
  retrieved_chunks : list of chunk dicts produced by retrieve() in retrieve.py
                     each dict may contain text, table content, or a raw base64 image
  chat_history     : the full list of past {"role", "content"} message dicts
                     ← from st.session_state["messages"] in streamlit.py
  api_key          : OpenAI API key string
                     ← from st.session_state / .env loaded in streamlit.py

Output  (→ back to streamlit.py)
──────
  answer    : str — the model's response, displayed in the chat UI
  ctx_texts : list[str] — the plain text of every context block used,
              shown in the "View sources" expander in streamlit.py
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


# ─────────────────────────────────────────
# 1. Constants
# ─────────────────────────────────────────

# CHAT_HISTORY_WINDOW: how many past conversation *turns* to include in each API call.
# 1 turn = 1 user message + 1 assistant message = 2 entries in the messages list.
# So a window of 3 means the last 3 * 2 = 6 messages are sent.
# Why 3: large windows increase token cost (GPT charges per token) and can push the
# retrieved context off the front of the context window. 3 turns gives enough
# continuity for follow-up questions ("what did you mean by that?") without bloat.
CHAT_HISTORY_WINDOW = 3

# Model used for answer generation — NOT the same as the vision model in Ingestion.py.
# GPT-4o-mini is used here because it is significantly cheaper and faster than GPT-4o
# for text generation, while still supporting multimodal input (images via image_url blocks).
# GPT-4o is only used in Ingestion.py for image description, where quality matters more.
Model = "gpt-4o-mini"


# ─────────────────────────────────────────
# 2. Context Builder
#    Converts the list of retrieved chunk dicts into a list of OpenAI content
#    blocks. This is the step that makes the pipeline "multimodal" — images are
#    sent as base64 data URIs alongside text and table blocks.
# ─────────────────────────────────────────

def _build_context_content(retrieved_chunks: list) -> tuple[list, list]:
    """
    Convert a list of retrieved chunk dicts into two parallel outputs:
      1. content  — a list of OpenAI API content blocks (text and image_url dicts)
                    ready to be placed in the "user" message sent to the LLM
      2. ctx_texts — a list of plain-text strings, one per chunk, used in the
                     "View sources" expander in streamlit.py

    Input
    ─────
      retrieved_chunks : list of chunk dicts from retrieve() in retrieve.py
                         each dict has keys: modality, source_pdf, page_number,
                         image_b64, raw_text, retrieval_text
                         ← produced by _make_result() in retrieve.py

    Output
    ──────
      content   : list of dicts in OpenAI content-block format
                  → passed to _build_messages() as context_content
                  → appended to the final user message sent to the LLM
      ctx_texts : list of plain-text strings (one per chunk)
                  → returned by generate_answer() to streamlit.py for the source expander

    Three chunk types are handled differently:

    IMAGE chunks  (modality == "image" and image_b64 is not None):
      The raw base64 image is sent directly to the LLM as an image_url content block
      using the data URI format "data:image/jpeg;base64,<base64string>".
      A short caption text block follows the image so the model knows which PDF and
      page it came from. detail="high" tells GPT-4o mini to process the image in
      high-resolution mode, important for charts with small text.
      ctx_texts gets the GPT-4o *description* (retrieval_text) rather than the raw
      base64, because the description is human-readable and fits in the UI expander.

    TEXT chunks   (modality == "text"):
      Sent as a plain [TEXT from <pdf>, page <n>] labelled text block.
      raw_text is preferred over retrieval_text because for text chunks they are the
      same, but the preference is consistent with the table case below.

    TABLE chunks  (modality == "table"):
      Sent as a [TABLE from <pdf>, page <n>] labelled text block.
      raw_text is the pipe-delimited version produced by html_table_to_text() in
      Ingestion.py — more readable by the LLM than raw HTML.
      The label tells the LLM to interpret this as structured data (e.g. "according
      to the table…") rather than a prose paragraph.
    """
    # content will become part of the final user message sent to the OpenAI API.
    # It is a list of content blocks — mixing text and image_url dicts is how
    # the multimodal API accepts mixed-modality input.
    content = []

    # ctx_texts is the human-readable source list returned to streamlit.py.
    # It has one entry per chunk, parallel to the content blocks above.
    ctx_texts = []

    for chunk in retrieved_chunks:
        source   = chunk["source_pdf"]    # filename for citation (e.g. "paper.pdf")
        page     = chunk["page_number"]   # page number for citation (None for images)
        modality = chunk["modality"]      # "text" | "table" | "image"

        # ── IMAGE CHUNK ───────────────────────────────────────────────────────
        # Only enter this branch if the chunk is an image AND we have the actual
        # base64 bytes. If image_b64 is None (shouldn't happen after the ingestion
        # fix, but defensive), fall through to the text/table branch which will
        # use the GPT-4o description as a text block instead.
        if modality == "image" and chunk.get("image_b64"):
            # OpenAI vision API expects images as data URIs inside image_url blocks.
            # The format is: "data:<mime-type>;base64,<base64string>"
            # This is the same format a browser uses to embed images inline in HTML.
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{chunk['image_b64']}",
                    # "high" detail mode tiles the image and processes each tile separately,
                    # needed for charts where axis labels or table values are in small text.
                    "detail": "high",
                },
            })

            # Follow the image with a short caption so the model can cite the source.
            # The model sees: [image] then [caption text] — the caption anchors the
            # image to a specific document and page in the model's response.
            content.append({
                "type": "text",
                "text": f"Image from {source} page {page}",
            })

            # For the sources expander in the UI: show the GPT-4o description
            # (retrieval_text) rather than the raw base64, which is not human-readable.
            ctx_texts.append(chunk["retrieval_text"])

        else:
            # ── TEXT or TABLE CHUNK ───────────────────────────────────────────
            # raw_text is the original extracted text (pipe-delimited for tables).
            # retrieval_text is the same for text/table chunks (no summarisation was applied).
            # We prefer raw_text because for tables it is the pipe-delimited version
            # from html_table_to_text(), which is more readable than the HTML original
            # that retrieval_text might contain if the fallback path ran in Ingestion.py.
            # The "or" fallback handles the edge case where raw_text is None.
            text = chunk.get("raw_text") or chunk.get("retrieval_text")

            # Label the block so the LLM knows the data type.
            # "TABLE" cues the model to treat this as structured rows/columns.
            # "TEXT" tells it this is prose — it will quote or paraphrase accordingly.
            label = "TABLE" if modality == "table" else "TEXT"

            content.append({
                "type": "text",
                "text": f"[{label} from {source}, page {page}]\n{text}\n",
            })

            # ctx_texts gets the same text shown to the LLM — what the user sees
            # in the "View sources" expander mirrors what the model received.
            ctx_texts.append(text)

    # → (content, ctx_texts) returned to generate_answer()
    #   content   → _build_messages() → appended to the final user message
    #   ctx_texts → returned by generate_answer() → streamlit.py source expander
    return content, ctx_texts


# ─────────────────────────────────────────
# 3. Message List Builder
#    Assembles the complete list of messages that gets sent to the OpenAI
#    Chat API in a single request. The list includes: a system instruction,
#    a sliding window of recent chat history, and the current query with
#    its retrieved context blocks.
# ─────────────────────────────────────────

def _build_messages(question: str, context_content: list, chat_history: list) -> list:
    """
    Assemble the complete messages list sent to the OpenAI Chat API for one turn.

    Input
    ─────
      question        : the user's current question string
                        ← from streamlit.py chat input, forwarded through generate_answer()
      context_content : list of OpenAI content blocks (text + image_url dicts)
                        ← produced by _build_context_content() just before this call
      chat_history    : the full chat history from st.session_state["messages"]
                        ← each entry is {"role": "user"|"assistant"|"system", "content": str}
                        ← the FULL history is passed in; windowing happens here

    Output: list of message dicts ready to pass to llm.invoke()
            → consumed by generate_answer() which passes it to ChatOpenAI

    Final message structure sent to the API:
    ┌─────────────────────────────────────────────────────────────┐
    │ { role: "system",    content: "<instructions>" }           │ ← always first
    │ { role: "user",      content: "<question N-2>" }           │ ← history window
    │ { role: "assistant", content: "<answer N-2>" }             │
    │ { role: "user",      content: "<question N-1>" }           │
    │ { role: "assistant", content: "<answer N-1>" }             │
    │ { role: "user",      content: [context blocks] + question }│ ← current turn
    └─────────────────────────────────────────────────────────────┘

    Why send history: allows the model to resolve anaphoric references
    ("what did you mean by that?", "compare it to the previous answer").
    Without history, every question is treated as independent.

    Why window the history: the API charges per input token. Sending the full
    unbounded history would grow the cost linearly with conversation length.
    A window of 3 turns covers most follow-up questions while staying cheap.

    Why content_content is a list (not a string) in the final user message:
    The OpenAI multimodal API expects the "content" field to be a list of
    content blocks when the message contains images. Strings are only valid
    for text-only messages.
    """
    messages = []

    # ── System message ────────────────────────────────────────────────────────
    # Sets the model's persona and hard constraints for this entire conversation.
    # It is always the first message so the model's behaviour is anchored before
    # it sees any history or the current question.
    # "strictly from the provided document context" instructs the model not to use
    # its parametric knowledge — reducing hallucinations about document content.
    messages.append({
        "role": "system",
        "content": (
            "You are a precise research assistant answering questions strictly "
            "from the provided document context (text, tables, and images). "
            "If the answer is not in the context, say 'Not found in context.' "
            "Be concise. Cite the source PDF and page where possible."
        ),
    })

    # ── Sliding history window ────────────────────────────────────────────────
    # CHAT_HISTORY_WINDOW = 3 turns.
    # 1 turn = 1 user message + 1 assistant message = 2 entries in the list.
    # [-6:] gives the 6 most recent messages (= 3 turns) from the full history.
    # "system" role messages in chat_history (ingestion notifications) are passed
    # through as-is — the API ignores role="system" mid-conversation by treating
    # it as additional context, which is acceptable behaviour.
    window = chat_history[-(CHAT_HISTORY_WINDOW) * 2:]

    for msg in window:
        # Re-package each history message into a clean dict.
        # The "sources" key present on assistant messages in session_state is
        # intentionally excluded — it is UI metadata, not conversation content.
        messages.append({
            "role":    msg["role"],
            "content": msg["content"],
        })

    # ── Current turn user message ─────────────────────────────────────────────
    # current_content is a copy of the context blocks so we don't mutate the
    # original list (which would corrupt it if generate_answer() were ever
    # called more than once with the same context_content reference).
    current_content = context_content.copy()

    # Append the actual question and instructions after all context blocks.
    # Placing the question AFTER the context is intentional: the model should
    # read the evidence first, then answer — reducing the chance of it generating
    # an answer from prior knowledge before processing the context.
    current_content.append({
        "type": "text",
        "text": (
            "\n─────────────────────────────────────\n"
            "Using only the context above, answer the question.\n"
            "If the context doesn't contain the answer, say 'Not found in context.'\n\n"
            f"Question: {question}\n\nAnswer:"
        ),
    })

    # The final user message uses a list of content blocks (not a plain string)
    # because it may contain image_url blocks. The OpenAI API requires this format
    # for multimodal messages — a plain string would cause the image blocks to be ignored.
    messages.append({
        "role":    "user",
        "content": current_content,    # list of text + image_url blocks
    })

    # → messages list returned to generate_answer() for llm.invoke()
    return messages


# ─────────────────────────────────────────
# 4. Main Entry Point
#    Orchestrates the full generation pipeline for one user turn:
#    retrieved chunks → context blocks → API messages → answer string.
# ─────────────────────────────────────────

def generate_answer(
    question: str,
    retrieved_chunks: list,
    chat_history: list,
    api_key: str,
) -> tuple[str, list]:
    """
    Full answer generation pipeline for one user turn.

    Input
    ─────
      question         : the user's current question (plain string)
                         ← from the chat input widget in streamlit.py
      retrieved_chunks : list of chunk dicts from retrieve() in retrieve.py
                         contains the top-k most relevant chunks, possibly
                         including text, table, and image modalities
      chat_history     : list of past {"role", "content"} dicts
                         ← st.session_state["messages"] from streamlit.py
                         The FULL history — windowing is done inside _build_messages()
      api_key          : OpenAI API key string
                         ← loaded from .env in streamlit.py, passed here per-call

    Output  (→ back to streamlit.py)
    ──────
      answer    : str — the model's text response, appended to session_state messages
                        and displayed in the chat UI
      ctx_texts : list[str] — plain text of every context block used, shown in the
                              "View sources" expander under each assistant message

    Call sequence inside this function:
      1. Initialise ChatOpenAI with the provided API key
      2. _build_context_content() → convert chunks into OpenAI content blocks
      3. Guard: if no content was built, return a canned "not found" message
      4. _build_messages() → assemble the full system + history + current-turn message list
      5. llm.invoke(messages) → send to GPT-4o mini and get an AIMessage response
      6. parser.invoke(response) → extract the plain text string from the AIMessage
      7. Return (answer, ctx_texts) to streamlit.py

    Why api_key is passed per-call rather than read from the environment:
      The key comes from st.session_state in streamlit.py (loaded from .env at startup).
      Passing it explicitly here keeps generate.py decoupled from Streamlit — it could
      be called from a script or a test without needing a running Streamlit session.

    Why temperature=0.1 (not 0):
      temperature=0 produces fully deterministic output, which can sometimes cause
      the model to repeat itself across similar queries. 0.1 adds minimal randomness
      to avoid robotic repetition while still keeping answers consistent and grounded.
    """

    # Initialise the LLM client for this call.
    # A new client is created per call because api_key is dynamic (from session_state).
    # LangChain's ChatOpenAI wraps the openai Python SDK and handles retries/formatting.
    llm = ChatOpenAI(model=Model, temperature=0.1, api_key=api_key)

    # StrOutputParser extracts the plain text string from LangChain's AIMessage object.
    # Without it, llm.invoke() returns an AIMessage — we need the .content string.
    parser = StrOutputParser()

    # Convert retrieved chunks into OpenAI API content blocks.
    # context_content: list of text/image_url dicts → sent to the LLM
    # ctx_texts:       list of plain strings          → returned to streamlit.py for the UI
    context_content, ctx_texts = _build_context_content(retrieved_chunks)

    # Guard: if retrieval found nothing (empty index, bad query, all chunks filtered out),
    # return early with a clear message rather than sending an empty context to the model.
    # An empty context would cause the model to either hallucinate from parametric knowledge
    # or produce a confused "not found" — we handle it explicitly for a cleaner UX message.
    if not context_content:
        return "No relevant context was found in the uploaded documents.", []

    # Assemble the complete messages list: system + history window + current turn.
    # This is what gets sent verbatim to the OpenAI Chat Completions API.
    messages = _build_messages(question, context_content, chat_history)

    try:
        # llm.invoke(messages) sends the entire message list to GPT-4o mini.
        # Returns an AIMessage object containing the model's response.
        response = llm.invoke(messages)

        # parser.invoke() extracts the plain .content string from the AIMessage object.
        answer = parser.invoke(response)

    except Exception as e:
        # Don't crash the Streamlit app on an API error.
        # Common causes: rate limit, invalid API key, network timeout, context too long.
        # The error is surfaced as the answer text so the user can see what went wrong.
        answer = f"Generation failed: {e}"

    # → (answer, ctx_texts) returned to streamlit.py
    #   answer    → appended to st.session_state["messages"] and rendered in the chat UI
    #   ctx_texts → stored in the assistant message dict under "sources" key,
    #               displayed in the "View sources" expander in streamlit.py
    return answer, ctx_texts