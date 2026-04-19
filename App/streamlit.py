"""
streamlit.py
────────────
The Streamlit application entry point — renders the chat UI and orchestrates
the full RAG pipeline from PDF upload through to answer display.

Pipeline position
─────────────────
  This is the top of the pipeline. It:
    1. Loads models once at startup (SentenceTransformer + CrossEncoder)
    2. Accepts PDF uploads and triggers ingest_pdfs() in Ingestion.py
    3. Accepts user questions and calls retrieve() in retrieve.py
    4. Passes retrieved chunks to generate_answer() in generate.py
    5. Renders answers and source citations in the chat UI

Input  (from the user via the browser)
──────
  PDF files : uploaded via st.file_uploader in the sidebar
  Query     : typed into st.chat_input in the main area
  top_k     : selected via the sidebar slider

Output  (rendered in the browser)
──────
  Chat UI : user messages, assistant answers, source expanders
  Sidebar  : ingestion status, chunk counts, top-k slider, reset button
"""

import streamlit as st
import os
import sys
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()

# OPENAI_API_KEY is loaded from .env at startup.
# It is passed to generate_answer() in generate.py on every query.
# streamlit.py does NOT validate the key — it shows a warning if missing
# and disables the "Ingest PDFs" button, so the user cannot proceed without it.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Add the App/ directory to the Python path so that `from Ingestion import …`
# and `from retrieve import …` resolve correctly regardless of where Streamlit
# is launched from. Without this, imports would fail if the working directory
# is not the App/ folder.
sys.path.append(os.path.join(os.path.dirname(__file__), "codebase"))
sys.path.append(os.path.join(os.path.dirname(__file__), "app", "utils"))

# ← ingest_pdfs() is the main entry point in Ingestion.py
from Ingestion import ingest_pdfs
# ← generate_answer() converts retrieved chunks + question → final answer string
from generate import generate_answer
# ← retrieve() is the unified entry point in retrieve.py
from retrieve import retrieve


# ─────────────────────────────────────────
# 1. Model Loading
#    Both models are large (~500 MB combined) and slow to load (~10–30 seconds).
#    @st.cache_resource ensures they are loaded ONCE when the app first starts
#    and then reused for every user interaction and every Streamlit rerun.
#    Without this decorator, Streamlit would reload the models on every button
#    click or slider change, making the app unusably slow.
# ─────────────────────────────────────────

@st.cache_resource
def load_models():
    """
    Load and cache the embedding and reranking models at application startup.

    Output: (embedder, reranker) — both are returned together so the cache
            holds a single resource entry rather than two separate ones.
            → embedder passed to ingest_pdfs() in Ingestion.py (for FAISS index building)
            → embedder stored in indexes["embedder"] and used by retrieve.py at query time
            → reranker passed to ingest_pdfs() in Ingestion.py (passed through to indexes)
            → reranker stored in indexes["reranker"] and used by retrieve_hybrid_reranked()

    Why BAAI/bge-base-en-v1.5:
        Strong general-purpose English embedding model. The "base" size (768 dims)
        gives good quality without the memory cost of "large" (1024 dims).
        bge models are specifically trained for retrieval tasks (bi-encoder style).

    Why BAAI/bge-reranker-v2-m3:
        Cross-encoder that jointly encodes (query, document) pairs. Significantly
        more accurate than bi-encoder cosine similarity for final reranking.
        v2-m3 is multilingual and handles English documents well.
    """
    # embedder: converts text strings → 768-dim numpy vectors
    # → used at index time (Ingestion.py → build_faiss) and query time (retrieve.py → retrieve_dense)
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")

    # reranker: scores (query, document) pairs → single relevance float
    # → used only at query time (retrieve.py → retrieve_hybrid_reranked)
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

    # → (embedder, reranker) stored by @st.cache_resource, unpacked at module level below
    return embedder, reranker


# ─────────────────────────────────────────
# 2. Page Configuration
#    Must be the first Streamlit call after imports.
#    "wide" layout uses the full browser width — important for the chat UI
#    which needs horizontal space for message bubbles.
# ─────────────────────────────────────────

st.set_page_config(
    page_title="Multimodal RAG",
    layout="wide",
)

# Unpack models from the cache — this call is instant after the first load.
# embedder and reranker are module-level variables used throughout the session.
# → passed to ingest_pdfs() in Ingestion.py when the user clicks "Ingest PDFs"
embedder, reranker = load_models()


# ─────────────────────────────────────────
# 3. Session State Initialisation
#    st.session_state persists across Streamlit reruns (e.g. button clicks,
#    slider moves). Without these defaults, accessing a key that doesn't exist
#    yet would raise a KeyError on the very first run.
# ─────────────────────────────────────────

# messages: the full conversation history as a list of {"role", "content"} dicts.
# Role values used: "user", "assistant", "system".
# → read by generate.py → _build_messages() for the sliding context window
# → written here after each user query and each assistant response
if "messages" not in st.session_state:
    st.session_state.messages = []

# ingestion_done: True once ingest_pdfs() has completed successfully.
# Controls whether the chat input and "Clear & Reset" button are shown.
if "ingestion_done" not in st.session_state:
    st.session_state.ingestion_done = False

# indexes: the complete index bundle returned by ingest_pdfs() in Ingestion.py.
# Contains bm25, faiss, meta, chunk_id_to_meta, image_b64_lookup, embedder, reranker.
# → passed to retrieve() in retrieve.py on every user query
if "indexes" not in st.session_state:
    st.session_state.indexes = None

# top_k: number of chunks to retrieve per query.
# Initial value of 5 is a practical default — enough context for most questions
# without overwhelming the LLM's context window or incurring excessive token costs.
# → controlled by the sidebar slider below
# → passed to retrieve() on every query
if "top_k" not in st.session_state:
    st.session_state.top_k = 5


# ─────────────────────────────────────────
# 4. Sidebar
#    Contains: instructions, PDF uploader, ingestion button, status metrics,
#    top-k slider, and a reset button.
#    Streamlit renders sidebar content immediately when `with st.sidebar:` runs,
#    before the main area content below.
# ─────────────────────────────────────────

with st.sidebar:
    st.title("Multimodal RAG with Hybrid Reranker and Cross Encoder ")
    st.markdown("---")
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. Upload one or more PDF files
    2. Click **Ingest PDFs** to build the index
    3. Ask questions about your documents
    """)
    st.markdown("---")
    st.markdown("**Developed by Amitoj Singh Chawla**")
    st.markdown("---")

    # ── PDF File Uploader ─────────────────────────────────────────────────────
    # Accepts multiple PDFs at once. uploaded_files is a list of UploadedFile
    # objects (in-memory byte buffers). Each is written to disk temporarily by
    # ingest_pdfs() in Ingestion.py because unstructured needs a real file path.
    # → uploaded_files passed to ingest_pdfs() when the button is clicked below
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    # ingest_ready is True only when both files are uploaded AND the API key exists.
    # It controls the disabled= state of the "Ingest PDFs" button below.
    # Without the API key, GPT-4o image descriptions would fail silently mid-ingestion.
    ingest_ready = bool(uploaded_files) and bool(OPENAI_API_KEY)

    # Surface a visible warning if the key is missing so the user knows why the button is disabled.
    if not OPENAI_API_KEY:
        st.warning("⚠️ OPENAI_API_KEY not found in .env")

    # Show the "Ingest PDFs" button only when files are uploaded AND ingestion hasn't run yet.
    # After successful ingestion, this block is skipped entirely so the user can't re-ingest
    # without first clicking "Clear & Reset" (which sets ingestion_done back to False).
    if uploaded_files and not st.session_state.ingestion_done:
        if st.button("Ingest PDFs", type="primary", use_container_width=True, disabled=not ingest_ready):

            # ── Reset state before a fresh ingestion ──────────────────────────
            # Clear any previous conversation and indexes so the new ingestion
            # starts with a clean slate. This handles the case where the user
            # uploads new files after a previous session without clicking Reset.
            st.session_state.messages = []
            st.session_state.ingestion_done = False
            st.session_state.indexes = None

            # st.status() creates a collapsible status panel that shows live progress
            # messages from inside the ingestion functions (st.write, st.spinner calls
            # in Ingestion.py appear here because Streamlit captures them from the
            # active execution context).
            with st.status("Ingesting PDFs...", expanded=True) as status:
                # ← ingest_pdfs() is the main entry point in Ingestion.py
                # It returns a dict on success, or {} on failure.
                indexes = ingest_pdfs(
                    uploaded_files=uploaded_files,
                    embedder=embedder,
                    reranker=reranker,
                )

                if indexes:
                    # Store the index bundle in session_state so retrieve() can access it.
                    # → st.session_state["indexes"] is passed to retrieve() on every query
                    st.session_state.indexes = indexes
                    st.session_state.ingestion_done = True
                    status.update(label="✅ Ingestion complete", state="complete", expanded=False)

                    # st.rerun() forces a full Streamlit rerun so the chat input
                    # (gated on ingestion_done below) becomes visible immediately.
                    st.rerun()
                else:
                    # ingest_pdfs() returned {} — it already showed an st.error() in the status panel.
                    status.update(label="❌ Ingestion failed", state="error")

    # ── Post-ingestion status display ─────────────────────────────────────────
    # Once ingestion completes, show a summary of what was indexed.
    # meta is the list of lightweight chunk dicts built in Ingestion.py.
    # We count by modality so the user can verify their PDFs were processed correctly
    # (e.g. if Images shows 0, the PDF may not have extractable images).
    if st.session_state.ingestion_done and st.session_state.indexes:
        meta = st.session_state.indexes["meta"]
        st.success("✅ Index ready")

        # Three side-by-side metric tiles — one per modality
        c1, c2, c3 = st.columns(3)
        c1.metric("Text",   sum(1 for m in meta if m["modality"] == "text"))
        c2.metric("Tables", sum(1 for m in meta if m["modality"] == "table"))
        c3.metric("Images", sum(1 for m in meta if m["modality"] == "image"))

    st.markdown("---")

    # ── Top-K Slider ──────────────────────────────────────────────────────────
    # Controls how many chunks retrieve() returns per query.
    # min=1 ensures at least one chunk is always retrieved.
    # max=10 is a practical upper bound — beyond ~10 chunks the LLM context fills up
    # and the answer quality degrades as irrelevant chunks dilute the signal.
    # The slider reads from and writes to session_state["top_k"] to persist the value
    # across reruns (slider interactions trigger a rerun in Streamlit).
    # → st.session_state.top_k passed to retrieve() on every user query below
    st.session_state.top_k = st.slider(
        "🎯 Top-K chunks",
        min_value=1, max_value=10,
        value=st.session_state.top_k,
    )

    st.markdown("---")

    # ── Reset Button ──────────────────────────────────────────────────────────
    # Only shown after ingestion so the user can start over with new PDFs.
    # Removes all four session_state keys — the "not in session_state" checks
    # at the top of this file will re-initialise them on the next rerun.
    if st.session_state.ingestion_done:
        if st.button("🗑️ Clear & Reset", use_container_width=True):
            for key in ["messages", "ingestion_done", "indexes", "top_k"]:
                st.session_state.pop(key, None)    # pop() with default avoids KeyError
            st.rerun()


# ─────────────────────────────────────────
# 5. Main Chat Area
#    Renders the page title, the full chat history, and the query input.
#    Streamlit re-renders this entire block on every rerun (every interaction).
# ─────────────────────────────────────────

st.markdown(
    "<h1 style='text-align: center;'> R.A.G </h1>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ─────────────────────────────────────────
# 6. Chat History Rendering
#    Replay every message in session_state["messages"] as styled HTML bubbles.
#    Streamlit does not have a built-in chat bubble component, so we use
#    st.markdown with inline CSS. unsafe_allow_html=True is required for this.
#    Each assistant message also shows a "View sources" expander below its bubble.
# ─────────────────────────────────────────

for message in st.session_state.messages:
    role    = message["role"]
    content = message["content"]

    if role == "system":
        # System messages (e.g. "Ingestion complete") are shown as yellow info banners.
        # They are not part of the LLM conversation — just UI notifications.
        st.markdown(
            f"""<div style='background-color: #FFECB3; padding: 15px; border-radius: 10px; margin: 10px 0;'>
            <p style='margin: 0; color: #333;'><strong>ℹ️ System:</strong> {content}</p>
            </div>""",
            unsafe_allow_html=True,
        )
    elif role == "user":
        # User messages are right-aligned with a green background to visually
        # distinguish them from assistant responses.
        st.markdown(
            f"""<div style='background-color: #DCF8C6; padding: 15px; border-radius: 10px; margin: 10px 0; margin-left: 20%;'>
            <p style='margin: 0; color: #333;'><strong>You:</strong> {content}</p>
            </div>""",
            unsafe_allow_html=True,
        )
    elif role == "assistant":
        # Assistant messages are left-aligned with a blue-grey background.
        st.markdown(
            f"""<div style='background-color: #E8EAF6; padding: 15px; border-radius: 10px; margin: 10px 0; margin-right: 20%;'>
            <p style='margin: 0; color: #333;'><strong>🤖 Bot:</strong> {content}</p>
            </div>""",
            unsafe_allow_html=True,
        )

        # Show the retrieved source chunks as a collapsed expander below the answer.
        # message["sources"] is the ctx_texts list returned by generate_answer() in generate.py.
        # It contains the plain text (or GPT-4o description for images) of each retrieved chunk.
        # [:300] truncates long chunks so the expander stays readable.
        if message.get("sources"):
            with st.expander("📄 View sources"):
                for i, src in enumerate(message["sources"], 1):
                    st.caption(f"**Source {i}:** {src[:300]}...")


# ─────────────────────────────────────────
# 7. Chat Input and Query Pipeline
#    The chat input widget is only shown after ingestion is complete.
#    When the user submits a question, the full RAG pipeline runs:
#      retrieve() → generate_answer() → append to session_state → rerun
# ─────────────────────────────────────────

if st.session_state.ingestion_done:
    # st.chat_input renders a fixed input bar at the bottom of the page.
    # It returns the submitted string (or None if the user hasn't submitted yet).
    # Each submission triggers a Streamlit rerun where user_query is non-None.
    user_query = st.chat_input("Ask a question about your documents...")

    if user_query:
        # Append the user's message to the history immediately so it appears
        # in the chat bubble rendering above on the next rerun.
        st.session_state.messages.append({"role": "user", "content": user_query})

        try:
            with st.spinner("Thinking..."):
                # ── Step 1: Retrieve ──────────────────────────────────────────
                # retrieve() unpacks st.session_state.indexes and calls
                # retrieve_hybrid_reranked() in retrieve.py.
                # Returns a list of top_k chunk dicts ordered by CrossEncoder score.
                # → chunks passed directly to generate_answer() below
                chunks = retrieve(
                    query=user_query,
                    method="hybrid_reranker",          # always use the most accurate method
                    indexes=st.session_state.indexes,  # ← built by ingest_pdfs() in Ingestion.py
                    top_k=st.session_state.top_k,      # ← set by the sidebar slider above
                )

                # ── Step 2: Generate ──────────────────────────────────────────
                # generate_answer() converts chunks → context blocks → API messages → answer.
                # Returns the answer string and a parallel list of plain-text source snippets.
                # → answer displayed in the chat bubble below
                # → ctx_texts shown in the "View sources" expander below the answer
                answer, ctx_texts = generate_answer(
                    question=user_query,
                    retrieved_chunks=chunks,
                    chat_history=st.session_state.messages,   # full history; windowed internally
                    api_key=OPENAI_API_KEY
                )

        except Exception as e:
            # Surface retrieval or generation errors as an assistant message
            # rather than crashing the app with an unhandled exception.
            # The chat history is preserved so the user can see what failed.
            st.session_state.messages.append({
                "role":    "assistant",
                "content": f"Error: {e}",
            })

        # Append the assistant's answer and its source snippets to the history.
        # "sources" is a custom key (not a standard OpenAI role key) — it is only
        # used by the rendering loop above to populate the "View sources" expander.
        # generate.py's _build_messages() skips non-standard keys when building API messages.
        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "sources": ctx_texts,   # ← ctx_texts from generate_answer() in generate.py
        })

        # Force a full rerun so the new assistant message is rendered in the chat history above.
        # Without st.rerun(), Streamlit would wait for the next user interaction before refreshing.
        st.rerun()

else:
    # Before ingestion: show a prompt guiding the user to upload files.
    # The chat input is intentionally absent here to prevent queries against an empty index.
    st.info("👆 Upload and ingest PDFs from the sidebar to start chatting.")