"""
streamlit_v2.py  ·  Updated Multimodal RAG UI
──────────────────────────────────────────────
Changes from streamlit.py (v1):
  1. METHOD SELECTOR  — sidebar radio lets the user pick from all 4 retrievers
                        instead of hardcoding "hybrid_reranker" on every query.
  2. METHOD SPEED HINT — caption below radio warns when a slower method is chosen.
  3. ACTIVE METHOD TAG — header shows which retriever + top-k is currently live.
  4. BETTER SOURCE CHIPS — each source chunk gets its own expander with modality
                           icon, source PDF name, and page number as the label.
  5. STYLED CHAT BUBBLES — cyan accent (#00BCD4 family) matching the redesign.
  6. INGESTION STEP LOG  — status panel prints each pipeline step as it completes.
  7. MINOR: reset also clears st.session_state.method so the sidebar re-renders
            cleanly on the next upload.

Logic changes summary (see inline comments for detail):
  · st.session_state.method is now set by the sidebar radio widget.
  · retrieve() receives st.session_state.method instead of the hardcoded string.
  · No changes to retrieve.py, Ingestion.py, or generate.py — all logic lives there.
"""

import streamlit as st
import os
import sys
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

sys.path.append(os.path.join(os.path.dirname(__file__), "codebase"))
sys.path.append(os.path.join(os.path.dirname(__file__), "app", "utils"))

from Ingestion import ingest_pdfs
from generate import generate_answer
from retrieve import retrieve

# ─────────────────────────────────────────
# 1. Model loading (unchanged from v1)
# ─────────────────────────────────────────

@st.cache_resource
def load_models():
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    reranker  = CrossEncoder("BAAI/bge-reranker-v2-m3")
    return embedder, reranker

st.set_page_config(page_title="Multimodal RAG", layout="wide")
embedder, reranker = load_models()

# ─────────────────────────────────────────
# 2. Session state — added `method` key
# ─────────────────────────────────────────

if "messages"       not in st.session_state: st.session_state.messages       = []
if "ingestion_done" not in st.session_state: st.session_state.ingestion_done = False
if "indexes"        not in st.session_state: st.session_state.indexes        = None
if "top_k"          not in st.session_state: st.session_state.top_k          = 5

# NEW: persists the selected retrieval method across reruns.
# Default is "hybrid_reranker" — the most accurate method.
# The sidebar radio below writes to this key on every change.
if "method" not in st.session_state: st.session_state.method = "hybrid_reranker"

# ─────────────────────────────────────────
# 3. CSS — cyan accent, styled bubbles
# ─────────────────────────────────────────

st.markdown("""
<style>
/* Import font */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

/* User bubble */
.bubble-user {
    background: #0e2a30;
    border: 1px solid #00838f;
    border-radius: 16px 4px 16px 16px;
    padding: 14px 18px;
    margin: 6px 0 6px 18%;
    color: #e0f7fa;
    line-height: 1.75;
    font-size: 14px;
}

/* Assistant bubble */
.bubble-bot {
    background: #1a1f2e;
    border: 1px solid #263145;
    border-radius: 4px 16px 16px 16px;
    padding: 14px 18px;
    margin: 6px 18% 6px 0;
    color: #e8eaf6;
    line-height: 1.75;
    font-size: 14px;
}

/* System / info banner */
.bubble-system {
    background: #1a2400;
    border: 1px solid #33691e;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 4px 0;
    color: #ccff90;
    font-size: 13px;
}

/* Role label above bubble */
.bubble-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 3px;
    color: #546e7a;
}
.bubble-label-bot  { color: #00acc1; }
.bubble-label-user { color: #26c6da; text-align: right; }

/* Source expander tweaks */
.stExpander { border: 1px solid #263145 !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# 4. Helper: render a single chat message
# ─────────────────────────────────────────

MODALITY_ICON = {"text": "≡", "table": "⊞", "image": "◫"}

def render_message(message: dict):
    role    = message["role"]
    content = message["content"]

    if role == "system":
        st.markdown(
            f"<div class='bubble-system'>ℹ️ {content}</div>",
            unsafe_allow_html=True,
        )

    elif role == "user":
        st.markdown("<div class='bubble-label bubble-label-user'>You</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='bubble-user'>{content}</div>", unsafe_allow_html=True)

    elif role == "assistant":
        st.markdown("<div class='bubble-label bubble-label-bot'>✦ Assistant</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='bubble-bot'>{content}</div>", unsafe_allow_html=True)

        # ── Source chips ────────────────────────────────────────────────────
        # CHANGE FROM V1: each source chunk gets its own expander with a
        # descriptive label (modality icon + PDF name + page) instead of one
        # big "View sources" expander that dumps all chunks in a list.
        sources = message.get("sources", [])    # list of chunk dicts
        metas   = message.get("source_meta", []) # list of meta dicts (new in v2)

        if sources:
            cols = st.columns(min(len(sources), 4))
            for i, (src_text, meta) in enumerate(zip(sources, metas or [{}] * len(sources))):
                modality    = meta.get("modality", "text")
                source_pdf  = meta.get("source_pdf", f"Source {i+1}")
                page        = meta.get("page_number")
                icon        = MODALITY_ICON.get(modality, "≡")
                page_label  = f" · p{page}" if page else ""
                label       = f"{icon} {source_pdf}{page_label}"

                with cols[i % 4]:
                    with st.expander(label, expanded=False):
                        st.caption(src_text[:400] + ("…" if len(src_text) > 400 else ""))


# ─────────────────────────────────────────
# 5. Retrieval method definitions
#    Kept in one place so sidebar + header both use the same labels/descriptions.
# ─────────────────────────────────────────

METHODS = {
    "hybrid_reranker": {
        "label": "⭐ Hybrid + Reranker",
        "desc":  "BM25 + Dense → RRF → CrossEncoder. Most accurate, ~2–4 s/query.",
        "speed": None,   # no warning — this is the recommended default
    },
    "hybrid": {
        "label": "Hybrid RRF",
        "desc":  "BM25 + Dense fused via Reciprocal Rank Fusion. Good balance.",
        "speed": None,
    },
    "dense": {
        "label": "Dense (semantic)",
        "desc":  "FAISS cosine search with BAAI/bge-base. Fast, semantic only.",
        "speed": "⚡ Fastest — no reranking. May miss keyword-specific answers.",
    },
    "bm25": {
        "label": "BM25 (keyword)",
        "desc":  "Sparse TF-IDF keyword search. Best for exact terms.",
        "speed": "⚡ Fastest — no embeddings. Misses paraphrase / semantic queries.",
    },
}


# ─────────────────────────────────────────
# 6. Sidebar
# ─────────────────────────────────────────

with st.sidebar:
    st.title("Multimodal RAG")
    st.caption("Hybrid Reranker · CrossEncoder · GPT-4o Vision")
    st.markdown("---")

    # ── Instructions ──────────────────────────────────────────────────────────
    with st.expander("📋 How to use", expanded=False):
        st.markdown("""
1. Upload one or more PDF files
2. Click **Ingest PDFs** to build the index
3. Select a retrieval method below
4. Ask questions about your documents
        """)

    st.markdown("---")

    # ── PDF uploader ──────────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    ingest_ready = bool(uploaded_files) and bool(OPENAI_API_KEY)
    if not OPENAI_API_KEY:
        st.warning("⚠️ OPENAI_API_KEY not found in .env")

    if uploaded_files and not st.session_state.ingestion_done:
        if st.button("Ingest PDFs", type="primary", use_container_width=True, disabled=not ingest_ready):
            st.session_state.messages       = []
            st.session_state.ingestion_done = False
            st.session_state.indexes        = None

            with st.status("Ingesting PDFs…", expanded=True) as status:
                # Each sub-step is written out so the user can see live progress.
                # ingest_pdfs() calls st.write() internally for its own progress lines.
                st.write(" Partitioning PDFs (hi_res mode)…")
                st.write(" Generating GPT-4o image descriptions…")
                st.write(" Extracting tables → pipe-delimited text…")
                st.write(" Building BM25 + FAISS indexes…")

                indexes = ingest_pdfs(
                    uploaded_files=uploaded_files,
                    embedder=embedder,
                    reranker=reranker,
                )

                if indexes:
                    st.session_state.indexes        = indexes
                    st.session_state.ingestion_done = True
                    status.update(label="✅ Ingestion complete", state="complete", expanded=False)
                    st.rerun()
                else:
                    status.update(label="❌ Ingestion failed", state="error")

    # ── Post-ingestion metrics ─────────────────────────────────────────────────
    if st.session_state.ingestion_done and st.session_state.indexes:
        meta = st.session_state.indexes["meta"]
        st.success("✅ Index ready")
        c1, c2, c3 = st.columns(3)
        c1.metric("Text",   sum(1 for m in meta if m["modality"] == "text"))
        c2.metric("Tables", sum(1 for m in meta if m["modality"] == "table"))
        c3.metric("Images", sum(1 for m in meta if m["modality"] == "image"))

    st.markdown("---")

    # ── Retrieval method selector — NEW IN V2 ──────────────────────────────────
    # LOGIC CHANGE: The chosen method is stored in st.session_state.method and
    # passed to retrieve() on every query (see the query pipeline below).
    # In v1 this was hardcoded to "hybrid_reranker" inside the retrieve() call.
    if st.session_state.ingestion_done:
        st.markdown("**Retrieval Method**")

        # Radio renders one option per method; format_func maps the key to the display label.
        selected_method = st.radio(
            label="retrieval_method",         # internal widget key
            options=list(METHODS.keys()),
            format_func=lambda m: METHODS[m]["label"],
            index=list(METHODS.keys()).index(st.session_state.method),
            label_visibility="collapsed",     # hide the raw label; title above is enough
        )

        # Persist the selection across reruns.
        # This is the ONLY place st.session_state.method is written.
        st.session_state.method = selected_method

        # Show description + optional speed warning under the selected option.
        st.caption(METHODS[selected_method]["desc"])
        if METHODS[selected_method]["speed"]:
            st.caption(METHODS[selected_method]["speed"])

        st.markdown("---")

    # ── Top-K slider ──────────────────────────────────────────────────────────
    st.session_state.top_k = st.slider(
        "🎯 Top-K chunks",
        min_value=1,
        max_value=10,
        value=st.session_state.top_k,
    )

    st.markdown("---")

    # ── Reset button ──────────────────────────────────────────────────────────
    if st.session_state.ingestion_done:
        if st.button("🗑️ Clear & Reset", use_container_width=True):
            # NEW: also clear `method` so the sidebar radio re-initialises cleanly.
            for key in ["messages", "ingestion_done", "indexes", "top_k", "method"]:
                st.session_state.pop(key, None)
            st.rerun()

    st.markdown("---")
    st.caption("Amitoj Singh Chawla · Bachelor's Thesis · CS AI & Data Science")


# ─────────────────────────────────────────
# 7. Main chat area
# ─────────────────────────────────────────

# Header — shows active method + top-k when the index is ready.
if st.session_state.ingestion_done:
    active_label = METHODS[st.session_state.method]["label"]
    st.markdown(
        f"<h2 style='text-align:center; letter-spacing:-0.5px;'>✦ Multimodal RAG</h2>"
        f"<p style='text-align:center; color:#546e7a; font-size:13px;'>"
        f"{active_label} &nbsp;·&nbsp; top-{st.session_state.top_k} chunks</p>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<h2 style='text-align:center; letter-spacing:-0.5px;'>✦ Multimodal RAG</h2>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Chat history rendering ─────────────────────────────────────────────────────
for message in st.session_state.messages:
    render_message(message)

# ── Query pipeline ─────────────────────────────────────────────────────────────
if st.session_state.ingestion_done:
    user_query = st.chat_input("Ask a question about your documents…")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})

        try:
            with st.spinner(f"Retrieving with {METHODS[st.session_state.method]['label']}…"):

                # ── LOGIC CHANGE: pass st.session_state.method instead of hardcoded string ──
                # In v1 this was: method="hybrid_reranker"
                # Now it reflects whatever the user selected in the sidebar radio above.
                chunks = retrieve(
                    query=user_query,
                    method=st.session_state.method,       # ← dynamic, from sidebar
                    indexes=st.session_state.indexes,
                    top_k=st.session_state.top_k,
                )

            with st.spinner("Generating answer…"):
                answer, ctx_texts = generate_answer(
                    question=user_query,
                    retrieved_chunks=chunks,
                    chat_history=st.session_state.messages,
                    api_key=OPENAI_API_KEY,
                )

        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
            st.rerun()

        # ── CHANGE: also store source_meta alongside source text ──────────────
        # In v1 only ctx_texts (plain strings) were stored.
        # Now we also store lightweight metadata (modality, source_pdf, page_number)
        # for each chunk so render_message() can label each expander properly.
        source_meta = [
            {
                "modality":   c.get("modality"),
                "source_pdf": c.get("source_pdf"),
                "page_number": c.get("page_number"),
            }
            for c in chunks
        ]

        st.session_state.messages.append({
            "role":        "assistant",
            "content":     answer,
            "sources":     ctx_texts,    # plain text strings (same as v1)
            "source_meta": source_meta,  # NEW: modality + PDF + page for labels
        })

        st.rerun()

else:
    st.info("👆 Upload and ingest PDFs from the sidebar to start chatting.")
