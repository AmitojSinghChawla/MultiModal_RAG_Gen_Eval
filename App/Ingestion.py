"""
Ingestion.py
────────────
Converts uploaded PDF files into a searchable in-memory index bundle.

Pipeline position
─────────────────
  Called by  : streamlit.py → ingest_pdfs()
               (triggered when the user clicks "Ingest PDFs" in the sidebar)
  Calls out to: partition_pdf()   — unstructured library, does the heavy PDF parsing
                describe_image()  — GPT-4o vision API, converts images into text descriptions
                build_bm25()      — creates the sparse keyword index
                build_faiss()     — creates the dense semantic index

Input
─────
  uploaded_files : list of Streamlit UploadedFile objects (in-memory PDF bytes)
  embedder       : SentenceTransformer model loaded once in streamlit.py via @st.cache_resource
  reranker       : CrossEncoder model loaded once in streamlit.py via @st.cache_resource

Output  (→ stored in st.session_state["indexes"] in streamlit.py)
──────
  A single dict containing:
    "bm25"             — BM25Okapi sparse index over all chunk texts
    "faiss"            — FAISS IndexFlatIP dense index over all chunk embeddings
    "meta"             — list of lightweight chunk metadata dicts (no image bytes)
    "chunk_id_to_meta" — dict mapping chunk_id → meta entry for O(1) lookup
    "image_b64_lookup" — dict mapping chunk_id → raw base64 string for image chunks only
    "embedder"         — the SentenceTransformer model (passed through for use at query time)
    "reranker"         — the CrossEncoder model (passed through for use at query time)
"""

import re
import shutil
import tempfile
import uuid
import os

import faiss
import nltk
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from unstructured.documents.elements import CompositeElement, Table
from unstructured.partition.pdf import partition_pdf


# ─────────────────────────────────────────
# 1. Environment Setup
#    Load the OpenAI key from .env so GPT-4o vision calls can authenticate.
#    Fail immediately at import time if the key is missing — better to crash
#    here with a clear message than to fail silently during the first API call.
# ─────────────────────────────────────────

load_dotenv(verbose=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Hard stop: without this key, describe_image() cannot call GPT-4o,
# and the entire image ingestion branch is broken.
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY environment variable not set. Please set it before running the script."
    )


# ─────────────────────────────────────────
# 2. NLTK Setup
#    Download the stopwords corpus once at import time (no-op if already cached).
#    These objects are module-level singletons — creating them once here avoids
#    re-initialising them on every call to tokenize().
# ─────────────────────────────────────────

nltk.download("stopwords", quiet=True)

# _stop_words: set of ~180 common English function words (the, is, at, …)
# Removing them from BM25 queries prevents common words from drowning out rare,
# discriminative terms.
_stop_words = set(stopwords.words("english"))

# _stemmer: maps inflected word forms to their root (running → run, studies → studi)
# Applied identically at index time and query time so "running" in a query still
# matches "run" in a chunk.
_stemmer = PorterStemmer()


# ─────────────────────────────────────────
# 3. Text Utilities
#    Small, pure helper functions used by the chunk-building loop below.
# ─────────────────────────────────────────


def clean_text(text: str) -> str:
    """
    Collapse all runs of whitespace (spaces, tabs, newlines) into a single space.

    Input : raw text string extracted from a PDF element — may contain multiple
            consecutive spaces or newlines from PDF rendering artefacts.
    Output: a single-line, whitespace-normalised string.
            → used as raw_text / retrieval_text in every chunk dict built by ingest_pdfs().

    Why this matters: BM25 and the sentence-transformer both tokenise on whitespace,
    so extra whitespace doesn't hurt correctness, but it wastes tokens and makes
    stored strings harder to read in the UI source expander.
    """
    return " ".join(text.split())


def html_table_to_text(html: str) -> str:
    """
    Convert an HTML table string (produced by unstructured) into a readable,
    pipe-delimited plain-text representation.

    Input : raw HTML string from table.metadata.text_as_html,
            e.g. "<table><tr><th>Name</th><th>Age</th></tr>…</table>"
            ← comes from the Table element's metadata in table_text_segregation()

    Output: multi-line pipe-delimited string, one row per line,
            e.g. "Name | Age\nAlice | 30"
            → stored as raw_text and retrieval_text in table chunk dicts
            → displayed in the UI source expander and sent to the LLM as [TABLE] context

    Why pipe-delimited: the sentence-transformer embeds the text as a sequence,
    and pipe-separated columns preserve the row structure in a way that both
    BM25 term matching and dense embeddings can reason over.
    """
    soup = BeautifulSoup(html, "html.parser")       # parse the raw HTML string into a tree
    rows = []
    for tr in soup.find_all("tr"):                  # iterate every table row <tr>
        # Pull the visible text from every cell — <td> (data cell) or <th> (header cell).
        # strip=True removes leading/trailing whitespace inside each cell.
        cells = [cell.get_text(strip=True) for cell in tr.find_all(["td", "th"])]
        if cells:
            rows.append(" | ".join(cells))          # join all cells in the row with a pipe delimiter
    return "\n".join(rows)                          # each row becomes its own line in the output


def tokenize(text: str) -> list[str]:
    """
    Convert a raw text string into a list of stemmed, stopword-filtered tokens
    suitable for BM25 indexing and querying.

    Input : any plain-text string — either a chunk's retrieval_text (at index time,
            called from build_bm25()) or the user's query string (at query time,
            called from retrieve_bm25() in retrieve.py).
    Output: list of lowercase, punctuation-stripped, stemmed tokens with stopwords removed.
            → at index time:  passed to BM25Okapi() as the corpus in build_bm25()
            → at query time:  passed to bm25.get_scores() in retrieve.py → retrieve_bm25()

    CRITICAL: this function MUST produce identical output for semantically equivalent
    input at both index time and query time. If the pipeline is changed (e.g. different
    stemmer), both callers must be updated together, or BM25 scores will be wrong.

    Step-by-step transformation example:
        "Hello, World! Running fast."
        → lower:   "hello, world! running fast."
        → strip:   "hello  world  running fast "
        → split:   ["hello", "world", "running", "fast"]
        → filter:  ["world", "running", "fast"]   (stopwords removed, len > 2)
        → stem:    ["world", "run", "fast"]
    """
    text = text.lower()                                                     # "Hello World" → "hello world"
    text = re.sub(r"[^a-z0-9\s]", " ", text)                               # "hello, world!" → "hello  world "
    tokens = text.split()                                                   # split on any whitespace
    tokens = [t for t in tokens if t not in _stop_words and len(t) > 2]    # drop stopwords & 1-2 char tokens
    tokens = [_stemmer.stem(t) for t in tokens]                             # "running" → "run"
    # → returned list is consumed by BM25Okapi at index time (build_bm25)
    #   or by bm25.get_scores() at query time (retrieve.py → retrieve_bm25)
    return tokens


# ─────────────────────────────────────────
# 4. PDF Partitioning
#    Uses the `unstructured` library to break a PDF into typed elements.
#    This is the most expensive step — it runs a layout-detection ML model
#    on every page, so it can take tens of seconds for long documents.
# ─────────────────────────────────────────


def create_chunks_from_pdf(file_path):
    """
    Partition a single PDF file into a flat list of structured elements.

    Input : file_path — absolute path to a PDF file on disk.
            ← written to a temp file by ingest_pdfs() because unstructured
              requires a real filesystem path, not in-memory bytes.

    Output: list of unstructured Element objects (Table and CompositeElement instances).
            → passed to table_text_segregation() and get_images() in ingest_pdfs()

    Key parameters explained:
      strategy="hi_res"
          Runs a deep-learning layout model (detectron2) on each page to identify
          regions: text blocks, tables, figures, headers.
          Alternative "fast" mode is faster but misses tables and images.

      extract_images_in_pdf=True
          Tells unstructured to extract image regions from the PDF.

      extract_image_block_types=["Image"]
          Only extract elements classified as images (not charts, which are handled
          as text by the layout model).

      extract_image_block_to_payload=True
          Embed the raw base64 image data directly onto the element object's metadata
          (element.metadata.image_base64), so we don't need a separate image file.

      chunking_strategy="by_title"
          After element detection, merge adjacent elements into chunks based on
          section headings — keeps related content together.

      max_characters=5000
          Hard upper size cap per chunk. Prevents oversized chunks that would exceed
          the embedding model's context window or LLM token limits.

      combine_text_under_n_chars=3000
          Merge very small adjacent chunks to avoid retrieving fragments.
          Without this, a section with a heading and two sentences would become
          three separate chunks, all retrieved independently.

      new_after_n_chars=1000
          Soft split target — start a new chunk after ~1000 chars even if no
          heading boundary is found. Balances chunk granularity vs. context richness.
    """
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=5000,
        combine_text_under_n_chars=3000,
        new_after_n_chars=1000,
    )
    # → elements is a flat list of Table and CompositeElement objects
    #   passed to table_text_segregation() and get_images() in ingest_pdfs()
    return elements


# ─────────────────────────────────────────
# 5. Element Segregation
#    After partitioning, unstructured gives us a flat list of mixed element
#    types. These two functions split that list into the three modalities
#    the pipeline handles separately: text, tables, and images.
# ─────────────────────────────────────────


def table_text_segregation(all_elements):
    """
    Split a flat list of unstructured elements into tables and text chunks.

    Input : all_elements — the raw list returned by create_chunks_from_pdf()
            ← called from ingest_pdfs() immediately after partitioning

    Output: (tables, texts) — a 2-tuple where:
              tables : list of Table elements (HTML table structure preserved in metadata)
              texts  : list of CompositeElement elements (text blocks; may also contain
                       embedded image sub-elements accessible via metadata.orig_elements)
            → tables fed into the table chunk loop in ingest_pdfs()
            → texts  fed into the text chunk loop AND into get_images() in ingest_pdfs()

    Why split: tables need HTML → pipe-text conversion; text blocks need plain
    extraction; images live inside CompositeElements and need a separate pass.
    Elements that are neither Table nor CompositeElement (e.g. Title elements
    before chunking) are silently dropped — they have already been merged into
    CompositeElements by the chunking strategy.
    """
    tables = []
    texts = []
    for el in all_elements:
        if isinstance(el, Table):               # HTML table detected by unstructured
            tables.append(el)
        elif isinstance(el, CompositeElement):  # main text block; may contain embedded images
            texts.append(el)
    # → (tables, texts) returned to ingest_pdfs() for separate chunk-building loops
    return tables, texts


def get_images(chunks):
    """
    Walk through a list of elements and extract unique, content-bearing base64 images.

    Input : chunks — the full elements list from create_chunks_from_pdf()
            ← called from ingest_pdfs() after table_text_segregation()
            Images live *inside* CompositeElements as sub-elements in
            chunk.metadata.orig_elements, not as top-level elements themselves.

    Output: images_b64 — list of raw base64 strings, one per unique meaningful image.
            → iterated in ingest_pdfs() to call describe_image() and build image chunk dicts

    Three filters are applied to avoid noise in the index:
      1. No base64 data   — unstructured failed to extract the image; skip it.
      2. Length < 1500 chars — base64 encodes ~0.75 bytes per char, so 1500 chars ≈ 1125 bytes ≈ 1 KB.
                              Images this small are almost always logos, icons, or decorative
                              dividers — not meaningful content.
      3. Duplicate fingerprint — uses the first 300 chars of the base64 string as a cheap hash.
                                 Identical images embedded multiple times in a PDF (e.g. a
                                 recurring logo in a header) are deduplicated here.
    """
    images_b64 = []
    # seen_hashes tracks fingerprints of images already collected to prevent duplicates
    seen_hashes = set()

    for chunk in chunks:
        # Images only live inside CompositeElements — skip Tables and any other types
        if not isinstance(chunk, CompositeElement):
            continue

        # orig_elements holds the original sub-elements that were merged into this composite chunk
        # (e.g. a heading, a paragraph, and an image figure that were grouped by "by_title" chunking)
        chunk_els = chunk.metadata.orig_elements or []
        for el in chunk_els:
            # Type-check by string because the Image class is not easily importable
            # from unstructured without specifying the exact sub-module path
            if "Image" not in str(type(el)):
                continue

            # image_base64 is populated by extract_image_block_to_payload=True in partition_pdf()
            img_b64 = el.metadata.image_base64

            if not img_b64:                     # unstructured couldn't extract data for this image
                continue
            if len(img_b64) < 1500:             # too small to be a meaningful content image (~1 KB)
                continue

            # First 300 chars of base64 are a fast, cheap fingerprint.
            # Two identical images will always share the same first 300 chars.
            img_hash = img_b64[:300]
            if img_hash in seen_hashes:         # already collected this image from an earlier chunk
                continue

            seen_hashes.add(img_hash)
            images_b64.append(img_b64)

    # → images_b64 is a deduplicated list of raw base64 strings
    #   passed back to ingest_pdfs() for GPT-4o description and chunk dict construction
    return images_b64


# ─────────────────────────────────────────
# 6. Image Description via GPT-4o Vision
#    Images cannot be embedded or keyword-searched directly.
#    GPT-4o converts each image into a rich text description that
#    captures all visible data — this description is what gets indexed
#    (stored as retrieval_text) so BM25 and FAISS can find the image
#    in response to a user's text query.
# ─────────────────────────────────────────


def describe_image(image_b64: str) -> str:
    """
    Send a single base64-encoded image to GPT-4o and return a detailed,
    retrieval-optimised text description.

    Input : image_b64 — a single raw base64 string for one image.
            ← comes from the images_b64 list produced by get_images()
            ← called in a loop inside ingest_pdfs()

    Output: a multi-sentence string describing the image's type, all visible
            text, numbers, axes, trends, and components.
            → stored as retrieval_text in the image chunk dict
            → indexed by both build_bm25() and build_faiss()
            → used by retrieve.py to match the image against user queries
            → displayed in the UI source expander as the "source" for image hits

    On API failure: returns a placeholder string so ingestion doesn't crash.
    The image chunk is still created with the placeholder as retrieval_text,
    meaning the image will not be retrievable but the pipeline continues.

    Why GPT-4o and not GPT-4o-mini:
        GPT-4o-mini produces vaguer descriptions ("a bar chart showing some values"),
        which hurts both BM25 (no specific terms to match) and dense retrieval
        (generic embeddings cluster together). GPT-4o names axes, cites values,
        and describes relationships — specifics that make the description searchable.

    Why temperature=0:
        Descriptions are used as a search index. Randomness could produce different
        descriptions on re-ingestion, making the index non-reproducible.

    Why max_tokens=1024:
        Complex figures (multi-series charts, architecture diagrams) need up to ~800
        tokens for a thorough description. 1024 gives a safe margin.

    Why detail="high":
        Low-res mode downsamples the image before sending it to the model.
        Charts with small-font axis labels or dense tables would lose legible text.
        High-res mode tiles the image and processes each tile separately.
    """
    _vision_llm = ChatOpenAI(
        model="gpt-4o",
        api_key=OPENAI_API_KEY,
        temperature=0,      # deterministic — want consistent, reproducible descriptions
        max_tokens=1024,    # enough room for a thorough description of complex figures
    )
    # StrOutputParser extracts the plain string from LangChain's AIMessage response object
    _parser = StrOutputParser()

    # Build a browser-style data URI so the OpenAI API can decode the raw bytes
    data_url = f"data:image/jpeg;base64,{image_b64}"

    # Construct the multimodal message: image block first, then the instruction text block.
    # The API processes content blocks in order, so the model sees the image before the prompt.
    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": data_url,
                    "detail": "high",   # high-res mode — needed for charts with small text
                },
            },
            {
                "type": "text",
                "text": (
                    "You are a precise technical analyst. "
                    "Describe this image in as much detail as possible for a document retrieval system. "
                    "Your description will be used as a search index, so specificity is critical.\n\n"
                    "Cover ALL of the following that apply:\n"
                    "1. Type of visual (bar chart, line graph, pie chart, table, flowchart, architecture diagram, photograph, etc.)\n"
                    "2. Title or heading visible in the image\n"
                    "3. All axis labels, units of measurement, and scale ranges (for charts/graphs)\n"
                    "4. Every data series name and its approximate values at key points\n"
                    "5. All text visible in the image — labels, legends, annotations, captions\n"
                    "6. Any specific numbers, percentages, dates, currency values, or named entities\n"
                    "7. The overall trend, conclusion, or key insight the visual communicates\n"
                    "8. For tables: the column headers and a summary of the row data\n"
                    "9. For diagrams: the components, their relationships, and directional flow\n\n"
                    "Be specific. Do not say 'some values' — state the actual values. "
                    "Do not say 'various categories' — name the categories. "
                    "A researcher should be able to answer factual questions about this figure "
                    "using only your description."
                ),
            },
        ]
    )

    try:
        response = _vision_llm.invoke([message])    # send image + prompt to GPT-4o
        # → description string returned to the image chunk loop in ingest_pdfs()
        return _parser.invoke(response)             # pull the plain string out of the response
    except Exception as e:
        # Do not crash ingestion on a single failed image — the chunk will be created
        # with a placeholder retrieval_text so the rest of the pipeline continues.
        print(f"   WARNING: GPT-4o vision failed — {e}")
        return "Image description unavailable due to API error."


# ─────────────────────────────────────────
# 7. Index Building
#    These two functions consume the final all_chunks list and build
#    the two complementary search indexes used at query time.
# ─────────────────────────────────────────


def build_bm25(chunks: list[dict]) -> BM25Okapi:
    """
    Build a BM25 sparse keyword index over all chunks' retrieval_text fields.

    Input : chunks — the complete all_chunks list assembled by ingest_pdfs(),
            containing text, table, and image chunk dicts.
            ← called from ingest_pdfs() after all chunk dicts are built

    Output: a fitted BM25Okapi object that can score any tokenized query against
            the corpus.
            → stored under key "bm25" in the indexes dict returned by ingest_pdfs()
            → consumed by retrieve_bm25() in retrieve.py at query time

    Why BM25: it excels at exact keyword matches — "Figure 3", "Table 2", specific
    model names, or rare technical terms that a dense embedding might paraphrase away.
    It complements FAISS (which handles semantic similarity) in the hybrid retrieval step.

    Position alignment: BM25Okapi internally maps corpus position → score, so the order
    of `chunks` here MUST match the order of `meta` built later in ingest_pdfs().
    Both are built from the same all_chunks list in the same order, so this holds.
    """
    # tokenize() produces the same normalised token list used at query time in retrieve.py
    corpus = [tokenize(chunk["retrieval_text"]) for chunk in chunks]
    # BM25Okapi fits term frequencies and document frequencies over the full corpus
    bm25 = BM25Okapi(corpus)
    print(f"BM25 index built over {len(corpus)} documents")
    # → bm25 object stored in indexes["bm25"], consumed by retrieve.py → retrieve_bm25()
    return bm25


def build_faiss(chunks: list[dict], model: SentenceTransformer) -> faiss.IndexFlatIP:
    """
    Build a FAISS dense vector index over all chunks' retrieval_text embeddings.

    Input : chunks — the complete all_chunks list from ingest_pdfs()
            model  — the SentenceTransformer embedder (BAAI/bge-base-en-v1.5)
                     ← loaded once in streamlit.py via @st.cache_resource
                     ← passed into ingest_pdfs() and forwarded here

    Output: a FAISS IndexFlatIP (exact inner-product search) loaded with one
            normalised embedding vector per chunk.
            → stored under key "faiss" in the indexes dict returned by ingest_pdfs()
            → consumed by retrieve_dense() and retrieve_hybrid() in retrieve.py

    Why L2 normalisation before IndexFlatIP:
        After normalising each vector to unit length (magnitude = 1),
        inner product (dot product) equals cosine similarity.
        IndexFlatIP then performs exact cosine search without any approximation,
        which is acceptable at the scale of a few thousand chunks.

    Why show_progress_bar=True:
        Encoding can take 10–30 seconds for large documents. The progress bar
        in the terminal gives the user feedback during the Streamlit spinner.

    Position alignment: FAISS assigns each vector an integer ID equal to its
    insertion order. This order MUST match the order of `meta` in ingest_pdfs()
    so that FAISS index position i corresponds to meta[i]. Both are built from
    the same all_chunks list in the same order.
    """
    # Extract only the text field that will be embedded — image chunks use their
    # GPT-4o description here, not the raw base64, so all modalities are embeddable.
    texts = [chunk["retrieval_text"] for chunk in chunks]

    # encode() converts each string to a fixed-size vector (768 dims for bge-base-en-v1.5)
    # convert_to_numpy=True is required — FAISS only accepts numpy arrays, not torch tensors
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Normalise every row vector to unit length so dot product == cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]       # vector dimension — 768 for bge-base-en-v1.5
    index = faiss.IndexFlatIP(dim)  # exact inner-product (cosine) search, no approximation
    index.add(embeddings)           # load all vectors; FAISS assigns position 0, 1, 2, …

    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    # → index stored in indexes["faiss"], consumed by retrieve.py → retrieve_dense / retrieve_hybrid
    return index


# ─────────────────────────────────────────
# 8. Main Ingestion Pipeline
#    Orchestrates everything above for each uploaded PDF,
#    then assembles and returns the complete index bundle.
# ─────────────────────────────────────────


def ingest_pdfs(uploaded_files: list, embedder, reranker):
    """
    Top-level ingestion function — converts a list of uploaded PDFs into a
    complete, query-ready index bundle stored in Streamlit's session state.

    Input
    ─────
      uploaded_files : list of Streamlit UploadedFile objects
                       ← passed from streamlit.py when user clicks "Ingest PDFs"
      embedder       : SentenceTransformer("BAAI/bge-base-en-v1.5")
                       ← loaded once in streamlit.py via @st.cache_resource
      reranker       : CrossEncoder("BAAI/bge-reranker-v2-m3")
                       ← loaded once in streamlit.py via @st.cache_resource

    Output  (→ stored as st.session_state["indexes"] in streamlit.py)
    ──────
      On success: dict with keys bm25, faiss, meta, chunk_id_to_meta,
                  image_b64_lookup, embedder, reranker
      On failure: empty dict {} — streamlit.py checks for this and shows an error

    Per-file pipeline:
      1. Write UploadedFile bytes to a temp file on disk
         (unstructured's partition_pdf requires a real file path, not BytesIO)
      2. Call partition_pdf via create_chunks_from_pdf() — most expensive step
      3. Split elements into tables, texts, images
      4. Build a chunk dict for each element
         → text chunks  : raw extracted text, indexed directly
         → table chunks : HTML converted to pipe-delimited text, indexed directly
         → image chunks : GPT-4o vision description used as retrieval text;
                          raw base64 kept in image_b64_lookup for the LLM at answer time
      5. Always delete the temp file (finally block) — no disk residue

    After all files are processed:
      6. Build meta list (lightweight copy of all chunk dicts, no image bytes)
      7. Build BM25 index over all retrieval_texts
      8. Build FAISS index over all retrieval_text embeddings
      9. Return the complete indexes dict
    """

    # all_chunks accumulates every chunk dict (text, table, image) across all uploaded PDFs.
    # This single flat list is the source of truth for both index builders.
    all_chunks = []

    # image_b64_lookup maps chunk_id → raw base64 string for image chunks only.
    # It is populated as image chunks are created below so that retrieve.py can
    # re-attach the actual image bytes to a retrieval result at query time.
    # → passed through to retrieve.py via indexes["image_b64_lookup"]
    image_b64_lookup = {}

    total_files = len(uploaded_files)

    # ── Per-file loop ─────────────────────────────────────────────────────────
    for file_index, uploaded_file in enumerate(uploaded_files):
        st.write(f"uploaded_file: {uploaded_file.name} ({file_index + 1}/{total_files})")

        # unstructured.partition_pdf() requires a real filesystem path, not in-memory bytes.
        # mkdtemp() creates a unique temporary directory that we can safely delete afterward.
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, uploaded_file.name)

        try:
            # Write Streamlit's in-memory bytes to a physical file so unstructured can read it
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.read())

            # Partition the PDF into typed elements — this is the slow ML-heavy step.
            # Wrapped in st.spinner so the user sees feedback during the wait.
            with st.spinner(f"Partitioning {uploaded_file.name}..."):
                try:
                    # → elements: flat list of Table and CompositeElement objects
                    elements = create_chunks_from_pdf(tmp_path)
                except Exception as e:
                    # If one PDF fails (corrupted, password-protected, etc.) skip it
                    # and continue with the remaining files rather than aborting everything.
                    st.warning(f"Skipping {uploaded_file.name} : {e}")
                    continue

        finally:
            # Always remove the temp directory, even if an exception occurred above.
            # ignore_errors=True means a missing directory (e.g. never created) won't raise.
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # Split the flat element list into the three modality buckets
        # tables : Table elements (HTML structure in metadata.text_as_html)
        # texts  : CompositeElement blocks (may contain embedded image sub-elements)
        tables, texts = table_text_segregation(elements)

        # Extract unique, content-bearing images from the CompositeElement blocks.
        # Images live inside CompositeElements as sub-elements in metadata.orig_elements.
        images = get_images(elements)

        print(f"   Texts: {len(texts)}, Tables: {len(tables)}, Images: {len(images)}")

        # ── Text chunks ───────────────────────────────────────────────────────
        # One chunk dict per CompositeElement text block.
        # retrieval_text == raw_text because no summarisation is applied to plain text.
        for text in texts:
            # Use .text attribute if available (standard for CompositeElement);
            # fall back to str() for any unexpected element types.
            raw_text = text.text if hasattr(text, "text") else str(text)
            raw_text = clean_text(raw_text)     # normalise whitespace

            chunk = {
                "chunk_id": str(uuid.uuid4()),                              # unique ID for BM25/FAISS position tracking
                "modality": "text",
                "source_pdf": uploaded_file.name,                           # used in UI source citations
                "page_number": getattr(text.metadata, "page_number", None), # None if unstructured couldn't determine
                "raw_text": raw_text,                                        # sent to LLM as [TEXT] context block
                "image_b64": None,                                           # not applicable for text chunks
                "retrieval_text": raw_text,                                  # indexed by BM25 and FAISS
            }
            all_chunks.append(chunk)

        # ── Table chunks ──────────────────────────────────────────────────────
        # One chunk dict per Table element.
        # Prefer the HTML version (converted to pipe-delimited text) over the raw string
        # because the HTML preserves row/column structure, making it more readable and
        # giving BM25 better keyword coverage over table cell values.
        for table in tables:
            if hasattr(table, "metadata") and getattr(table.metadata, "text_as_html", None):
                # Convert HTML table → pipe-delimited text so the LLM and search indexes
                # can reason over the structure without parsing HTML
                table_text = html_table_to_text(table.metadata.text_as_html)
            else:
                # Fallback for tables where unstructured didn't produce HTML metadata
                table_text = clean_text(str(table))

            chunk = {
                "chunk_id": str(uuid.uuid4()),
                "modality": "table",
                "source_pdf": uploaded_file.name,
                "page_number": getattr(table.metadata, "page_number", None),
                "raw_text": table_text,           # sent to LLM as [TABLE] context block
                "image_b64": None,                # not applicable for table chunks
                "retrieval_text": table_text,     # indexed by BM25 and FAISS
            }
            all_chunks.append(chunk)

        # ── Image chunks ──────────────────────────────────────────────────────
        # One chunk dict per unique image.
        # GPT-4o converts the raw pixels into a text description that gets indexed.
        # The raw base64 is stored separately in image_b64_lookup so it can be
        # sent directly to the answering LLM (GPT-4o mini) at query time.
        if images:
            st.write(f"   Describing {len(images)} image(s) with GPT-4o...")

        for i, image in enumerate(images):
            print(f"   Image {i + 1}/{len(images)}...", end=" ", flush=True)

            # Call GPT-4o vision — most expensive per-image step (~1–3 seconds per image)
            # Returns a detailed text description that acts as the search index for this image
            description = describe_image(image)
            print("done")

            # Generate the chunk_id before building the dict so we can register it
            # in image_b64_lookup with the same ID used inside the chunk dict.
            chunk_id = str(uuid.uuid4())
            chunk = {
                "chunk_id": chunk_id,
                "modality": "image",
                "source_pdf": uploaded_file.name,
                "page_number": None,              # unstructured doesn't track image page numbers
                "raw_text": None,                 # no raw text for image chunks
                "image_b64": image,               # kept here for reference; lookup is the authoritative source
                "retrieval_text": description,    # GPT-4o description — indexed by BM25 and FAISS
            }

            # Register this image in the lookup dict so retrieve.py can re-attach
            # the raw base64 to retrieval results. Without this, the LLM would
            # receive only the text description and never see the actual image.
            # → consumed by _make_result() in retrieve.py via indexes["image_b64_lookup"]
            image_b64_lookup[chunk_id] = image

            all_chunks.append(chunk)

    # ── Guard: nothing was extracted ──────────────────────────────────────────
    if not all_chunks:
        # This happens if every uploaded file failed to partition, or all were empty.
        st.error(f"No Chunks Found, Check Your Pdfs")
        # Return empty dict — streamlit.py checks `if indexes:` and shows an error banner
        return {}

    # ─────────────────────────────────────────
    # 9. Build Metadata List
    #    meta is a lightweight copy of all_chunks that strips out the raw image
    #    bytes (image_b64). It is used by retrieve.py to reconstruct result dicts
    #    without needing to carry large binary data through the retrieval loop.
    # ─────────────────────────────────────────

    # meta has the same order as all_chunks — position i in meta corresponds to
    # position i in the FAISS index and BM25 corpus. Do not sort or reorder.
    meta = [
        {
            "chunk_id":       c["chunk_id"],
            "modality":       c["modality"],
            "source_pdf":     c["source_pdf"],
            "page_number":    c["page_number"],
            "retrieval_text": c["retrieval_text"],
            "raw_text":       c.get("raw_text"),  # None for image chunks
        }
        for c in all_chunks
    ]

    # chunk_id_to_meta enables O(1) lookup of metadata by chunk_id during RRF fusion
    # in retrieve_hybrid() — avoids scanning the full meta list for every candidate.
    # → consumed by retrieve.py → retrieve_hybrid() and retrieve_hybrid_reranked()
    chunk_id_to_meta = {m["chunk_id"]: m for m in meta}

    # ── Build indexes ─────────────────────────────────────────────────────────
    with st.spinner("Building Bm25 Index"):
        # → bm25 stored in indexes["bm25"]
        bm25 = build_bm25(all_chunks)

    with st.spinner("Building Faiss Index"):
        # → faiss_index stored in indexes["faiss"]
        faiss_index = build_faiss(all_chunks, embedder)

    st.success(f"Ingestion Complete - {len(all_chunks)} chunks indexed")

    # ── Return the complete index bundle ──────────────────────────────────────
    # This dict is stored in st.session_state["indexes"] by streamlit.py and
    # passed verbatim to retrieve() in retrieve.py on every user query.
    return {
        "bm25":             bm25,             # sparse keyword index → retrieve.py → retrieve_bm25
        "faiss":            faiss_index,      # dense semantic index → retrieve.py → retrieve_dense
        "meta":             meta,             # list of chunk metadata dicts (no image bytes)
        "chunk_id_to_meta": chunk_id_to_meta, # chunk_id → meta dict for O(1) lookup in hybrid retrieval
        "image_b64_lookup": image_b64_lookup, # chunk_id → base64 string for image modality only
        "embedder":         embedder,         # passed through so retrieve.py can embed queries
        "reranker":         reranker,         # passed through so retrieve.py can rerank candidates
    }