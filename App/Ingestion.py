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

# ─── Environment Setup ───────────────────────────────────────────────────────

load_dotenv(verbose=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY environment variable not set. Please set it before running the script."
    )

# ─── NLTK Setup ──────────────────────────────────────────────────────────────

nltk.download("stopwords", quiet=True)

_stop_words = set(stopwords.words("english"))   # common English words to ignore (the, is, at …)
_stemmer = PorterStemmer()                       # reduces words to their root form (running → run)


# ─── Text Utilities ───────────────────────────────────────────────────────────


def clean_text(text: str) -> str:
    """Collapse multiple whitespace characters into a single space."""
    return " ".join(text.split())


def html_table_to_text(html: str) -> str:
    """Convert an HTML table string to a plain-text pipe-delimited representation.

    Example:
        <table><tr><th>Name</th><th>Age</th></tr><tr><td>Alice</td><td>30</td></tr></table>
        →  "Name | Age\\nAlice | 30"
    """
    soup = BeautifulSoup(html, "html.parser")       # parse the raw HTML string into a tree
    rows = []
    for tr in soup.find_all("tr"):                  # iterate every table row <tr>
        # pull the text from every cell — <td> (data) or <th> (header)
        cells = [cell.get_text(strip=True) for cell in tr.find_all(["td", "th"])]
        if cells:
            rows.append(" | ".join(cells))          # join all cells in the row with a pipe
    return "\n".join(rows)                          # each row becomes its own line


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 — must be called identically at index time and query time.

    Steps: lowercase → strip punctuation → split → drop stopwords & short tokens → stem.
    """
    text = text.lower()                                                     # "Hello World" → "hello world"
    text = re.sub(r"[^a-z0-9\s]", " ", text)                               # "hello, world!" → "hello  world "
    tokens = text.split()                                                   # split on any whitespace
    tokens = [t for t in tokens if t not in _stop_words and len(t) > 2]    # drop stopwords & 1-2 char tokens
    tokens = [_stemmer.stem(t) for t in tokens]                             # "running" → "run"
    return tokens


# ─── PDF Partition ────────────────────────────────────────────────────────────


def create_chunks_from_pdf(file_path):
    """Partition a PDF into structured elements using unstructured.

    strategy="hi_res"                    — layout-detection model, best accuracy
    extract_images_in_pdf=True           — pull out image blocks from the PDF
    extract_image_block_to_payload=True  — embed the base64 image data onto the element
    chunking_strategy="by_title"         — group content by document headings/sections
    max_characters=5000                  — hard size cap per chunk
    combine_text_under_n_chars=3000      — merge tiny fragments so chunks are meaningful
    new_after_n_chars=1000               — soft split target
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
    return elements


# ─── Element Segregation ──────────────────────────────────────────────────────


def table_text_segregation(all_elements):
    """Split a flat element list into two lists: tables and text (CompositeElement) chunks."""
    tables = []
    texts = []
    for el in all_elements:
        if isinstance(el, Table):               # HTML table detected by unstructured
            tables.append(el)
        elif isinstance(el, CompositeElement):  # main text block; may also contain embedded images
            texts.append(el)
    return tables, texts


def get_images(chunks):
    """Extract unique, meaningful base64 images from CompositeElement chunks.

    Filters applied:
      - Skip images with no base64 data
      - Skip images < ~1 KB (logos, icons) — 1500 base64 chars ≈ 1125 raw bytes
      - Skip duplicates using the first 300 chars as a fast fingerprint
    """
    images_b64 = []
    seen_hashes = set()     # tracks fingerprints so we don't add the same image twice

    for chunk in chunks:
        if not isinstance(chunk, CompositeElement):
            continue

        chunk_els = chunk.metadata.orig_elements or []  # original sub-elements inside this composite
        for el in chunk_els:
            if "Image" not in str(type(el)):            # skip anything that isn't an Image element
                continue

            img_b64 = el.metadata.image_base64          # raw base64 string for this image

            if not img_b64:                             # skip if unstructured didn't extract data
                continue
            if len(img_b64) < 1500:                     # skip tiny images (icons, decorative)
                continue

            img_hash = img_b64[:300]                    # cheap fingerprint — first 300 chars
            if img_hash in seen_hashes:                 # skip if we've already collected this image
                continue

            seen_hashes.add(img_hash)
            images_b64.append(img_b64)

    return images_b64


# ─── Image Description via GPT-4o ────────────────────────────────────────────
# GPT-4o is used here because it has the strongest vision capability.
# gpt-4o-mini is cheaper but produces vaguer descriptions for charts and
# diagrams, which hurts retrieval quality.


def describe_image(image_b64: str) -> str:
    """Send a base64 image to GPT-4o and return a detailed retrieval-ready description.

    The prompt extracts chart axes, table contents, diagram relationships,
    and all visible numbers/labels — specifics that make BM25 (exact term
    match) and dense embeddings (semantic richness) both work well.

    Returns a placeholder string on API failure so ingestion doesn't crash.
    """
    _vision_llm = ChatOpenAI(
        model="gpt-4o",
        api_key=OPENAI_API_KEY,
        temperature=0,      # deterministic — want consistent, reproducible descriptions
        max_tokens=1024,    # enough room for a thorough description of complex figures
    )
    _parser = StrOutputParser()     # extracts the plain string from the LLM response object

    data_url = f"data:image/jpeg;base64,{image_b64}"   # browser-style data URI the API expects

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
        return _parser.invoke(response)             # pull the plain string out of the response
    except Exception as e:
        print(f"   WARNING: GPT-4o vision failed — {e}")
        return "Image description unavailable due to API error."


# ─── Index Building ───────────────────────────────────────────────────────────


def build_bm25(chunks: list[dict]) -> BM25Okapi:
    """Build a BM25 sparse index from the retrieval_text of each chunk."""
    corpus = [tokenize(chunk["retrieval_text"]) for chunk in chunks]    # tokenize every chunk's text
    bm25 = BM25Okapi(corpus)    # BM25Okapi is the standard TF-IDF-style sparse retriever
    print(f"BM25 index built over {len(corpus)} documents")
    return bm25


def build_faiss(chunks: list[dict], model: SentenceTransformer) -> faiss.IndexFlatIP:
    """Build a FAISS dense index from sentence-transformer embeddings.

    Embeddings are L2-normalised so dot product == cosine similarity,
    letting IndexFlatIP (inner product) serve as exact cosine search.
    """
    texts = [chunk["retrieval_text"] for chunk in chunks]                           # text to embed
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True) # encode → numpy array

    faiss.normalize_L2(embeddings)  # scale every vector to unit length (magnitude = 1)
    # after normalisation: dot(a, b) == cosine_similarity(a, b)

    dim = embeddings.shape[1]               # vector dimension (e.g. 768 for most sentence models)
    index = faiss.IndexFlatIP(dim)          # flat inner-product index — exact search, no approximation
    index.add(embeddings)                   # load all vectors into the index

    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


# ─── Main Ingestion Pipeline ──────────────────────────────────────────────────


def ingest_pdfs(uploaded_files: list, api_key: str, embedder, reranker):
    """Full pipeline: PDFs → chunks → BM25 + FAISS indexes.

    For each uploaded file:
      1. Write it to a temp file (unstructured needs a real disk path, not bytes)
      2. Partition into elements (text, tables, images)
      3. Build a chunk dict for every element
      4. Describe images via GPT-4o vision

    Then build both indexes over all collected chunks and return everything.
    """
    all_chunks = []         # accumulates chunk dicts across all uploaded files
    image_b64_lookup = {}   # reserved for future lookup by chunk_id

    total_files = len(uploaded_files)

    for file_index, uploaded_file in enumerate(uploaded_files):
        st.write(f"uploaded_file: {uploaded_file.name} ({file_index + 1}/{total_files})")

        # Write the uploaded bytes to a temp file — unstructured needs a real file path
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, uploaded_file.name)

        try:
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.read())       # dump Streamlit's in-memory bytes to disk

            with st.spinner(f"Partitioning {uploaded_file.name}..."):
                try:
                    elements = partition_pdf(tmp_path)  # break the PDF into structured elements
                except Exception as e:
                    st.warning(f"Skipping {uploaded_file.name} : {e}")
                    continue

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)  # always clean up the temp folder

        tables, texts = table_text_segregation(elements)    # split elements by type
        images = get_images(elements)                        # extract unique base64 images

        print(f"   Texts: {len(texts)}, Tables: {len(tables)}, Images: {len(images)}")

        # ── Text chunks ───────────────────────────────────────────────────────
        for text in texts:
            raw_text = text.text if hasattr(text, "text") else str(text)    # safely get text content
            raw_text = clean_text(raw_text)     # collapse extra whitespace

            chunk = {
                "chunk_id": str(uuid.uuid4()),                              # unique ID for this chunk
                "modality": "text",
                "source_pdf": uploaded_file.name,
                "page_number": getattr(text.metadata, "page_number", None),
                "raw_text": raw_text,
                "image_b64": None,
                "retrieval_text": raw_text,                                 # what gets indexed and searched
            }
            all_chunks.append(chunk)

        # ── Table chunks ──────────────────────────────────────────────────────
        for table in tables:
            if hasattr(table, "metadata") and getattr(table.metadata, "text_as_html", None):
                table_text = html_table_to_text(table.metadata.text_as_html)    # prefer the HTML version
            else:
                table_text = clean_text(str(table))     # fallback: stringify the raw element

            chunk = {
                "chunk_id": str(uuid.uuid4()),
                "modality": "table",
                "source_pdf": uploaded_file.name,
                "page_number": getattr(table.metadata, "page_number", None),
                "raw_text": table_text,
                "image_b64": None,
                "retrieval_text": table_text,
            }
            all_chunks.append(chunk)

        # ── Image chunks ──────────────────────────────────────────────────────
        if images:
            st.write(f"   Describing {len(images)} image(s) with GPT-4o...")

        for i, image in enumerate(images):
            print(f"   Image {i + 1}/{len(images)}...", end=" ", flush=True)
            description = describe_image(image)     # GPT-4o vision → rich text description
            print("done")

            chunk = {
                "chunk_id": str(uuid.uuid4()),
                "modality": "image",
                "source_pdf": uploaded_file.name,
                "page_number": None,
                "raw_text": None,
                "image_b64": image,             # keep the raw image bytes for display
                "retrieval_text": description,  # GPT-4o description is what gets indexed
            }
            all_chunks.append(chunk)

    if not all_chunks:
        st.error(f"No Chunks Found, Check Your Pdfs")
        return {}

    # Build a lightweight metadata list (no image bytes) for fast lookup
    meta = [
        {
            "chunk_id": c["chunk_id"],
            "modality": c["modality"],
            "source_pdf": c["source_pdf"],
            "page_number": c["page_number"],
            "retrieval_text": c["retrieval_text"],
            "raw_text": c.get("raw_text"),
        }
        for c in all_chunks
    ]

    chunk_id_to_meta = {m["chunk_id"]: m for m in meta}    # dict for O(1) lookup by chunk_id

    with st.spinner("Building Bm25 Index"):
        bm25 = build_bm25(all_chunks)

    with st.spinner("Building Faiss Index"):
        faiss_index = build_faiss(all_chunks)

    st.success(f"Ingestion Complete - {len(all_chunks)} chunks indexed")

    return {
        "bm25": bm25,
        "faiss": faiss_index,
        "meta": meta,
        "chunk_id_to_meta": chunk_id_to_meta,
        "image_b64_lookup": image_b64_lookup,
        "embedder": embedder,
        "reranker": reranker,
    }