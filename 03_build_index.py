"""
03_build_index.py
-----------------
Reads chunks.json and builds two indexes:

  bm25_index.pkl   — BM25 sparse index (keyword matching)
  faiss_index.bin  — FAISS dense index (semantic vectors)
  index_meta.json  — chunk metadata list aligned to both indexes

Run this once after ingestion. Re-run whenever chunks.json changes.

Fixes applied vs original:
  - tokenize() now imported from utils.py so query and document tokens
    are ALWAYS processed identically (was the critical BM25 mismatch bug)
  - nltk.download uses quiet=True so it doesn't spam the terminal on every run

Usage:
    python 03_build_index.py
"""

import json
import pickle
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from utils import tokenize          # shared tokenizer — MUST match retrieve.py


CHUNKS_FILE = "chunks.json"
BM25_FILE   = "bm25_index.pkl"
FAISS_FILE  = "faiss_index.bin"
META_FILE   = "index_meta.json"

# Small, fast, good enough for a bachelor's thesis
EMBED_MODEL = "all-MiniLM-L6-v2"


# ===============================
# Load chunks
# ===============================

def load_chunks(path: str) -> list[dict]:
    """Read chunks.json (one JSON object per line) into a list of dicts."""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


# ===============================
# Build BM25 index
# ===============================

def build_bm25(chunks: list[dict]) -> BM25Okapi:
    """
    Tokenize every chunk's retrieval_text and build a BM25Okapi index.

    Uses utils.tokenize() which applies:
      lowercase → remove punctuation → remove stopwords → Porter stemming

    retrieve.py imports the SAME tokenize() function so query tokens
    and document tokens are always in the same form.
    """
    corpus = [tokenize(chunk["retrieval_text"]) for chunk in chunks]
    bm25   = BM25Okapi(corpus)
    print(f"BM25 index built over {len(corpus)} documents")
    return bm25


# ===============================
# Build FAISS index
# ===============================

def build_faiss(chunks: list[dict], model: SentenceTransformer) -> faiss.IndexFlatIP:
    """
    Encode every chunk's retrieval_text into a dense vector and build a
    FAISS flat inner-product index.

    Vectors are L2-normalised so inner product == cosine similarity.
    IndexFlatIP is brute-force (exact) — fine for thesis-scale datasets.
    """
    texts      = [chunk["retrieval_text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Normalise so dot product = cosine similarity
    faiss.normalize_L2(embeddings)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


# ===============================
# Build metadata list
# ===============================

def build_meta(chunks: list[dict]) -> list[dict]:
    """
    Build a metadata list that is positionally aligned with both indexes.

    index 0 in bm25 / faiss  ←→  meta[0]
    index 1 in bm25 / faiss  ←→  meta[1]
    ...

    image_b64 is intentionally excluded — look it up from chunks.json by
    chunk_id only when you actually need to display the image.
    """
    meta = []
    for chunk in chunks:
        meta.append({
            "chunk_id"      : chunk["chunk_id"],
            "modality"      : chunk["modality"],
            "source_pdf"    : chunk["source_pdf"],
            "page_number"   : chunk["page_number"],
            "retrieval_text": chunk["retrieval_text"],
            "raw_text"      : chunk.get("raw_text"),
        })
    return meta


# ===============================
# Main
# ===============================

def main():

    # 1. Load all chunks
    chunks = load_chunks(CHUNKS_FILE)

    # 2. Build and save BM25
    bm25 = build_bm25(chunks)
    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)
    print(f"Saved BM25 index → {BM25_FILE}")

    # 3. Load embedding model and build FAISS
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model       = SentenceTransformer(EMBED_MODEL)
    faiss_index = build_faiss(chunks, model)
    faiss.write_index(faiss_index, FAISS_FILE)
    print(f"Saved FAISS index → {FAISS_FILE}")

    # 4. Save metadata (positionally aligned to both indexes)
    meta = build_meta(chunks)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved index metadata → {META_FILE}")

    print("\nIndexing complete. Files created:")
    print(f"  {BM25_FILE}")
    print(f"  {FAISS_FILE}")
    print(f"  {META_FILE}")


if __name__ == "__main__":
    main()
