"""
retrieve.py
-----------
All 4 retrieval methods in one file.

Each function returns a ranked list of result dicts. Every result has the
same structure regardless of modality, so downstream code (chatbot,
evaluator) can handle them uniformly.

Result format:
    {
        "rank"          : 1,
        "chunk_id"      : "abc-123",
        "score"         : 0.91,
        "modality"      : "text" | "table" | "image",
        "source_pdf"    : "paper.pdf",
        "page_number"   : 4,
        "retrieval_text": "...",       # LLaVA description for images
        "raw_text"      : "...",       # None for image chunks
        "image_b64"     : "...",       # base64 string for image chunks, None otherwise
    }

MULTIMODAL NOTE
---------------
image_b64 is now populated for image chunks by looking up the chunk_id
in a fast in-memory dict built from chunks.json at startup.
The LLM (chatbot / evaluator) can then send the raw image directly to
a vision model (e.g. llava, gemma3) so it *sees* the image rather than
relying only on LLaVA's pre-generated text description.

Retrieval itself is unchanged — BM25 and FAISS still operate on
retrieval_text — but generation becomes truly multimodal.

Usage:
    from retrieve import load_indexes, retrieve
    indexes = load_indexes()
    results = retrieve("What is the revenue?", method="hybrid", indexes=indexes, top_k=5)
"""

import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

from utils import tokenize          # shared tokenizer — must match 03_build_index.py


BM25_FILE    = "bm25_index.pkl"
FAISS_FILE   = "faiss_index.bin"
META_FILE    = "index_meta.json"
CHUNKS_FILE  = "chunks.json"       # source of truth for image_b64

EMBED_MODEL  = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ─────────────────────────────────────────
# Load all indexes once
# ─────────────────────────────────────────

def load_indexes() -> dict:
    """
    Load every artifact needed for retrieval and keep it in memory.

    Call this ONCE at startup; pass the returned dict to every retrieve()
    call. Loading models and indexes on every query would be far too slow.

    What gets loaded
    ----------------
    bm25             : BM25Okapi object — the sparse keyword index
    faiss            : FAISS IndexFlatIP — the dense vector index
    meta             : list of dicts, positionally aligned to both indexes
    chunk_id_to_meta : fast O(1) lookup from chunk_id → meta entry
    image_b64_lookup : fast O(1) lookup from chunk_id → base64 image string
                       (only populated for modality="image" chunks)
    embedder         : SentenceTransformer — encodes queries for FAISS
    reranker         : CrossEncoder — re-scores (query, text) pairs
    """
    print("Loading indexes and models...")

    # ── Sparse index ──────────────────────────────────────────────────────
    with open(BM25_FILE, "rb") as f:
        bm25 = pickle.load(f)

    # ── Dense index ───────────────────────────────────────────────────────
    faiss_index = faiss.read_index(FAISS_FILE)

    # ── Metadata (positionally aligned to both indexes) ───────────────────
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Pre-build O(1) chunk_id → meta lookup so hybrid doesn't do a linear
    # scan on every query
    chunk_id_to_meta = {m["chunk_id"]: m for m in meta}

    # ── image_b64 lookup ──────────────────────────────────────────────────
    # chunks.json is the only place that stores the raw base64 image bytes.
    # index_meta.json intentionally omits them to keep that file small.
    # We read chunks.json once here and build a lookup dict so that any
    # image chunk that surfaces in retrieval results can have its b64 data
    # attached with a single dict lookup — no repeated file I/O at query time.
    image_b64_lookup = {}
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("modality") == "image" and chunk.get("image_b64"):
                image_b64_lookup[chunk["chunk_id"]] = chunk["image_b64"]

    print(f"  image_b64_lookup populated: {len(image_b64_lookup)} image chunk(s)")

    # ── ML models ─────────────────────────────────────────────────────────
    embedder = SentenceTransformer(EMBED_MODEL)
    reranker = CrossEncoder(RERANK_MODEL)

    print("All indexes loaded.\n")

    return {
        "bm25"            : bm25,
        "faiss"           : faiss_index,
        "meta"            : meta,
        "chunk_id_to_meta": chunk_id_to_meta,
        "image_b64_lookup": image_b64_lookup,   # ← new
        "embedder"        : embedder,
        "reranker"        : reranker,
    }


# ─────────────────────────────────────────
# Helper: build a single result dict
# ─────────────────────────────────────────

def _make_result(rank: int, score: float, meta_entry: dict, image_b64_lookup: dict) -> dict:
    """
    Construct one result dict from a metadata entry.

    For image chunks, attach the base64 image from the lookup dict.
    For text / table chunks, image_b64 stays None.

    Parameters
    ----------
    rank             : 1-based position in the result list
    score            : relevance score from the retrieval method
    meta_entry       : one entry from index_meta.json
    image_b64_lookup : chunk_id → base64 string (built in load_indexes)
    """
    chunk_id = meta_entry["chunk_id"]
    modality = meta_entry["modality"]

    # Attach b64 only for image chunks; None for text/table
    image_b64 = image_b64_lookup.get(chunk_id) if modality == "image" else None

    return {
        "rank"          : rank,
        "chunk_id"      : chunk_id,
        "score"         : round(float(score), 4),
        "modality"      : modality,
        "source_pdf"    : meta_entry["source_pdf"],
        "page_number"   : meta_entry.get("page_number"),
        "retrieval_text": meta_entry["retrieval_text"],
        "raw_text"      : meta_entry.get("raw_text"),
        "image_b64"     : image_b64,        # populated for image chunks
    }


# ─────────────────────────────────────────
# 1. Sparse — BM25
# ─────────────────────────────────────────

def retrieve_bm25(query: str, bm25, meta: list, image_b64_lookup: dict, top_k: int = 5) -> list:
    """
    BM25 keyword retrieval.

    Tokenize the query with the same pipeline used at index-build time
    (lowercase → strip punctuation → remove stopwords → Porter stem),
    then score every document and return the top_k highest-scored chunks.

    Image chunks participate in BM25 just like text chunks — their
    retrieval_text (the LLaVA description) was indexed the same way.
    """
    tokenized_query = tokenize(query)
    scores          = bm25.get_scores(tokenized_query)
    top_indices     = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        results.append(_make_result(rank, scores[idx], meta[idx], image_b64_lookup))

    return results


# ─────────────────────────────────────────
# 2. Dense — FAISS
# ─────────────────────────────────────────

def retrieve_dense(query: str, faiss_index, meta: list, embedder, image_b64_lookup: dict, top_k: int = 5) -> list:
    """
    Dense semantic retrieval using cosine similarity.

    The query is encoded by the same SentenceTransformer that encoded all
    chunks at index-build time. Both query and document vectors are
    L2-normalised so inner product == cosine similarity.

    Image chunks participate via the embedding of their LLaVA description.
    Dense retrieval has a semantic advantage: if the user asks a conceptual
    question whose words differ from LLaVA's description, FAISS can still
    surface the image because the meaning is aligned in vector space.
    """
    query_vec = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)

    scores, indices = faiss_index.search(query_vec, top_k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        if idx == -1:
            continue
        results.append(_make_result(rank, score, meta[idx], image_b64_lookup))

    return results


# ─────────────────────────────────────────
# 3. Hybrid — BM25 + Dense (RRF)
# ─────────────────────────────────────────

def retrieve_hybrid(
    query: str,
    bm25,
    faiss_index,
    meta: list,
    embedder,
    chunk_id_to_meta: dict,
    image_b64_lookup: dict,
    top_k: int = 5,
    rrf_k: int = 60,
) -> list:
    """
    Reciprocal Rank Fusion (RRF) over BM25 and dense ranked lists.

    RRF formula: score(chunk) = Σ  1 / (k + rank_in_system)
    k = 60 is the standard default (Cormack 2009).

    fetch_k = top_k * 3 so the fusion step has enough candidates from each
    system to meaningfully re-rank rather than just concatenating two lists.
    """
    fetch_k = top_k * 3

    bm25_results  = retrieve_bm25(query, bm25, meta, image_b64_lookup, top_k=fetch_k)
    dense_results = retrieve_dense(query, faiss_index, meta, embedder, image_b64_lookup, top_k=fetch_k)

    rrf_scores: dict = {}

    for result in bm25_results:
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + result["rank"])

    for result in dense_results:
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + result["rank"])

    sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:top_k]

    results = []
    for rank, cid in enumerate(sorted_ids, start=1):
        m = chunk_id_to_meta[cid]
        results.append(_make_result(rank, rrf_scores[cid], m, image_b64_lookup))

    return results


# ─────────────────────────────────────────
# 4. Hybrid + Reranker
# ─────────────────────────────────────────

def retrieve_hybrid_reranked(
    query: str,
    bm25,
    faiss_index,
    meta: list,
    embedder,
    reranker,
    chunk_id_to_meta: dict,
    image_b64_lookup: dict,
    top_k: int = 5,
) -> list:
    """
    Two-stage pipeline:
      Stage 1 — Hybrid RRF produces a candidate pool (top_k * 3 chunks)
      Stage 2 — Cross-encoder reranker scores every (query, retrieval_text)
                pair and we keep the top_k highest-scored results.

    The cross-encoder reads query + document jointly (full attention across
    both), giving a much more accurate relevance signal than bi-encoder
    cosine similarity — but it's too slow to run on the whole corpus.

    For images, the reranker scores against retrieval_text (LLaVA description).
    image_b64 is carried through untouched and used downstream by the LLM.
    """
    candidates = retrieve_hybrid(
        query, bm25, faiss_index, meta, embedder, chunk_id_to_meta,
        image_b64_lookup, top_k=top_k * 3,
    )

    if not candidates:
        return []

    pairs  = [(query, c["retrieval_text"]) for c in candidates]
    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["score"] = round(float(score), 4)

    candidates.sort(key=lambda x: x["score"], reverse=True)

    results = []
    for rank, candidate in enumerate(candidates[:top_k], start=1):
        candidate["rank"] = rank
        results.append(candidate)

    return results


# ─────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────

def retrieve(query: str, method: str, indexes: dict, top_k: int = 5) -> list:
    """
    Single entry point — call this from the chatbot and evaluator.

    Parameters
    ----------
    query   : the user's question as a plain string
    method  : "bm25" | "dense" | "hybrid" | "hybrid_reranker"
    indexes : dict returned by load_indexes()
    top_k   : number of results to return

    Returns
    -------
    List of result dicts. image_b64 is populated for image chunks so the
    LLM can receive the actual image rather than just a text description.
    """
    bm25             = indexes["bm25"]
    faiss_index      = indexes["faiss"]
    meta             = indexes["meta"]
    chunk_id_to_meta = indexes["chunk_id_to_meta"]
    image_b64_lookup = indexes["image_b64_lookup"]
    embedder         = indexes["embedder"]
    reranker         = indexes["reranker"]

    if method == "bm25":
        return retrieve_bm25(query, bm25, meta, image_b64_lookup, top_k)

    elif method == "dense":
        return retrieve_dense(query, faiss_index, meta, embedder, image_b64_lookup, top_k)

    elif method == "hybrid":
        return retrieve_hybrid(
            query, bm25, faiss_index, meta, embedder,
            chunk_id_to_meta, image_b64_lookup, top_k,
        )

    elif method == "hybrid_reranker":
        return retrieve_hybrid_reranked(
            query, bm25, faiss_index, meta, embedder, reranker,
            chunk_id_to_meta, image_b64_lookup, top_k,
        )

    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            "Choose from: bm25, dense, hybrid, hybrid_reranker"
        )
