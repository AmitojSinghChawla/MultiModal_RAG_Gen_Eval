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
        "retrieval_text": "...",       # GPT-4o description for images, raw text for others
        "raw_text"      : "...",       # None for image chunks
        "image_b64"     : "...",       # base64 string for image chunks, None otherwise
    }
"""

import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from utils import tokenize  # shared tokenizer — must match 02_build_index.py

# ── File paths ────────────────────────────────────────────────────────────────
# index_meta.json is positionally aligned to both indexes (meta[0] describes
# the chunk at position 0 in BM25 and position 0 in FAISS).
BM25_FILE   = r"C:\Users\amito\PycharmProjects\MultiModal_RAG_Gen_Eval\Indexes\bm25_index.pkl"
FAISS_FILE  = r"C:\Users\amito\PycharmProjects\MultiModal_RAG_Gen_Eval\Indexes\faiss_index.bin"
META_FILE   = r"C:\Users\amito\PycharmProjects\MultiModal_RAG_Gen_Eval\Indexes\index_meta.json"
CHUNKS_FILE = r"C:\Users\amito\PycharmProjects\MultiModal_RAG_Gen_Eval\Chunks\chunks.json"
# chunks.json is the original source of truth written by 01chunk_exporter.py.
# It is the ONLY place that stores raw base64 image bytes. index_meta.json
# deliberately omits them to stay small and human-readable.

# ── Model names ───────────────────────────────────────────────────────────────
EMBED_MODEL  = "BAAI/bge-base-en-v1.5"

# RERANK_MODEL: the cross-encoder used in the final reranking stage.
#   Unlike the bi-encoder (which encodes query and document separately),
#   a cross-encoder takes BOTH as one input and produces a single relevance
#   score. This is much more accurate but also much slower — too slow to run
#   on the whole corpus, which is why we only run it on the top candidates
#   from the hybrid stage (a "two-stage" pipeline).
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


# ─────────────────────────────────────────
# Load all indexes once
# ─────────────────────────────────────────

def load_indexes() -> dict:
    """
    Load every artifact needed for retrieval and return them in one dict.

    This is called ONCE at startup (chatbot loop or evaluator main).
    Everything — indexes, metadata, image bytes, ML models — is loaded into
    memory here and reused for every query. You never want to reload a
    SentenceTransformer or a FAISS index on every query; that would take
    several seconds each time.
    """

    print("Loading indexes and models...")

    # ── BM25 ──────────────────────────────────────────────────────────────
    # BM25Okapi is a Python object (from the rank_bm25 library).
    # pickle.load deserialises it back from the binary file that
    # 02_build_index.py wrote with pickle.dump.
    with open(BM25_FILE, "rb") as f:
        bm25 = pickle.load(f)

    # ── FAISS ─────────────────────────────────────────────────────────────
    # faiss.read_index loads the vector index from disk.
    # The index type here is IndexFlatIP (inner product), which does exact
    # brute-force search — no approximation. Fine for thousands of chunks.
    faiss_index = faiss.read_index(FAISS_FILE)

    # ── Metadata ──────────────────────────────────────────────────────────
    # index_meta.json is a JSON array. meta[N] describes the chunk at
    # position N in both BM25 and FAISS — this positional alignment is the
    # critical invariant that makes retrieval work correctly.
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # ── chunk_id_to_meta ──────────────────────────────────────────────────
    # meta is a list, so looking up a chunk by its chunk_id would normally
    # require iterating through every entry — O(N) per lookup.
    # We build a dict here so lookups are O(1).
    # Used in hybrid retrieval where RRF works with chunk_ids (not positions).
    chunk_id_to_meta = {m["chunk_id"]: m for m in meta}

    # ── image_b64 lookup ──────────────────────────────────────────────────
    # index_meta.json omits image bytes to stay small. chunks.json has
    # everything. We scan it once here and pull out only image chunks,
    # building {chunk_id: base64_string} for O(1) lookup after retrieval.
    image_b64_lookup = {}
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            # Only store entries that are images and actually have b64 data.
            # (Some image extractions can fail and leave image_b64 as None.)
            if chunk.get("modality") == "image" and chunk.get("image_b64"):
                image_b64_lookup[chunk["chunk_id"]] = chunk["image_b64"]

    print(f"  image_b64_lookup populated: {len(image_b64_lookup)} image chunk(s)")

    # ── ML models ─────────────────────────────────────────────────────────
    # SentenceTransformer downloads model weights on first use and caches
    # them locally. Subsequent loads are fast (reads from disk cache).
    # The embedder is used by dense and hybrid retrieval to encode queries.
    embedder = SentenceTransformer(EMBED_MODEL)

    # CrossEncoder loads the reranker weights similarly.
    # use_fp16=True halves memory usage and speeds up inference with
    # negligible accuracy difference on modern hardware.
    reranker = CrossEncoder(RERANK_MODEL)

    print("All indexes loaded.\n")

    # Return everything in one dict so callers (chatbot, evaluator) can pass
    # it around as a single object rather than juggling 7 separate variables.
    return {
        "bm25":             bm25,
        "faiss":            faiss_index,
        "meta":             meta,
        "chunk_id_to_meta": chunk_id_to_meta,
        "image_b64_lookup": image_b64_lookup,
        "embedder":         embedder,
        "reranker":         reranker,
    }


# ─────────────────────────────────────────
# Helper: build a single result dict
# ─────────────────────────────────────────

def _make_result(rank: int, score: float, meta_entry: dict, image_b64_lookup: dict) -> dict:
    """
    Construct one result dict from a metadata entry.

    Every retrieval method calls this to turn a raw (score, metadata) pair
    into the standardised result format that the chatbot and evaluator expect.
    Centralising this here means the format is guaranteed consistent across
    all four methods.

    The leading underscore signals it is a private helper — only meant to
    be called from within this file, not imported elsewhere.
    """
    chunk_id = meta_entry["chunk_id"]
    modality = meta_entry["modality"]

    # For image chunks, look up the base64 bytes from the lookup dict.
    # For text and table chunks, image_b64 stays None — there are no bytes
    # to attach, and the LLM will receive raw_text instead.
    image_b64 = image_b64_lookup.get(chunk_id) if modality == "image" else None

    return {
        "rank":            rank,
        "chunk_id":        chunk_id,
        "score":           round(float(score), 4),  # 4dp is enough for display/debugging
        "modality":        modality,
        "source_pdf":      meta_entry["source_pdf"],
        "page_number":     meta_entry.get("page_number"),   # .get() because it can be None
        "retrieval_text":  meta_entry["retrieval_text"],    # what was indexed and searched
        "raw_text":        meta_entry.get("raw_text"),      # original text; None for images
        "image_b64":       image_b64,
    }


# ─────────────────────────────────────────
# Method 1: BM25
# ─────────────────────────────────────────

def retrieve_bm25(
    query: str, bm25, meta: list, image_b64_lookup: dict, top_k: int = 5
) -> list:
    """
    BM25 (Best Match 25) keyword retrieval.

    BM25 is a classical ranking function based on term frequency and inverse
    document frequency. It scores every document in the corpus against the
    query and returns the highest scorers. No neural network involved —
    it is purely counting and weighting word occurrences.

    Strengths: exact keyword matching, very fast, no GPU needed.
    Weaknesses: misses synonyms and paraphrases ("car" won't match "automobile").
    """

    # tokenize() applies the same lowercasing, stopword removal, and stemming
    # that was applied to documents at index-build time. This is critical —
    # if the query tokens don't match the index tokens, BM25 scores everything
    # as zero.
    tokenized_query = tokenize(query)

    # bm25.get_scores() returns a numpy array of length = number of chunks.
    # scores[i] is the BM25 relevance score of chunk i against the query.
    # Higher = more relevant.
    scores = bm25.get_scores(tokenized_query)

    # np.argsort returns indices that would sort the array ascending.
    # [::-1] reverses to descending (highest score first).
    # [:top_k] takes only the top_k indices.
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        # idx is the integer position in both the BM25 corpus and meta list.
        results.append(_make_result(rank, scores[idx], meta[idx], image_b64_lookup))

    return results


# ─────────────────────────────────────────
# Method 2: Dense (semantic) retrieval
# ─────────────────────────────────────────

def retrieve_dense(
    query: str,
    faiss_index,
    meta: list,
    embedder,
    image_b64_lookup: dict,
    top_k: int = 5,
) -> list:
    """
    Dense semantic retrieval using vector similarity.

    Instead of matching keywords, this method encodes the query into a
    vector and finds the chunks whose vectors are most similar. Similarity
    is measured by cosine similarity, capturing meaning regardless of exact
    word choice.

    Strengths: handles synonyms, paraphrases, conceptual questions.
    Weaknesses: can miss exact technical terms; slower than BM25.
    """

    # embedder.encode() runs the query through the SentenceTransformer model
    # and returns a numpy array of shape (1, 768).
    # convert_to_numpy=True ensures we get numpy, not a torch tensor,
    # since FAISS expects numpy.
    query_vec = embedder.encode([query], convert_to_numpy=True)

    # L2 normalisation scales the vector to unit length so the dot product
    # between two vectors equals their cosine similarity. The document vectors
    # were also L2-normalised at index-build time, so both sides match.
    faiss.normalize_L2(query_vec)

    # faiss_index.search() finds the top_k most similar vectors.
    # Returns two arrays of shape (1, top_k):
    #   scores  — cosine similarity scores
    #   indices — integer positions of matching chunks in the index
    scores, indices = faiss_index.search(query_vec, top_k)

    results = []
    # indices[0] and scores[0] unpack the single-query batch dimension.
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        # FAISS returns -1 as a sentinel when it can't fill all top_k slots
        # (e.g. if the index has fewer than top_k vectors). Skip these.
        if idx == -1:
            continue
        results.append(_make_result(rank, score, meta[idx], image_b64_lookup))

    return results


# ─────────────────────────────────────────
# Method 3: Hybrid (BM25 + Dense via RRF)
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
    Reciprocal Rank Fusion (RRF) — combines BM25 and dense ranked lists.

    The core insight: BM25 and dense retrieval fail in complementary ways.
    BM25 misses synonyms; dense misses exact keywords. A chunk that appears
    in both lists is almost certainly relevant. RRF fuses the two lists by
    converting ranks to scores and summing them.

    RRF formula for each chunk:
        rrf_score = 1/(k + rank_in_bm25) + 1/(k + rank_in_dense)

    k=60 is the standard constant from the original paper (Cormack 2009).
    Chunks only in one list still get a partial score.
    """

    # We fetch 3x more candidates than needed from each system so the
    # fusion step has a wide enough pool to meaningfully re-rank.
    fetch_k = top_k * 3

    bm25_results  = retrieve_bm25(query, bm25, meta, image_b64_lookup, top_k=fetch_k)
    dense_results = retrieve_dense(query, faiss_index, meta, embedder, image_b64_lookup, top_k=fetch_k)

    # rrf_scores accumulates the RRF score for each chunk_id seen in either list.
    # We use chunk_id (UUID string) as the key rather than integer position
    # because the same chunk appears at different positions in BM25 vs dense.
    rrf_scores: dict = {}

    # For each chunk in BM25 results: add 1/(60 + rank) to its RRF score.
    for result in bm25_results:
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + result["rank"])

    # Same for dense results. If a chunk appeared in both lists, its two
    # contributions are summed — that is how RRF rewards consensus.
    for result in dense_results:
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + result["rank"])

    # Sort all seen chunk_ids by accumulated RRF score, highest first,
    # and keep only top_k.
    sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:top_k]

    results = []
    for rank, cid in enumerate(sorted_ids, start=1):
        # chunk_id_to_meta gives O(1) metadata lookup — built in load_indexes()
        # exactly for this purpose.
        m = chunk_id_to_meta[cid]
        results.append(_make_result(rank, rrf_scores[cid], m, image_b64_lookup))

    return results


# ─────────────────────────────────────────
# Method 4: Hybrid + Reranker
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
    Two-stage retrieval pipeline — the most accurate method.

    Stage 1 (Recall): Hybrid RRF casts a wide net and returns top_k * 3
    candidates. Speed matters here — BM25 and bi-encoder are both fast.

    Stage 2 (Precision): The cross-encoder reranker scores every candidate
    against the query with full attention across both — it reads the query
    and document together, not separately. This is far more accurate than
    bi-encoder cosine similarity but also far slower. Running it on only
    top_k*3 candidates keeps it practical.

    The tradeoff: this is the slowest method but typically the most accurate.
    """

    # Stage 1: get top_k*3 candidates from hybrid retrieval.
    # e.g. if top_k=5, we get 15 candidates for the reranker to score.
    candidates = retrieve_hybrid(
        query, bm25, faiss_index, meta, embedder,
        chunk_id_to_meta, image_b64_lookup, top_k=top_k * 3,
    )

    # Guard: if hybrid returned nothing (e.g. empty index), return immediately.
    if not candidates:
        return []

    # Stage 2: build (query, document) pairs for the cross-encoder.
    # We use retrieval_text for every modality — for images that is the
    # GPT-4o description, for text/tables it is the raw text.
    # The reranker never sees actual image pixels; it scores based on how
    # well the image's text description answers the query.
    pairs = [(query, c["retrieval_text"]) for c in candidates]

    # reranker.predict() runs all pairs through the cross-encoder in one batch.
    # Returns a 1D array of relevance scores — higher = more relevant.
    scores = reranker.predict(pairs)

    # Overwrite the RRF scores from stage 1 with the more accurate
    # cross-encoder scores.
    for candidate, score in zip(candidates, scores):
        candidate["score"] = round(float(score), 4)

    # Re-sort by the new cross-encoder scores.
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Take top_k and fix up rank numbers to reflect the new order.
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
    Single entry point called by 03_evaluate.py and 04_chatbot.py for every query.

    The chatbot and evaluator only ever call this function — they don't
    need to know which method is which or what arguments each one needs.
    This function handles the dispatch and unpacks the indexes dict so
    callers don't have to.

    Parameters
    ----------
    query   : the user's question as a plain string
    method  : "bm25" | "dense" | "hybrid" | "hybrid_reranker"
    indexes : the dict returned by load_indexes()
    top_k   : how many results to return
    """

    # Unpack the indexes dict into named variables for readability.
    # All of these were loaded once at startup by load_indexes().
    bm25             = indexes["bm25"]
    faiss_index      = indexes["faiss"]
    meta             = indexes["meta"]
    chunk_id_to_meta = indexes["chunk_id_to_meta"]
    image_b64_lookup = indexes["image_b64_lookup"]
    embedder         = indexes["embedder"]
    reranker         = indexes["reranker"]

    if method == "bm25":
        # Only needs BM25 and metadata — no neural model involved.
        return retrieve_bm25(query, bm25, meta, image_b64_lookup, top_k)

    elif method == "dense":
        # Only needs FAISS and the embedding model.
        return retrieve_dense(query, faiss_index, meta, embedder, image_b64_lookup, top_k)

    elif method == "hybrid":
        # Needs both BM25 and FAISS, plus chunk_id_to_meta for the RRF merge step.
        return retrieve_hybrid(
            query, bm25, faiss_index, meta, embedder,
            chunk_id_to_meta, image_b64_lookup, top_k,
        )

    elif method == "hybrid_reranker":
        # Needs everything — hybrid for stage 1, reranker for stage 2.
        return retrieve_hybrid_reranked(
            query, bm25, faiss_index, meta, embedder,
            reranker, chunk_id_to_meta, image_b64_lookup, top_k,
        )

    else:
        # Fail loudly on an unrecognised method so bugs surface immediately.
        raise ValueError(
            f"Unknown method '{method}'. "
            "Choose from: bm25, dense, hybrid, hybrid_reranker"
        )