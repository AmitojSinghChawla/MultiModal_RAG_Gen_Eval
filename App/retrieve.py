"""
retrieve.py
───────────
Implements all four retrieval methods used by the RAG pipeline to find the
most relevant chunks for a given user query.

Pipeline position
─────────────────
  Called by  : streamlit_app.py → retrieve()
               (called once per user message, before generate_answer())
  Calls out to: tokenize() in Ingestion.py (for BM25 query tokenisation)
                embedder.encode()          (SentenceTransformer, for dense queries)
                faiss_index.search()       (FAISS, for dense nearest-neighbour lookup)
                bm25.get_scores()          (BM25Okapi, for sparse keyword scoring)
                reranker.predict()         (CrossEncoder, for final reranking)

Input  (via the unified retrieve() entry point)
──────
  query   : the user's plain-text question string
  method  : one of "bm25" | "dense" | "hybrid" | "hybrid_reranker"
  indexes : the dict produced by ingest_pdfs() in Ingestion.py,
            stored in st.session_state["indexes"] by streamlit_app.py
  top_k   : how many results to return (set by the sidebar slider in streamlit_app.py)

Output  (→ passed to generate_answer() in generate.py)
──────
  A list of result dicts, one per retrieved chunk, each containing:
    "rank"          : int   — 1-based position in the ranked result list
    "chunk_id"      : str   — UUID matching the chunk built in Ingestion.py
    "score"         : float — retrieval score (BM25 / cosine / RRF / reranker)
    "modality"      : str   — "text" | "table" | "image"
    "source_pdf"    : str   — filename of the PDF this chunk came from
    "page_number"   : int | None — page number, or None for image chunks
    "retrieval_text": str   — the text that was indexed (GPT-4o description for images)
    "raw_text"      : str | None — original extracted text; None for image chunks
    "image_b64"     : str | None — raw base64 string for image chunks; None otherwise
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from Ingestion import tokenize

# ─────────────────────────────────────────
# 1. Result Dict Builder
#    Every retrieval method produces its own raw scores and indices,
#    but all of them use this single helper to package each hit into
#    a uniform result dict. This ensures the output format is identical
#    regardless of which method was used — generate.py and streamlit_app.py
#    can read the same keys without knowing which retrieval path ran.
# ─────────────────────────────────────────


def _make_result(
    rank: int, score: float, meta_entry: dict, image_b64_lookup: dict
) -> dict:
    """
    Build a single standardised result dict from retrieval metadata.

    Input
    ─────
      rank           : 1-based rank of this result in the final ordered list
      score          : the retrieval score for this chunk (BM25 / cosine / RRF / reranker)
      meta_entry     : a single entry from the meta list built in Ingestion.py
                       ← comes from indexes["meta"][i] or indexes["chunk_id_to_meta"][chunk_id]
      image_b64_lookup : dict mapping chunk_id → base64 string
                         ← comes from indexes["image_b64_lookup"] built in Ingestion.py

    Output: a dict with all keys that generate.py and streamlit_app.py expect.
            → appended to the results list by every retrieval method
            → ultimately passed to generate_answer() in generate.py

    Key behaviour: image_b64 is only populated for "image" modality chunks.
    For text and table chunks it is set to None to avoid carrying large binary
    data through the result list unnecessarily.
    The lookup uses .get() — if the chunk_id is somehow missing from the lookup
    (e.g. ingestion bug), it returns None rather than raising a KeyError.
    """
    chunk_id = meta_entry["chunk_id"]
    modality = meta_entry["modality"]

    # Only fetch the raw image bytes for image chunks — text/table chunks have no base64
    image_b64 = image_b64_lookup.get(chunk_id) if modality == "image" else None

    # → this dict is the fundamental unit consumed by generate.py and displayed in streamlit_app.py
    return {
        "rank": rank,
        "chunk_id": chunk_id,
        "score": round(
            float(score), 4
        ),  # 4 decimal places is enough for display/debugging
        "modality": modality,
        "source_pdf": meta_entry["source_pdf"],
        "page_number": meta_entry.get(
            "page_number"
        ),  # .get() returns None if key missing
        "retrieval_text": meta_entry[
            "retrieval_text"
        ],  # used as the reranker input pair
        "raw_text": meta_entry.get("raw_text"),  # None for image chunks
        "image_b64": image_b64,  # base64 string → sent to LLM as image_url block
    }


# ─────────────────────────────────────────
# 2. Method 1 — BM25 Sparse Retrieval
#    Scores every chunk in the corpus against the tokenized query using
#    TF-IDF-style term frequency weighting. Best for exact keyword matches
#    (model names, figure numbers, rare technical terms).
# ─────────────────────────────────────────


def retrieve_bm25(
    query: str, bm25, meta: list, image_b64_lookup: dict, top_k: int = 5
) -> list:
    """
    Retrieve the top-k chunks using BM25 sparse keyword scoring.

    Input
    ─────
      query          : the user's plain-text question string
                       ← comes from streamlit_app.py via retrieve()
      bm25           : a fitted BM25Okapi object
                       ← indexes["bm25"] built by build_bm25() in Ingestion.py
      meta           : the ordered list of chunk metadata dicts
                       ← indexes["meta"] — position i in meta must match position i in the BM25 corpus
      image_b64_lookup : chunk_id → base64 mapping
                         ← indexes["image_b64_lookup"] from Ingestion.py
      top_k          : number of results to return
                       ← set by the sidebar slider in streamlit_app.py

    Output: list of top_k result dicts ordered by BM25 score descending.
            → used directly when method="bm25" in retrieve()
            → used as one of two candidate lists when method="hybrid" or "hybrid_reranker"

    How BM25 scoring works:
      tokenize() produces the same normalised token list used at index time in Ingestion.py.
      bm25.get_scores() returns one float per corpus document — higher means more term overlap.
      np.argsort(scores)[::-1][:top_k] sorts descending and takes the top-k indices.
      Each index i directly maps to meta[i] because BM25 and meta share the same ordering.
    """
    # Tokenize the query identically to how chunks were tokenized at index time
    # (same function, same stemmer, same stopword list) — mismatches would break scoring
    tokenized_query = tokenize(query)

    # get_scores() computes BM25(query, doc) for every doc in the corpus at once
    # Returns a numpy array of length == len(all_chunks) from ingest time
    scores = bm25.get_scores(tokenized_query)

    # argsort returns ascending order; [::-1] reverses to descending; [:top_k] takes the best
    top_indices = np.argsort(scores)[::-1][:top_k]

    # → list of result dicts passed back to retrieve() or to retrieve_hybrid()
    return [
        _make_result(rank, scores[idx], meta[idx], image_b64_lookup)
        for rank, idx in enumerate(top_indices, start=1)
    ]


# ─────────────────────────────────────────
# 3. Method 2 — Dense Semantic Retrieval
#    Embeds the query with the same SentenceTransformer used at index time,
#    then finds the nearest chunk vectors in FAISS using cosine similarity.
#    Best for paraphrases and semantic questions that don't use exact keywords.
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
    Retrieve the top-k chunks using dense vector similarity (cosine search via FAISS).

    Input
    ─────
      query        : the user's plain-text question string
                     ← comes from streamlit_app.py via retrieve()
      faiss_index  : FAISS IndexFlatIP loaded with L2-normalised chunk embeddings
                     ← indexes["faiss"] built by build_faiss() in Ingestion.py
      meta         : ordered list of chunk metadata dicts
                     ← indexes["meta"] — position i in meta matches FAISS vector i
      embedder     : SentenceTransformer("BAAI/bge-base-en-v1.5")
                     ← indexes["embedder"] passed through from streamlit_app.py
      image_b64_lookup : chunk_id → base64 mapping
                         ← indexes["image_b64_lookup"] from Ingestion.py
      top_k        : number of results to return

    Output: list of top_k result dicts ordered by cosine similarity descending.
            → used directly when method="dense" in retrieve()
            → used as one of two candidate lists when method="hybrid" or "hybrid_reranker"

    How dense scoring works:
      embedder.encode() converts the query string to the same 768-dim vector space
      used at index time. faiss.normalize_L2() makes it unit-length so inner product
      equals cosine similarity. faiss_index.search() returns the top_k nearest neighbours
      as (scores, indices) where index i maps to meta[i].

    Edge case: FAISS returns idx == -1 when the index has fewer vectors than top_k.
    These are skipped to avoid an out-of-bounds access on meta[-1].
    """
    # Encode the query into the same embedding space used at index time.
    # [query] wraps it in a list because encode() expects an iterable of strings.
    # convert_to_numpy=True is required — FAISS only accepts numpy arrays.
    query_vec = embedder.encode([query], convert_to_numpy=True)

    # Normalise the query vector to unit length so the inner product with each
    # chunk vector equals cosine similarity (matching the normalisation applied
    # during build_faiss() in Ingestion.py)
    faiss.normalize_L2(query_vec)

    # search() returns two arrays of shape (1, top_k):
    #   scores  — cosine similarity values (higher = more similar)
    #   indices — positions in the FAISS index (map directly to meta list positions)
    scores, indices = faiss_index.search(query_vec, top_k)

    results = []
    # [0] unpacks the single-query batch dimension — we only have one query at a time
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        # FAISS returns -1 for empty slots when fewer vectors exist than top_k
        if idx == -1:
            continue
        results.append(_make_result(rank, score, meta[idx], image_b64_lookup))

    # → list of result dicts passed back to retrieve() or to retrieve_hybrid()
    return results


# ─────────────────────────────────────────
# 4. Method 3 — Hybrid Retrieval (BM25 + Dense via RRF)
#    Runs both BM25 and dense retrieval independently, then merges the two
#    ranked lists using Reciprocal Rank Fusion (RRF). RRF rewards chunks that
#    rank highly in BOTH lists — these tend to be genuinely relevant rather
#    than strong in just one modality. This is the foundation for method 4.
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
    Fuse BM25 and dense results using Reciprocal Rank Fusion (RRF).

    Input
    ─────
      query, bm25, faiss_index, meta, embedder, image_b64_lookup
                       — same as retrieve_bm25() and retrieve_dense() above
      chunk_id_to_meta : dict mapping chunk_id → metadata entry
                         ← indexes["chunk_id_to_meta"] from Ingestion.py
                         needed here because RRF works by chunk_id (not by list index),
                         so we need to look up metadata for each winning chunk_id
      top_k            : final number of results to return
      rrf_k            : RRF damping constant — controls how much early ranks are rewarded
                         vs late ranks. 60 is the standard published default (Cormack 2009).

    Output: list of top_k result dicts ordered by RRF score descending.
            → used directly when method="hybrid" in retrieve()
            → used as the candidate pool for retrieve_hybrid_reranked()

    How RRF works:
      Each chunk's RRF score = Σ  1 / (rrf_k + rank_in_list)
      summed over each list it appears in. A chunk ranked 1st in both lists scores
      1/(60+1) + 1/(60+1) ≈ 0.033; a chunk ranked 10th in one list and absent from
      the other scores 1/(60+10) ≈ 0.014. The RRF constant 60 was chosen because
      it prevents a single top-ranked result from dominating when the other list
      ranks it poorly — empirically optimal across many IR benchmarks.

    Why fetch_k = top_k * 3:
      Each individual retriever only sees top_k * 3 candidates. This gives enough
      overlap between the two lists for RRF to be meaningful — if both lists only
      returned top_k items, many chunks would appear in only one list and RRF
      would degrade to simple score averaging.
    """
    # Over-fetch from each retriever so there is meaningful overlap to fuse.
    # top_k * 3 means: for top_k=5 we fetch 15 BM25 candidates and 15 FAISS candidates.
    fetch_k = top_k * 3

    # Run both retrievers independently — each sees the same query but uses a
    # completely different scoring mechanism (keyword vs. semantic)
    bm25_results = retrieve_bm25(query, bm25, meta, image_b64_lookup, top_k=fetch_k)
    dense_results = retrieve_dense(
        query, faiss_index, meta, embedder, image_b64_lookup, top_k=fetch_k
    )

    # rrf_scores maps chunk_id → cumulative RRF score across both lists
    rrf_scores: dict = {}

    # Accumulate RRF contributions from the BM25 ranked list.
    # A chunk ranked 1st contributes 1/(60+1) ≈ 0.0164; ranked 15th contributes 1/(60+15) ≈ 0.0133.
    for result in bm25_results:
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + result["rank"])

    # Accumulate RRF contributions from the dense ranked list.
    # If a chunk appeared in both lists, its score is the SUM of both contributions,
    # rewarding consensus between the two retrieval methods.
    for result in dense_results:
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + result["rank"])

    # Sort chunk_ids by their total RRF score descending, take the best top_k
    sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:top_k]

    # Rebuild result dicts using chunk_id_to_meta for O(1) metadata lookup.
    # We can't use list indices here because RRF re-orders chunks from both lists
    # — chunk_id is the only stable identifier across both retrieval results.
    # → list of result dicts passed back to retrieve() or to retrieve_hybrid_reranked()
    return [
        _make_result(rank, rrf_scores[cid], chunk_id_to_meta[cid], image_b64_lookup)
        for rank, cid in enumerate(sorted_ids, start=1)
    ]


# ─────────────────────────────────────────
# 5. Method 4 — Hybrid + CrossEncoder Reranking
#    Extends the hybrid method with a CrossEncoder reranker as a final scoring
#    pass. The CrossEncoder reads both the query and each candidate chunk
#    together (joint encoding), which is far more accurate than the independent
#    encodings used in BM25 and dense retrieval — but too slow to run over the
#    full corpus, so it only reruns the top candidates from the hybrid step.
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
    Retrieve and rerank: hybrid RRF fusion → CrossEncoder reranking → top-k.

    Input
    ─────
      All inputs from retrieve_hybrid() plus:
      reranker : CrossEncoder("BAAI/bge-reranker-v2-m3")
                 ← indexes["reranker"] passed through from streamlit_app.py
                 A CrossEncoder jointly encodes (query, document) pairs and
                 produces a relevance score that is much more accurate than
                 cosine similarity or BM25 — but requires one forward pass
                 per candidate, making it too slow to run over the full corpus.

    Output: list of top_k result dicts re-ordered by CrossEncoder relevance score.
            → passed to generate_answer() in generate.py as retrieved_chunks
            → this is the only method used by streamlit_app.py (method="hybrid_reranker")

    Two-stage approach:
      Stage 1 (hybrid): fetch top_k * 3 candidates efficiently (BM25 + FAISS + RRF).
                        These may include some noise — hybrid is accurate but not perfect.
      Stage 2 (rerank): run the CrossEncoder over the candidates to produce a precise
                        relevance score for each (query, candidate_text) pair.
                        The CrossEncoder sees both the query and the document at once,
                        allowing it to model fine-grained semantic interactions that
                        bi-encoder cosine similarity misses.

    Why retrieval_text for reranking:
      For image chunks, retrieval_text is the GPT-4o description (not raw pixels).
      The CrossEncoder can score text, not images, so the description is the right
      thing to pair with the query. For text/table chunks, retrieval_text is the
      same as raw_text.
    """
    # Stage 1: get top_k * 3 hybrid candidates to feed into the reranker.
    # More candidates = higher recall going into stage 2, at the cost of more
    # reranker forward passes. top_k * 3 is a practical balance.
    candidates = retrieve_hybrid(
        query,
        bm25,
        faiss_index,
        meta,
        embedder,
        chunk_id_to_meta,
        image_b64_lookup,
        top_k=top_k * 3,
    )

    # Guard: if hybrid returned nothing (e.g. empty index), return immediately.
    # The reranker would fail on an empty list.
    if not candidates:
        return []

    # Build (query, document_text) pairs — the CrossEncoder requires both strings
    # as a tuple to do joint encoding. We always pair against retrieval_text because
    # that is what was indexed (and what the model was trained to score against queries).
    pairs = [(query, c["retrieval_text"]) for c in candidates]

    # reranker.predict() runs one CrossEncoder forward pass per pair and returns
    # a list of raw relevance logits (unbounded floats; higher = more relevant).
    # This is the most accurate — but slowest — step in the retrieval pipeline.
    scores = reranker.predict(pairs)

    # Overwrite each candidate's score with the CrossEncoder's relevance logit.
    # The original RRF score is no longer meaningful after this pass.
    for candidate, score in zip(candidates, scores):
        candidate["score"] = round(float(score), 4)

    # Sort candidates by CrossEncoder score descending — highest relevance first
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Re-assign 1-based ranks to reflect the new CrossEncoder ordering.
    # Only update the top_k that will be returned — the rest are discarded.
    for rank, candidate in enumerate(candidates[:top_k], start=1):
        candidate["rank"] = rank

    # → final top_k result dicts passed back to retrieve(), then to generate_answer()
    return candidates[:top_k]


# ─────────────────────────────────────────
# 6. Unified Entry Point
#    streamlit_app.py always calls this single function.
#    It unpacks the indexes dict and dispatches to the correct method.
#    All retrieval complexity is hidden behind this interface.
# ─────────────────────────────────────────


def retrieve(query: str, method: str, indexes: dict, top_k: int = 5) -> list:
    """
    Single entry point called by streamlit_app.py for every user query.

    Input
    ─────
      query   : the user's plain-text question string
                ← comes from the chat input widget in streamlit_app.py
      method  : which retrieval algorithm to run
                ← hardcoded to "hybrid_reranker" in streamlit_app.py
                   but any of the four values below are valid
      indexes : the complete index bundle built by ingest_pdfs() in Ingestion.py
                ← stored in st.session_state["indexes"] by streamlit_app.py
      top_k   : number of results to return
                ← set by the sidebar slider in streamlit_app.py

    Output: list of result dicts from whichever retrieval method was chosen.
            → passed directly to generate_answer() in generate.py

    Unpacking the indexes dict here — rather than in each retrieval function —
    keeps the individual retrieval functions testable in isolation (they accept
    plain arguments, not a nested dict).
    """
    # Unpack the indexes dict into named variables for clarity.
    # All of these were built by ingest_pdfs() in Ingestion.py and stored in session_state.
    bm25 = indexes["bm25"]  # ← Ingestion.py → build_bm25()
    faiss_index = indexes["faiss"]  # ← Ingestion.py → build_faiss()
    meta = indexes["meta"]  # ← Ingestion.py → lightweight chunk metadata list
    chunk_id_to_meta = indexes[
        "chunk_id_to_meta"
    ]  # ← Ingestion.py → O(1) chunk lookup dict
    image_b64_lookup = indexes[
        "image_b64_lookup"
    ]  # ← Ingestion.py → chunk_id → base64 for images
    embedder = indexes[
        "embedder"
    ]  # ← streamlit_app.py → load_models() → passed through ingest
    reranker = indexes[
        "reranker"
    ]  # ← streamlit_app.py → load_models() → passed through ingest

    # Dispatch to the correct retrieval function based on the method string.
    # "hybrid_reranker" is the default used by streamlit_app.py — it is the most accurate.
    if method == "bm25":
        return retrieve_bm25(query, bm25, meta, image_b64_lookup, top_k)
    elif method == "dense":
        return retrieve_dense(
            query, faiss_index, meta, embedder, image_b64_lookup, top_k
        )
    elif method == "hybrid":
        return retrieve_hybrid(
            query,
            bm25,
            faiss_index,
            meta,
            embedder,
            chunk_id_to_meta,
            image_b64_lookup,
            top_k,
        )
    elif method == "hybrid_reranker":
        return retrieve_hybrid_reranked(
            query,
            bm25,
            faiss_index,
            meta,
            embedder,
            reranker,
            chunk_id_to_meta,
            image_b64_lookup,
            top_k,
        )
    else:
        # Fail loudly on an unrecognised method so bugs surface immediately
        raise ValueError(
            f"Unknown method '{method}'. Choose from: bm25, dense, hybrid, hybrid_reranker"
        )
