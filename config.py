"""
config.py
---------
Centralized configuration for the multimodal RAG system.

Import this in all modules to ensure consistency:
    from config import EMBED_MODEL, RERANK_MODEL, LLM_MODEL
"""

# ═══════════════════════════════════════════════════════════════
# File Paths
# ═══════════════════════════════════════════════════════════════

CHUNKS_FILE = "chunks.json"
BM25_FILE   = "bm25_index.pkl"
FAISS_FILE  = "faiss_index.bin"
META_FILE   = "index_meta.json"

QUESTIONS_FILE        = "ground_truth.json"
RETRIEVAL_RESULTS_FILE = "retrieval_results.json"
GENERATION_RESULTS_FILE = "generation_results.json"


# ═══════════════════════════════════════════════════════════════
# Model Configurations
# ═══════════════════════════════════════════════════════════════

# Embedding Model (for dense retrieval)
# Options: all-MiniLM-L6-v2 (fast), all-mpnet-base-v2 (better quality)
EMBED_MODEL = "all-MiniLM-L6-v2"

# Reranker Model (for hybrid_reranker method)
# Options: ms-marco-MiniLM-L-6-v2 (fast), ms-marco-MiniLM-L-12-v2 (better)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# LLM Model (for generation)
# Options: gpt-4o-mini (fast/cheap), gpt-4o (better quality)
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0  # 0 = deterministic, higher = more creative


# ═══════════════════════════════════════════════════════════════
# Retrieval Configuration
# ═══════════════════════════════════════════════════════════════

# Default retrieval parameters
DEFAULT_TOP_K = 5
DEFAULT_METHOD = "hybrid_reranker"
RRF_K = 60  # Reciprocal Rank Fusion parameter

# Available methods
RETRIEVAL_METHODS = ["bm25", "dense", "hybrid", "hybrid_reranker"]


# ═══════════════════════════════════════════════════════════════
# Evaluation Configuration
# ═══════════════════════════════════════════════════════════════

# Metrics to compute
RETRIEVAL_METRICS = ["recall", "precision", "mrr", "ndcg"]
GENERATION_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# Evaluation parameters
EVAL_TOP_K = 5


# ═══════════════════════════════════════════════════════════════
# System Configuration
# ═══════════════════════════════════════════════════════════════

# Logging
VERBOSE = True
SHOW_PROGRESS = True

# Error handling
CONTINUE_ON_ERROR = True  # Continue evaluation even if some questions fail
MAX_RETRIES = 3  # For LLM API calls
RETRY_DELAY = 2  # Seconds between retries