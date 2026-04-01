import json
import pickle
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from utils import tokenize  # shared tokenizer — MUST match retrieve.py

CHUNKS_FILE = "chunks.json"
BM25_FILE = "bm25_index.pkl"
FAISS_FILE = "faiss_index.bin"
META_FILE = "index_meta.json"

# Small, fast, good enough for a bachelor's thesis
EMBED_MODEL = "BAAI/bge-base-en-v1.5"


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
    corpus = [tokenize(chunk["retrieval_text"]) for chunk in chunks]
    bm25 = BM25Okapi(corpus)
    print(f"BM25 index built over {len(corpus)} documents")
    return bm25


# ===============================
# Build FAISS index
# ===============================


def build_faiss(chunks: list[dict], model: SentenceTransformer) -> faiss.IndexFlatIP:
    texts = [chunk["retrieval_text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # model here is the sentence transformer model, which is used to encode the retrieval_text of each chunk into a vector embedding. The resulting embeddings are stored in a numpy array.

    # Normalise so dot product = cosine similarity
    faiss.normalize_L2(embeddings)
    # normalise means that we are scaling the embeddings to have a length of 1. This is done so that when we compute the dot product between two embeddings, it will be equivalent to computing the cosine similarity between them.

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    # index flat IP (inner product) is a simple FAISS index that computes the inner product between query and indexed vectors. Since we normalised the embeddings, this inner product will effectively give us the cosine similarity.
    index.add(embeddings)

    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def build_meta(chunks: list[dict]) -> list[dict]:
    meta = []
    for chunk in chunks:
        meta.append(
            {
                "chunk_id": chunk["chunk_id"],
                "modality": chunk["modality"],
                "source_pdf": chunk["source_pdf"],
                "page_number": chunk["page_number"],
                "retrieval_text": chunk["retrieval_text"],
                "raw_text": chunk.get("raw_text"),
            }
        )
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
    model = SentenceTransformer(EMBED_MODEL)
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
