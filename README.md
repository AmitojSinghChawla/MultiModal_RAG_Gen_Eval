# Multimodal RAG Pipeline

A Retrieval-Augmented Generation system built for a bachelor's thesis. Ingests academic PDFs containing text, tables, and images, builds hybrid search indexes, and evaluates four retrieval methods against a 50-question ground truth dataset.

---

## Overview

The pipeline extracts and indexes three content modalities from PDFs:

- **Text** — cleaned prose chunks via `unstructured` (`by_title` chunking strategy)
- **Tables** — HTML tables converted to pipe-delimited plain text
- **Images** — raw base64 images described by GPT-4o vision and indexed as text

At query time, one of four retrieval methods finds the most relevant chunks. GPT-4o Mini generates the final answer using retrieved context, with full multimodal support (text, table content, and raw images are all passed to the LLM).

---

## Architecture

```
PDFs
 │
 ▼
01chunk_exporter.py       partition_pdf (hi_res) → text / table / image chunks
 │                        GPT-4o vision → image descriptions
 │                        → chunks.json
 ▼
02_build_index.py         BM25Okapi     → bm25_index.pkl
 │                        BAAI/bge-base → faiss_index.bin
 │                        metadata      → index_meta.json
 ▼
03_evaluate.py            retrieve() → generate() → string metrics + RAGAS
 │                        → evaluation_results.json
 ▼
04_chatbot.py             CLI chatbot (all 4 methods, configurable top-k)
streamlit.py              Web UI (PDF upload, in-memory ingestion, chat)
```

---

## Project Structure

```
MultiModal_RAG_Gen_Eval/
│
├── codebase/
│   ├── 01chunk_exporter.py     # PDF partitioning and chunk export
│   ├── 02_build_index.py       # BM25 + FAISS index construction
│   ├── 03_evaluate.py          # Evaluation runner (all 4 methods, resume support)
│   ├── 04_chatbot.py           # CLI chatbot
│   ├── retrieve.py             # All 4 retrieval methods
│   └── utils.py                # Shared BM25 tokenizer
│
├── App/
│   ├── streamlit.py            # Streamlit web UI
│   ├── Ingestion.py            # In-memory ingestion for the web app
│   ├── retrieve.py             # Retrieval logic for the web app
│   └── generate.py             # Answer generation with sliding chat history
│
├── documents/                  # Source PDFs (11 academic papers)
├── Indexes/
│   ├── bm25_index.pkl
│   ├── faiss_index.bin
│   └── index_meta.json
├── Questions/
│   └── gold_questions.json     # 50-question ground truth dataset
└── Result/
    └── evaluation_results.json # Per-question + summary evaluation results
```

---

## Setup

**Requirements:** Python 3.10+

```bash
pip install unstructured[pdf] faiss-cpu rank-bm25 sentence-transformers \
            langchain langchain-openai openai ragas datasets rouge-score \
            beautifulsoup4 nltk python-dotenv streamlit
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_key_here
```

> **Model usage note:** GPT-4o is used at ingestion time for image descriptions. GPT-4o Mini is used for answer generation and RAGAS evaluation scoring.

---

## Usage

### 1. Export Chunks

Partitions all PDFs in `documents/` using `unstructured` (hi_res strategy) and writes `chunks.json`:

```bash
python codebase/01chunk_exporter.py
```

### 2. Build Indexes

Reads `chunks.json` and builds a BM25 sparse index and a FAISS dense index:

```bash
python codebase/02_build_index.py
```

### 3. Run Evaluation

Evaluates all four retrieval methods against `gold_questions.json` (50 questions). Supports resume — completed questions are skipped automatically:

```bash
python codebase/03_evaluate.py
```

### 4. CLI Chatbot

```bash
python codebase/04_chatbot.py
python codebase/04_chatbot.py --method hybrid --top_k 3
```

Available methods: `bm25` · `dense` · `hybrid` · `hybrid_reranker`

### 5. Streamlit Web App

Upload PDFs directly in the browser. Ingestion runs in-memory (no pre-built index required):

```bash
streamlit run App/streamlit.py
```

---

## Retrieval Methods

All four methods share a unified `retrieve(query, method, indexes, top_k)` entry point in `retrieve.py`.

| Method | Description |
|---|---|
| `bm25` | BM25Okapi sparse keyword search. Tokenised with Porter stemming and stopword removal. Best for exact terms, model names, figure numbers. |
| `dense` | FAISS IndexFlatIP cosine search using `BAAI/bge-base-en-v1.5` (768-dim). Best for paraphrases and semantic queries. |
| `hybrid` | Runs BM25 and dense independently, then merges ranked lists using Reciprocal Rank Fusion (RRF, k=60). |
| `hybrid_reranker` | Hybrid RRF followed by CrossEncoder reranking (`BAAI/bge-reranker-v2-m3`). Joint query-document encoding for the most accurate final ranking. |

For `hybrid` and `hybrid_reranker`, each retriever fetches `top_k × 3` candidates before fusion to ensure meaningful overlap between the two ranked lists.

---

## Evaluation Results

Evaluated over 50 questions across 11 source PDFs with Top-K = 5.

| Method | Exact Match | Token F1 | ROUGE-L | Ans. Correctness | Faithfulness | Ctx. Precision | Ctx. Recall |
|---|---|---|---|---|---|---|---|
| BM25 | 0.040 | 0.459 | 0.461 | 0.729 | 0.776 | 0.844 | 0.870 |
| Dense | 0.020 | 0.363 | 0.357 | 0.601 | 0.606 | 0.813 | 0.890 |
| Hybrid | 0.040 | 0.428 | 0.425 | 0.662 | 0.689 | 0.840 | 0.860 |
| **Hybrid + Reranker** | **0.040** | **0.472** | **0.468** | **0.761** | **0.770** | **0.901** | **0.960** |

> String metrics (exact match, token F1, ROUGE-L) are low across all methods because GPT-4o Mini rephrases ground truth answers rather than reproducing them verbatim. RAGAS metrics (answer correctness, faithfulness, context precision/recall) reflect semantic quality more reliably for generative RAG systems.

---

## Models

| Role | Model |
|---|---|
| PDF image description | `gpt-4o` |
| Answer generation | `gpt-4o-mini` |
| Dense embeddings | `BAAI/bge-base-en-v1.5` |
| CrossEncoder reranker | `BAAI/bge-reranker-v2-m3` |
| RAGAS evaluation scoring | `gpt-4o-mini` + `text-embedding-ada-002` |

---

## Ingestion Pipeline (Web App)

The `App/Ingestion.py` module handles in-memory ingestion for the Streamlit app without requiring pre-built index files. Per uploaded PDF:

1. Writes the uploaded bytes to a temporary file on disk (required by `unstructured`)
2. Runs `partition_pdf` with `strategy="hi_res"` and `chunking_strategy="by_title"`
3. Separates elements into text, table, and image buckets
4. Converts table HTML to pipe-delimited text via `html_table_to_text()`
5. Deduplicates images by base64 fingerprint; skips images smaller than ~1 KB
6. Calls GPT-4o vision to produce a retrieval-optimised text description for each image
7. Builds BM25 and FAISS indexes over all chunks from all uploaded files
8. Returns a single `indexes` dict stored in `st.session_state`

The temporary file is always deleted in a `finally` block. If one PDF fails to partition, it is skipped and ingestion continues with the remaining files.

---

## Generation

`App/generate.py` constructs the OpenAI Chat API message list for each query:

- **System message** — instructs the model to answer strictly from provided context
- **Sliding history window** — last 3 conversation turns (6 messages) for follow-up resolution
- **Context blocks** — retrieved chunks formatted as `[TEXT]`, `[TABLE]`, or inline `image_url` blocks; images are sent as base64 data URIs with `detail="high"`
- **Answer model** — `gpt-4o-mini` at `temperature=0.1`

---

## Source PDFs

11 academic papers covering: RAG survey, BLIP-2, DPR, BM25/BERT re-ranking, RAGAS, Reciprocal Rank Fusion, ResNet, ROUGE, TCN, MobileNetV2, and related topics.

---

*Bachelor's Thesis — Computer Science (AI & Data Science) · Amitoj Singh Chawla*
