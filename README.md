# Multimodal RAG Pipeline

A multimodal Retrieval-Augmented Generation system built for a bachelor's thesis. Processes PDFs containing text, tables, and images, builds hybrid search indexes, and evaluates four retrieval methods against a 50-question ground truth dataset.

---

## Overview

The pipeline ingests academic PDFs, extracts and indexes three modalities (text chunks, tables, and images described via GPT-4o vision), and answers questions using GPT-4o Mini with retrieved context. Four retrieval methods are benchmarked: BM25, dense semantic search, hybrid RRF fusion, and hybrid with CrossEncoder reranking.

---

## Architecture

```
PDFs
 │
 ▼
01chunk_exporter.py          partition_pdf (hi_res) → text / table / image chunks
 │                           GPT-4o vision → image descriptions
 │                           → chunks.json
 ▼
02_build_index.py            BM25Okapi       → bm25_index.pkl
 │                           BAAI/bge-base   → faiss_index.bin
 │                           metadata        → index_meta.json
 ▼
03_evaluate.py               retrieve() → generate() → string metrics + RAGAS
 │                           → evaluation_results.json
 ▼
04_chatbot.py / streamlit.py interactive Q&A over indexed documents
```

---

## Project Structure

```
MultiModal_RAG_Gen_Eval/
│
├── codebase/
│   ├── 01chunk_exporter.py     # PDF partitioning and chunk export
│   ├── 02_build_index.py       # BM25 + FAISS index construction
│   ├── 03_evaluate.py          # Evaluation runner (all 4 methods)
│   ├── 04_chatbot.py           # CLI chatbot
│   ├── retrieve.py             # All 4 retrieval methods
│   └── utils.py                # Shared BM25 tokenizer
│
├── App/
│   ├── streamlit.py            # Streamlit web UI
│   ├── Ingestion.py            # In-memory ingestion for the web app
│   ├── retrieve.py             # Retrieval logic for the web app
│   └── generate.py             # Answer generation with chat history
│
├── documents/                  # Source PDFs (11 papers)
├── Chunks/
│   └── chunks.json             # Exported chunks (text + table + image)
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

> GPT-4o is used for image description at ingestion time. GPT-4o Mini is used for answer generation and RAGAS scoring.

---

## Usage

### 1. Export Chunks

Partitions all PDFs in `documents/` and writes `chunks.json`:

```bash
python codebase/01chunk_exporter.py
```

### 2. Build Indexes

Reads `chunks.json` and builds BM25 + FAISS indexes:

```bash
python codebase/02_build_index.py
```

### 3. Run Evaluation

Evaluates all 4 retrieval methods against `gold_questions.json`:

```bash
python codebase/03_evaluate.py
```

Supports resume — already-completed questions are skipped automatically.

### 4. CLI Chatbot

```bash
python codebase/04_chatbot.py
python codebase/04_chatbot.py --method hybrid --top_k 3
```

Available methods: `bm25`, `dense`, `hybrid`, `hybrid_reranker`

### 5. Streamlit Web App

Supports PDF upload, in-memory ingestion, and a chat interface:

```bash
streamlit run App/streamlit.py
```

---

## Evaluation Results

Evaluated over 50 questions across 11 source PDFs. Top-K = 5.

| Method            | Exact Match | Token F1 | ROUGE-L | Ans. Correctness | Faithfulness | Ctx. Precision | Ctx. Recall |
|-------------------|-------------|----------|---------|-----------------|--------------|----------------|-------------|
| BM25              | 0.040       | 0.459    | 0.461   | 0.729           | 0.776        | 0.844          | 0.870       |
| Dense             | 0.020       | 0.363    | 0.357   | 0.601           | 0.606        | 0.813          | 0.890       |
| Hybrid            | 0.040       | 0.428    | 0.425   | 0.662           | 0.689        | 0.840          | 0.860       |
| Hybrid + Reranker | **0.040**   | **0.472**| **0.468**| **0.761**      | **0.770**    | **0.901**      | **0.960**   |

> String metrics (exact match, token F1, ROUGE-L) are low across all methods due to answer verbosity — the LLM rephrases ground truth rather than reproducing it verbatim. RAGAS metrics reflect semantic correctness more accurately.

---

## Models

| Role | Model |
|------|-------|
| PDF image description | `gpt-4o` |
| Answer generation | `gpt-4o-mini` |
| Dense embeddings | `BAAI/bge-base-en-v1.5` |
| CrossEncoder reranker | `BAAI/bge-reranker-v2-m3` |
| RAGAS scoring | `gpt-4o-mini` + `text-embedding-ada-002` |

---

## Source PDFs

11 academic papers covering: RAG survey, TCN, BLIP-2, DPR, MobileNetV2, BERT re-ranking, RAGAS, RRF, ResNet, ROUGE, and related topics.

---

*Bachelor's Thesis — Computer Science (AI & Data Science)*
