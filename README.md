<h1 align="center">Evaluating Retrieval Strategies in a Multimodal RAG Pipeline</h1>
<h3 align="center">Bachelor's Thesis Research Synopsis</h3>

<p align="center">
<b>Amitoj Singh Chawla</b> | 01013211922 <br>
B.Tech Artificial Intelligence & Data Science <br>
Guru Tegh Bahadur Institute of Technology, GGSIPU <br><br>
<b>Supervisor:</b> Ms. Upasana Singh <br>
<b>Expected Completion:</b> June 2026
</p>

---

<h2> Research Objective</h2>

<p>
This thesis presents a systematic empirical evaluation of four retrieval strategies within a production-grade <b>Multimodal Retrieval-Augmented Generation (RAG)</b> system for academic research paper analysis.
</p>

<p>
The system processes <b>text, tables, and images</b> from PDF papers and benchmarks retrieval performance across keyword-based, semantic, hybrid, and reranked configurations using both classical metrics and the <b>RAGAS</b> framework.
</p>

---

<h2> Research Questions</h2>

<ul>
<li><b>RQ1:</b> Impact of retrieval strategies (BM25, FAISS dense, RRF hybrid, reranker) on generation quality</li>
<li><b>RQ2:</b> Does multimodal indexing improve retrieval coverage and faithfulness?</li>
<li><b>RQ3:</b> Can a single-call multimodal LLM (GPT-4o-mini) handle full context synthesis?</li>
</ul>

---

<h2>System Architecture</h2>

<h3>Data Ingestion Pipeline</h3>

<ul>
<li><b>Document Partitioning:</b> hi_res layout detection, title-based chunking</li>
<li><b>Table Processing:</b> HTML → pipe-delimited text (BeautifulSoup)</li>
<li><b>Image Processing:</b> Base64 extraction + GPT-4o-generated descriptions</li>
</ul>

<h3>Indexing</h3>

<ul>
<li><b>BM25:</b> Tokenized sparse retrieval (stemming + filtering)</li>
<li><b>FAISS:</b> all-MiniLM-L6-v2 embeddings with cosine similarity</li>
</ul>

<h3>Retrieval Methods</h3>

<ol>
<li><b>BM25:</b> Exact keyword matching</li>
<li><b>Dense Retrieval:</b> Semantic similarity (FAISS)</li>
<li><b>Hybrid (RRF):</b> Score fusion → <code>1 / (k + rank)</code></li>
<li><b>Hybrid + Reranker:</b> Cross-encoder refinement</li>
</ol>

---

<h2>Answer Generation</h2>

<ul>
<li>Single-call multimodal inference using <b>GPT-4o-mini</b></li>
<li>Unified context: text + tables + images</li>
<li>Strict grounding: <i>“Not found in context”</i> fallback</li>
</ul>

---

<h2>Evaluation</h2>

<h3>Dataset</h3>

<p>
32 curated questions across 5 major deep learning papers (RAG, TCN, MobileNetV2, ResNet, U-Net), testing numerical, visual, and cross-modal reasoning.
</p>

<h3>Metrics</h3>

<ul>
<li><b>Exact Match (EM)</b></li>
<li><b>Token F1</b></li>
<li><b>ROUGE-L</b></li>
<li><b>Answer Correctness</b></li>
<li><b>Faithfulness</b></li>
<li><b>Context Precision / Recall</b></li>
</ul>

---

<h2> Preliminary Findings</h2>

<ul>
<li><b>BM25 dominates:</b> Faithfulness = 0.85, Context Precision = 0.94</li>
<li>Keyword retrieval still wins for technical domains</li>
<li>Cross-encoder underperforms due to <b>domain mismatch</b></li>
</ul>

---

<h2> Tech Stack</h2>

<ul>
<li><b>Core:</b> Python, FAISS, BM25, SentenceTransformers</li>
<li><b>Frameworks:</b> LangChain, OpenAI API, Gemini Flash</li>
<li><b>Processing:</b> Unstructured, PyMuPDF, BeautifulSoup</li>
<li><b>Evaluation:</b> RAGAS + custom metrics</li>
<li><b>Deployment:</b> Streamlit + FastAPI</li>
</ul>

---

<h2> Key Contributions</h2>

<ul>
<li>Empirical comparison of 4 retrieval strategies</li>
<li>Production-grade multimodal RAG system</li>
<li>Single-call multimodal inference design</li>
<li>Strong evidence supporting BM25 in academic QA</li>
</ul>

---

<h2>📬 Contact</h2>

<p>
📧 amitoj1503@gmail.com <br>
</p>

---
