# 🚀 Production-Grade RAG System with Hybrid Search, Reranking & RAGAS Evaluation

A production-ready **Retrieval-Augmented Generation (RAG) & Evaluation Framework** built with Python, LangChain, FAISS, BM25, SentenceTransformers, Groq LLM, FastAPI, and Streamlit.

This system addresses core enterprise RAG challenges—retrieval quality, context relevance, and LLM hallucinations—by integrating **Hybrid Search (BM25 + Dense Vectors)**, **Cross-Encoder Reranking**, **Real-Time Embedding-based Hallucination Detection**, and empirical **RAGAS Benchmark A/B Testing**.

---

## 🌟 Key Features & Architecture

```
                               ┌───────────────────────────┐
                               │     PDF Documents (Data)  │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Document Loader & Chunking │
                               │ (Fixed / Recursive/Sem.)  │
                               └─────────────┬─────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
         ┌───────────────────────────┐               ┌───────────────────────────┐
         │   BM25 Keyword Search     │               │    FAISS Vector Store     │
         │     (Lexical Index)       │               │ (BAAI/bge-small-en-v1.5)  │
         └─────────────┬─────────────┘               └─────────────┬─────────────┘
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  Reciprocal Rank Fusion   │
                               │      (Hybrid Search)      │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │   Cross-Encoder Reranker  │
                               │ (ms-marco-MiniLM-L-6-v2)  │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │   Groq LLM Generation     │
                               │     (LangChain LCEL)      │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Real-Time Hallucination   │
                               │   Checker (Cos Sim Score) │
                               └─────────────┬─────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
         ┌───────────────────────────┐               ┌───────────────────────────┐
         │     FastAPI REST Server   │               │   Streamlit Web UI        │
         │    (Live Upload & Query)  │               │ (Metrics & Live Query)    │
         └───────────────────────────┘               └───────────────────────────┘
```

### 1. 📑 Document Ingestion & Advanced Chunking
- **PDF Extraction:** Extracts structured text and page metadata using `pdfplumber`.
- **Multiple Chunking Strategies:**
  - **Fixed-Size Chunking:** Fixed character length with overlap.
  - **Recursive Character Chunking:** Hierarchical splitting on natural paragraph/sentence boundaries (`\n\n`, `\n`, `.`).
  - **Semantic Chunking:** Sentence-embedding similarity splitting for contextual coherence.

### 2. 🔍 Hybrid Search & Reciprocal Rank Fusion (RRF)
- **Dense Vector Search:** Uses `FAISS` with normalized L2 cosine embeddings (`BAAI/bge-small-en-v1.5`).
- **Lexical Keyword Search:** Uses `BM25Okapi` for exact keyword match retrieval.
- **RRF Re-ranking:** Merges top results using Reciprocal Rank Fusion score: $RRF(d) = \sum \frac{1}{k + rank(d)}$.

### 3. 🎯 Cross-Encoder Reranking
- Re-ranks candidate retrieved chunks using `sentence-transformers/cross-encoder/ms-marco-MiniLM-L-6-v2` (~90MB lightweight model) to compute deep query-passage cross-attention relevancy.

### 4. ⚡ Groq LLM Generation Pipeline
- Uses **LangChain LCEL** connected to **Groq API** (`openai/gpt-oss-20b` / `llama-3.3-70b-versatile`) for sub-second, grounded generation with source citations.

### 5. 🛡️ Real-Time Embedding Hallucination Detection
- Millisecond-latency local sentence-to-context embedding verification (`HallucinationChecker`).
- Splits generated answers into sentences, calculates max cosine similarity against retrieved contexts, and flags answers if $>30\%$ sentences fall below similarity threshold ($0.4$).

### 6. 📊 RAGAS Benchmark & A/B Testing Framework
- Evaluates system performance across standard RAG metrics:
  - **Faithfulness:** Groundedness of generated answers in context.
  - **Response Relevancy:** Direct applicability of answer to user prompt.
  - **Context Recall:** Coverage of ground-truth reference context.
- **A/B Testing Module (`evaluation/ab_testing.py`):** Empirically compares chunking strategies side-by-side to select the winning configuration.

### 7. 🚀 FastAPI REST Server & Live PDF Upload
- **Lifespan Startup Pre-loading:** Pre-loads heavy embedding models, FAISS indices, and Rerankers into global state once at server launch.
- **`POST /upload-pdf` Endpoint:** Supports live PDF uploads, updating FAISS vector store and BM25 index on the fly.

### 8. 💻 Interactive Streamlit Dashboard
- **4 Dashboard Views:**
  1. **Evaluation Results:** RAGAS metrics summary cards, detailed per-question tables, and score bar charts.
  2. **A/B Test Comparison:** Comparative strategy charts and automated winner identification.
  3. **Live Query Test:** Real-time query testing with answer rendering, hallucination alerts, and expandable source citations.
  4. **Upload Document:** Drag-and-drop live PDF upload & instant indexing UI.

---

## 📁 Project Directory Structure

```
rag-eval-system/
├── api/
│   └── main.py                 # FastAPI REST server with Lifespan & endpoints
├── dashboard/
│   └── app.py                  # Streamlit 4-page interactive dashboard UI
├── evaluation/
│   ├── ab_testing.py           # A/B testing framework for chunking strategies
│   ├── eval_dataset.py         # Ground-truth evaluation dataset
│   ├── hallucination_check.py  # Real-time embedding cosine similarity checker
│   └── ragas_eval.py           # RAGAS evaluation runner with compatibility shims
├── generation/
│   ├── rag_chain.py            # RAG Chain pipeline (Search -> Rerank -> LCEL -> Groq)
│   └── test_ragchain.py        # Standalone test script for RAGChain
├── ingestion/
│   ├── chunking.py             # Fixed, Recursive, & Semantic chunkers
│   ├── pdf_loader.py           # PDF text & metadata extractor via pdfplumber
│   └── test_chunking.py        # Chunking test script
├── retrieval/
│   ├── hybrid_search.py        # BM25 + FAISS Hybrid Searcher & RRF fusion
│   ├── reranker.py             # SentenceTransformers CrossEncoder reranker
│   ├── vector_store.py         # FAISS vector store with dynamic add_chunks()
│   └── Test_vector_store.py    # Vector store test script
├── config.py                   # Centralized Pydantic configuration settings
├── requirements.txt            # Project Python dependencies
└── .env                        # Environment variables (Groq API key, models)
```

---

## ⚙️ Setup & Installation Guide

### Prerequisites
- Python 3.10+
- Groq API Key ([Get Groq Key](https://console.groq.com/))

### 1. Clone & Navigate to Repository
```powershell
git clone https://github.com/your-username/rag-eval-system.git
cd rag-eval-system
```

### 2. Create Virtual Environment & Install Dependencies
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment Variables Setup
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

## 🚀 Running the Project

### 1. Run Pipeline Unit Tests
```powershell
python ingestion/test_chunking.py
python retrieval/Test_vector_store.py
python retrieval/test_hybridsearch.py
python generation/test_ragchain.py
```

### 2. Run RAGAS Benchmark Evaluation
```powershell
python evaluation/ragas_eval.py
```

### 3. Run Chunking Strategy A/B Test
```powershell
python -m evaluation.ab_testing
```

### 4. Start FastAPI REST Backend
```powershell
uvicorn api.main:app --reload --port 8000
```
- Interactive Swagger API Docs: `http://127.0.0.1:8000/docs`

### 5. Start Streamlit Web Dashboard
```powershell
streamlit run dashboard/app.py
```
- Web UI: `http://localhost:8501`

---

## 🔗 API Documentation Summary

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Server health check and component initialization status. |
| `/query` | `POST` | Executes RAG pipeline, returns answer, sources, & hallucination check. |
| `/upload-pdf` | `POST` | Uploads PDF live, extracts text, generates chunks, and updates FAISS index. |

---

## 📜 License & Author

Developed by **Mohit** for Advanced RAG Portfolio & AI Engineering.  
Licensed under the MIT License.
#   r a g - e v a l - s y s t e m  
 