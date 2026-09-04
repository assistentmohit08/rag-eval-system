"""
======================================================================
FASTAPI BACKEND SERVER (api/main.py)
======================================================================
Purpose:
  Production-grade REST API backend that wraps the RAG Evaluation System:
  - Startup Lifespan Event: Pre-loads heavy models (Embeddings, FAISS VectorStore, 
    BM25 Index, Cross-Encoder Reranker, Groq LLM RAGChain, HallucinationChecker) once 
    at server startup to ensure low latency per query request.
  - Endpoints:
      GET  /health : System status & component loading health check.
      POST /query  : Accepts user question, executes RAG Pipeline, runs 
                     real-time hallucination check, and returns answer + citations.

Usage:
  Run using Uvicorn server:
    uvicorn api.main:app --reload --port 8000
======================================================================
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

# --------------------------------------------------------------------
# STEP 1: PATH RESOLUTION & MODULE IMPORTS
# --------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, Field, ConfigDict

from ingestion.pdf_loader import load_pdf, load_all_pdfs
from ingestion.chunking import recursive_chunk
from retrieval.vector_store import VectorStore
from retrieval.hybrid_search import HybridSearcher
from retrieval.reranker import Reranker
from generation.rag_chain import RAGChain
from evaluation.hallucination_check import HallucinationChecker
from config import settings


# ====================================================================
# GLOBAL APPLICATION STATE
# ====================================================================
# Server startup pe models & indices yahan pre-load hoke share hote hain
app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Context Manager:
    Server start hone par heavy models/index ek baar load karta hai (Startup Event),
    aur server stop hone par resource cleanup manage karta hai (Shutdown Event).
    """
    print("\n[STARTUP]: Initializing RAG Evaluation System Components...")

    # 1. Load or Build VectorStore FAISS Index
    store = VectorStore()
    index_file = project_root / "data" / "faiss_index" / "index.faiss"

    if index_file.exists():
        print(f"[STARTUP]: Loading existing FAISS index from {index_file.parent}...")
        store.load()
    else:
        print("[STARTUP]: FAISS index not found. Building index from PDFs in data/pdfs...")
        pdf_dir = project_root / "data" / "pdfs"
        pages = load_all_pdfs(str(pdf_dir))
        all_chunks, all_metadata = [], []
        for page in pages:
            chunks = recursive_chunk(page["text"])
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    "source_file": page["source_file"],
                    "page_number": page["page_number"],
                    "chunking_strategy": "recursive",
                })
        store.build_index(all_chunks, all_metadata)
        store.save()

    # 2. Build BM25 Keyword Search Index
    searcher = HybridSearcher(store)
    searcher.build_bm25_index()

    # 3. Initialize Cross-Encoder Reranker, RAGChain, and HallucinationChecker
    reranker = Reranker()
    rag_chain = RAGChain(searcher, reranker)
    hallucination_checker = HallucinationChecker()

    # 4. Store initialized components in global app_state dictionary
    app_state["rag_chain"] = rag_chain
    app_state["hallucination_checker"] = hallucination_checker

    print("[STARTUP]: RAG Evaluation System is ready to handle requests!\n")
    yield  # Server active state (handling API requests)
    print("\n[SHUTDOWN]: Cleaning up resources...")


# Initialize FastAPI Instance
app = FastAPI(
    title="RAG Evaluation System API",
    description="Production-grade RAG with Hybrid Search, Cross-Encoder Reranking, Real-time Hallucination Check, and Live PDF Ingestion",
    version="1.0.0",
    lifespan=lifespan,
)


# ====================================================================
# REQUEST & RESPONSE PYDANTIC SCHEMAS
# ====================================================================
class QueryRequest(BaseModel):
    """User Question Input Payload Schema."""
    question: str = Field(..., min_length=1, description="Natural language question")
    top_k: int = Field(default=4, ge=1, le=10, description="Number of top reranked sources to return")


class SourceInfo(BaseModel):
    """Retrieved Source Document Citation Metadata Schema."""
    model_config = ConfigDict(extra="ignore")
    source_file: str
    page_number: int
    rerank_score: float
    text_snippet: str


class QueryResponse(BaseModel):
    """API Query Endpoint Output Payload Schema."""
    question: str
    answer: str
    sources: list[SourceInfo]
    hallucination_check: dict


# ====================================================================
# API ENDPOINTS
# ====================================================================
@app.get("/health")
def health_check():
    """Health check endpoint to verify server status and component initialization."""
    return {
        "status": "healthy",
        "rag_chain_loaded": "rag_chain" in app_state,
        "hallucination_checker_loaded": "hallucination_checker" in app_state,
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Main RAG Endpoint:
    1. Receives user query.
    2. Runs complete RAG Pipeline (Retrieve -> Rerank -> LLM Generation).
    3. Executes Real-Time Embedding Hallucination Check against retrieved contexts.
    4. Returns structured JSON payload with answer, source citations, and hallucination metrics.
    """
    rag_chain: RAGChain = app_state.get("rag_chain")
    checker: HallucinationChecker = app_state.get("hallucination_checker")

    if rag_chain is None or checker is None:
        raise HTTPException(status_code=503, detail="RAG system is not initialized yet")

    try:
        # STEP 1: Execute RAG Pipeline (Hybrid Search + Rerank + Groq LLM)
        result = rag_chain.answer(request.question, top_k_final=request.top_k)

        # STEP 2: Extract contexts & execute Real-time Hallucination Check
        contexts = [s.get("text", s.get("text_snippet", "")) for s in result["sources"]]
        hallucination_result = checker.check(result["answer"], contexts)

        # STEP 3: Construct and return structured Pydantic response
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources=[SourceInfo(**s) for s in result["sources"]],
            hallucination_check=hallucination_result,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Live PDF Document Ingestion Endpoint:
    1. Saves uploaded PDF into data/pdfs/ directory.
    2. Extracts text pages, chunks content, and adds vectors to FAISS index live.
    3. Re-builds BM25 index and updates saved index files on disk.
    """
    rag_chain: RAGChain = app_state.get("rag_chain")
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG system is not initialized yet")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files (.pdf) are supported")

    try:
        # 1. Save uploaded file to data/pdfs directory
        pdf_dir = project_root / "data" / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        file_path = pdf_dir / file.filename

        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        # 2. Extract pages from uploaded PDF
        pages = load_pdf(str(file_path))
        if not pages:
            raise HTTPException(status_code=400, detail="Could not extract text from uploaded PDF")

        # 3. Chunk extracted text
        new_chunks, new_metadata = [], []
        for page in pages:
            chunks = recursive_chunk(page["text"])
            for chunk in chunks:
                new_chunks.append(chunk)
                new_metadata.append({
                    "source_file": page["source_file"],
                    "page_number": page["page_number"],
                    "chunking_strategy": "recursive",
                })

        # 4. Update FAISS VectorStore dynamically & save index
        store = rag_chain.searcher.vector_store
        store.add_chunks(new_chunks, new_metadata)
        store.save()

        # 5. Re-build BM25 Keyword Search Index
        rag_chain.searcher.build_bm25_index()

        return {
            "status": "success",
            "message": f"PDF '{file.filename}' uploaded and indexed successfully!",
            "filename": file.filename,
            "total_pages": len(pages),
            "chunks_added": len(new_chunks),
            "total_vectors_in_index": store.index.ntotal,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading and indexing PDF: {str(e)}")