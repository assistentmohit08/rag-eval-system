"""
======================================================================
TEST RAG CHAIN SCRIPT (generation/test_ragchain.py)
======================================================================
Purpose:
  Is script ka maqsad complete End-to-End RAG System test karna hai:
  1. VectorStore load karna.
  2. BM25 Keyword Search index initialize karna.
  3. Cross-Encoder Reranker load karna (~90MB model).
  4. RAGChain instance create karna.
  5. Sample question ask karke Groq LLM generated answer + Source Citations view karna.

Usage:
  Run from project root:
    python -m generation.test_ragchain
    OR
    python generation/test_ragchain.py
======================================================================
"""

import sys
from pathlib import Path

# --------------------------------------------------------------------
# STEP 1: PATH RESOLUTION & MODULE IMPORTS
# --------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ingestion.pdf_loader import load_all_pdfs
from ingestion.chunking import recursive_chunk
from retrieval.vector_store import VectorStore
from retrieval.hybrid_search import HybridSearcher
from retrieval.reranker import Reranker
from generation.rag_chain import RAGChain

# --------------------------------------------------------------------
# STEP 2: LOAD OR BUILD VECTOR STORE
# --------------------------------------------------------------------
store = VectorStore()
index_file = project_root / "data" / "faiss_index" / "index.faiss"

if index_file.exists():
    print(f"Loading FAISS Vector Index from: {index_file.parent}")
    store.load()
else:
    print("FAISS index not found. Building index from PDFs in data/pdfs...")
    pdf_dir = project_root / "data" / "pdfs"
    pages = load_all_pdfs(str(pdf_dir))
    if not pages:
        print("[ERROR]: No PDFs found in data/pdfs. Please add PDF documents.")
        sys.exit(1)

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

# --------------------------------------------------------------------
# STEP 3: INITIALIZE HYBRID SEARCHER & RERANKER
# --------------------------------------------------------------------
searcher = HybridSearcher(store)
searcher.build_bm25_index()

reranker = Reranker()

# --------------------------------------------------------------------
# STEP 4: INITIALIZE RAG CHAIN PIPELINE
# --------------------------------------------------------------------
rag = RAGChain(searcher, reranker)

# --------------------------------------------------------------------
# STEP 5: TEST QUESTION & GENERATE ANSWER
# --------------------------------------------------------------------
test_question = "What is the experience and skills mentioned in the document?"
print(f"\nAsking Question: '{test_question}'\n")

result = rag.answer(test_question)

# --------------------------------------------------------------------
# STEP 6: PRINT ANSWER & CITATIONS
# --------------------------------------------------------------------
print("=" * 65)
print("GROQ LLM GENERATED ANSWER:")
print("=" * 65)
print(result["answer"])

print("\n" + "=" * 65)
print("SUPPORTING SOURCE CITATIONS:")
print("=" * 65)
for s in result["sources"]:
    print(f"  - {s['source_file']} (Page {s['page_number']}, Rerank Score: {s['rerank_score']:.3f})")
print("=" * 65)