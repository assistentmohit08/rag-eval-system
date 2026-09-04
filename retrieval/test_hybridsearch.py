"""
======================================================================
TEST HYBRID SEARCH & RERANKER SCRIPT (retrieval/test_hybridsearch.py)
======================================================================
Purpose:
  Is script ka maqsad complete Retrieval Pipeline test karna hai:
  1. Saved FAISS Vector Store load karna.
  2. BM25 Index build karna on existing chunks.
  3. Hybrid Search (FAISS + BM25) run karke top 10 candidates retrieve karna (using RRF).
  4. Cross-Encoder Reranker run karke candidates ko rerank karke Top 4 final results nikalna.

Usage:
  Run from project root:
    python -m retrieval.test_hybridsearch
    OR
    python retrieval/test_hybridsearch.py
======================================================================
"""

import sys
from pathlib import Path

# --------------------------------------------------------------------
# STEP 1: PATH RESOLUTION & MODULE IMPORTS
# --------------------------------------------------------------------
# Project root ko sys.path mein add karo taaki module imports resolve ho sakein
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ingestion.pdf_loader import load_all_pdfs
from ingestion.chunking import recursive_chunk
from retrieval.vector_store import VectorStore
from retrieval.hybrid_search import HybridSearcher
from retrieval.reranker import Reranker

# --------------------------------------------------------------------
# STEP 2: LOAD VECTOR STORE (LOAD OR BUILD IF NOT EXISTS)
# --------------------------------------------------------------------
store = VectorStore()
index_file = project_root / "data" / "faiss_index" / "index.faiss"

if index_file.exists():
    print(f"Loading existing FAISS index from: {index_file.parent}")
    store.load()
else:
    print("FAISS index not found on disk. Building index from PDFs in data/pdfs...")
    pdf_dir = project_root / "data" / "pdfs"
    pages = load_all_pdfs(str(pdf_dir))
    if not pages:
        print("[ERROR]: No PDFs found in data/pdfs. Cannot build index.")
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
# STEP 3: INITIALIZE HYBRID SEARCHER & BUILD BM25 INDEX
# --------------------------------------------------------------------
# Vector Store ke metadata se BM25 Index build karte hain
searcher = HybridSearcher(store)  # 'store' variable from Step 2
searcher.build_bm25_index()

# --------------------------------------------------------------------
# STEP 4: EXECUTE HYBRID SEARCH
# --------------------------------------------------------------------
query = "resume experience skills"
print(f"\n1. Executing Hybrid Search for Query: '{query}'...")

# RRF (Reciprocal Rank Fusion) se FAISS + BM25 scores combine honge
hybrid_results = searcher.search(query, top_k=10)
print(f"Retrieved {len(hybrid_results)} candidates from Hybrid Search.")

# --------------------------------------------------------------------
# STEP 5: RERANK CANDIDATES USING CROSS-ENCODER MODEL
# --------------------------------------------------------------------
print("\n2. Reranking candidates using Cross-Encoder Model...")
reranker = Reranker()

# Initial 10 candidates ko rerank karke final top 4 select karte hain
final_results = reranker.rerank(query, hybrid_results, top_k=4)

# --------------------------------------------------------------------
# STEP 6: DISPLAY FINAL RERANKED RESULTS
# --------------------------------------------------------------------
print("\n" + "=" * 65)
print("FINAL RERANKED RETRIEVAL RESULTS:")
print("=" * 65)
for i, r in enumerate(final_results, start=1):
    clean_text = r['text'].replace('\n', ' ')
    print(f"Rank #{i} | Rerank Score: {r['rerank_score']:.4f} | Page {r['page_number']} | Source: {r['source_file']}")
    print(f"Snippet: {clean_text[:120]}...\n")
print("=" * 65)