"""
======================================================================
TEST VECTOR STORE SCRIPT (retrieval/Test_vector_store.py)
======================================================================
Purpose:
  Is script ka maqsad poora ingestion to vector index pipeline test karna hai:
  1. PDFs se text extract karna (PDF Loader)
  2. Text ko chunks aur metadata mein split karna (Chunking)
  3. Chunks ko Embeddings mein convert karke FAISS Index mein build karna (VectorStore)
  4. FAISS index ko disk pe save karna (Save Index)
  5. Search query chala kar similarity results calculate karna (Vector Search)

Usage:
  Run from project root:
    python -m retrieval.Test_vector_store
    OR
    python retrieval/Test_vector_store.py
======================================================================
"""

import sys
from pathlib import Path

# --------------------------------------------------------------------
# STEP 1: PATH RESOLUTION & MODULE IMPORTS
# --------------------------------------------------------------------
# Project root ko sys.path mein append kar rahe hain taaki imports properly resolve ho payein.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ingestion.pdf_loader import load_all_pdfs
from ingestion.chunking import recursive_chunk
from retrieval.vector_store import VectorStore

# --------------------------------------------------------------------
# STEP 2: LOAD PDFS & PREPARE CHUNKS WITH METADATA
# --------------------------------------------------------------------
# data/pdfs/ folder se saari PDF files load kar rahe hain
pdf_dir = project_root / "data" / "pdfs"
print(f"Loading PDFs from: {pdf_dir}")
pages = load_all_pdfs(str(pdf_dir))

if not pages:
    print("\n[WARNING]: No PDF pages loaded. Please check if PDF files exist in data/pdfs/")
else:
    print(f"Loaded {len(pages)} pages. Preparing text chunks and metadata...")
    all_chunks, all_metadata = [], []

    # Har page par Recursive Chunking apply karte hain aur chunk ke saath metadata link karte hain
    for page in pages:
        chunks = recursive_chunk(page["text"])
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({
                "source_file": page["source_file"],
                "page_number": page["page_number"],
                "chunking_strategy": "recursive",
            })

    print(f"Total chunks created: {len(all_chunks)}")

    # --------------------------------------------------------------------
    # STEP 3: BUILD & SAVE VECTOR INDEX IN FAISS
    # --------------------------------------------------------------------
    print("\nInitializing Vector Store & Building Embeddings...")
    store = VectorStore()
    
    # Sentence Transformer embedding model har chunk ko vector representation mein transform karega
    store.build_index(all_chunks, all_metadata)
    
    # Built index ko ./data/faiss_index path par persist (save) kar lenge
    store.save()

    # --------------------------------------------------------------------
    # STEP 4: EXECUTE TEST SEMANTIC SEARCH QUERY
    # --------------------------------------------------------------------
    test_query = "resume experience skills"
    print(f"\nRunning Similarity Search for Query: '{test_query}'")
    
    # Top 3 most relevant chunks retrieve karte hain based on Cosine Similarity
    results = store.search(test_query, top_k=3)

    print("\n" + "=" * 60)
    print("TOP SEARCH RESULTS FROM VECTOR STORE:")
    print("=" * 60)
    for i, r in enumerate(results, start=1):
        clean_text = r['text'].replace('\n', ' ')
        print(f"Result #{i} | Cosine Score: {r['score']:.4f} | Source: {r['source_file']} (Page {r['page_number']})")
        print(f"Content Snippet: {clean_text[:120]}...\n")
    print("=" * 60)