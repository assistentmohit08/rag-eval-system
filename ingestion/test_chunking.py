"""
======================================================================
TEST CHUNKING SCRIPT (ingestion/test_chunking.py)
======================================================================
Purpose:
  Is script ka maqsad teeno Chunking Strategies ko testing/comparison karna hai:
  1. Fixed-Size Chunking   (Fixed character length + overlap)
  2. Recursive Chunking    (Smart splitting using natural boundaries like \n\n, \n, .)
  3. Semantic Chunking     (Embedding-based similarity drop detection)

Usage:
  Run from project root:
    python -m ingestion.test_chunking
    OR
    python ingestion/test_chunking.py
======================================================================
"""

import sys
from pathlib import Path

# --------------------------------------------------------------------
# STEP 1: PATH RESOLUTION (System Path Setup)
# --------------------------------------------------------------------
# Agar script directly ingestion/ folder se execute hoti hai,
# toh Python root folder (rag-eval-system) ko sys.path mein nahi dhoond patar.
# Isliye path resolve karke root directory ko sys.path mein add kar rahe hain.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Custom Modules Import
from ingestion.pdf_loader import load_all_pdfs
from ingestion.chunking import (
    fixed_size_chunk,
    recursive_chunk,
    semantic_chunk,
)

# --------------------------------------------------------------------
# STEP 2: LOAD PDF DATA
# --------------------------------------------------------------------
# Project ke data/pdfs directory se saari PDF files load karenge.
pdf_dir = project_root / "data" / "pdfs"
print(f"Loading PDFs from directory: {pdf_dir}")

pages = load_all_pdfs(str(pdf_dir))
print(f"Total pages loaded successfully: {len(pages)}")

# --------------------------------------------------------------------
# STEP 3: RUN CHUNKING STRATEGIES COMPARISON
# --------------------------------------------------------------------
if pages:
    # Testing ke liye pehle page ka text sample select karte hain
    sample_text = pages[0]["text"]
    print("\n" + "=" * 50)
    print("CHUNKING RESULTS FOR FIRST PAGE SAMPLE TEXT:")
    print("=" * 50)

    # Strategy 1: Fixed Size
    fixed_result = fixed_size_chunk(sample_text)
    print(f"1. Fixed-Size Chunking   : {len(fixed_result)} chunks created")

    # Strategy 2: Recursive Splitter
    recursive_result = recursive_chunk(sample_text)
    print(f"2. Recursive Chunking    : {len(recursive_result)} chunks created")

    # Strategy 3: Semantic Embeddings Splitter
    semantic_result = semantic_chunk(sample_text)
    print(f"3. Semantic Chunking     : {len(semantic_result)} chunks created")
    print("=" * 50)
else:
    print("\n[WARNING]: No pages loaded. Please check if PDF files exist inside data/pdfs/")