"""
======================================================================
A/B TESTING EVALUATION MODULE (evaluation/ab_testing.py)
======================================================================
Purpose:
  A/B Testing Framework for Chunking Strategies:
  - Compares 3 chunking strategies side-by-side using empirical data:
      1. Fixed-Size Chunking
      2. Recursive Chunking
      3. Semantic Chunking
  - Process per Strategy:
      1. Generate chunks and metadata.
      2. Build isolated FAISS Index saved to `./data/faiss_index_{strategy_name}`.
      3. Build Hybrid Searcher & RAGChain.
      4. Execute RAGAS Evaluation benchmark.
      5. Compute average scores for Faithfulness, Response Relevancy, and Context Recall.
  - Generates comparative summary CSV report at `evaluation/ab_test_results.csv`.

Usage:
  Run from project root:
    python -m evaluation.ab_testing
    OR
    python evaluation/ab_testing.py
======================================================================
"""

import sys
import types
from pathlib import Path

# --------------------------------------------------------------------
# STEP 1: PATH RESOLUTION & COMPATIBILITY SHIM
# --------------------------------------------------------------------
# Add project root to sys.path so imports work regardless of execution folder
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Compatibility shim for RAGAS + langchain-community
if "langchain_community.chat_models.vertexai" not in sys.modules:
    dummy_mod = types.ModuleType("langchain_community.chat_models.vertexai")
    dummy_mod.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = dummy_mod

import pandas as pd

from ingestion.pdf_loader import load_all_pdfs
from ingestion.chunking import fixed_size_chunk, recursive_chunk, semantic_chunk
from retrieval.vector_store import VectorStore
from retrieval.hybrid_search import HybridSearcher
from retrieval.reranker import Reranker
from generation.rag_chain import RAGChain
from evaluation.ragas_eval import run_evaluation
from config import settings


# Mapping of strategy name to its respective chunking function
CHUNKING_STRATEGIES = {
    "fixed_size": fixed_size_chunk,
    "recursive": recursive_chunk,
    "semantic": semantic_chunk,
}


def build_chunks_for_strategy(pages: list[dict], strategy_name: str, chunk_fn) -> tuple[list[str], list[dict]]:
    """
    Applies selected chunking function to extracted PDF pages and attaches strategy metadata.

    Args:
        pages (list[dict]): List of extracted PDF page dictionaries.
        strategy_name (str): Name identifier of the chunking strategy.
        chunk_fn (function): Target chunking function to execute.

    Returns:
        tuple[list[str], list[dict]]: Generated text chunks list and metadata list.
    """
    all_chunks, all_metadata = [], []

    for page in pages:
        chunks = chunk_fn(page["text"])
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({
                "source_file": page["source_file"],
                "page_number": page["page_number"],
                "chunking_strategy": strategy_name,  # Tracks strategy lineage
            })

    return all_chunks, all_metadata


def run_ab_test() -> pd.DataFrame:
    """
    Runs end-to-end A/B test pipeline across all chunking strategies and returns comparative DataFrame.

    Returns:
        pd.DataFrame: Comparative evaluation metrics summary table.
    """
    # 1. Load PDF pages once for all strategy iterations
    pages = load_all_pdfs(settings.pdf_data_path)

    # 2. Instantiate Cross-Encoder Reranker once (shared across iterations)
    reranker = Reranker()

    comparison_rows = []

    # 3. Iterate through each chunking strategy
    for strategy_name, chunk_fn in CHUNKING_STRATEGIES.items():
        print(f"\n{'=' * 60}")
        print(f"RUNNING A/B TEST FOR STRATEGY: {strategy_name.upper()}")
        print(f"{'=' * 60}")

        # Step A: Generate Chunks & Metadata
        chunks, metadata = build_chunks_for_strategy(pages, strategy_name, chunk_fn)
        print(f"  Total chunks generated: {len(chunks)}")

        # Step B: Build & Save Strategy-Specific FAISS Index
        store = VectorStore()
        store.build_index(chunks, metadata)
        save_path = project_root / "data" / f"faiss_index_{strategy_name}"
        store.save(path=str(save_path))

        # Step C: Build Hybrid Searcher & RAGChain Instance
        searcher = HybridSearcher(store)
        searcher.build_bm25_index()
        rag_chain = RAGChain(searcher, reranker)

        # Step D: Run RAGAS Benchmark Evaluation
        results_df = run_evaluation(rag_chain)

        # Step E: Calculate Average Metric Scores
        comparison_rows.append({
            "strategy": strategy_name,
            "num_chunks": len(chunks),
            "avg_faithfulness": results_df["faithfulness"].mean() if "faithfulness" in results_df else 0.0,
            "avg_answer_relevancy": results_df["answer_relevancy"].mean() if "answer_relevancy" in results_df else 0.0,
            "avg_context_recall": results_df["context_recall"].mean() if "context_recall" in results_df else 0.0,
        })

    # 4. Generate & Save Comparison CSV
    comparison_df = pd.DataFrame(comparison_rows)
    csv_path = project_root / "evaluation" / "ab_test_results.csv"
    comparison_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 60)
    print("FINAL A/B TEST COMPARISON RESULTS:")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print(f"\nSummary results saved to: {csv_path}")

    return comparison_df


if __name__ == "__main__":
    run_ab_test()