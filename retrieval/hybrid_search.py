"""
======================================================================
HYBRID SEARCH MODULE (retrieval/hybrid_search.py)
======================================================================
Purpose:
  Hybrid Search combine karta hai 2 search techniques ko:
  1. Dense Semantic Search (FAISS): Context and meaning match karta hai (embeddings se).
  2. Keyword Search (BM25Okapi): Exact word matches, acronyms, and rare terms match karta hai.

Why Hybrid?
  - Embeddings kabhi-kabhi specific keywords/numbers skip kar sakti hain.
  - BM25 exact keywords dhoondta hai par context nahi samajhta.
  - Dono ko RRF (Reciprocal Rank Fusion) se blend karke best recall and precision milta hai.
======================================================================
"""

from rank_bm25 import BM25Okapi

from retrieval.vector_store import VectorStore
from config import settings


class HybridSearcher:
    """
    Hybrid Search Engine combining FAISS VectorStore and BM25Okapi.
    """

    def __init__(self, vector_store: VectorStore):
        """
        Existing VectorStore instance attach karta hai.
        """
        self.vector_store = vector_store
        self.bm25: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []

    # ------------------------------------------------------------
    def build_bm25_index(self):
        """
        VectorStore ke `chunk_metadata` se hi BM25 index build karta hai.
        Ensure karta hai ki FAISS and BM25 dono exact SAME chunks list par operate karein.
        """
        # 1. Saare chunks ka text extract karo
        texts = [meta["text"] for meta in self.vector_store.chunk_metadata]

        # 2. Text ko lower-case karke words mein tokenize karo
        self._tokenized_corpus = [text.lower().split() for text in texts]

        # 3. BM25 Index initialize karo (BM25Okapi algorithm)
        self.bm25 = BM25Okapi(self._tokenized_corpus)
        print(f"BM25 index built on {len(texts)} chunks successfully!")

    # ------------------------------------------------------------
    def search(self, query: str, top_k: int = None) -> list[dict]:
        """
        Semantic Search + BM25 Search results ko merge karta hai using RRF (Reciprocal Rank Fusion).

        Args:
            query (str): Search query string.
            top_k (int): Number of top combined results to return.

        Returns:
            list[dict]: Rank-fused chunk metadata with 'fused_score'.
        """
        if self.bm25 is None:
            raise ValueError("BM25 Index empty: Pehle build_bm25_index() call karo")

        top_k = top_k or settings.top_k_retrieval

        # STEP 1: FAISS Semantic Results retrieve karo
        semantic_results = self.vector_store.search(query, top_k=top_k)

        # STEP 2: BM25 Keyword Search execute karo
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Top-k indices array descending order mein sort karo
        bm25_top_indices = bm25_scores.argsort()[::-1][:top_k]

        # STEP 3: Reciprocal Rank Fusion (RRF) Algorithm
        # RRF Score Formula: RRF_Score = 1 / (60 + Rank)
        rrf_k = 60  # Smoothing constant to prevent low ranks from dominating
        fused_scores: dict[int, float] = {}

        # 3a. Semantic Search ranks accumulate karo
        for rank, result in enumerate(semantic_results):
            # Text matching se chunk index identify karo
            idx = self.vector_store.chunk_metadata.index(
                next(m for m in self.vector_store.chunk_metadata if m["text"] == result["text"])
            )
            fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (rrf_k + rank + 1)

        # 3b. BM25 Keyword Search ranks accumulate karo
        for rank, idx in enumerate(bm25_top_indices):
            idx = int(idx)
            fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (rrf_k + rank + 1)

        # STEP 4: Fused RRF scores ke according top_k chunks select karo
        sorted_indices = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]

        # STEP 5: Final merged results return dictionary
        return [
            {**self.vector_store.chunk_metadata[idx], "fused_score": fused_scores[idx]}
            for idx in sorted_indices
        ]