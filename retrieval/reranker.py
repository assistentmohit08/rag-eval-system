"""
======================================================================
RERANKER MODULE (retrieval/reranker.py)
======================================================================
Purpose:
  Cross-Encoder Reranking Engine:
  - Model used: `cross-encoder/ms-marco-MiniLM-L-6-v2` (Lightweight ~90MB)
  - Alternate model option: `BAAI/bge-reranker-base` (Heavy ~1.11GB)

Why Cross-Encoder?
  - Bi-Encoder (Embedding models) query aur document ko ALAG-ALAG vector space mein embed karte hain.
  - Cross-Encoder query aur text chunk ko TOGETHER pair-wise process karta hai.
  - Cross-attention mechanism query key words ko document key words se directly evaluate karta hai, 
    producing significantly higher relevance precision.

======================================================================
"""

from sentence_transformers import CrossEncoder

from config import settings


class Reranker:
    """
    Cross-Encoder Reranker using lightweight MS-MARCO MiniLM model (~90MB).
    """

    def __init__(self, model_name: str = None):
        """
        Loads CrossEncoder model (default: cross-encoder/ms-marco-MiniLM-L-6-v2).
        Lightweight model downloads fast (~90MB) and provides high ranking accuracy.
        """
        model_path = model_name or settings.reranker_model
        print(f"Loading CrossEncoder Reranker model: {model_path}...")
        self.model = CrossEncoder(model_path)

    def rerank(self, query: str, candidates: list[dict], top_k: int = None) -> list[dict]:
        """
        Initial retrieved candidates list ko query ke relative score ke base pe rerank karta hai.

        Args:
            query (str): Original user search query string.
            candidates (list[dict]): Candidates retrieved from Hybrid Search / VectorStore (must contain 'text' key).
            top_k (int): Number of top ranked candidates to return.

        Returns:
            list[dict]: Reranked candidates with added 'rerank_score', sorted from highest to lowest score.
        """
        if not candidates:
            return []

        top_k = top_k or settings.top_k_rerank

        # 1. Query-Document Pairs list generate karo: [[query, text1], [query, text2], ...]
        pairs = [[query, c["text"]] for c in candidates]

        # 2. Cross-Encoder model se Joint Attention scores predict/compute karo
        scores = self.model.predict(pairs)

        # Handle edge case: Agar single pair input ho, predict float return kar sakta hai list/array nahi
        if isinstance(scores, (float, int)):
            scores = [scores]

        # 3. Candidate metadata dictionary mein 'rerank_score' inject karo
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        # 4. Candidates ko rerank_score ke basis par Descending order (best first) mein sort karo
        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]