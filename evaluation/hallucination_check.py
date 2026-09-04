"""
======================================================================
HALLUCINATION CHECKER MODULE (evaluation/hallucination_check.py)
======================================================================
Purpose:
  Real-Time Lightweight Hallucination Detection Engine:
  - RAGAS Faithfulness evaluation LLM-based hoti hai (accurate but slow & costly).
  - Yeh module Embedding-Similarity based hai — fast, local, free, and executes in milliseconds 
    per query inside the FastAPI endpoint.

How it Works:
  1. Refusal Check: Agar LLM ne "I don't have enough information..." kaha, toh 
     hallucination risk = False (Model refused correctly).
  2. Sentence Splitting: Answer text ko individual sentences mein divide karta hai.
  3. Sentence & Context Embedding: Answer sentences aur retrieved context chunks ko embed karta hai.
  4. Max Cosine Similarity Match: Har answer sentence ki max similarity retrieved contexts se measure karta hai.
  5. Unsupported Threshold Check: Agar sentence ki max similarity < threshold (e.g. 0.4) hai, 
     toh us sentence ko "unsupported/hallucinated" mark karta hai.
  6. Verdict: Agar > 30% sentences unsupported hain, overall answer ko hallucinated flag karta hai.
======================================================================
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from config import settings


class HallucinationChecker:
    """
    Embedding-similarity based real-time hallucination detector.
    """

    def __init__(self, model_name: str = None):
        """
        SentenceTransformer embedding model initialize karta hai (default: BAAI/bge-small-en-v1.5).
        """
        self.model = SentenceTransformer(model_name or settings.embedding_model)

    # ------------------------------------------------------------
    def _split_sentences(self, text: str) -> list[str]:
        """
        Text block ko regex sentence boundaries (. ! ?) par split karta hai.

        Args:
            text (str): Full answer text string.

        Returns:
            list[str]: Clean list of non-empty sentence strings.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    # ------------------------------------------------------------
    def check(self, answer: str, contexts: list[str], threshold: float = 0.4) -> dict:
        """
        Answer ke har sentence ko retrieved contexts ke saath compare karke support score compute karta hai.

        Args:
            answer (str): LLM-generated answer text.
            contexts (list[str]): List of retrieved context text chunks.
            threshold (float): Minimum cosine similarity score (0.0 to 1.0) to consider a sentence supported.

        Returns:
            dict: Evaluation results containing 'is_likely_hallucinated', 'overall_support_score', 
                  'sentence_details', and 'note'.
        """
        # STEP 1: Refusal Answer Detection
        # Agar LLM ne explicitly information na hone ki baat ki hai, toh hallucination check bypass karo
        if "don't have enough information" in answer.lower():
            return {
                "is_likely_hallucinated": False,
                "overall_support_score": 1.0,
                "sentence_details": [],
                "note": "Model refused to answer — no hallucination risk.",
            }

        # STEP 2: Sentence Tokenization & Empty Validation
        sentences = self._split_sentences(answer)
        if not sentences or not contexts:
            return {
                "is_likely_hallucinated": True,
                "overall_support_score": 0.0,
                "sentence_details": [],
                "note": "Empty answer or no contexts available to verify against.",
            }

        # STEP 3: Compute Embeddings for Sentences and Context Chunks (Batch Processing)
        sentence_embeddings = self.model.encode(sentences)
        context_embeddings = self.model.encode(contexts)

        sentence_details = []
        unsupported_count = 0

        # STEP 4: Sentence-to-Context Cosine Similarity Matching
        for sentence, sent_emb in zip(sentences, sentence_embeddings):
            # Calculate Cosine Similarity: (sent_emb . ctx_emb) / (||sent_emb|| * ||ctx_emb||)
            similarities = [
                np.dot(sent_emb, ctx_emb) / (np.linalg.norm(sent_emb) * np.linalg.norm(ctx_emb) + 1e-8)
                for ctx_emb in context_embeddings
            ]
            max_similarity = float(max(similarities))
            is_supported = max_similarity >= threshold

            if not is_supported:
                unsupported_count += 1

            sentence_details.append({
                "sentence": sentence,
                "max_similarity": round(max_similarity, 3),
                "is_supported": is_supported,
            })

        # STEP 5: Calculate Overall Support Score & Hallucination Flag
        overall_score = 1 - (unsupported_count / len(sentences))
        # Flag as hallucinated if > 30% of sentences are unsupported
        is_likely_hallucinated = (unsupported_count / len(sentences)) > 0.3

        return {
            "is_likely_hallucinated": is_likely_hallucinated,
            "overall_support_score": round(overall_score, 3),
            "sentence_details": sentence_details,
            "note": f"{unsupported_count}/{len(sentences)} sentences below similarity threshold ({threshold}).",
        }