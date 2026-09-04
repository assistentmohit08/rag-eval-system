"""
Ab teen strategies likhte hain:

Strategy 1 — Fixed-Size Chunking (sabse simple)

Bas har N characters pe kaat do, thoda overlap rakh ke (taaki context na tuute).

Strategy 2 — Recursive Chunking (smart splitting)

Pehle paragraph breaks pe todne ki koshish karta hai, agar chunk phir bhi bada hai toh sentence pe, phir word pe — matlab natural boundaries respect karta hai.

Strategy 3 — Semantic Chunking (sabse advanced)

Sentences ke embeddings nikaal ke dekhta hai kahan "topic change" ho raha hai (consecutive sentences ke beech similarity drop) — wahi pe chunk todta hai. Isse related content ek saath rehta hai.
"""

# 
"""
Teen alag chunking strategies — Step 8 mein hum A/B test karenge
ki kaunsi RAGAS metrics pe best perform karti hai.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings

# =====================================================================
# STRATEGY 1 — FIXED-SIZE CHUNKING
# =====================================================================
def fixed_size_chunk(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """
    Fixed-Size Chunking (Sabse Simple Strategy):
    - Har chunk mein fixed number of characters (e.g., 500) hote hain.
    - Chunks ke beech mein character overlap (e.g., 50) rakhte hain taaki 
      boundary pe Context breaks na hon.

    Args:
        text (str): Input full text from document/page.
        chunk_size (int): Max character length per chunk (default from settings).
        overlap (int): Number of overlapping characters between adjacent chunks.

    Returns:
        list[str]: Filtered list of text chunk strings.
    """
    # 1. Config settings se default chunk size aur overlap specify karo
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    chunks = []
    start = 0

    # 2. Sliding window technique se string slicing karo
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        
        # Next chunk ki starting position setup karo (Overlap minus karke)
        start = end - overlap

    # 3. Empty ya whitespace-only chunks ko filter out karke return karo
    return [c.strip() for c in chunks if c.strip()]


# =====================================================================
# STRATEGY 2 — RECURSIVE CHUNKING
# =====================================================================
def recursive_chunk(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """
    Recursive Chunking (Smart Boundary Splitting):
    - Text ko natural breaks par hierarchical order mein split karta hai:
        1. Paragraph breaks ("\n\n")
        2. Line breaks ("\n")
        3. Sentences (". ")
        4. Words (" ")
        5. Characters ("") - Fallback
    - Isse semantic context preserve hota hai aur sentences beech mein nahi katt-te.

    Args:
        text (str): Input text string.
        chunk_size (int): Maximum target character limit per chunk.
        overlap (int): Overlap threshold (if applicable).

    Returns:
        list[str]: Natural boundary respected chunks.
    """
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    # Priority list of separators (largest boundary -> smallest boundary)
    separators = ["\n\n", "\n", ".", " ", ""]

    def split_text(text: str, seps: list[str]) -> list[str]:
        # Base Case: Agar text pehle se chunk_size se chhota hai, directly return karo
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        # Current highest-priority separator pick karo
        sep = seps[0]
        remaining_seps = seps[1:]
        parts = text.split(sep) if sep else [text]

        chunks, current = [], ""
        for part in parts:
            piece = part + sep
            # Check karo agar current chunk mein naya piece fit ho sakta hai
            if len(current) + len(piece) <= chunk_size:
                current += piece
            else:
                if current:
                    chunks.append(current)
                # Agar koi single piece (e.g. bada paragraph) chunk_size se bada hai,
                # toh baaki separators (remaining_seps) se us specific piece ko recursive break karo
                if len(piece) > chunk_size and remaining_seps:
                    chunks.extend(split_text(piece, remaining_seps))
                    current = ""
                else:
                    current = piece
        if current.strip():
            chunks.append(current)
        return chunks

    chunks = split_text(text, separators)
    return [c.strip() for c in chunks if c.strip()]


# =====================================================================
# STRATEGY 3 — SEMANTIC CHUNKING
# =====================================================================
_embedder = None  # Global lazy-loaded model instance


def _get_embedder():
    """Embedding model ko tabhi memory mein load karo jab Semantic Chunking execute ho."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


def semantic_chunk(text: str, similarity_threshold: float = 0.5) -> list[str]:
    """
    Semantic Chunking (Advanced Topic-Change Detection):
    - Har sentence ko vector embedding mein convert karta hai.
    - Adjacent sentences ke beech Cosine Similarity calculate karta hai.
    - Jab similarity score threshold se niche girta hai (meaning topic change ho gaya),
      tab naya chunk start karta hai.

    Args:
        text (str): Full text block to chunk.
        similarity_threshold (float): Minimum cosine similarity to keep sentences in same chunk.

    Returns:
        list[str]: Topic-aligned semantic chunks.
    """
    # 1. Text ko sentences mein split karo
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    if len(sentences) <= 1:
        return sentences

    # 2. Sentences ke Embeddings calculate karo
    embedder = _get_embedder()
    embeddings = embedder.encode(sentences)

    chunks, current_chunk = [], [sentences[0]]

    # 3. Consecutive sentences ki Cosine Similarity compute karo
    for i in range(1, len(sentences)):
        # Cosine Similarity Formula: (A . B) / (||A|| * ||B||)
        sim = np.dot(embeddings[i - 1], embeddings[i]) / (
            np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(embeddings[i]) + 1e-8
        )

        # Drop check: Similarity < threshold means Topic Shift
        if sim < similarity_threshold:
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")

    return chunks