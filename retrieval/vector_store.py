"""
FAISS-based vector store.

Yeh module 3 kaam karta hai:
1. Chunks ko embed karke FAISS index banana
2. Index ko disk pe save karna (taaki restart pe dobara embed na karna pade)
3. Query aake usse similar chunks dhoondhna
"""

import pickle 
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings

class VectorStore:
    """
    FAISS-based Vector Store Engine.
    
    Yeh class 3 main responsibilities nibhati hai:
    1. Text Chunks ko dense vector embeddings mein encode karna aur FAISS Index mein add karna.
    2. Built index aur associate metadata ko disk pe save/load karna.
    3. User Search Queries ke basis pe Cosine Similarity Compute karke top matching chunks return karna.
    """

    def __init__(self, embedding_model: str = None):
        """
        Embedding model load karta hai (default: BAAI/bge-small-en-v1.5).
        SentenceTransformer model query aur chunks dono ko identical vector space mein map karta hai.
        """
        self.model = SentenceTransformer(embedding_model or settings.embedding_model)
        self.index: faiss.Index | None = None
        self.chunk_metadata: list[dict] = []  # Chunks text + source/page metadata mapping

    def build_index(self, chunks: list[str], metadata: list[dict]):
        """
        Chunks and metadata list se FAISS vector index tayyar karta hai.
        
        Args:
            chunks (list[str]): Text chunks ki list.
            metadata (list[dict]): Har chunk ki extra info (e.g. source file name, page number).
        """
        assert len(chunks) == len(metadata), "Chunks aur metadata ki length same honi chahiye"

        print(f"Generating embeddings for {len(chunks)} chunks...")

        # 1. Chunks ko Embeddings mein convert karo (Batch encoding)
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        # 2. Embeddings ko Normalize karo (Cosine Similarity calculation ke liye zaroori hai)
        faiss.normalize_L2(embeddings)

        # 3. Vector Embedding dimension measure karo (e.g., 384 for bge-small)
        dimension = embeddings.shape[1]

        # 4. FAISS Index (Inner Product / Cosine Distance) initialize & populate karo
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        # 5. Metadata list mein actual text combine karke memory mein track karo
        self.chunk_metadata = [
            {**meta, "text": chunk} for chunk, meta in zip(chunks, metadata)
        ] 
        print(f"FAISS Index successfully built with {self.index.ntotal} vectors!")

    def add_chunks(self, chunks: list[str], metadata: list[dict]):
        """
        New text chunks aur metadata ko existing FAISS index mein dynamically add karta hai.
        
        Args:
            chunks (list[str]): New text chunks list.
            metadata (list[dict]): Extra info dictionaries for new chunks.
        """
        assert len(chunks) == len(metadata), "Chunks aur metadata ki length same honi chahiye"
        if not chunks:
            return

        print(f"Adding {len(chunks)} new chunks to FAISS index...")

        # 1. Embeddings generate karo
        embeddings = self.model.encode(chunks)
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)

        # 2. If index doesn't exist yet, build new; otherwise add to existing index
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        # 3. Metadata append karo
        new_chunk_metadata = [{**meta, "text": chunk} for chunk, meta in zip(chunks, metadata)]
        self.chunk_metadata.extend(new_chunk_metadata)
        print(f"Successfully added {len(chunks)} chunks! Total vectors in index: {self.index.ntotal}")

    def search(self, query: str, top_k: int = None) -> list[dict]:
        """
        Query text receive karke FAISS Index se Nearest Top-K Matches find karta hai.

        Args:
            query (str): User natural language search question/query.
            top_k (int): Number of top matching context chunks to return.

        Returns:
            list[dict]: List of matched metadata dictionary with similarity scores.
        """
        if self.index is None:
            raise ValueError("Index abhi build nahi hua — pehle build_index() ya load() call karo")

        top_k = top_k or settings.top_k_retrieval

        # 1. Query ko vector mein convert aur normalize karo
        query_vec = self.model.encode([query]).astype("float32")
        faiss.normalize_L2(query_vec)

        # 2. FAISS similarity search (Returns Scores & Matching Vector Indices)
        scores, indices = self.index.search(query_vec, top_k)

        # 3. Matches format karke return list generate karo
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS -1 return karta hai jab adequate matches na hon
                continue
            result = {**self.chunk_metadata[idx], "score": float(score)}
            results.append(result)

        return results

    def save(self, path: str = None):
        """FAISS binary index and metadata pickle file ko disk par save karta hai."""
        path = Path(path or settings.faiss_index_path)
        path.mkdir(parents=True, exist_ok=True)

        # FAISS Index write to file
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Metadata dictionary serialized to pickle file
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(self.chunk_metadata, f)
        print(f"Index & metadata saved successfully to directory: {path}")

    def load(self, path: str = None):
        """Disk se saved index and metadata load karta hai (saves re-computation time)."""
        path = Path(path or settings.faiss_index_path)

        # Read binary FAISS index
        self.index = faiss.read_index(str(path / "index.faiss"))
        
        # Unpickle metadata
        with open(path / "metadata.pkl", "rb") as f:
            self.chunk_metadata = pickle.load(f)
        print(f"Vector Store loaded: {self.index.ntotal} vectors available in index.")