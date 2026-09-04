"""
Central configuration for the RAG Evaluation System.

Har module (ingestion, retrieval, generation, evaluation, api)
yahan se hi settings import karega — taaki agar kabhi model ya
path change karna ho, toh sirf ek jagah (.env file) update karni pade.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # -- LLM (Groq) --
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"

    # -- Embeddings & Reranking (dono local models, free) --
    embedding_model: str = "BAAI/bge-small-en-v1.5"        # ~133MB small embedding model
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # ~90MB lightweight fast reranker

    # -- Storage paths --
    faiss_index_path: str = "./data/faiss_index"
    pdf_data_path: str = "./data/pdfs"

    # -- Chunking defaults (baad mein A/B testing mein override honge) --
    chunk_size: int = 500
    chunk_overlap: int = 50

    # -- Retrieval --
    top_k_retrieval: int = 10   # initial retrieval count (before rerank)
    top_k_rerank: int = 4       # final count after reranking

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Poore project mein isi single instance ko import karna hai:
# from config import settings
settings = Settings()