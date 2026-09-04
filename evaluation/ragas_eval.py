"""
======================================================================
RAGAS EVALUATION MODULE (evaluation/ragas_eval.py)
======================================================================
Purpose:
  RAGAS Assessment & Metric Evaluation Pipeline:
  1. Executes test benchmark questions from `eval_dataset.py` through `RAGChain`.
  2. Collects system answers, ground truth references, and retrieved context chunks.
  3. Evaluates 3 Core RAGAS Metrics:
     a. Faithfulness: Is answer strictly derived from context? (Prevents Hallucinations)
     b. Response Relevancy: How directly does the answer address the question?
     c. Context Recall: Were all ground-truth details present in the retrieved chunks?
  4. Saves detailed per-question metrics to `evaluation/eval_results.csv`.

Usage:
  Run from project root:
    python -m evaluation.ragas_eval
    OR
    python evaluation/ragas_eval.py
======================================================================
"""

import sys
import types
from pathlib import Path

# --------------------------------------------------------------------
# STEP 1: PATH RESOLUTION & COMPATIBILITY SHIMS
# --------------------------------------------------------------------
# Project root setup so ingestion, retrieval, generation imports resolve
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Compatibility shim for RAGAS + langchain-community:
# Prevents ModuleNotFoundError for legacy vertexai module in newer langchain-community versions
if "langchain_community.chat_models.vertexai" not in sys.modules:
    dummy_mod = types.ModuleType("langchain_community.chat_models.vertexai")
    dummy_mod.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = dummy_mod

import pandas as pd
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
try:
    from ragas.metrics.collections import Faithfulness, ResponseRelevancy, LLMContextRecall
except ImportError:
    from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextRecall

from langchain_groq import ChatGroq
from generation.rag_chain import RAGChain
from evaluation.eval_dataset import EVAL_QUESTIONS
from config import settings


# ====================================================================
# LANGCHAIN EMBEDDINGS WRAPPER FOR RAGAS
# ====================================================================
class SentenceTransformerEmbeddings(Embeddings):
    """
    LangChain-compatible wrapper class around `SentenceTransformer` 
    required by RAGAS embedding-based metric evaluation.
    """

    def __init__(self, model_name: str = None):
        """Initializes SentenceTransformer embedding model."""
        self.model = SentenceTransformer(model_name or settings.embedding_model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encodes multiple text chunks into floating point embedding vectors."""
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Encodes single query string into embedding vector."""
        return self.model.encode([text])[0].tolist()


# ====================================================================
# STEP A: COLLECT RAG OUTPUTS FOR EVALUATION
# ====================================================================
def collect_rag_outputs(rag_chain: RAGChain) -> list[SingleTurnSample]:
    """
    Executes each evaluation test question through `RAGChain` and 
    wraps the output into RAGAS `SingleTurnSample` data format.

    Args:
        rag_chain (RAGChain): RAG Pipeline instance.

    Returns:
        list[SingleTurnSample]: List of benchmark samples for RAGAS.
    """
    samples = []

    for item in EVAL_QUESTIONS:
        question = item["question"]
        reference = item["reference"]

        # Execute full RAG pipeline (Retrieve -> Rerank -> LLM Answer)
        result = rag_chain.answer(question)

        # Extract full context text from sources list safely
        retrieved_texts = [
            source.get("text", source.get("text_snippet", "")) 
            for source in result["sources"]
        ]

        # Package into RAGAS SingleTurnSample structure
        samples.append(
            SingleTurnSample(
                user_input=question,
                response=result["answer"],
                retrieved_contexts=retrieved_texts,
                reference=reference,
            )
        )
        print(f"Collected sample for question: '{question[:50]}...'")

    return samples


# ====================================================================
# STEP B: RUN RAGAS EVALUATION
# ====================================================================
def run_evaluation(rag_chain: RAGChain) -> pd.DataFrame:
    """
    Runs full RAGAS evaluation pipeline across Faithfulness, Relevancy, 
    and Context Recall metrics.

    Args:
        rag_chain (RAGChain): Active RAGChain instance.

    Returns:
        pd.DataFrame: Pandas DataFrame containing metric scores per sample.
    """
    # 1. Collect samples from RAGChain
    samples = collect_rag_outputs(rag_chain)
    dataset = EvaluationDataset(samples=samples)

    # 2. Setup Evaluator LLM and Embeddings Wrapper
    evaluator_llm = LangchainLLMWrapper(
        ChatGroq(api_key=settings.groq_api_key, model=settings.groq_model, temperature=0)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(SentenceTransformerEmbeddings())

    # 3. Initialize Target Metrics
    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        LLMContextRecall(llm=evaluator_llm),
    ]

    # 4. Execute RAGAS Evaluation
    print("\nRunning RAGAS evaluation metrics... (this may take a short moment)")
    result = evaluate(dataset=dataset, metrics=metrics)

    # 5. Convert results to Pandas DataFrame
    df = result.to_pandas()
    return df


# ====================================================================
# MAIN EXECUTION BLOCK
# ====================================================================
if __name__ == "__main__":
    # Resolve system path to project root
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from ingestion.pdf_loader import load_all_pdfs
    from ingestion.chunking import recursive_chunk
    from retrieval.vector_store import VectorStore
    from retrieval.hybrid_search import HybridSearcher
    from retrieval.reranker import Reranker

    # 1. Load or Build VectorStore
    store = VectorStore()
    index_file = project_root / "data" / "faiss_index" / "index.faiss"

    if index_file.exists():
        print(f"Loading FAISS Vector Index from: {index_file.parent}")
        store.load()
    else:
        print("Building index from PDFs in data/pdfs...")
        pdf_dir = project_root / "data" / "pdfs"
        pages = load_all_pdfs(str(pdf_dir))
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

    # 2. Setup Searcher, Reranker, and RAGChain
    searcher = HybridSearcher(store)
    searcher.build_bm25_index()

    reranker = Reranker()
    rag = RAGChain(searcher, reranker)

    # 3. Run Evaluation
    results_df = run_evaluation(rag)

    # 4. Print Summary Averages
    print("\n" + "=" * 55)
    print("RAGAS EVALUATION AVERAGE METRIC SCORES:")
    print("=" * 55)
    if "faithfulness" in results_df.columns:
        print(f"Faithfulness:      {results_df['faithfulness'].mean():.3f}")
    if "answer_relevancy" in results_df.columns:
        print(f"Answer Relevancy:  {results_df['answer_relevancy'].mean():.3f}")
    if "context_recall" in results_df.columns:
        print(f"Context Recall:    {results_df['context_recall'].mean():.3f}")
    print("=" * 55)

    # Save CSV Results File
    csv_output = project_root / "evaluation" / "eval_results.csv"
    results_df.to_csv(csv_output, index=False)
    print(f"\nDetailed evaluation results saved to: {csv_output}")