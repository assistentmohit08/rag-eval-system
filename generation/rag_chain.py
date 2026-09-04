"""
======================================================================
RAG CHAIN MODULE (generation/rag_chain.py)
======================================================================
Purpose:
  Full End-to-End RAG Generation Pipeline:
  1. Retrieve: Hybrid Search (FAISS + BM25) se initial context chunks fetch karna.
  2. Rerank: Cross-Encoder model se top relevant context chunks filter/reorder karna.
  3. Format Context: Chunks ko source tags ([Source N: file, page]) ke saath prompt string mein join karna.
  4. Generate: Groq LLM (llama-3.3-70b-versatile) ka use karke strict factual answer generate karna.

Why Groq LLM?
  - Ultra-fast inference speed (LPU hardware acceleration).
  - Free API tier for rapid prototyping and production evaluation.
======================================================================
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieval.hybrid_search import HybridSearcher
from retrieval.reranker import Reranker
from config import settings


# ====================================================================
# PROMPT TEMPLATE (Grounding & Anti-Hallucination Rules)
# ====================================================================
# Yeh Prompt Template LLM ko strict instructions deta hai ki sirf 
# provided Context documents ke basis par answer kare (Zero Hallucination).
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based ONLY on the provided context.

Rules:
1. Answer using ONLY information present in the context below.
2. If the context does not contain enough information to answer, say
   "I don't have enough information in the provided documents to answer this."
   Do NOT make up information.
3. Be concise and direct.
4. When possible, mention which part of the context supports your answer.

Context:
{context}

Question: {question}

Answer:
""")


class RAGChain:
    """
    RAG Chain combining Hybrid Retrieval, Cross-Encoder Reranking, and Groq LLM Generation.
    """

    def __init__(self, searcher: HybridSearcher, reranker: Reranker):
        """
        Initializes searcher, reranker, and LangChain LCEL chain with ChatGroq.

        Args:
            searcher (HybridSearcher): Hybrid searcher instance (FAISS + BM25).
            reranker (Reranker): Cross-Encoder Reranker instance.
        """
        self.searcher = searcher
        self.reranker = reranker

        # 1. ChatGroq LLM Client Setup
        self.llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=0,  # 0 = Deterministic factual output (Ideal for RAG QA)
        )

        # 2. LangChain Expression Language (LCEL) Chain
        # Pipeline: Prompt Template -> Groq LLM -> String Output Parser
        self.chain = RAG_PROMPT | self.llm | StrOutputParser()

    # ------------------------------------------------------------
    def _format_context(self, chunks: list[dict]) -> str:
        """
        Retrieved context chunks ko single structured string mein convert karta hai 
        with explicit file name and page number references.

        Args:
            chunks (list[dict]): List of reranked top context chunks.

        Returns:
            str: Formatted context block with source labels.
        """
        formatted = []
        for i, chunk in enumerate(chunks, start=1):
            formatted.append(
                f"[Source {i}: {chunk['source_file']}, Page {chunk['page_number']}]\n{chunk['text']}"
            )
        return "\n\n".join(formatted)

    # ------------------------------------------------------------
    def answer(self, question: str, top_k_retrieve: int = None, top_k_final: int = None) -> dict:
        """
        Executes complete RAG Pipeline: Retrieve -> Rerank -> Format -> LLM Generate.

        Args:
            question (str): User natural language question.
            top_k_retrieve (int): Number of initial chunks to retrieve from Hybrid Search.
            top_k_final (int): Final top reranked chunks to feed into LLM prompt.

        Returns:
            dict: Dictionary containing 'question', 'answer', and 'sources' list.
        """
        # STEP 1: Hybrid Retrieval (FAISS Vector + BM25 Keyword)
        candidates = self.searcher.search(
            question, 
            top_k=top_k_retrieve or settings.top_k_retrieval
        )

        # STEP 2: Cross-Encoder Reranking
        top_chunks = self.reranker.rerank(
            question, 
            candidates, 
            top_k=top_k_final or settings.top_k_rerank
        )

        # STEP 3: Format Context for LLM Prompt
        context = self._format_context(top_chunks)

        # STEP 4: Invoke LangChain Groq LLM Pipeline
        answer_text = self.chain.invoke({"context": context, "question": question})

        # STEP 5: Return Answer along with Source Citation Metadata
        return {
            "question": question,
            "answer": answer_text,
            "sources": [
                {
                    "source_file": c["source_file"],
                    "page_number": c["page_number"],
                    "rerank_score": c["rerank_score"],
                    "text": c["text"],
                    "text_snippet": c["text"][:200],
                }
                for c in top_chunks
            ],
        }