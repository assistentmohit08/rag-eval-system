"""
======================================================================
EVALUATION DATASET MODULE (evaluation/eval_dataset.py)
======================================================================
Purpose:
  Ground Truth Evaluation Test Benchmark Dataset.
  - RAGAS (Retrieval-Augmented Generation Assessment) evaluation framework 
    requires a set of reference questions and ground truth answers.
  - The framework compares RAG generated answers against these ground truth references 
    to evaluate metrics like Faithfulness, Response Relevancy, and Context Recall.

Dataset Guidelines:
  1. Factual Questions: Simple direct questions from document content.
  2. Multi-Part Questions: Complex queries covering multiple parts of the document.
  3. Out-Of-Scope Questions: Questions NOT answered in the document (used for hallucination detection).
======================================================================
"""

# Ground Truth Dataset List for RAGAS Evaluation
EVAL_QUESTIONS = [
    {
        "question": "What primary skills and qualifications are listed in the document?",
        "reference": "The document lists technical skills, project experience, and educational background.",
    },
    {
        "question": "What candidate background or project experience is highlighted?",
        "reference": "The candidate has experience working on software engineering and AI projects.",
    },
    {
        "question": "What is the capital city of France?",
        "reference": "I don't have enough information in the provided documents to answer this.",
    },
]