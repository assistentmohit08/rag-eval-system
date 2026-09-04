"""
======================================================================
STREAMLIT DASHBOARD (dashboard/app.py)
======================================================================
Purpose:
  Interactive Web UI for RAG System Monitoring & Evaluation:
  1. Evaluation Results Page: Displays RAGAS evaluation metrics summary cards, 
     per-question breakdown table, and score distribution bar charts.
  2. A/B Test Comparison Page: Displays comparative performance metrics across 
     Chunking Strategies (Fixed-Size, Recursive, Semantic) and highlights the winner.
  3. Live Query Test Page: Connects directly to the FastAPI Backend (`http://127.0.0.1:8000/query`) 
     to send questions, display generated answers, show real-time hallucination warnings, 
     and inspect retrieved source citations.

Usage:
  Run using Streamlit CLI:
    streamlit run dashboard/app.py
======================================================================
"""

import sys
import types
from pathlib import Path

# --------------------------------------------------------------------
# STEP 1: PATH RESOLUTION & COMPATIBILITY SHIMS
# --------------------------------------------------------------------
# Add project root to sys.path so files can resolve project paths
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Compatibility shim for Starlette middleware gzip in Streamlit:
# Prevents ImportError for DEFAULT_EXCLUDED_CONTENT_TYPES across starlette releases
import starlette.middleware.gzip
if not hasattr(starlette.middleware.gzip, "DEFAULT_EXCLUDED_CONTENT_TYPES"):
    starlette.middleware.gzip.DEFAULT_EXCLUDED_CONTENT_TYPES = (
        "text/html", "text/css", "text/plain", "application/javascript", "application/json"
    )

import pandas as pd
import requests
import streamlit as st

# Configure Page Title and Wide Layout
st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")
st.title("📊 RAG System — Evaluation & Monitoring Dashboard")

# FastAPI Backend REST Endpoint URL
API_URL = "http://127.0.0.1:8000"

# --------------------------------------------------------------------
# STEP 2: SIDEBAR NAVIGATION
# --------------------------------------------------------------------
page = st.sidebar.radio(
    "Select Navigation Page:", 
    ["Evaluation Results", "A/B Test Comparison", "Live Query Test", "Upload Document"]
)

# --------------------------------------------------------------------
# PAGE 1: RAGAS EVALUATION RESULTS
# --------------------------------------------------------------------
if page == "Evaluation Results":
    st.header("📈 RAGAS Benchmark Evaluation Results")
    eval_csv = project_root / "evaluation" / "eval_results.csv"

    try:
        df = pd.read_csv(eval_csv)

        # 1. Summary Metric Cards
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Faithfulness", f"{df['faithfulness'].mean():.3f}")
        col2.metric("Avg Answer Relevancy", f"{df['answer_relevancy'].mean():.3f}")
        col3.metric("Avg Context Recall", f"{df['context_recall'].mean():.3f}")

        # 2. Per-Question Detailed Table
        st.subheader("Per-Question Detailed Breakdown")
        st.dataframe(df, use_container_width=True)

        # 3. Metric Distribution Chart
        st.subheader("Metric Distribution Chart")
        st.bar_chart(df[["faithfulness", "answer_relevancy", "context_recall"]])

    except FileNotFoundError:
        st.warning("No evaluation results found. Please run `python evaluation/ragas_eval.py` first.")

# --------------------------------------------------------------------
# PAGE 2: A/B TEST CHUNKING STRATEGY COMPARISON
# --------------------------------------------------------------------
elif page == "A/B Test Comparison":
    st.header("🔬 Chunking Strategy Comparison (A/B Test)")
    ab_csv = project_root / "evaluation" / "ab_test_results.csv"

    try:
        df = pd.read_csv(ab_csv)

        # 1. Comparative Metrics Dataframe
        st.dataframe(df, use_container_width=True)

        # 2. Comparative Bar Chart
        st.subheader("Strategy Metrics Comparison Chart")
        chart_df = df.set_index("strategy")[["avg_faithfulness", "avg_answer_relevancy", "avg_context_recall"]]
        st.bar_chart(chart_df)

        # 3. Calculate Overall Best Strategy
        df["overall_score"] = df[["avg_faithfulness", "avg_answer_relevancy", "avg_context_recall"]].mean(axis=1)
        best_strategy = df.loc[df["overall_score"].idxmax(), "strategy"]
        st.success(f"🏆 Best Performing Strategy: **{best_strategy.upper()}**")

    except FileNotFoundError:
        st.warning("No A/B test results found. Please run `python evaluation/ab_testing.py` first.")

# --------------------------------------------------------------------
# PAGE 3: LIVE QUERY TESTING VIA FASTAPI BACKEND
# --------------------------------------------------------------------
elif page == "Live Query Test":
    st.header("⚡ Live RAG Query & Real-time Evaluation")
    question = st.text_input("Enter your question:")

    if st.button("Ask System") and question:
        with st.spinner("Executing RAG Pipeline & Hallucination Check..."):
            try:
                # Send HTTP POST payload to FastAPI /query endpoint
                response = requests.post(f"{API_URL}/query", json={"question": question})
                response.raise_for_status()
                data = response.json()

                # Display Generated Answer
                st.subheader("Generated Answer")
                st.write(data["answer"])

                # Display Real-time Hallucination Checker Result
                check = data["hallucination_check"]
                if check["is_likely_hallucinated"]:
                    st.error(f"⚠️ Hallucination Risk Detected! (Support Score: {check['overall_support_score']})")
                else:
                    st.info(f"✅ Answer Well Supported by Context (Support Score: {check['overall_support_score']})")

                # Display Retrieved Source Citations
                st.subheader("Retrieved Source Citations")
                for source in data["sources"]:
                    with st.expander(f"📄 {source['source_file']} — Page {source['page_number']} (Score: {source['rerank_score']:.3f})"):
                        st.write(source["text_snippet"])

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI server. Please start the server using `uvicorn api.main:app --reload`.")
            except Exception as e:
                st.error(f"Error: {e}")

# --------------------------------------------------------------------
# PAGE 4: LIVE PDF DOCUMENT UPLOAD
# --------------------------------------------------------------------
elif page == "Upload Document":
    st.header("📤 Upload & Index New PDF Document")
    st.write("Upload any PDF document live. The server will extract text, generate chunks, update vector embeddings, and make it instantly searchable.")

    uploaded_file = st.file_uploader("Choose a PDF file:", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Upload & Index Document"):
            with st.spinner("Uploading and indexing PDF..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{API_URL}/upload-pdf", files=files)
                    response.raise_for_status()
                    res_data = response.json()

                    st.success(f"🎉 {res_data['message']}")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Pages", res_data["total_pages"])
                    col2.metric("Chunks Added", res_data["chunks_added"])
                    col3.metric("Total Vectors in Index", res_data["total_vectors_in_index"])

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to FastAPI server. Please start the server using `uvicorn api.main:app --reload`.")
                except Exception as e:
                    st.error(f"Error uploading document: {e}")