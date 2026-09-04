"""
======================================================================
PDF LOADER MODULE (ingestion/pdf_loader.py)
======================================================================
Purpose:
  PDF files se structured text aur metadata extract karna:
  - `pdfplumber` use kar rahe hain kyunki yeh layout and tabular data preserved rakhta hai.
  - Har page ke content ke saath file name and page number capture karta hai.
======================================================================
"""

import pdfplumber
from pathlib import Path
from typing import TypedDict


# TypedDict structure for type-checking and code clarity
class PageContent(TypedDict):
    text: str          # Extracted page text
    page_number: int   # 1-indexed page number
    source_file: str   # Name of the PDF file


def load_pdf(file_path: str) -> list[PageContent]:
    """
    Single PDF file load karke uske har page ka text & metadata dictionary return karta hai.

    Args:
        file_path (str): File system path to the target PDF file.

    Returns:
        list[PageContent]: List of dictionaries containing page text and metadata.
    """
    pages: list[PageContent] = []
    file_name = Path(file_path).name

    # 1. pdfplumber stream context manager kholna
    with pdfplumber.open(file_path) as pdf:
        # 2. Har page par iterate karna (1-indexed counter)
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""  # Fallback for empty/scanned pages
            
            # 3. Non-empty text pages filter karna
            if text.strip():
                pages.append({
                    "text": text,
                    "page_number": i,
                    "source_file": file_name,
                })

    return pages


def load_all_pdfs(folder_path: str) -> list[PageContent]:
    """
    Target folder se saari `.pdf` files iterate karke combine pages list return karta hai.

    Args:
        folder_path (str): Target directory containing PDF documents.

    Returns:
        list[PageContent]: Combined list of all extracted pages across all PDFs.
    """
    all_pages: list[PageContent] = []
    folder = Path(folder_path)

    # Folder ke andar har *.pdf file discover karke text extract karo
    for pdf_file in folder.glob("*.pdf"):
        print(f"Reading & Extraction PDF: {pdf_file.name}")
        all_pages.extend(load_pdf(str(pdf_file)))

    return all_pages