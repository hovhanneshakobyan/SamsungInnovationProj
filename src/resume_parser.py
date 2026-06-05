"""
resume_parser.py
Extracts clean text from PDF, DOCX, or plain text input.
"""

import re
import fitz          # PyMuPDF
import pdfplumber
from docx import Document as DocxDocument


def parse_pdf(file_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF with pdfplumber as fallback."""
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception:
        # fallback
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
    return _clean_text(text)


def parse_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""
    doc = DocxDocument(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return _clean_text("\n".join(paragraphs))


def parse_text(raw_text: str) -> str:
    """Clean and return pasted plain text."""
    return _clean_text(raw_text)


def parse_resume(source, file_type: str = "text") -> str:
    """
    Universal entry point.
    source    : file path (str) or raw text (str)
    file_type : 'pdf' | 'docx' | 'text'
    """
    file_type = file_type.lower()
    if file_type == "pdf":
        return parse_pdf(source)
    elif file_type == "docx":
        return parse_docx(source)
    else:
        return parse_text(source)


def _clean_text(text: str) -> str:
    """Remove excessive whitespace and non-printable characters."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\u0400-\u04FF\n]', '', text)
    return text.strip()


# ── quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = "  John   Doe \n\n Software Engineer  \n Python, PyTorch, Docker  "
    print(parse_resume(sample, "text"))
