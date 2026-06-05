from __future__ import annotations

import fitz
import pdfplumber
from docx import Document as DocxDocument

from src.utils import looks_like_section_header, normalize_whitespace


class ResumeParser:
    def parse_pdf(self, file_path: str) -> str:
        text = ""
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text("text") + "\n"
            doc.close()
        except Exception:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
        return self._clean_structured_text(text)

    def parse_docx(self, file_path: str) -> str:
        doc = DocxDocument(file_path)
        text = "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
        return self._clean_structured_text(text)

    def parse_text(self, text: str) -> str:
        return self._clean_structured_text(text)

    def parse(self, source: str, file_type: str) -> str:
        kind = file_type.lower()
        if kind == "pdf":
            return self.parse_pdf(source)
        if kind == "docx":
            return self.parse_docx(source)
        return self.parse_text(source)

    def _clean_structured_text(self, text: str) -> str:
        text = normalize_whitespace(text)
        raw_lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

        repaired = []
        i = 0
        while i < len(raw_lines):
            line = raw_lines[i]
            if i + 1 < len(raw_lines):
                nxt = raw_lines[i + 1]
                if (
                    not looks_like_section_header(line)
                    and not looks_like_section_header(nxt)
                    and len(line.split()) <= 4
                    and line
                    and nxt
                    and line[-1].isalnum()
                    and nxt[0].islower()
                ):
                    line = f"{line} {nxt}"
                    i += 1
            repaired.append(line)
            i += 1

        final_lines = []
        for line in repaired:
            if looks_like_section_header(line):
                if final_lines and final_lines[-1] != "":
                    final_lines.append("")
                final_lines.append(line.upper())
                final_lines.append("")
            else:
                final_lines.append(line)

        result = "\n".join(final_lines)
        return normalize_whitespace(result)
