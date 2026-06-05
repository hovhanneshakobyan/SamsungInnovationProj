from __future__ import annotations

import re
from typing import Dict, List


SECTION_PATTERNS = {
    "summary": r"\b(summary|profile|objective|about me|professional summary)\b",
    "experience": r"\b(experience|work experience|employment|professional experience|work history)\b",
    "education": r"\b(education|academic background|qualifications|degree)\b",
    "skills": r"\b(skills|technical skills|programming languages|core competencies|technologies|expertise)\b",
    "projects": r"\b(projects|personal projects|selected projects)\b",
    "certifications": r"\b(certifications|licenses|certificates|professional trainings)\b",
    "contact": r"\b(email|phone|linkedin|github|address)\b",
    "publications": r"\b(publications)\b",
    "awards": r"\b(awards|grants|scholarships)\b",
    "languages": r"\b(languages)\b",
}


class SectionExtractor:
    def detect_sections(self, text: str) -> Dict[str, bool]:
        low = text.lower()
        return {name: bool(re.search(pattern, low)) for name, pattern in SECTION_PATTERNS.items()}

    def split_lines(self, text: str) -> List[str]:
        return [ln.rstrip() for ln in text.splitlines() if ln.strip()]

    def extract_section_block(self, text: str, section_name: str) -> str:
        lines = self.split_lines(text)
        pattern = SECTION_PATTERNS.get(section_name)
        if not pattern:
            return ""

        start = None
        end = None

        for i, line in enumerate(lines):
            if re.search(pattern, line.lower()):
                start = i
                break

        if start is None:
            return ""

        for j in range(start + 1, len(lines)):
            low = lines[j].lower()
            if any(name != section_name and re.search(pat, low) for name, pat in SECTION_PATTERNS.items()):
                end = j
                break

        block = lines[start:end] if end is not None else lines[start:]
        return "\n".join(block)

    def extract_experience_block(self, text: str) -> str:
        return self.extract_section_block(text, "experience")

    def extract_education_block(self, text: str) -> str:
        return self.extract_section_block(text, "education")

    def extract_skills_block(self, text: str) -> str:
        block = self.extract_section_block(text, "skills")
        if block:
            return block
        return ""

    def extract_education_lines(self, text: str) -> List[str]:
        block = self.extract_education_block(text)
        if not block:
            return []
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        out = []
        for line in lines:
            low = line.lower()
            if any(k in low for k in ["experience", "skills", "publications", "awards"]):
                continue
            out.append(line)
        return out
