from __future__ import annotations

import re
from typing import Dict, List


MANDATORY_HINTS = [
    "required",
    "must",
    "hands on experience",
    "hands-on experience",
    "strong knowledge",
    "proven knowledge",
    "ability to",
    "3+ years",
    "2+ years",
    "experience in",
    "experience with",
]

OPTIONAL_HINTS = [
    "nice to have",
    "plus",
    "preferred",
    "bonus",
    "will be plus",
    "would be a plus",
]


def split_jd_lines(jd_text: str) -> List[str]:
    lines = []
    for line in jd_text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = line.strip(" •-\t")
        if line and len(line.split()) >= 3:
            lines.append(line)
    return lines


def classify_requirement_priority(line: str) -> str:
    low = line.lower()

    if any(hint in low for hint in OPTIONAL_HINTS):
        return "optional"

    if any(hint in low for hint in MANDATORY_HINTS):
        return "mandatory"

    return "context"


def build_requirement_table(jd_text: str, extract_skills_fn) -> List[Dict]:
    rows = []
    for line in split_jd_lines(jd_text):
        rows.append({
            "text": line,
            "priority": classify_requirement_priority(line),
            "skills": extract_skills_fn(line),
        })
    return rows