from __future__ import annotations

import difflib
import json
import random
import re
from pathlib import Path
from typing import Any, List

import numpy as np


ACTION_VERBS = {
    "built", "developed", "designed", "implemented", "optimized", "led",
    "created", "managed", "improved", "launched", "analyzed", "automated",
    "deployed", "scaled", "reduced", "increased", "delivered", "collaborated",
    "owned", "architected", "fixed", "maintained", "extended", "integrated",
    "tested", "debugged", "supported",
}

SECTION_HEADERS = {
    "SUMMARY", "PROFILE", "OBJECTIVE", "EDUCATION", "EXPERIENCE",
    "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE", "SKILLS",
    "TECHNICAL SKILLS", "PROGRAMMING LANGUAGES", "LANGUAGES",
    "CERTIFICATIONS", "PROJECTS", "PUBLICATIONS", "AWARDS",
    "AWARDS, GRANTS AND SCHOLARSHIPS", "PROFESSIONAL TRAININGS",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text_for_matching(text: str) -> str:
    text = text.lower()
    text = text.replace("javasript", "javascript")
    text = text.replace("java script", "javascript")
    text = text.replace("dotnet", ".net")
    text = text.replace("dot net", ".net")
    text = text.replace("asp net", "asp.net")
    text = text.replace("restful apis", "rest api")
    text = text.replace("restful api", "rest api")
    text = text.replace("object oriented", "object-oriented")
    text = re.sub(r"[^a-z0-9+#./\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def looks_like_section_header(line: str) -> bool:
    line = line.strip()
    if line.upper() in SECTION_HEADERS:
        return True
    if len(line) > 50:
        return False
    return bool(re.fullmatch(r"[A-Z ,&/()\-\+]+", line))


def looks_like_title_only(line: str) -> bool:
    if len(line.split()) <= 6:
        return True
    if "," in line and len(line.split()) <= 9:
        return True
    return False


def count_numeric_impact(text: str) -> int:
    return len(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", text))


def unified_diff(old: str, new: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            old.splitlines(),
            new.splitlines(),
            fromfile="original",
            tofile="optimized",
            lineterm="",
        )
    )


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))