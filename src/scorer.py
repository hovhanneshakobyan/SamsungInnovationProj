from __future__ import annotations

import re
from typing import Dict, List, Optional
import numpy as np

from src.jd_rules import build_requirement_table
from src.sections import SectionExtractor
from src.semantic_model import SemanticResumeMatcher
from src.skills import SkillExtractor
from src.utils import ACTION_VERBS, count_numeric_impact


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    if a.shape[1] != b.shape[1]:
        return 0.0
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return float(np.mean(np.max(np.dot(a, b.T), axis=1)))


class ATSScorer:
    def __init__(self):
        self.sections = SectionExtractor()
        self.semantic = SemanticResumeMatcher()
        self.skills   = SkillExtractor()

    def normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower())

    def extract_skills(self, text: str) -> List[str]:
        return self.skills.extract_skills(text)

    def extract_bullets(self, text: str) -> List[str]:
        if not text:
            return []
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return [ln for ln in lines if len(ln.split()) > 6]

    def safe_encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((1, 384))
        emb = self.semantic.encode(texts)
        if emb is None or len(emb) == 0:
            return np.zeros((1, 384))
        return np.array(emb)

    def semantic_score(self, resume_text: str, jd_text: str,
                       bullets: List[str], edu: List[str]) -> Dict:

        jd_rows = build_requirement_table(jd_text, self.extract_skills)

        jd_resp, jd_skill, jd_edu = [], [], []
        for r in jd_rows:
            t = r.get("text", "").lower()
            if any(x in t for x in ["degree", "education", "bachelor", "master", "diploma"]):
                jd_edu.append(r["text"])
            elif r.get("skills"):
                jd_skill.append(r["text"])
            else:
                jd_resp.append(r["text"])

        resp  = cosine(self.safe_encode(jd_resp),  self.safe_encode(bullets))
        resume_skill_strs = self.extract_skills(resume_text)
        skill = cosine(
            self.safe_encode(jd_skill),
            self.safe_encode(resume_skill_strs) if resume_skill_strs else np.zeros((1, 384)),
        )

        # --- Education: None when JD doesn't mention it ---
        # If jd_edu is empty the JD has no education requirement.
        # Returning 0 would penalise the candidate unfairly, so we return
        # None and redistribute its weight to resp + skill in overall.
        edu_required = bool(jd_edu)
        edu_score: Optional[float]
        if edu_required:
            edu_score = cosine(self.safe_encode(jd_edu), self.safe_encode(edu))
        else:
            edu_score = None          # "not applicable"

        # Weight redistribution when education is N/A
        if edu_score is None:
            overall = 0.5625 * resp + 0.4375 * skill   # 0.45/0.80 and 0.35/0.80
        else:
            overall = 0.45 * resp + 0.35 * skill + 0.20 * edu_score

        return {
            "overall_similarity":   round(min(overall, 1.0), 4),
            "responsibility_score": round(resp, 4),
            "skill_score":          round(skill, 4),
            # None means "JD did not require education" — shown as N/A in UI
            "education_score":      round(edu_score, 4) if edu_score is not None else None,
            "education_required":   edu_required,
        }

    def achievement_score(self, bullets: List[str]) -> float:
        if not bullets:
            return 0.25
        v = sum(1 for b in bullets if b.split()[0].lower() in ACTION_VERBS)
        n = sum(1 for b in bullets if count_numeric_impact(b) > 0)
        return min(1.0, (v + n) / (2 * len(bullets)))

    def keyword_report(self, resume_text: str, jd_text: str) -> Dict:
        report         = self.skills.jd_skill_report(resume_text, jd_text)
        matched        = report["matched"]
        missing        = report["missing"]
        related        = report["related_matches"]
        base_coverage  = report["coverage_pct"]

        related_bonus  = sum(0.5 for item in related if item["related_resume_skills"])
        jd_skill_count = max(len(report["jd_skills"]), 1)
        adjusted_coverage = min(100.0, base_coverage + (related_bonus / jd_skill_count) * 100)

        return {
            "matched":           matched,
            "missing":           missing,
            "related_matches":   related,
            "coverage_pct":      round(adjusted_coverage, 1),
            "base_coverage_pct": round(base_coverage, 1),
        }

    def score(self, resume_text: str, jd_text: str) -> Dict:
        exp     = self.sections.extract_experience_block(resume_text)
        bullets = self.extract_bullets(exp)
        edu     = self.sections.extract_education_lines(resume_text)

        keyword     = self.keyword_report(resume_text, jd_text)
        semantic    = self.semantic_score(resume_text, jd_text, bullets, edu)
        achievement = self.achievement_score(bullets)

        section_score = sum(
            1 for s in ["experience", "education", "skills"]
            if self.sections.detect_sections(resume_text).get(s, False)
        ) / 3

        overall = (
            0.40 * (keyword["coverage_pct"] / 100) +
            0.30 * semantic["overall_similarity"] +
            0.15 * section_score +
            0.15 * achievement
        ) * 100

        return {
            "overall_score": round(overall, 1),
            "breakdown": {
                "keyword":     round(keyword["coverage_pct"], 1),
                "semantic":    round(semantic["overall_similarity"] * 100, 1),
                "sections":    round(section_score * 100, 1),
                "achievement": round(achievement * 100, 1),
            },
            "keyword_report":  keyword,
            "semantic_report": semantic,
            "format_warnings": [],
        }
