from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.skills import SkillExtractor


class SemanticMatcher:
    """
    Section-aware semantic matcher.

    Main ideas:
    - Compare JD responsibilities to experience bullets
    - Compare JD skills to resume skills
    - Give partial credit for related technologies
    - Avoid matching arbitrary JD sentences to arbitrary random lines
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.skills = SkillExtractor()

    def _embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        return np.array(self.model.encode(texts, normalize_embeddings=True))

    def requirement_to_bullet_matches(self, resume_text: str, jd_text: str) -> Dict:
        resume_units = self._extract_resume_units(resume_text)
        jd_units = self._extract_jd_requirements(jd_text)

        if not resume_units or not jd_units:
            return {
                "overall_similarity": 0.0,
                "matches": [],
                "uncovered_requirements": [],
            }

        r_emb = self._embed(resume_units)
        j_emb = self._embed(jd_units)
        sim_matrix = np.matmul(j_emb, r_emb.T)

        matches = []
        uncovered = []
        scores = []

        for i, jd_req in enumerate(jd_units):
            best_idx = int(np.argmax(sim_matrix[i]))
            base_score = float(sim_matrix[i][best_idx])
            best_resume_match = resume_units[best_idx]

            # blend semantic sentence similarity with skill relatedness
            skill_score = self.skills.relatedness_score(jd_req, best_resume_match)
            final_score = 0.75 * base_score + 0.25 * skill_score

            item = {
                "jd_requirement": jd_req,
                "best_resume_match": best_resume_match,
                "similarity": round(final_score, 4),
                "base_similarity": round(base_score, 4),
                "skill_relatedness": round(skill_score, 4),
            }

            matches.append(item)
            scores.append(final_score)

            if final_score < 0.45:
                uncovered.append(item)

        overall = float(np.mean(scores)) if scores else 0.0

        return {
            "overall_similarity": round(overall, 4),
            "matches": matches,
            "uncovered_requirements": uncovered,
        }

    def match_sections(
        self,
        experience_bullets: List[str],
        resume_skills: List[str],
        education_lines: List[str],
        jd_text: str,
    ) -> Dict:
        jd_parts = self.classify_jd_requirements(jd_text)

        responsibility_report = self._match_responsibilities(
            experience_bullets,
            jd_parts["responsibilities"],
        )
        skill_report = self._match_skills(
            resume_skills,
            jd_parts["skills"],
        )
        education_report = self._match_education(
            education_lines,
            jd_parts["education"],
        )

        section_scores = []
        for value in [
            responsibility_report["score"],
            skill_report["score"],
            education_report["score"],
        ]:
            if value is not None:
                section_scores.append(value)

        overall = float(np.mean(section_scores)) if section_scores else 0.0

        return {
            "overall_similarity": round(overall, 4),
            "responsibility_report": responsibility_report,
            "skill_report": skill_report,
            "education_report": education_report,
        }

    def classify_jd_requirements(self, jd_text: str) -> Dict[str, List[str]]:
        units = self._extract_jd_requirements(jd_text)

        responsibilities = []
        skills = []
        education = []
        other = []

        for unit in units:
            low = unit.lower()

            if any(word in low for word in ["bachelor", "master", "degree", "computer science", "related field"]):
                education.append(unit)
            elif self.skills.extract_skills(unit):
                skills.append(unit)
            elif any(word in low for word in [
                "develop", "create", "test", "validate", "reporting", "participate",
                "identify", "implement", "work independently", "communicate"
            ]):
                responsibilities.append(unit)
            else:
                other.append(unit)

        return {
            "responsibilities": responsibilities,
            "skills": skills,
            "education": education,
            "other": other,
        }

    def _match_responsibilities(self, experience_bullets: List[str], jd_reqs: List[str]) -> Dict:
        if not jd_reqs:
            return {"score": None, "matches": []}
        if not experience_bullets:
            return {"score": 0.0, "matches": [], "note": "No experience bullets found."}

        r_emb = self._embed(experience_bullets)
        j_emb = self._embed(jd_reqs)
        sim_matrix = np.matmul(j_emb, r_emb.T)

        matches = []
        scores = []

        for i, req in enumerate(jd_reqs):
            best_idx = int(np.argmax(sim_matrix[i]))
            base_score = float(sim_matrix[i][best_idx])
            best_bullet = experience_bullets[best_idx]

            skill_score = self.skills.relatedness_score(req, best_bullet)
            final_score = 0.75 * base_score + 0.25 * skill_score

            matches.append({
                "jd_requirement": req,
                "best_resume_match": best_bullet,
                "similarity": round(final_score, 4),
                "base_similarity": round(base_score, 4),
                "skill_relatedness": round(skill_score, 4),
            })
            scores.append(final_score)

        return {
            "score": round(float(np.mean(scores)), 4) if scores else 0.0,
            "matches": matches,
        }

    def _match_skills(self, resume_skills: List[str], jd_skill_lines: List[str]) -> Dict:
        if not jd_skill_lines:
            return {"score": None, "matches": []}

        matches = []
        scores = []

        for jd_line in jd_skill_lines:
            line_score = 0.0
            related_hits = []

            for resume_skill in resume_skills:
                s = self.skills.relatedness_score(jd_line, resume_skill)
                if s > line_score:
                    line_score = s
                if s > 0:
                    related_hits.append(resume_skill)

            matches.append({
                "jd_requirement": jd_line,
                "related_resume_skills": sorted(set(related_hits)),
                "similarity": round(line_score, 4),
            })
            scores.append(line_score)

        return {
            "score": round(float(np.mean(scores)), 4) if scores else 0.0,
            "matches": matches,
        }

    def _match_education(self, education_lines: List[str], jd_education_lines: List[str]) -> Dict:
        if not jd_education_lines:
            return {"score": None, "matches": []}
        if not education_lines:
            return {"score": 0.0, "matches": [], "note": "No education lines found."}

        edu_text = " ".join(education_lines).lower()
        matches = []
        scores = []

        for line in jd_education_lines:
            low = line.lower()
            score = 0.0
            if "bachelor" in low or "master" in low or "degree" in low:
                if any(word in edu_text for word in ["university", "degree", "education", "bachelor", "master"]):
                    score = 0.8
                elif any(word in edu_text for word in ["polytechnic", "college", "academy"]):
                    score = 0.7

            matches.append({
                "jd_requirement": line,
                "best_resume_match": " | ".join(education_lines[:3]),
                "similarity": round(score, 4),
            })
            scores.append(score)

        return {
            "score": round(float(np.mean(scores)), 4) if scores else 0.0,
            "matches": matches,
        }

    def _extract_resume_units(self, text: str) -> List[str]:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        units = []

        for line in lines:
            if len(line.split()) >= 5 and not re.fullmatch(r"[A-Z ,&/-]+", line):
                units.append(line)

        return units

    def _extract_jd_requirements(self, jd_text: str) -> List[str]:
        jd_text = jd_text.replace("\r\n", "\n").replace("\r", "\n")

        raw_parts = []
        for line in jd_text.splitlines():
            line = line.strip(" •-\t")
            if not line:
                continue
            if len(line.split()) >= 4:
                raw_parts.append(line)

        cleaned = []
        seen = set()
        for part in raw_parts:
            key = part.lower()
            if key not in seen:
                seen.add(key)
                cleaned.append(part)

        return cleaned
