from __future__ import annotations

import re
from typing import Dict, List

from src.scorer import ATSScorer


# Lines we prepend for human readability — we must strip them before re-scoring
# so the scorer doesn't penalise "Missing from resume: ios, swift" as bad content.
_HEADER_SENTINEL = "TARGET SKILLS ALIGNMENT"
_SUMMARY_SENTINEL = "PROFESSIONAL SUMMARY\nSoftware Engineer with hands-on experience in"


class ResumeOptimizer:
    def __init__(self):
        self.scorer = ATSScorer()

    # ------------------------------------------------------------------
    def _strip_optimizer_header(self, text: str) -> str:
        """Remove the TARGET SKILLS block we prepended so scoring is clean."""
        if _HEADER_SENTINEL not in text:
            return text
        # Everything from the sentinel up to the first blank line after it
        # forms our header block. Find where the original resume starts.
        idx = text.find(_HEADER_SENTINEL)
        # Walk forward two blank lines — that's past our block
        after = text[idx:]
        # Find the double-newline that ends our block
        end = after.find("\n\nHOV")   # original resume usually starts with the name
        if end == -1:
            # fallback: find the second \n\n after the sentinel
            first = after.find("\n\n")
            end   = after.find("\n\n", first + 2) if first != -1 else -1
        if end != -1:
            return after[end:].lstrip()
        return text

    # ------------------------------------------------------------------
    def _inject_skills_section(self, text: str, skill_line: str) -> str:
        """Append missing skills into the SKILLS / TECHNICAL SKILLS section."""
        pattern = re.compile(
            r"((?:TECHNICAL\s+)?SKILLS?[:\s]*\n?)(.*?)(\n\n|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        m = pattern.search(text)
        if m:
            old_body = m.group(2).strip()
            new_body = (old_body + ", " + skill_line) if old_body else skill_line
            return text[: m.start(2)] + new_body + text[m.end(2):]
        return text.rstrip() + f"\n\nSKILLS\n{skill_line}\n"

    # ------------------------------------------------------------------
    def optimize(self, resume_text: str, jd_text: str) -> Dict:
        # Score on the clean original — no headers injected yet
        score   = self.scorer.score(resume_text, jd_text)
        keyword = score["keyword_report"]

        matched = keyword.get("matched", [])
        missing = keyword.get("missing", [])
        related = keyword.get("related_matches", [])

        optimized   = resume_text.strip()
        suggestions: List[str] = []

        # ── 1. Add PROFESSIONAL SUMMARY if absent ────────────────────────
        if "SUMMARY" not in optimized.upper():
            skill_preview = ", ".join(matched[:5]) or "software development"
            optimized = (
                f"PROFESSIONAL SUMMARY\n"
                f"Software Engineer with hands-on experience in {skill_preview}.\n\n"
                + optimized
            )
            suggestions.append(
                "A Professional Summary was added — personalise it with your years of "
                "experience and a specific achievement."
            )

        # ── 2. Inject missing skills into the SKILLS section ─────────────
        # Do this BEFORE prepending the alignment header so the regex finds
        # the real SKILLS section in the original resume text.
        if missing:
            missing_str = ", ".join(missing[:8])
            optimized = self._inject_skills_section(optimized, missing_str)
            suggestions.append(
                f"Add evidence for: {missing_str}. Even a side project or course counts."
            )

        # ── 3. Prepend a TARGET SKILLS block (human-readable header only) ─
        # This is for the candidate to read, NOT for the ATS scorer.
        target_lines: List[str] = []
        if matched:
            target_lines.append(f"Matched: {', '.join(matched[:8])}")
        if missing:
            target_lines.append(f"Missing from resume: {', '.join(missing[:8])}")
        for item in related[:4]:
            jd_s   = item["jd_skill"]
            covers = ", ".join(item["related_resume_skills"])
            target_lines.append(f"  \u2022 Your {covers} experience covers '{jd_s}' requirements")

        if target_lines:
            optimized = (
                _HEADER_SENTINEL + "\n"
                + "\n".join(target_lines)
                + "\n\n"
                + optimized
            )

        # ── 4. Suggestions: related-tech tips ────────────────────────────
        for item in related:
            jd_s   = item["jd_skill"]
            covers = ", ".join(item["related_resume_skills"])
            suggestions.append(
                f"JD asks for '{jd_s}' — your '{covers}' experience is related. "
                f"Add a bullet that explicitly names '{jd_s}' to turn partial credit into full."
            )

        # ── 5. Generic quality tips ───────────────────────────────────────
        suggestions += [
            "Add measurable impact to each bullet (%, $, time saved, scale).",
            "Use a single-column layout — multi-column resumes confuse most ATS parsers.",
            "Expand the Education section with GPA, relevant courses, or honours if available.",
        ]

        # ── 6. Score AFTER: strip the human-readable header first ────────
        clean_for_scoring = self._strip_optimizer_header(optimized)
        score_after = self.scorer.score(clean_for_scoring, jd_text)

        return {
            "optimized_resume": optimized,
            "suggestions": suggestions,
            "score_before": score["overall_score"],
            "score_after":  score_after["overall_score"],
        }
