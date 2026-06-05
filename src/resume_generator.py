"""
resume_generator.py
Generates a complete ATS-friendly resume from scratch using fine-tuned T5.

Training data: data/processed/generation_pairs.csv  (prompt → full resume)
Checkpoint:    models/generator_t5/

Pipeline:
  User fills form → _build_prompt() → T5 generates full resume text
  Fallback: clean template if T5 not yet trained
"""

import os
import re
import torch
from loguru import logger

from src.resume_optimizer import T5_GEN_CKPT, _t5_is_ready


class ResumeGenerator:
    """
    Generates a full resume from structured user input using fine-tuned T5.
    Falls back to a clean template if T5 generator is not yet trained.
    """

    def __init__(self):
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._use_t5 = _t5_is_ready(T5_GEN_CKPT)

        if self._use_t5:
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            logger.info("ResumeGenerator: loading fine-tuned T5 generator …")
            self.tokenizer = T5Tokenizer.from_pretrained(T5_GEN_CKPT)
            self.model     = T5ForConditionalGeneration.from_pretrained(T5_GEN_CKPT).to(self.device)
            self.model.eval()
        else:
            logger.info("ResumeGenerator: T5 generator not trained — template fallback active.")
            self.tokenizer = None
            self.model     = None

    @torch.no_grad()
    def generate(self,
                 full_name:      str,
                 job_title:      str,
                 email:          str,
                 phone:          str,
                 location:       str,
                 linkedin:       str,
                 years_exp:      int,
                 summary:        str,
                 skills:         list,
                 experiences:    list,
                 education:      list,
                 certifications: list,
                 jd_text:        str = "") -> tuple:
        """
        Returns (resume_text: str, used_ai: bool).
        used_ai=True  → T5 generated the resume
        used_ai=False → template used (T5 not yet trained)
        """
        prompt = _build_prompt(
            full_name=full_name, job_title=job_title,
            email=email, phone=phone, location=location, linkedin=linkedin,
            years_exp=years_exp, summary=summary, skills=skills,
            experiences=experiences, education=education,
            certifications=certifications, jd_text=jd_text,
        )

        if self._use_t5:
            try:
                input_ids = self.tokenizer(
                    prompt, max_length=512, truncation=True, return_tensors="pt"
                ).input_ids.to(self.device)

                output = self.model.generate(
                    input_ids,
                    max_new_tokens=512,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    length_penalty=1.5,
                )
                generated = self.tokenizer.decode(output[0], skip_special_tokens=True)

                # Strip any echoed prompt prefix
                if generated.lower().startswith("generate resume:"):
                    generated = generated[len("generate resume:"):].strip()

                # Post-process: ensure JD keywords present
                if jd_text:
                    generated = _inject_missing_keywords(generated, jd_text)

                return generated, True

            except Exception as e:
                logger.warning(f"T5 generator failed ({e}) — using template fallback")

        # Template fallback
        resume = _build_template(
            full_name=full_name, job_title=job_title,
            email=email, phone=phone, location=location, linkedin=linkedin,
            years_exp=years_exp, summary=summary, skills=skills,
            experiences=experiences, education=education,
            certifications=certifications, jd_text=jd_text,
        )
        return resume, False


# ── Prompt builder (must match generate_synthetic_data.py format) ─────────────
def _build_prompt(full_name, job_title, email, phone, location, linkedin,
                  years_exp, summary, skills, experiences, education,
                  certifications, jd_text) -> str:
    exp_parts = []
    for e in experiences:
        bullets   = "; ".join(b.strip() for b in e.get("bullets", []) if b.strip())
        exp_parts.append(
            f"{e.get('title','')} at {e.get('company','')} "
            f"({e.get('start','')}–{e.get('end','Present')}): {bullets}"
        )
    exp_str = " | ".join(exp_parts)

    edu_str = " | ".join(
        f"{e.get('degree','')} from {e.get('institution','')} ({e.get('year','')})"
        for e in education
    )

    jd_hint = f" target_job: {jd_text[:200]}" if jd_text.strip() else ""

    return (
        f"generate resume: "
        f"name: {full_name} "
        f"title: {job_title} "
        f"email: {email} "
        f"phone: {phone} "
        f"location: {location} "
        f"linkedin: {linkedin} "
        f"years: {years_exp} "
        f"summary: {summary} "
        f"skills: {', '.join(skills)} "
        f"experience: {exp_str} | "
        f"education: {edu_str} "
        f"certifications: {', '.join(certifications)}"
        f"{jd_hint}"
    )


# ── Template fallback ─────────────────────────────────────────────────────────
def _build_template(full_name, job_title, email, phone, location, linkedin,
                    years_exp, summary, skills, experiences, education,
                    certifications, jd_text) -> str:
    lines = []
    lines.append(full_name.upper())
    lines.append(job_title)
    contacts = [p for p in [email, phone, location, linkedin] if p.strip()]
    lines.append("  |  ".join(contacts))
    lines.append("")

    if summary.strip():
        if jd_text:
            kws     = _top_jd_keywords(jd_text, 3)
            missing = [k for k in kws if k.lower() not in summary.lower()]
            if missing:
                summary = summary.rstrip(".") + f". Experienced with {', '.join(missing)}."
        lines += ["SUMMARY", "-" * 40, summary.strip(), ""]

    if skills:
        if jd_text:
            kws  = _top_jd_keywords(jd_text, 15)
            low  = [s.lower() for s in skills]
            for kw in kws:
                if kw.lower() not in low:
                    skills.append(kw)
        lines += ["SKILLS", "-" * 40]
        for i in range(0, len(skills), 4):
            lines.append("  •  ".join(skills[i:i+4]))
        lines.append("")

    if experiences:
        lines += ["EXPERIENCE", "-" * 40]
        for exp in experiences:
            lines.append(f"{exp.get('title','')}  —  {exp.get('company','')}  "
                         f"({exp.get('start','')} – {exp.get('end','Present')})")
            for b in exp.get("bullets", []):
                if b.strip():
                    lines.append(f"• {b.strip()}")
            lines.append("")

    if education:
        lines += ["EDUCATION", "-" * 40]
        for edu in education:
            lines.append(f"{edu.get('degree','')}  —  "
                         f"{edu.get('institution','')}  ({edu.get('year','')})")
        lines.append("")

    if certifications:
        lines += ["CERTIFICATIONS", "-" * 40]
        for c in certifications:
            if c.strip():
                lines.append(f"• {c.strip()}")
        lines.append("")

    return "\n".join(lines)


# ── Keyword helpers ───────────────────────────────────────────────────────────
_STOPWORDS = {
    "a","an","the","and","or","in","on","at","to","for","of","with","is","are",
    "was","were","be","been","have","has","had","do","does","did","will","would",
    "can","could","should","may","might","we","you","they","he","she","it","this",
    "that","which","who","from","by","as","our","your","their","all","also","not",
    "but","if","then","so","its","looking","hiring","seeking","required","must",
    "please","able","work","team","strong","good","excellent","responsibilities",
    "requirements","candidate","experience","years","skills",
}

def _top_jd_keywords(jd_text: str, n: int = 15) -> list:
    tokens = re.findall(r'\b[A-Za-z][A-Za-z0-9+#.\-]{1,}\b', jd_text)
    seen, result = set(), []
    for t in tokens:
        tl = t.lower()
        if tl not in _STOPWORDS and len(tl) > 2 and tl not in seen:
            seen.add(tl); result.append(t)
        if len(result) >= n:
            break
    return result

def _inject_missing_keywords(resume_text: str, jd_text: str) -> str:
    keywords   = _top_jd_keywords(jd_text, 15)
    resume_low = resume_text.lower()
    missing    = [kw for kw in keywords if kw.lower() not in resume_low]
    if not missing:
        return resume_text
    skills_pat = re.compile(r'(SKILLS?[:\s]*\n?)(.*?)(\n\n|\Z)',
                             re.IGNORECASE | re.DOTALL)
    m = skills_pat.search(resume_text)
    if m:
        old = m.group(2).strip()
        new = old + (", " if old else "") + ", ".join(missing)
        resume_text = resume_text[:m.start(2)] + new + resume_text[m.end(2):]
    else:
        resume_text = resume_text.rstrip() + f"\n\nSKILLS\n{', '.join(missing)}\n"
    return resume_text
