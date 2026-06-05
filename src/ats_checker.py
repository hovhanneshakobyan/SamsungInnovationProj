"""
ats_checker.py
ATS Compatibility Checker — scores an optimized resume on ATS-friendliness.

Checks
------
1. Keyword coverage   – % of JD keywords found in resume
2. Section presence   – required sections detected (Experience, Education, Skills)
3. Format warnings    – tables, images, multi-column layouts detected (plain-text heuristics)
4. Semantic score     – cosine similarity from the trained Siamese model (optional)
5. Overall ATS score  – weighted combination → [0, 100]

The module can be used standalone (returns a dict) or via the Streamlit UI.
"""

import re
from collections import Counter
from loguru import logger

# Optional: use trained siamese model for semantic scoring
try:
    from src.siamese_model import get_match_score
    _SIAMESE_AVAILABLE = True
except ImportError:
    _SIAMESE_AVAILABLE = False


# ── Keyword extraction (simple TF-IDF-style) ─────────────────────────────────
_STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for",
    "of", "with", "is", "are", "was", "were", "be", "been", "have",
    "has", "had", "do", "does", "did", "will", "would", "can", "could",
    "should", "may", "might", "we", "you", "they", "he", "she", "it",
    "this", "that", "which", "who", "from", "by", "as", "our", "your",
}

def extract_keywords(text: str, top_n: int = 30) -> list[str]:
    """Return top_n most frequent non-stopword tokens."""
    tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9+#.\-]{1,}\b', text.lower())
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    freq   = Counter(tokens)
    return [w for w, _ in freq.most_common(top_n)]


# ── Section detection ─────────────────────────────────────────────────────────
_SECTION_PATTERNS = {
    "experience":  r'\b(experience|work history|employment|professional background)\b',
    "education":   r'\b(education|academic|degree|university|college|school)\b',
    "skills":      r'\b(skills|technologies|technical skills|competencies|expertise)\b',
    "summary":     r'\b(summary|objective|profile|about me)\b',
    "contact":     r'\b(email|phone|linkedin|github|address)\b',
}

def detect_sections(resume_text: str) -> dict[str, bool]:
    text_lower = resume_text.lower()
    return {section: bool(re.search(pattern, text_lower))
            for section, pattern in _SECTION_PATTERNS.items()}


# ── Format warnings ───────────────────────────────────────────────────────────
def detect_format_issues(resume_text: str) -> list[str]:
    """Heuristic checks for ATS-unfriendly formatting."""
    warnings = []
    if re.search(r'\|.*\|.*\|', resume_text):
        warnings.append("Possible table detected — ATS may misparse columns.")
    if len(re.findall(r'\n', resume_text)) < 5:
        warnings.append("Very few line breaks — document may be image-based PDF.")
    if re.search(r'[^\x00-\x7F]{10,}', resume_text):
        warnings.append("Non-ASCII characters detected — may cause parsing issues.")
    if len(resume_text) < 200:
        warnings.append("Resume text is very short — ensure full content was extracted.")
    return warnings


# ── Keyword coverage score ────────────────────────────────────────────────────
def keyword_coverage(resume_text: str, jd_text: str) -> dict:
    jd_keywords      = extract_keywords(jd_text, top_n=30)
    resume_lower     = resume_text.lower()
    matched          = [kw for kw in jd_keywords if kw in resume_lower]
    coverage_pct     = len(matched) / max(len(jd_keywords), 1) * 100
    missing          = [kw for kw in jd_keywords if kw not in matched]
    return {
        "jd_keywords":    jd_keywords,
        "matched":        matched,
        "missing":        missing,
        "coverage_pct":   round(coverage_pct, 1),
    }


# ── ATS Score aggregation ─────────────────────────────────────────────────────
WEIGHTS = {
    "keyword_coverage":  0.40,   # 40%
    "section_presence":  0.25,   # 25%
    "no_format_issues":  0.15,   # 15%
    "semantic_score":    0.20,   # 20%  (from Siamese model)
}

def ats_score(resume_text: str, jd_text: str,
              siamese_model=None, siamese_tokenizer=None) -> dict:
    """
    Returns a full ATS report dict with an overall score [0–100].
    """

    # 1. Keyword coverage
    kw_result   = keyword_coverage(resume_text, jd_text)
    kw_score    = kw_result["coverage_pct"] / 100   # [0,1]

    # 2. Section presence
    sections     = detect_sections(resume_text)
    core         = ["experience", "education", "skills"]
    section_score = sum(sections[s] for s in core) / len(core)   # [0,1]

    # 3. Format issues
    warnings      = detect_format_issues(resume_text)
    format_score  = max(0.0, 1.0 - len(warnings) * 0.25)

    # 4. Semantic score (Siamese model)
    if _SIAMESE_AVAILABLE and siamese_model is not None:
        try:
            sem_score = get_match_score(resume_text, jd_text,
                                        model=siamese_model,
                                        tokenizer=siamese_tokenizer)
        except Exception as e:
            logger.warning(f"Siamese scoring failed: {e}")
            sem_score = kw_score   # fallback to keyword score
    else:
        sem_score = kw_score       # fallback

    # Weighted overall
    overall = (
        WEIGHTS["keyword_coverage"] * kw_score
      + WEIGHTS["section_presence"] * section_score
      + WEIGHTS["no_format_issues"] * format_score
      + WEIGHTS["semantic_score"]   * sem_score
    ) * 100

    return {
        "overall_score":    round(overall, 1),
        "keyword_coverage": kw_result,
        "sections":         sections,
        "format_warnings":  warnings,
        "semantic_score":   round(sem_score * 100, 1),
        "breakdown": {
            "keyword":  round(kw_score * WEIGHTS["keyword_coverage"] * 100, 1),
            "sections": round(section_score * WEIGHTS["section_presence"] * 100, 1),
            "format":   round(format_score  * WEIGHTS["no_format_issues"] * 100, 1),
            "semantic": round(sem_score     * WEIGHTS["semantic_score"]   * 100, 1),
        }
    }


if __name__ == "__main__":
    r = "Software Engineer. Skills: Python, Docker. Education: BSc CS. Experience: 3 years at TechCorp."
    j = "We need a Python developer with Docker and Kubernetes skills."
    report = ats_score(r, j)
    print(f"Overall ATS Score: {report['overall_score']}/100")
    print(f"Missing keywords:  {report['keyword_coverage']['missing']}")
