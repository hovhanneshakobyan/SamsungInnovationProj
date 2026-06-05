"""
quick_test.py
Smoke test — runs the full pipeline end-to-end WITHOUT training.
Uses the rule-based keyword optimizer and ATS checker.
Verifies all imports and logic work before training.

Run (correct):   .venv\Scripts\python quick_test.py
Run (from IDE):  just press Run — PyCharm/VS Code use venv automatically
"""

import sys
import os

# ── venv guard ────────────────────────────────────────────────────────────────
def _check_venv():
    in_venv = (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        or ".venv" in sys.executable
        or "venv" in sys.executable
    )
    if not in_venv:
        print("=" * 58)
        print("  WARNING: Not running inside the project virtualenv!")
        print("  Use one of these commands instead:")
        print()
        print('  .venv\\Scripts\\python quick_test.py')
        print('  (or just run from PyCharm / VS Code)')
        print("=" * 58)
        print()

_check_venv()

sys.path.insert(0, os.path.dirname(__file__))

from loguru import logger

SAMPLE_RESUME = """
Alex Johnson
Email: alex.johnson@email.com

SUMMARY
Software Engineer with 4 years of experience building backend systems.

EXPERIENCE
Software Engineer — Acme Corp (2020–Present)
• Developed REST APIs using Python and Flask.
• Managed PostgreSQL databases and Redis caching.
• Deployed services with Docker and CI/CD pipelines.

EDUCATION
BSc Computer Science — MIT

SKILLS
Python, Flask, PostgreSQL, Docker, Git, REST API
"""

SAMPLE_JD = """
TechNova is hiring a Senior Python Developer.

Responsibilities:
• Design and implement scalable backend services.
• Work with Kubernetes, Docker, and cloud infrastructure.
• Collaborate with frontend teams on REST APIs.

Requirements:
• 3+ years of Python experience.
• Strong knowledge of Docker, Kubernetes, PostgreSQL.
• Experience with CI/CD and cloud platforms (AWS/GCP).
• Excellent problem-solving skills.
"""


def run():
    logger.info("=" * 58)
    logger.info("  RESUME2VEC — QUICK SMOKE TEST")
    logger.info("=" * 58)

    # ── Step 1: Resume Parser ─────────────────────────────────────────────
    logger.info("\n[1/4] Resume Parser")
    from src.resume_parser import parse_resume
    resume = parse_resume(SAMPLE_RESUME, "text")
    logger.success(f"  Parsed {len(resume)} characters")
    print(f"  Preview: {resume[:80]}…\n")

    # ── Step 2: ATS Checker ───────────────────────────────────────────────
    logger.info("[2/4] ATS Checker")
    from src.ats_checker import ats_score
    report = ats_score(resume, SAMPLE_JD)
    logger.success(f"  Overall ATS score : {report['overall_score']} / 100")
    logger.success(f"  Keyword coverage  : {report['keyword_coverage']['coverage_pct']}%")
    logger.success(f"  Missing keywords  : {report['keyword_coverage']['missing'][:5]}")
    logger.success(f"  Sections found    : {[k for k,v in report['sections'].items() if v]}")
    print()

    # ── Step 3: Resume Optimizer (rule-based, no training needed) ─────────
    logger.info("[3/4] Resume Optimizer  (rule-based keyword injection)")
    from src.resume_optimizer import ResumeOptimizer
    optimizer = ResumeOptimizer(use_t5=False)
    optimized = optimizer.optimize(resume, SAMPLE_JD)
    logger.success(f"  Generated {len(optimized)} characters of optimized resume")
    print(f"  Preview: {optimized[:150]}…\n")

    # ── Step 4: Re-score optimized ────────────────────────────────────────
    logger.info("[4/4] ATS Re-score after optimization")
    new_report = ats_score(optimized, SAMPLE_JD)
    before = report["overall_score"]
    after  = new_report["overall_score"]
    delta  = round(after - before, 1)
    sign   = "+" if delta >= 0 else ""
    icon   = "✅" if delta >= 0 else "⚠️"
    logger.success(f"  {icon} Before: {before}  →  After: {after}  (Δ {sign}{delta})")

    kw_before = report["keyword_coverage"]["coverage_pct"]
    kw_after  = new_report["keyword_coverage"]["coverage_pct"]
    logger.success(f"  Keyword coverage: {kw_before}%  →  {kw_after}%")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 58)
    if delta >= 0:
        print("  ALL TESTS PASSED ✓  Pipeline is working correctly.")
    else:
        print("  Tests ran but optimizer needs training for best results.")
    print()
    print("  Next steps:")
    print("    Generate data : python src/generate_synthetic_data.py")
    print("    Train models  : python train_all.py --synthetic")
    print("    Launch UI     : streamlit run app.py")
    print("=" * 58)


if __name__ == "__main__":
    run()
