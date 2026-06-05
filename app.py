from __future__ import annotations

import io
import os
import tempfile
from typing import Dict, List

import streamlit as st
import pandas as pd

from src.diff_utils import make_unified_diff
from src.optimizer import ResumeOptimizer
from src.parser import ResumeParser
from src.scorer import ATSScorer
from src.job_scraper import scrape_job_description


# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="ATS Tool", layout="wide", page_icon="📄")

# ── session state defaults ────────────────────────────────────────────────────
for key, default in [("jd_text", ""), ("mode", "Job Seeker (B2C)")]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── shared helpers ────────────────────────────────────────────────────────────
parser    = ResumeParser()
scorer    = ATSScorer()
optimizer = ResumeOptimizer()


def parse_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        if suffix == ".pdf":
            return parser.parse(tmp_path, "pdf")
        if suffix == ".docx":
            return parser.parse(tmp_path, "docx")
        return parser.parse(
            uploaded_file.getvalue().decode("utf-8", errors="ignore"), "text"
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def score_badge(s: float) -> str:
    return "🟢" if s >= 75 else ("🟡" if s >= 50 else "🔴")


def edu_display(semantic_report: Dict) -> str:
    """Return a human-readable education score string."""
    if not semantic_report.get("education_required", True):
        return "N/A (not required)"
    v = semantic_report.get("education_score")
    return f"{v * 100:.0f}%" if v is not None else "N/A"


# ─────────────────────────────────────────────────────────────────────────────
# MODE SELECTOR  (top of page, outside sidebar)
# ─────────────────────────────────────────────────────────────────────────────
st.title("ATS Resume Tool")

mode = st.radio(
    "Select mode",
    ["Job Seeker (B2C)", "Recruiter / HR (B2B)"],
    horizontal=True,
    key="mode",
    help=(
        "**Job Seeker:** Upload your own CV and get a score + improvement tips.\n\n"
        "**Recruiter / HR:** Upload multiple CVs and rank them against one job description."
    ),
)

st.divider()


# ═════════════════════════════════════════════════════════════════════════════
#  B2C  —  Job Seeker mode
# ═════════════════════════════════════════════════════════════════════════════
if mode == "Job Seeker (B2C)":

    st.caption("Upload your CV · paste a job description · get your ATS score and improvement tips")

    # ── sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Your inputs")

        uploaded_resume = st.file_uploader(
            "Upload your resume", type=["pdf", "docx", "txt"]
        )

        st.subheader("Job Description")
        job_url = st.text_input(
            "Paste job URL (optional)",
            placeholder="https://staff.am/... or https://linkedin.com/..."
        )
        if st.button("Fetch from URL"):
            if job_url.strip():
                with st.spinner("Fetching…"):
                    scraped = scrape_job_description(job_url)
                st.session_state.jd_text = scraped
                if any(scraped.startswith(p) for p in ("LinkedIn requires", "Error", "Could not", "Failed")):
                    st.warning(scraped)
                else:
                    st.success(f"Fetched {len(scraped.split())} words.")
            else:
                st.warning("Please enter a URL first.")

        jd_text = st.text_area(
            "Or paste / edit job description",
            value=st.session_state.jd_text,
            height=300,
        )
        run_btn = st.button("Analyze Resume", type="primary")

    # ── main panel ────────────────────────────────────────────────────────────
    if run_btn:
        if uploaded_resume is None:
            st.error("Please upload a resume file.")
            st.stop()
        if not jd_text.strip():
            st.error("Please provide a job description.")
            st.stop()

        resume_text = parse_file(uploaded_resume)

        with st.spinner("Scoring…"):
            before = scorer.score(resume_text, jd_text)

        with st.spinner("Optimising…"):
            opt             = optimizer.optimize(resume_text, jd_text)
            optimized_resume = opt["optimized_resume"]
            score_after      = opt["score_after"]

        # ── score header ──────────────────────────────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"{score_badge(before['overall_score'])} Original Score")
            st.metric("ATS Score", f"{before['overall_score']} / 100")
        with c2:
            delta = round(score_after - before["overall_score"], 1)
            st.subheader(f"{score_badge(score_after)} Optimised Score")
            st.metric("ATS Score", f"{score_after} / 100", delta=delta,
                      delta_color="normal" if delta >= 0 else "inverse")

        # ── breakdown ─────────────────────────────────────────────────────────
        st.subheader("Score breakdown")
        bd = before["breakdown"]
        label_map = {
            "keyword": "Keyword match", "semantic": "Semantic match",
            "sections": "Sections present", "achievement": "Achievement quality",
        }
        cols = st.columns(len(bd))
        for i, (k, v) in enumerate(bd.items()):
            cols[i].metric(label_map.get(k, k.title()), f"{v}%")

        # ── tabs ──────────────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Skills & Gaps", "Related Skills", "Semantic Analysis",
            "Suggestions", "Resume Text", "Diff",
        ])

        kw = before["keyword_report"]
        sr = before["semantic_report"]

        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### ✅ Matched skills")
                for s in sorted(kw["matched"]):
                    st.markdown(f"- `{s}`")
                if not kw["matched"]:
                    st.info("No direct matches found.")
            with col_b:
                st.markdown("#### ❌ Missing skills")
                for s in sorted(kw["missing"]):
                    st.markdown(f"- `{s}`")
                if not kw["missing"]:
                    st.success("No missing skills!")

            base = kw.get("base_coverage_pct", kw["coverage_pct"])
            if kw["coverage_pct"] > base:
                st.caption(
                    f"Base coverage: **{base}%** → "
                    f"Adjusted with partial credit: **{kw['coverage_pct']}%**"
                )
            for w in before["format_warnings"]:
                st.warning(w)

        with tab2:
            related = kw.get("related_matches", [])
            if not related:
                st.info("No related-technology matches. This appears when you have skills "
                        "conceptually similar to the JD — e.g. MAUI covering Android.")
            else:
                st.markdown("These are **not exact matches** but related technologies. "
                            "Each counts as **0.5 of a full match** in your score.")
                for item in related:
                    with st.container(border=True):
                        st.markdown(f"**JD requires:** `{item['jd_skill']}`")
                        st.markdown(f"**You have:** `{', '.join(item['related_resume_skills'])}`")
                        st.caption(f"Add a bullet explicitly mentioning "
                                   f"'{item['jd_skill']}' to turn this into a full match.")

        with tab3:
            st.markdown("#### Semantic similarity")
            c3a, c3b, c3c = st.columns(3)
            c3a.metric("Responsibility match", f"{sr['responsibility_score'] * 100:.0f}%")
            c3b.metric("Skill match",          f"{sr['skill_score'] * 100:.0f}%")

            # Education: show N/A clearly with explanation when not required
            edu_val = edu_display(sr)
            c3c.metric("Education match", edu_val)
            if not sr.get("education_required", True):
                st.caption(
                    "ℹ️ Education score shows **N/A** because this job description "
                    "does not mention a degree requirement. Your degree is still on "
                    "your resume and visible to the recruiter — it just wasn't "
                    "factored into the score since the JD didn't ask for it."
                )

            with st.expander("Full JSON"):
                st.json(sr)

        with tab4:
            st.markdown("#### Improvement suggestions")
            for i, s in enumerate(opt["suggestions"], 1):
                st.markdown(f"**{i}.** {s}")

        with tab5:
            c5a, c5b = st.columns(2)
            with c5a:
                st.markdown("#### Original")
                st.text_area("orig", resume_text,       height=500, label_visibility="collapsed")
            with c5b:
                st.markdown("#### Optimised")
                st.text_area("opt",  optimized_resume,  height=500, label_visibility="collapsed")

        with tab6:
            diff_text = make_unified_diff(resume_text, optimized_resume)
            if not diff_text.strip():
                st.info("No differences detected.")
            else:
                st.code(diff_text, language="diff")


# ═════════════════════════════════════════════════════════════════════════════
#  B2B  —  Recruiter / HR mode
# ═════════════════════════════════════════════════════════════════════════════
else:

    st.caption("Upload multiple CVs · paste or scrape a job description · get a ranked shortlist")

    # ── sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Recruiter inputs")

        uploaded_resumes = st.file_uploader(
            "Upload candidate resumes (multiple allowed)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )

        st.subheader("Job Description")
        job_url_b2b = st.text_input(
            "Paste job URL (optional)",
            placeholder="https://staff.am/... or https://linkedin.com/...",
            key="b2b_url",
        )
        if st.button("Fetch from URL", key="b2b_fetch"):
            if job_url_b2b.strip():
                with st.spinner("Fetching…"):
                    scraped = scrape_job_description(job_url_b2b)
                st.session_state.jd_text = scraped
                if any(scraped.startswith(p) for p in ("LinkedIn requires", "Error", "Could not", "Failed")):
                    st.warning(scraped)
                else:
                    st.success(f"Fetched {len(scraped.split())} words.")
            else:
                st.warning("Please enter a URL first.")

        jd_text_b2b = st.text_area(
            "Or paste / edit job description",
            value=st.session_state.jd_text,
            height=280,
            key="b2b_jd",
        )

        min_score = st.slider(
            "Minimum score threshold",
            min_value=0, max_value=100, value=50, step=5,
            help="Candidates below this score are flagged as 'Below threshold'.",
        )

        rank_btn = st.button("Rank Candidates", type="primary")

    # ── main panel ────────────────────────────────────────────────────────────
    if not rank_btn:
        # instructions when idle
        st.markdown("""
### How it works

1. **Upload** all candidate CVs on the left (PDF, DOCX, or TXT — as many as you need).
2. **Paste or scrape** the job description.
3. Click **Rank Candidates** to score every CV against the JD automatically.

You'll get:
- A ranked table with ATS score, keyword coverage, and semantic match per candidate
- Colour-coded pass / review / reject tiers
- A downloadable CSV of the full results
- Per-candidate drill-down with matched / missing skills and detailed breakdown
        """)

    elif not uploaded_resumes:
        st.error("Please upload at least one resume.")

    elif not jd_text_b2b.strip():
        st.error("Please provide a job description.")

    else:
        # ── score every resume ────────────────────────────────────────────────
        results: List[Dict] = []
        progress = st.progress(0, text="Scoring candidates…")

        for i, f in enumerate(uploaded_resumes):
            resume_text = parse_file(f)
            s = scorer.score(resume_text, jd_text_b2b)
            kw = s["keyword_report"]
            sr = s["semantic_report"]

            results.append({
                "filename":       f.name,
                "overall_score":  s["overall_score"],
                "keyword_pct":    kw["coverage_pct"],
                "semantic_pct":   round(sr["overall_similarity"] * 100, 1),
                "sections_pct":   s["breakdown"]["sections"],
                "achievement_pct":s["breakdown"]["achievement"],
                "matched_skills": ", ".join(kw["matched"]),
                "missing_skills": ", ".join(kw["missing"]),
                "related_skills": "; ".join(
                    f"{r['jd_skill']} ← {', '.join(r['related_resume_skills'])}"
                    for r in kw.get("related_matches", [])
                ),
                "education_score": edu_display(sr),
                "_resume_text": resume_text,      # kept for drill-down, not shown in table
                "_score_obj":   s,
            })
            progress.progress((i + 1) / len(uploaded_resumes),
                               text=f"Scored {i+1}/{len(uploaded_resumes)}: {f.name}")

        progress.empty()

        # Sort descending
        results.sort(key=lambda x: x["overall_score"], reverse=True)

        # ── summary metrics ───────────────────────────────────────────────────
        scores = [r["overall_score"] for r in results]
        above  = sum(1 for s in scores if s >= min_score)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Candidates scored",    len(results))
        m2.metric("Above threshold",       above,
                  help=f"Score ≥ {min_score}")
        m3.metric("Top score",            f"{max(scores):.1f}")
        m4.metric("Average score",        f"{sum(scores)/len(scores):.1f}")

        # ── ranked table ──────────────────────────────────────────────────────
        st.subheader("Ranked candidates")

        def tier(score):
            if score >= 75: return "✅ Strong match"
            if score >= min_score: return "🟡 Review"
            return "🔴 Below threshold"

        table_rows = []
        for rank, r in enumerate(results, 1):
            table_rows.append({
                "Rank":            rank,
                "Candidate":       r["filename"],
                "Score":           r["overall_score"],
                "Tier":            tier(r["overall_score"]),
                "Keyword %":       r["keyword_pct"],
                "Semantic %":      r["semantic_pct"],
                "Matched skills":  r["matched_skills"],
                "Missing skills":  r["missing_skills"],
            })

        df = pd.DataFrame(table_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ── CSV download ──────────────────────────────────────────────────────
        csv_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
        csv_df   = pd.DataFrame(csv_rows)
        csv_buf  = io.StringIO()
        csv_df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download full results CSV",
            data=csv_buf.getvalue(),
            file_name="ats_ranking.csv",
            mime="text/csv",
        )

        # ── per-candidate drill-down ──────────────────────────────────────────
        st.subheader("Candidate drill-down")
        names = [r["filename"] for r in results]
        chosen = st.selectbox("Select a candidate to inspect", names)

        if chosen:
            r  = next(x for x in results if x["filename"] == chosen)
            s  = r["_score_obj"]
            kw = s["keyword_report"]
            sr = s["semantic_report"]

            rank_num = next(i+1 for i, x in enumerate(results) if x["filename"] == chosen)
            st.markdown(f"**Rank #{rank_num}** · Score: **{r['overall_score']} / 100** "
                        f"· {tier(r['overall_score'])}")

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Keyword match",     f"{kw['coverage_pct']}%")
            d2.metric("Semantic match",    f"{sr['overall_similarity']*100:.0f}%")
            d3.metric("Sections present",  f"{s['breakdown']['sections']}%")
            d4.metric("Achievement score", f"{s['breakdown']['achievement']}%")

            dc1, dc2 = st.columns(2)
            with dc1:
                st.markdown("**✅ Matched skills**")
                for sk in sorted(kw["matched"]):
                    st.markdown(f"- `{sk}`")
                if not kw["matched"]:
                    st.info("No direct matches.")
            with dc2:
                st.markdown("**❌ Missing skills**")
                for sk in sorted(kw["missing"]):
                    st.markdown(f"- `{sk}`")
                if not kw["missing"]:
                    st.success("All JD skills present.")

            related = kw.get("related_matches", [])
            if related:
                st.markdown("**🔁 Related skills (partial credit)**")
                for item in related:
                    st.markdown(
                        f"- JD needs `{item['jd_skill']}` · "
                        f"candidate has `{', '.join(item['related_resume_skills'])}`"
                    )

            # Education note
            if not sr.get("education_required", True):
                st.caption(
                    "ℹ️ Education score: **N/A** — the JD does not state a degree requirement."
                )
            else:
                st.caption(f"Education match: **{edu_display(sr)}**")

            with st.expander("Full resume text"):
                st.text(r["_resume_text"])
