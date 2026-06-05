"""
generate_synthetic_data.py
Generates synthetic training data for all three models:

  data/processed/pairs.csv              → Siamese model  (resume, jd, label)
  data/processed/optimized_pairs.csv    → T5 optimizer   (resume, jd, optimized_resume)
  data/processed/generation_pairs.csv   → T5 generator   (prompt → full resume)
"""

import os
import random
import pandas as pd
from loguru import logger

random.seed(42)

# ── Domain templates ───────────────────────────────────────────────────────────
DOMAINS = {
    "software_engineer": {
        "skills":  ["Python", "Java", "C++", "Docker", "Kubernetes", "REST API",
                    "Git", "CI/CD", "PostgreSQL", "Redis", "React", "Node.js"],
        "titles":  ["Software Engineer", "Backend Developer", "Full-Stack Developer",
                    "Software Developer", "Platform Engineer"],
        "jd_verbs": ["develop", "maintain", "design", "implement", "optimize"],
        "bullets": [
            "Developed and maintained REST APIs serving 1M+ daily requests",
            "Reduced deployment time by 40% through CI/CD pipeline automation",
            "Designed microservices architecture improving system scalability",
            "Optimized PostgreSQL queries reducing average response time by 35%",
            "Led migration from monolith to containerized Docker services",
            "Implemented unit and integration tests achieving 90% code coverage",
        ],
    },
    "data_scientist": {
        "skills":  ["Python", "PyTorch", "TensorFlow", "scikit-learn", "SQL",
                    "Pandas", "NumPy", "Spark", "Tableau", "A/B testing",
                    "NLP", "Computer Vision", "MLflow"],
        "titles":  ["Data Scientist", "ML Engineer", "AI Researcher",
                    "Applied Scientist", "Research Engineer"],
        "jd_verbs": ["build", "train", "analyze", "research", "deploy"],
        "bullets": [
            "Built and deployed machine learning models improving prediction accuracy by 25%",
            "Analyzed large datasets using Spark and Pandas to extract business insights",
            "Designed A/B testing framework that increased conversion rate by 18%",
            "Trained NLP models for text classification achieving 94% accuracy",
            "Created interactive dashboards in Tableau for executive reporting",
            "Automated data pipelines reducing manual processing time by 60%",
        ],
    },
    "devops": {
        "skills":  ["AWS", "Azure", "GCP", "Terraform", "Ansible", "Jenkins",
                    "Docker", "Kubernetes", "Linux", "Bash", "Prometheus", "Grafana"],
        "titles":  ["DevOps Engineer", "SRE", "Cloud Engineer",
                    "Infrastructure Engineer", "Platform Engineer"],
        "jd_verbs": ["automate", "manage", "monitor", "deploy", "scale"],
        "bullets": [
            "Managed AWS infrastructure for 50+ microservices using Terraform",
            "Reduced system downtime by 70% through proactive Prometheus monitoring",
            "Automated server provisioning with Ansible saving 20 hours per week",
            "Implemented Kubernetes cluster supporting 200+ pods in production",
            "Built Jenkins CI/CD pipelines reducing release cycles from weeks to hours",
            "Improved cloud cost efficiency by 30% through resource optimization",
        ],
    },
    "product_manager": {
        "skills":  ["roadmapping", "stakeholder management", "Agile", "Scrum",
                    "JIRA", "user research", "A/B testing", "SQL", "Figma",
                    "OKRs", "PRD writing", "market analysis"],
        "titles":  ["Product Manager", "Senior PM", "Product Lead",
                    "Group Product Manager", "Technical PM"],
        "jd_verbs": ["drive", "prioritize", "define", "collaborate", "launch"],
        "bullets": [
            "Defined and executed product roadmap resulting in 40% revenue growth",
            "Led cross-functional teams of 15+ engineers, designers, and analysts",
            "Conducted 50+ user interviews to identify key pain points and opportunities",
            "Launched 3 major features that increased DAU by 25%",
            "Prioritized backlog using OKR framework aligning with company strategy",
            "Collaborated with engineering to define technical requirements and PRDs",
        ],
    },
}

COMPANIES    = ["Acme Corp", "TechNova", "DataSphere", "CloudBase",
                "BuildIt", "ScaleUp", "NexGen", "AlphaWorks", "MetaSoft", "ByteCore"]
UNIVERSITIES = ["MIT", "Stanford", "Carnegie Mellon", "UC Berkeley",
                "Georgia Tech", "University of Waterloo", "ETH Zurich", "Columbia University"]
DEGREES      = ["BSc Computer Science", "MSc Data Science", "BSc Software Engineering",
                "MSc Artificial Intelligence", "BSc Mathematics", "MSc Computer Science"]
NAMES        = ["Alex Johnson", "Maria Garcia", "Wei Chen", "Priya Patel",
                "Daniel Kim", "Sophie Müller", "James Wilson", "Aisha Okafor",
                "Carlos Rivera", "Yuki Tanaka"]


def _pick(lst, n=3):
    return random.sample(lst, min(n, len(lst)))

def _years():
    return random.randint(1, 10)


# ── Resume builder ────────────────────────────────────────────────────────────
def make_resume(domain: str) -> str:
    cfg     = DOMAINS[domain]
    title   = random.choice(cfg["titles"])
    skills  = _pick(cfg["skills"], 5)
    name    = random.choice(NAMES)
    company = random.choice(COMPANIES)
    uni     = random.choice(UNIVERSITIES)
    degree  = random.choice(DEGREES)
    yrs     = _years()
    bullets = _pick(cfg["bullets"], 3)

    return (
        f"{name}\n"
        f"Email: {name.lower().replace(' ', '.')}@email.com  |  "
        f"LinkedIn: linkedin.com/in/{name.lower().replace(' ', '-')}\n\n"
        f"SUMMARY\n"
        f"Experienced {title} with {yrs}+ years of industry experience "
        f"specializing in {skills[0]} and {skills[1]}.\n\n"
        f"EXPERIENCE\n"
        f"{title} — {company}  ({2024 - yrs}–Present)\n"
        + "\n".join(f"• {b}" for b in bullets) + "\n\n"
        f"EDUCATION\n"
        f"{degree} — {uni}\n\n"
        f"SKILLS\n"
        f"{', '.join(skills)}\n"
    )


# ── JD builder ────────────────────────────────────────────────────────────────
def make_jd(domain: str, match: bool = True) -> str:
    cfg     = DOMAINS[domain]
    title   = random.choice(cfg["titles"])
    verb    = random.choice(cfg["jd_verbs"])
    company = random.choice(COMPANIES)

    if match:
        skills = _pick(cfg["skills"], 4)
    else:
        other  = random.choice([d for d in DOMAINS if d != domain])
        skills = _pick(DOMAINS[other]["skills"], 4)
        title  = random.choice(DOMAINS[other]["titles"])

    return (
        f"{company} is hiring a {title}.\n\n"
        f"Responsibilities:\n"
        f"• {verb.capitalize()} scalable systems and services.\n"
        f"• Work closely with engineering and product teams.\n"
        f"• Mentor junior team members.\n\n"
        f"Requirements:\n"
        f"• 3+ years of experience as a {title}.\n"
        f"• Proficiency in {', '.join(skills[:2])}.\n"
        f"• Strong knowledge of {', '.join(skills[2:])}.\n"
        f"• Excellent communication skills.\n"
    )


# ── Optimized resume (T5 optimizer target) ────────────────────────────────────
def make_optimized_resume(resume: str, jd: str, domain: str) -> str:
    cfg         = DOMAINS[domain]
    jd_keywords = _pick(cfg["skills"], 4)
    extra       = f"\nKey skills matching this role: {', '.join(jd_keywords)}.\n"
    return resume.rstrip() + extra


# ── Generation prompt + target (T5 generator training) ────────────────────────
def make_generation_pair(domain: str) -> dict:
    """
    Creates a (prompt → full resume) pair for T5 generation training.
    The prompt uses the same format as resume_generator.py's _build_prompt().
    The target is the full structured resume text (gold label).
    """
    cfg     = DOMAINS[domain]
    title   = random.choice(cfg["titles"])
    skills  = _pick(cfg["skills"], 5)
    name    = random.choice(NAMES)
    company = random.choice(COMPANIES)
    uni     = random.choice(UNIVERSITIES)
    degree  = random.choice(DEGREES)
    yrs     = _years()
    bullets = _pick(cfg["bullets"], 3)
    email   = f"{name.lower().replace(' ', '.')}@email.com"
    linkedin = f"linkedin.com/in/{name.lower().replace(' ', '-')}"

    # ── Build prompt (matches _build_prompt() in resume_generator.py) ────────
    exp_str = (f"{title} at {company} ({2024-yrs}–Present): "
               + "; ".join(bullets))
    edu_str = f"{degree} from {uni} ({2024 - yrs - 4})"

    prompt = (
        f"generate resume: "
        f"name: {name} "
        f"title: {title} "
        f"email: {email} "
        f"phone: +1 555 000 0000 "
        f"location: San Francisco, CA "
        f"linkedin: {linkedin} "
        f"years: {yrs} "
        f"summary: Experienced {title} with {yrs}+ years in {skills[0]} and {skills[1]}. "
        f"skills: {', '.join(skills)} "
        f"experience: {exp_str} | "
        f"education: {edu_str} "
        f"certifications: "
    )

    # ── Build target (full formatted resume) ─────────────────────────────────
    target = (
        f"{name.upper()}\n"
        f"{title}\n"
        f"{email}  |  +1 555 000 0000  |  San Francisco, CA  |  {linkedin}\n\n"
        f"SUMMARY\n"
        f"Experienced {title} with {yrs}+ years of industry experience "
        f"specializing in {skills[0]} and {skills[1]}.\n\n"
        f"SKILLS\n"
        f"{', '.join(skills)}\n\n"
        f"EXPERIENCE\n"
        f"{title}  —  {company}  ({2024 - yrs}–Present)\n"
        + "\n".join(f"• {b}" for b in bullets) + "\n\n"
        f"EDUCATION\n"
        f"{degree}  —  {uni}  ({2024 - yrs - 4})\n"
    )

    return {"prompt": prompt, "target": target}


# ── Main generate function ────────────────────────────────────────────────────
def generate(n_per_domain: int = 500, out_dir: str = "data/processed"):
    os.makedirs(out_dir, exist_ok=True)

    siamese_rows   = []
    optimizer_rows = []
    generation_rows = []

    for domain in DOMAINS:
        logger.info(f"Generating data for domain: {domain}")
        for _ in range(n_per_domain):
            resume = make_resume(domain)
            jd_pos = make_jd(domain, match=True)
            jd_neg = make_jd(domain, match=False)

            # Siamese pairs
            siamese_rows.append({"resume_text": resume, "jd_text": jd_pos, "label": 1})
            siamese_rows.append({"resume_text": resume, "jd_text": jd_neg, "label": 0})

            # T5 optimizer pairs
            optimized = make_optimized_resume(resume, jd_pos, domain)
            optimizer_rows.append({
                "resume_text":      resume,
                "jd_text":          jd_pos,
                "optimized_resume": optimized,
            })

            # T5 generation pairs (prompt → full resume)
            gen_pair = make_generation_pair(domain)
            generation_rows.append(gen_pair)

    # Shuffle and save
    df_siamese    = pd.DataFrame(siamese_rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df_opt        = pd.DataFrame(optimizer_rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df_generation = pd.DataFrame(generation_rows).sample(frac=1, random_state=42).reset_index(drop=True)

    df_siamese.to_csv(   os.path.join(out_dir, "pairs.csv"),            index=False)
    df_opt.to_csv(       os.path.join(out_dir, "optimized_pairs.csv"),  index=False)
    df_generation.to_csv(os.path.join(out_dir, "generation_pairs.csv"), index=False)

    logger.success(f"Siamese pairs    → pairs.csv            ({len(df_siamese)} rows)")
    logger.success(f"Optimizer pairs  → optimized_pairs.csv  ({len(df_opt)} rows)")
    logger.success(f"Generation pairs → generation_pairs.csv ({len(df_generation)} rows)")
    return df_siamese, df_opt, df_generation


if __name__ == "__main__":
    generate(n_per_domain=500)
