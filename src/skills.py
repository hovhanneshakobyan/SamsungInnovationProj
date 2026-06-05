from __future__ import annotations

import re
from typing import Dict, List, Set


class SkillExtractor:
    """
    General-purpose skill extractor with:
    - normalisation & alias mapping
    - phrase-first matching
    - light typo repair
    - concept grouping for related technologies (partial-credit scoring)
    """

    def __init__(self):
        # canonical name → list of surface forms
        self.skill_aliases: Dict[str, List[str]] = {
            # .NET ecosystem
            "c#":               ["c#", "c sharp", "c-sharp"],
            ".net":             [".net", "dotnet", "dot net"],
            "asp.net":          ["asp.net", "asp net", "aspnet"],
            "entity framework": ["entity framework", "ef core", "entityframework"],
            "wpf":              ["wpf", "windows presentation foundation"],
            "maui":             ["maui", ".net maui", "dotnet maui"],
            "xamarin":          ["xamarin"],
            "blazor":           ["blazor"],
            # Mobile
            "android":          ["android", "android development", "android developer"],
            "kotlin":           ["kotlin"],
            "java":             ["java"],
            "swift":            ["swift"],
            "ios":              ["ios", "ios development"],
            "react native":     ["react native", "reactnative"],
            "flutter":          ["flutter"],
            # Web frontend
            "javascript":       ["javascript", "java script", "js", "javasript"],
            "typescript":       ["typescript", "ts"],
            "html5":            ["html5", "html"],
            "css3":             ["css3", "css"],
            "react":            ["react", "reactjs", "react.js"],
            "vue":              ["vue", "vuejs", "vue.js"],
            "angular":          ["angular", "angularjs"],
            # Web backend / APIs
            "rest api":         ["rest api", "restful api", "restful apis", "api development"],
            "graphql":          ["graphql"],
            "node.js":          ["node.js", "nodejs", "node"],
            "django":           ["django"],
            "flask":            ["flask"],
            # Databases
            "sql":              ["sql"],
            "sql server":       ["sql server", "mssql"],
            "postgresql":       ["postgresql", "postgres"],
            "mysql":            ["mysql"],
            "nosql":            ["nosql", "no sql"],
            "mongodb":          ["mongodb", "mongo"],
            "firebase":         ["firebase", "fire base"],
            # DevOps / cloud
            "azure devops":     ["azure devops", "azuredevops"],
            "git":              ["git", "github", "gitlab"],
            "docker":           ["docker"],
            "kubernetes":       ["kubernetes", "k8s"],
            "aws":              ["aws", "amazon web services"],
            "azure":            ["azure", "microsoft azure"],
            # Languages
            "python":           ["python"],
            "c++":              ["c++", "cpp"],
            "go":               ["golang", "go language"],
            "rust":             ["rust"],
            # Practices
            "unit testing":     ["unit testing", "automated testing", "xunit", "nunit", "junit"],
            "oop":              ["object-oriented", "object oriented", "oop"],
            "design patterns":  ["design patterns", "software design patterns"],
            "full stack":       ["full stack", "full-stack", "fullstack"],
            "desktop ui":       ["desktop applications", "desktop ui", "desktop development"],
            "web ui":           ["web applications", "web ui", "frontend", "front-end"],
            "agile":            ["agile", "scrum", "kanban"],
            # Data / ML
            "tensorflow":       ["tensorflow"],
            "pandas":           ["pandas"],
            "selenium":         ["selenium"],
        }

        # Conceptual relatedness graph — used for partial-credit scoring.
        # Key: a JD skill.  Value: set of resume skills that cover it partially.
        # Relationships are bidirectional so both directions are listed.
        self.related_concepts: Dict[str, Set[str]] = {
            # Cross-platform / mobile ←→ Android
            "android":      {"maui", "xamarin", "kotlin", "java", "flutter", "react native"},
            "kotlin":       {"android", "java", "maui", "xamarin"},
            "java":         {"kotlin", "android", "c#", ".net"},
            "ios":          {"swift", "maui", "xamarin", "flutter", "react native"},
            "swift":        {"ios", "maui", "xamarin"},
            "maui":         {"android", "ios", "xamarin", ".net", "c#"},
            "xamarin":      {"android", "ios", "maui", ".net", "c#"},
            "react native": {"android", "ios", "javascript", "react"},
            "flutter":      {"android", "ios", "dart"},
            # .NET ecosystem
            ".net":         {"c#", "asp.net", "maui", "wpf", "blazor"},
            "c#":           {".net", "asp.net", "java", "kotlin"},
            "asp.net":      {".net", "c#", "sql", "web ui", "rest api"},
            "wpf":          {"desktop ui", "maui", ".net", "c#"},
            "blazor":       {".net", "c#", "web ui", "javascript"},
            # Web
            "web ui":       {"html5", "css3", "javascript", "typescript", "react",
                             "angular", "vue", "asp.net", "blazor"},
            "desktop ui":   {"wpf", "maui", "electron"},
            "react":        {"javascript", "typescript", "web ui"},
            "angular":      {"javascript", "typescript", "web ui"},
            "vue":          {"javascript", "typescript", "web ui"},
            "javascript":   {"typescript", "react", "angular", "vue", "node.js"},
            "typescript":   {"javascript", "react", "angular", "vue"},
            # Backend / APIs
            "rest api":     {"asp.net", "c#", ".net", "python", "node.js", "java"},
            "graphql":      {"rest api", "node.js", "javascript"},
            "node.js":      {"javascript", "typescript", "rest api"},
            "django":       {"python", "rest api", "web ui"},
            "flask":        {"python", "rest api"},
            # Databases
            "sql":          {"sql server", "postgresql", "mysql"},
            "sql server":   {"sql", "postgresql", "mysql"},
            "postgresql":   {"sql", "sql server", "mysql"},
            "mysql":        {"sql", "postgresql", "sql server"},
            "nosql":        {"mongodb", "firebase"},
            "mongodb":      {"nosql", "firebase"},
            "firebase":     {"nosql", "mongodb"},
            # OOP / patterns
            "oop":          {"c#", "java", "kotlin", "python", "c++", "design patterns"},
            "design patterns": {"oop", "c#", "java", "c++"},
            # DevOps
            "azure devops": {"git", "docker", "kubernetes", "azure"},
            "docker":       {"kubernetes", "azure devops"},
            "kubernetes":   {"docker", "azure devops"},
            "azure":        {"azure devops", "aws"},
            "aws":          {"azure", "docker", "kubernetes"},
            # Full stack
            "full stack":   {"html5", "css3", "javascript", "typescript",
                             "asp.net", "sql", "rest api"},
            # Testing
            "unit testing": {"selenium", "oop"},
            # Agile
            "agile":        {"azure devops", "git"},
        }

    # ------------------------------------------------------------------
    def normalize_text(self, text: str) -> str:
        text = text.lower()
        # common typos / alternate spellings
        for src, dst in [
            ("javasript", "javascript"),
            ("java script", "javascript"),
            ("asp net", "asp.net"),
            ("dotnet", ".net"),
            ("dot net", ".net"),
            ("entityframework", "entity framework"),
            ("restful apis", "rest api"),
            ("restful api", "rest api"),
            ("object oriented", "object-oriented"),
            ("react.js", "react"),
            ("vue.js", "vue"),
            ("node.js", "node.js"),   # keep
        ]:
            text = text.replace(src, dst)
        text = re.sub(r"[^a-z0-9+#./\-\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------------------------------------------------------
    def extract_skills(self, text: str) -> List[str]:
        norm = self.normalize_text(text)
        found = []

        # phrase-first: longest aliases matched first so "react native" beats "react"
        for canonical, aliases in sorted(
            self.skill_aliases.items(),
            key=lambda x: max(len(a) for a in x[1]),
            reverse=True,
        ):
            if any(alias in norm for alias in aliases):
                found.append(canonical)

        # dedupe preserving order, remove weaker overlaps
        seen: Set[str] = set()
        result = []
        for s in found:
            if s not in seen:
                seen.add(s)
                result.append(s)
        return result

    # ------------------------------------------------------------------
    def jd_skill_report(
        self, resume_text: str, jd_text: str
    ) -> Dict:
        resume_skills = set(self.extract_skills(resume_text))
        jd_skills     = set(self.extract_skills(jd_text))

        matched  = sorted(resume_skills & jd_skills)
        missing  = sorted(jd_skills - resume_skills)
        coverage = len(matched) / max(len(jd_skills), 1) * 100.0

        related_matches = []
        for jd_skill in sorted(missing):
            related = self.find_related_resume_skills(jd_skill, resume_skills)
            if related:
                related_matches.append({
                    "jd_skill": jd_skill,
                    "related_resume_skills": related,
                })

        return {
            "resume_skills": sorted(resume_skills),
            "jd_skills":     sorted(jd_skills),
            "matched":       matched,
            "missing":       missing,
            "related_matches": related_matches,
            "coverage_pct":  round(coverage, 1),
        }

    # ------------------------------------------------------------------
    def find_related_resume_skills(
        self, jd_skill: str, resume_skills: Set[str]
    ) -> List[str]:
        related = self.related_concepts.get(jd_skill, set())
        return sorted(s for s in resume_skills if s in related)

    # ------------------------------------------------------------------
    def relatedness_score(self, left: str, right: str) -> float:
        """
        Structured skill-level relatedness:
          1.0  exact canonical match
          0.7  directly related concept
          0.0  no known relationship
        """
        left_skills  = self.extract_skills(left)
        right_skills = self.extract_skills(right)

        if not left_skills or not right_skills:
            return 0.0

        if set(left_skills) & set(right_skills):
            return 1.0

        for ls in left_skills:
            rel = self.related_concepts.get(ls, set())
            if any(r in rel for r in right_skills):
                return 0.7

        for rs in right_skills:
            rel = self.related_concepts.get(rs, set())
            if any(l in rel for l in left_skills):
                return 0.7

        return 0.0
