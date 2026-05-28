from __future__ import annotations

import re
from collections.abc import Iterable


CORE_SKILLS = {
    "python",
    "sql",
    "machine learning",
    "data cleaning",
    "feature engineering",
    "scikit-learn",
    "xgboost",
    "pandas",
    "numpy",
    "fastapi",
    "streamlit",
    "opencv",
    "nlp",
    "deep learning",
    "model deployment",
    "git",
    "docker",
    "statistics",
}

ROLE_SKILLS = {
    "ml_engineer": CORE_SKILLS,
    "data_analyst": {"python", "sql", "pandas", "statistics", "excel", "power bi", "tableau"},
    "backend_engineer": {"python", "sql", "fastapi", "docker", "git", "apis", "databases"},
}


def normalize_text(text: str | None) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_skills(skills: Iterable[str] | str | None) -> list[str]:
    if skills is None:
        return []
    if isinstance(skills, str):
        skills = re.split(r"[,|;/]", skills)
    cleaned = {normalize_text(skill) for skill in skills if normalize_text(skill)}
    return sorted(cleaned)


def skills_to_text(skills: Iterable[str] | str | None) -> str:
    return " ".join(normalize_skills(skills))


def skill_match_score(skills: Iterable[str] | str | None, role: str = "ml_engineer") -> float:
    normalized = set(normalize_skills(skills))
    required = ROLE_SKILLS.get(role, CORE_SKILLS)
    if not required:
        return 0.0
    matched = sum(1 for skill in required if skill in normalized)
    return round((matched / len(required)) * 100, 2)


def missing_skills(
    skills: Iterable[str] | str | None,
    role: str = "ml_engineer",
    limit: int = 6,
) -> list[str]:
    normalized = set(normalize_skills(skills))
    required = ROLE_SKILLS.get(role, CORE_SKILLS)
    return sorted(skill for skill in required if skill not in normalized)[:limit]


def resume_score(
    *,
    cgpa: float,
    internships: int,
    projects: int,
    certifications: int,
    hackathons: int,
    aptitude_score: float,
    communication_score: float,
    skills: Iterable[str] | str | None,
    resume_text: str = "",
    **_: object,
) -> float:
    text = normalize_text(resume_text)
    has_deployment = any(token in text for token in ["deploy", "api", "fastapi", "streamlit", "docker"])
    has_metrics = any(token in text for token in ["accuracy", "f1", "roc", "precision", "recall", "cross validation"])

    score = 0.0
    score += min(max(cgpa, 0), 10) * 4.0
    score += min(internships, 4) * 6.0
    score += min(projects, 6) * 4.0
    score += min(certifications, 5) * 2.0
    score += min(hackathons, 4) * 2.0
    score += min(max(aptitude_score, 0), 100) * 0.10
    score += min(max(communication_score, 0), 100) * 0.10
    score += skill_match_score(skills) * 0.20
    score += 4.0 if has_deployment else 0.0
    score += 3.0 if has_metrics else 0.0
    return round(min(score, 100.0), 2)
