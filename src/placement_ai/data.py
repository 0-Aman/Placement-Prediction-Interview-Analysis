from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from placement_ai.features import CORE_SKILLS, normalize_text, skills_to_text


BRANCHES = ["CSE", "IT", "ECE", "EEE", "ME", "Civil", "AI&DS"]
DEGREES = ["B.Tech", "M.Tech", "BCA", "MCA", "B.Sc"]


def generate_synthetic_students(rows: int = 1200, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    skill_pool = sorted(CORE_SKILLS | {"excel", "tableau", "power bi", "java", "cloud", "linux"})
    records = []

    for _ in range(rows):
        cgpa = float(np.clip(rng.normal(7.2, 1.05), 4.0, 10.0))
        internships = int(np.clip(rng.poisson(1.2), 0, 6))
        projects = int(np.clip(rng.poisson(3.0), 0, 12))
        certifications = int(np.clip(rng.poisson(2.0), 0, 10))
        hackathons = int(np.clip(rng.poisson(1.0), 0, 8))
        aptitude_score = float(np.clip(rng.normal(68, 14), 20, 100))
        communication_score = float(np.clip(rng.normal(65, 15), 20, 100))
        skill_count = int(np.clip(rng.normal(7, 2.4), 2, 16))
        skills = sorted(rng.choice(skill_pool, size=skill_count, replace=False).tolist())
        resume_text = (
            f"Built projects using {', '.join(skills[:6])}. "
            f"Completed {projects} projects, {internships} internships, and worked on model evaluation."
        )

        ml_skill_hits = len(set(skills) & CORE_SKILLS)
        logit = (
            -11.0
            + 0.55 * cgpa
            + 0.55 * internships
            + 0.28 * projects
            + 0.16 * certifications
            + 0.22 * hackathons
            + 0.028 * aptitude_score
            + 0.025 * communication_score
            + 0.22 * ml_skill_hits
        )
        probability = 1 / (1 + np.exp(-logit))
        placed = int(rng.random() < probability)

        records.append(
            {
                "cgpa": round(cgpa, 2),
                "internships": internships,
                "projects": projects,
                "certifications": certifications,
                "hackathons": hackathons,
                "aptitude_score": round(aptitude_score, 1),
                "communication_score": round(communication_score, 1),
                "branch": rng.choice(BRANCHES),
                "degree": rng.choice(DEGREES),
                "skills": ", ".join(skills),
                "resume_text": resume_text,
                "placed": placed,
            }
        )

    return pd.DataFrame(records)


def clean_student_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    numeric_defaults = {
        "cgpa": cleaned.get("cgpa", pd.Series(dtype=float)).median(),
        "internships": 0,
        "projects": 0,
        "certifications": 0,
        "hackathons": 0,
        "aptitude_score": 0,
        "communication_score": 0,
    }

    for column, default in numeric_defaults.items():
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").fillna(default)

    cleaned["cgpa"] = cleaned["cgpa"].clip(0, 10)
    for column in ["aptitude_score", "communication_score"]:
        cleaned[column] = cleaned[column].clip(0, 100)

    for column in ["branch", "degree", "skills", "resume_text"]:
        cleaned[column] = cleaned[column].fillna("").astype(str).map(normalize_text)

    cleaned["skills_text"] = cleaned["skills"].map(skills_to_text)
    return cleaned


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return clean_student_frame(pd.read_csv(path))


def save_synthetic_dataset(path: str | Path, rows: int = 1200, random_state: int = 42) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_synthetic_students(rows=rows, random_state=random_state)
    frame.to_csv(path, index=False)
    return path
