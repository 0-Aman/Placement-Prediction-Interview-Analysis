from placement_ai.features import missing_skills, resume_score, skill_match_score


def test_skill_match_score_counts_required_skills() -> None:
    score = skill_match_score(["python", "sql", "fastapi", "pandas"])
    assert 0 < score < 100


def test_missing_skills_excludes_present_skills() -> None:
    gaps = missing_skills(["python", "sql", "fastapi"], limit=10)
    assert "python" not in gaps
    assert "sql" not in gaps


def test_resume_score_is_bounded() -> None:
    score = resume_score(
        cgpa=10,
        internships=10,
        projects=30,
        certifications=20,
        hackathons=20,
        aptitude_score=100,
        communication_score=100,
        skills=["python", "sql", "fastapi", "pandas", "machine learning"],
        resume_text="Deployed a FastAPI model with accuracy and F1 metrics.",
    )
    assert 0 <= score <= 100
