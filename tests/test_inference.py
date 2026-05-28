from placement_ai.inference import score_without_model
from placement_ai.schemas import StudentProfile


def test_score_without_model_accepts_full_student_profile() -> None:
    profile = StudentProfile(
        cgpa=8.1,
        internships=1,
        projects=4,
        certifications=2,
        hackathons=1,
        aptitude_score=75,
        communication_score=72,
        branch="CSE",
        degree="B.Tech",
        skills=["python", "sql", "fastapi"],
        resume_text="Deployed a FastAPI model with evaluation metrics.",
    )

    prediction = score_without_model(profile)

    assert 0 <= prediction.placement_probability <= 1
    assert prediction.resume_score > 0
