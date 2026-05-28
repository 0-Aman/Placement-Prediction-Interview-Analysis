from __future__ import annotations

from placement_ai.features import missing_skills, resume_score, skill_match_score
from placement_ai.schemas import StudentProfile


def build_suggestions(profile: StudentProfile) -> list[str]:
    suggestions: list[str] = []
    score = resume_score(**profile.model_dump())
    skill_score = skill_match_score(profile.skills)
    gaps = missing_skills(profile.skills, limit=5)

    if score < 60:
        suggestions.append("Add two end-to-end projects with data cleaning, model evaluation, and deployment.")
    if skill_score < 65:
        suggestions.append(f"Strengthen missing ML engineer skills: {', '.join(gaps)}.")
    if profile.internships < 1:
        suggestions.append("Apply for internships or open-source tasks to show practical experience.")
    if profile.projects < 3:
        suggestions.append("Build at least three portfolio projects and include GitHub links in the resume.")
    if profile.aptitude_score < 70:
        suggestions.append("Practice aptitude topics weekly: probability, statistics, logical reasoning, and SQL.")
    if profile.communication_score < 70:
        suggestions.append("Practice mock interviews and explain every project with problem, approach, metrics, and impact.")
    if "fastapi" not in {skill.lower() for skill in profile.skills}:
        suggestions.append("Deploy one model with FastAPI so your resume demonstrates model serving.")

    return suggestions[:6] or ["Profile is strong. Focus on interview practice and measurable project outcomes."]


def interview_suggestions(confidence_score: float, face_detected: bool, signals: dict[str, float | int | bool]) -> list[str]:
    suggestions: list[str] = []
    if not face_detected:
        return ["Keep your face centered in the frame and ensure the camera is at eye level."]
    if signals.get("brightness", 0) < 70:
        suggestions.append("Improve lighting so your face is clearly visible.")
    if signals.get("sharpness", 0) < 45:
        suggestions.append("Keep the camera steady and avoid sitting too close or too far.")
    if signals.get("face_area_ratio", 0) < 0.06:
        suggestions.append("Move slightly closer to the camera and keep shoulders relaxed.")
    if confidence_score < 65:
        suggestions.append("Practice answers with a steady gaze, slower pace, and clear project explanations.")
    return suggestions or ["Good camera presence. Keep answers concise and back claims with project metrics."]
