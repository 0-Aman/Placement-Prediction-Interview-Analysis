from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class StudentProfile(BaseModel):
    cgpa: float = Field(..., ge=0, le=10)
    internships: int = Field(..., ge=0, le=10)
    projects: int = Field(..., ge=0, le=30)
    certifications: int = Field(..., ge=0, le=30)
    hackathons: int = Field(..., ge=0, le=30)
    aptitude_score: float = Field(..., ge=0, le=100)
    communication_score: float = Field(..., ge=0, le=100)
    branch: str = Field(..., min_length=1)
    degree: str = Field(..., min_length=1)
    skills: list[str] = Field(default_factory=list)
    resume_text: str = ""

    @field_validator("skills", mode="before")
    @classmethod
    def split_skill_string(cls, value: object) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value  # type: ignore[return-value]


class PlacementPrediction(BaseModel):
    placement_probability: float
    predicted_label: int
    resume_score: float
    skill_match_score: float
    top_missing_skills: list[str]
    suggestions: list[str]


class InterviewAnalysis(BaseModel):
    face_detected: bool
    confidence_score: float
    emotion_label: str
    signals: dict[str, float | int | bool]
    suggestions: list[str]
