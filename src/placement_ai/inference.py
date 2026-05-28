from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib

from placement_ai.features import missing_skills, resume_score, skill_match_score
from placement_ai.schemas import PlacementPrediction, StudentProfile
from placement_ai.suggestions import build_suggestions
from placement_ai.train import profile_to_frame

DEFAULT_MODEL_PATH = Path("models/placement_model.joblib")


@lru_cache(maxsize=2)
def load_model(model_path: str = str(DEFAULT_MODEL_PATH)):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Run scripts/train_model.py before serving predictions."
        )
    return joblib.load(path)


def predict_placement(profile: StudentProfile, model_path: str = str(DEFAULT_MODEL_PATH)) -> PlacementPrediction:
    model = load_model(model_path)
    frame = profile_to_frame(profile.model_dump())
    probability = float(model.predict_proba(frame)[0][1])
    return PlacementPrediction(
        placement_probability=round(probability, 4),
        predicted_label=int(probability >= 0.5),
        resume_score=resume_score(**profile.model_dump()),
        skill_match_score=skill_match_score(profile.skills),
        top_missing_skills=missing_skills(profile.skills),
        suggestions=build_suggestions(profile),
    )


def score_without_model(profile: StudentProfile) -> PlacementPrediction:
    score = resume_score(**profile.model_dump())
    probability = min(max(score / 100, 0), 1)
    return PlacementPrediction(
        placement_probability=round(probability, 4),
        predicted_label=int(probability >= 0.6),
        resume_score=score,
        skill_match_score=skill_match_score(profile.skills),
        top_missing_skills=missing_skills(profile.skills),
        suggestions=build_suggestions(profile),
    )
