from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile

from placement_ai.inference import DEFAULT_MODEL_PATH, predict_placement, score_without_model
from placement_ai.interview import analyze_image_bytes
from placement_ai.schemas import InterviewAnalysis, PlacementPrediction, StudentProfile
from placement_ai.suggestions import build_suggestions

app = FastAPI(
    title="Smart Placement Prediction API",
    description="ML model serving API for placement prediction and interview frame analysis.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, bool | str]:
    return {"ok": True, "model_available": DEFAULT_MODEL_PATH.exists()}


@app.post("/predict", response_model=PlacementPrediction)
def predict(profile: StudentProfile) -> PlacementPrediction:
    try:
        return predict_placement(profile)
    except FileNotFoundError:
        return score_without_model(profile)


@app.post("/suggestions")
def suggestions(profile: StudentProfile) -> dict[str, list[str]]:
    return {"suggestions": build_suggestions(profile)}


@app.post("/interview/analyze-image", response_model=InterviewAnalysis)
async def interview_image(file: UploadFile = File(...)) -> InterviewAnalysis:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")
    content = await file.read()
    try:
        return analyze_image_bytes(content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not analyze image: {exc}") from exc
