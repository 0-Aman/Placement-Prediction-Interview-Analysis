from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from placement_ai.schemas import InterviewAnalysis
from placement_ai.suggestions import interview_suggestions


def _cascade_path(filename: str) -> str:
    return str(Path(cv2.data.haarcascades) / filename)


def analyze_frame(frame: np.ndarray) -> InterviewAnalysis:
    if frame is None or frame.size == 0:
        raise ValueError("Frame is empty.")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    face_cascade = cv2.CascadeClassifier(_cascade_path("haarcascade_frontalface_default.xml"))
    smile_cascade = cv2.CascadeClassifier(_cascade_path("haarcascade_smile.xml"))
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    face_detected = len(faces) > 0
    face_area_ratio = 0.0
    smile_detected = False

    if face_detected:
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        face_area_ratio = float((w * h) / (frame.shape[0] * frame.shape[1]))
        roi_gray = gray[y : y + h, x : x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=18, minSize=(25, 25))
        smile_detected = len(smiles) > 0

    lighting_score = np.interp(brightness, [35, 150], [0, 30]).clip(0, 30)
    sharpness_score = np.interp(sharpness, [15, 140], [0, 25]).clip(0, 25)
    framing_score = np.interp(face_area_ratio, [0.03, 0.18], [0, 30]).clip(0, 30) if face_detected else 0
    expression_score = 15 if smile_detected else 8 if face_detected else 0
    confidence_score = round(float(lighting_score + sharpness_score + framing_score + expression_score), 2)

    if not face_detected:
        label = "face_not_detected"
    elif confidence_score >= 75:
        label = "confident"
    elif confidence_score >= 55:
        label = "neutral"
    else:
        label = "low_confidence"

    signals: dict[str, float | int | bool] = {
        "faces": int(len(faces)),
        "brightness": round(brightness, 2),
        "sharpness": round(sharpness, 2),
        "face_area_ratio": round(face_area_ratio, 4),
        "smile_detected": smile_detected,
    }
    return InterviewAnalysis(
        face_detected=face_detected,
        confidence_score=confidence_score,
        emotion_label=label,
        signals=signals,
        suggestions=interview_suggestions(confidence_score, face_detected, signals),
    )


def analyze_image_bytes(content: bytes) -> InterviewAnalysis:
    array = np.frombuffer(content, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return analyze_frame(frame)
