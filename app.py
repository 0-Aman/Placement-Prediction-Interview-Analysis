from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from placement_ai.inference import DEFAULT_MODEL_PATH, predict_placement, score_without_model
from placement_ai.interview import analyze_image_bytes
from placement_ai.schemas import StudentProfile


st.set_page_config(page_title="Placement AI", page_icon="ML", layout="wide")

st.title("ML Engineer Intern Readiness Analyzer")

left, right = st.columns([1.05, 0.95])

with left:
    st.subheader("Student Profile")
    with st.form("profile_form"):
        c1, c2, c3 = st.columns(3)
        cgpa = c1.number_input("CGPA", 0.0, 10.0, 8.0, 0.1)
        internships = c2.number_input("Internships", 0, 10, 1)
        projects = c3.number_input("Projects", 0, 30, 4)

        c4, c5, c6 = st.columns(3)
        certifications = c4.number_input("Certifications", 0, 30, 2)
        hackathons = c5.number_input("Hackathons", 0, 30, 1)
        aptitude_score = c6.slider("Aptitude Score", 0, 100, 72)

        c7, c8 = st.columns(2)
        communication_score = c7.slider("Communication Score", 0, 100, 70)
        branch = c8.selectbox("Branch", ["CSE", "IT", "AI&DS", "ECE", "EEE", "ME", "Civil"])
        degree = st.selectbox("Degree", ["B.Tech", "M.Tech", "BCA", "MCA", "B.Sc"])
        skills_raw = st.text_input(
            "Skills",
            "python, machine learning, pandas, sql, scikit-learn, fastapi",
        )
        resume_text = st.text_area(
            "Resume Text",
            "Built ML projects using Python, pandas, scikit-learn, FastAPI and model evaluation.",
            height=130,
        )
        submitted = st.form_submit_button("Analyze Placement Readiness")

    if submitted:
        profile = StudentProfile(
            cgpa=cgpa,
            internships=internships,
            projects=projects,
            certifications=certifications,
            hackathons=hackathons,
            aptitude_score=aptitude_score,
            communication_score=communication_score,
            branch=branch,
            degree=degree,
            skills=skills_raw,
            resume_text=resume_text,
        )
        try:
            prediction = predict_placement(profile)
            model_note = "trained model"
        except FileNotFoundError:
            prediction = score_without_model(profile)
            model_note = "resume scoring fallback"

        m1, m2, m3 = st.columns(3)
        m1.metric("Placement Probability", f"{prediction.placement_probability * 100:.1f}%")
        m2.metric("Resume Score", f"{prediction.resume_score:.1f}/100")
        m3.metric("Skill Match", f"{prediction.skill_match_score:.1f}%")
        st.caption(f"Prediction source: {model_note}")
        st.progress(prediction.placement_probability)

        st.write("Missing Skills")
        st.write(", ".join(prediction.top_missing_skills) or "No major gaps found.")

        st.write("Improvement Suggestions")
        for item in prediction.suggestions:
            st.write(f"- {item}")

with right:
    st.subheader("Interview Snapshot Analysis")
    st.caption("Uses OpenCV heuristics for camera presence, lighting, sharpness and face framing.")
    image = st.camera_input("Capture interview frame")
    uploaded = st.file_uploader("Or upload an interview frame", type=["jpg", "jpeg", "png"])
    source = image or uploaded

    if source is not None:
        result = analyze_image_bytes(source.getvalue())
        st.metric("Confidence Score", f"{result.confidence_score:.1f}/100")
        st.write(f"Detected state: `{result.emotion_label}`")
        st.write("Signals")
        st.json(json.loads(result.model_dump_json())["signals"])
        st.write("Interview Suggestions")
        for item in result.suggestions:
            st.write(f"- {item}")

if DEFAULT_MODEL_PATH.exists():
    st.sidebar.success("Model found")
else:
    st.sidebar.warning("Model not trained yet. Run scripts/train_model.py.")
