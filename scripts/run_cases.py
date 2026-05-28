from __future__ import annotations

import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from placement_ai.inference import predict_placement, score_without_model
from placement_ai.schemas import StudentProfile


OUTPUT_DIR = Path("screenshots")

CASES = [
    {
        "name": "Case 1 - Strong ML Intern Profile",
        "profile": StudentProfile(
            cgpa=8.7,
            internships=2,
            projects=5,
            certifications=3,
            hackathons=2,
            aptitude_score=82,
            communication_score=78,
            branch="CSE",
            degree="B.Tech",
            skills=[
                "python",
                "sql",
                "machine learning",
                "pandas",
                "numpy",
                "scikit-learn",
                "xgboost",
                "fastapi",
                "streamlit",
                "git",
            ],
            resume_text="Built and deployed ML models with FastAPI, Streamlit, cross validation, F1 score and ROC AUC metrics.",
        ),
    },
    {
        "name": "Case 2 - Good Academics, Missing Deployment",
        "profile": StudentProfile(
            cgpa=8.4,
            internships=1,
            projects=3,
            certifications=2,
            hackathons=1,
            aptitude_score=76,
            communication_score=70,
            branch="IT",
            degree="B.Tech",
            skills=["python", "sql", "machine learning", "pandas", "numpy", "statistics"],
            resume_text="Created machine learning notebooks for classification and regression with pandas and scikit-learn.",
        ),
    },
    {
        "name": "Case 3 - Beginner Profile",
        "profile": StudentProfile(
            cgpa=6.9,
            internships=0,
            projects=1,
            certifications=1,
            hackathons=0,
            aptitude_score=58,
            communication_score=55,
            branch="ECE",
            degree="B.Tech",
            skills=["python", "excel", "sql"],
            resume_text="Learning Python and SQL. Completed one mini project on student marks analysis.",
        ),
    },
    {
        "name": "Case 4 - Project Heavy, Low Aptitude",
        "profile": StudentProfile(
            cgpa=7.6,
            internships=1,
            projects=6,
            certifications=2,
            hackathons=3,
            aptitude_score=52,
            communication_score=68,
            branch="AI&DS",
            degree="B.Tech",
            skills=["python", "machine learning", "opencv", "streamlit", "pandas", "git"],
            resume_text="Built computer vision and NLP demos in Streamlit. Added GitHub links but limited evaluation metrics.",
        ),
    },
    {
        "name": "Case 5 - Non CS Career Switcher",
        "profile": StudentProfile(
            cgpa=7.9,
            internships=1,
            projects=4,
            certifications=4,
            hackathons=1,
            aptitude_score=73,
            communication_score=82,
            branch="ME",
            degree="B.Tech",
            skills=["python", "sql", "pandas", "statistics", "fastapi", "docker"],
            resume_text="Transitioned to ML with data cleaning, statistics, API deployment and Dockerized FastAPI project.",
        ),
    },
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    font: ImageFont.ImageFont,
    fill: str,
    width: int,
    line_gap: int = 8,
) -> int:
    x, y = xy
    line_height = getattr(font, "size", 18) + line_gap
    for paragraph in text.split("\n"):
        for line in textwrap.wrap(paragraph, width=width):
            draw.text((x, y), line, fill=fill, font=font)
            y += line_height
        y += line_gap
    return y


def _pill(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: str) -> None:
    x, y = xy
    font = _font(22, bold=True)
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rounded_rectangle((x - 18, y - 10, bbox[2] + 18, bbox[3] + 10), radius=18, fill=fill)
    draw.text((x, y), text, fill="#ffffff", font=font)


def create_screenshot(case_name: str, profile: StudentProfile, index: int) -> dict[str, object]:
    try:
        prediction = predict_placement(profile)
        source = "XGBoost trained model"
    except FileNotFoundError:
        prediction = score_without_model(profile)
        source = "resume scoring fallback"

    image = Image.new("RGB", (1366, 768), "#f5f7fb")
    draw = ImageDraw.Draw(image)
    title_font = _font(40, bold=True)
    h_font = _font(27, bold=True)
    body_font = _font(23)
    small_font = _font(19)

    draw.rectangle((0, 0, 1366, 105), fill="#122033")
    draw.text((54, 30), "ML Engineer Intern Readiness Analyzer", fill="#ffffff", font=title_font)
    _pill(draw, (1080, 38), source, "#2f7d5c")

    draw.rounded_rectangle((44, 135, 646, 704), radius=18, fill="#ffffff", outline="#dce3ef", width=2)
    draw.rounded_rectangle((688, 135, 1322, 704), radius=18, fill="#ffffff", outline="#dce3ef", width=2)

    draw.text((76, 166), case_name, fill="#122033", font=h_font)
    y = 222
    profile_lines = [
        f"CGPA: {profile.cgpa}    Internships: {profile.internships}    Projects: {profile.projects}",
        f"Certifications: {profile.certifications}    Hackathons: {profile.hackathons}",
        f"Aptitude: {profile.aptitude_score}/100    Communication: {profile.communication_score}/100",
        f"Branch: {profile.branch}    Degree: {profile.degree}",
        "Skills: " + ", ".join(profile.skills),
    ]
    for line in profile_lines:
        y = _draw_wrapped(draw, line, (76, y), body_font, "#243447", width=47, line_gap=6)

    draw.text((720, 166), "Prediction Output", fill="#122033", font=h_font)
    probability = prediction.placement_probability
    bar_x, bar_y, bar_w, bar_h = 720, 238, 540, 38
    draw.rounded_rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), radius=19, fill="#e8edf5")
    draw.rounded_rectangle((bar_x, bar_y, bar_x + int(bar_w * probability), bar_y + bar_h), radius=19, fill="#2f7d5c")
    draw.text((720, 295), f"Placement Probability: {probability * 100:.1f}%", fill="#122033", font=h_font)
    draw.text((720, 340), f"Resume Score: {prediction.resume_score:.1f}/100", fill="#243447", font=body_font)
    draw.text((720, 378), f"Skill Match: {prediction.skill_match_score:.1f}%", fill="#243447", font=body_font)
    draw.text((720, 430), "Missing Skills", fill="#122033", font=h_font)
    missing = ", ".join(prediction.top_missing_skills) or "No major gap"
    y = _draw_wrapped(draw, missing, (720, 472), body_font, "#415166", width=43, line_gap=6)

    draw.text((720, y + 12), "Suggestions", fill="#122033", font=h_font)
    y += 54
    for suggestion in prediction.suggestions[:3]:
        y = _draw_wrapped(draw, "- " + suggestion, (720, y), small_font, "#415166", width=58, line_gap=5)

    output_path = OUTPUT_DIR / f"case_{index}_ml_engineer_intern.png"
    image.save(output_path)
    return {
        "case": case_name,
        "screenshot": str(output_path),
        "placement_probability": prediction.placement_probability,
        "resume_score": prediction.resume_score,
        "skill_match_score": prediction.skill_match_score,
        "predicted_label": prediction.predicted_label,
        "missing_skills": prediction.top_missing_skills,
        "suggestions": prediction.suggestions,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [
        create_screenshot(case["name"], case["profile"], index)
        for index, case in enumerate(CASES, start=1)
    ]
    (OUTPUT_DIR / "case_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
