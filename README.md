# Smart Placement Prediction & Interview Analysis System

ML Engineer Intern readiness project for students who want to evaluate whether their resume, skills, projects, and interview presence are strong enough for ML engineering internship roles.

An end-to-end ML engineering project for ML Engineer Intern placement readiness:

- scores a student's resume profile
- extracts and evaluates skills
- predicts ML Engineer Intern selection probability
- analyzes interview confidence from webcam/snapshot frames
- serves predictions through FastAPI
- provides an interactive Streamlit dashboard

The project is designed to show the full pipeline, not just model training: data cleaning, feature engineering, model evaluation, model persistence, REST serving, and real-time-ish inference.

## Tech Stack Fit For ML Engineer Intern Roles

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- OpenCV
- FastAPI
- Streamlit

Why this stack fits the intern role:

- `Pandas` and `NumPy` show data cleaning and feature engineering basics.
- `Scikit-learn` shows classical ML pipeline, cross validation, preprocessing, and evaluation.
- `XGBoost` adds a stronger production-style tabular classifier.
- `FastAPI` shows model serving through REST APIs.
- `Streamlit` gives a fast ML demo interface for recruiters/interviewers.
- `OpenCV` covers real-time inference basics for webcam-based interview analysis.

## Project Structure

```text
.
|-- app.py                         # Streamlit UI
|-- requirements.txt
|-- pyproject.toml
|-- data/
|   `-- .gitkeep
|-- models/
|   `-- .gitkeep
|-- scripts/
|   |-- make_dataset.py            # Generate synthetic training data
|   |-- run_cases.py               # Run 5 ML intern readiness cases and save screenshots
|   `-- train_model.py             # Train/evaluate/save pipeline
|-- src/
|   `-- placement_ai/
|       |-- api.py                 # FastAPI app
|       |-- data.py                # Data generation/loading/cleaning
|       |-- features.py            # Resume score and skill features
|       |-- inference.py           # Model loading + prediction service
|       |-- interview.py           # OpenCV confidence analysis
|       |-- schemas.py             # API/request schemas
|       |-- suggestions.py         # Improvement recommendations
|       `-- train.py               # ML pipeline + evaluation
`-- tests/
    `-- test_features.py
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Train The Model

Generate a synthetic dataset:

```powershell
python scripts/make_dataset.py --rows 1200 --output data/students.csv
```

Train and evaluate:

```powershell
python scripts/train_model.py --data data/students.csv --model-path models/placement_model.joblib
```

The training script prints cross-validation scores and saves:

- `models/placement_model.joblib`
- `models/metrics.json`

## Run API

```powershell
uvicorn placement_ai.api:app --reload --host 127.0.0.1 --port 8000
```

Example request:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/predict `
  -Method POST `
  -ContentType "application/json" `
  -Body '{
    "cgpa": 8.2,
    "internships": 2,
    "projects": 4,
    "certifications": 3,
    "hackathons": 2,
    "aptitude_score": 78,
    "communication_score": 72,
    "branch": "CSE",
    "degree": "B.Tech",
    "skills": ["python", "machine learning", "sql", "fastapi"],
    "resume_text": "Built ML projects with Python, pandas, scikit-learn, FastAPI and deployment."
  }'
```

## Run Streamlit App

```powershell
streamlit run app.py
```

The app can use a trained model if `models/placement_model.joblib` exists. If not, it still shows profile scoring, skill gaps, and suggestions.

## Run Demo Cases And Screenshots

```powershell
python scripts/run_cases.py
```

This creates five PNG screenshots in `screenshots/` and a machine-readable summary in `screenshots/case_results.json`.

## Notes

The interview analysis module is an OpenCV heuristic baseline. It estimates confidence from face visibility, lighting, sharpness, and smile cues. It should be presented as an assistive signal, not a medical or psychological emotion detector.
