from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from placement_ai.data import clean_student_frame, load_dataset

NUMERIC_FEATURES = [
    "cgpa",
    "internships",
    "projects",
    "certifications",
    "hackathons",
    "aptitude_score",
    "communication_score",
]
CATEGORICAL_FEATURES = ["branch", "degree"]
TEXT_FEATURES = ["skills_text", "resume_text"]
TARGET = "placed"


def _build_classifier(random_state: int = 42) -> Any:
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        )
    except ImportError:
        return GradientBoostingClassifier(random_state=random_state)


def build_pipeline(random_state: int = 42) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("skills", TfidfVectorizer(ngram_range=(1, 2), min_df=2), "skills_text"),
            ("resume", TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=2), "resume_text"),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", _build_classifier(random_state=random_state)),
        ]
    )


def train_model(
    data_path: str | Path,
    model_path: str | Path = "models/placement_model.joblib",
    metrics_path: str | Path = "models/metrics.json",
    random_state: int = 42,
) -> dict[str, Any]:
    df = load_dataset(data_path)
    if TARGET not in df.columns:
        raise ValueError(f"Dataset must contain target column '{TARGET}'.")

    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES]
    y = df[TARGET].astype(int)

    pipeline = build_pipeline(random_state=random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_validate(
        pipeline,
        x,
        y,
        cv=cv,
        scoring=["accuracy", "f1", "roc_auc"],
        n_jobs=-1,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "cv_accuracy_mean": round(float(cv_scores["test_accuracy"].mean()), 4),
        "cv_f1_mean": round(float(cv_scores["test_f1"].mean()), 4),
        "cv_roc_auc_mean": round(float(cv_scores["test_roc_auc"].mean()), 4),
        "test_accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "test_f1": round(float(f1_score(y_test, predictions)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES,
        "model_type": type(pipeline.named_steps["model"]).__name__,
    }

    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def profile_to_frame(profile: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame([profile])
    return clean_student_frame(frame)[NUMERIC_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES]
