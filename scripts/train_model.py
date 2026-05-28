from __future__ import annotations

import argparse
import json

from placement_ai.train import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train placement prediction model.")
    parser.add_argument("--data", default="data/students.csv")
    parser.add_argument("--model-path", default="models/placement_model.joblib")
    parser.add_argument("--metrics-path", default="models/metrics.json")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_model(
        data_path=args.data,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        random_state=args.random_state,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
