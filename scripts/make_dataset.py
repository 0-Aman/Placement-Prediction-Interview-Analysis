from __future__ import annotations

import argparse

from placement_ai.data import save_synthetic_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic placement dataset.")
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--output", default="data/students.csv")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = save_synthetic_dataset(args.output, rows=args.rows, random_state=args.random_state)
    print(f"Saved dataset to {output}")


if __name__ == "__main__":
    main()
