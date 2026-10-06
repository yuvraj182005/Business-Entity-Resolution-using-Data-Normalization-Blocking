from __future__ import annotations

import argparse
from pathlib import Path

from code.business_entity_resolution.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the end-to-end business entity resolution pipeline.")
    parser.add_argument("--train-dir", required=True, help="Directory containing train source files.")
    parser.add_argument("--test-dir", required=True, help="Directory containing test source files.")
    parser.add_argument("--ground-truth", default=None, help="Optional ground truth TSV for matching evaluation.")
    parser.add_argument("--output-dir", default="output", help="Directory for generated outputs.")
    parser.add_argument("--low-threshold", type=float, default=0.45, help="Threshold for review status.")
    parser.add_argument("--high-threshold", type=float, default=0.72, help="Threshold for match status.")
    args = parser.parse_args()

    result = run_pipeline(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        ground_truth_path=args.ground_truth,
        output_dir=args.output_dir,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
    )
    print(result)


if __name__ == "__main__":
    main()
