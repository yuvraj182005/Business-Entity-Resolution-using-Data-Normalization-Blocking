from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .data_loader import read_tsv
from .blocking import generate_candidate_pairs


def _split_truth_ids(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    return {item.strip() for item in text.split(",") if item.strip()}


def build_truth_lookup(ground_truth_df: pd.DataFrame) -> dict[str, set[str]]:
    lookup: dict[str, set[str]] = defaultdict(set)
    for _, row in ground_truth_df.iterrows():
        source1 = str(row.get("source1_entity_id", "")).strip()
        if not source1:
            continue
        lookup[source1].update(_split_truth_ids(row.get("matched_entity_ids")))
    return lookup


def evaluate_blocking(source1_df: pd.DataFrame, source2_df: pd.DataFrame, source3_df: pd.DataFrame, ground_truth_df: pd.DataFrame, output_dir: str | Path | None = None) -> dict:
    candidates = generate_candidate_pairs(source1_df, source2_df, source3_df)
    truth_lookup = build_truth_lookup(ground_truth_df)

    matched_true_pairs = 0
    total_true_pairs = 0
    covered_source1 = 0
    for source1_id, true_ids in truth_lookup.items():
        total_true_pairs += len(true_ids)
        candidate_ids = set(candidates.get(source1_id, []))
        matched_true_pairs += len(true_ids & candidate_ids)
        if true_ids:
            covered_source1 += 1

    all_candidates = sum(len(v) for v in candidates.values())
    total_source1 = len(source1_df)
    total_possible = total_source1 * (len(source2_df) + len(source3_df))

    blocking_recall = (matched_true_pairs / total_true_pairs) if total_true_pairs else 1.0
    avg_candidates = (all_candidates / total_source1) if total_source1 else 0.0
    max_candidates = max((len(v) for v in candidates.values()), default=0)
    reduction_ratio = 1.0 - (all_candidates / total_possible) if total_possible else 1.0

    metrics = {
        "blocking_recall": blocking_recall,
        "average_candidates_per_source1": avg_candidates,
        "max_candidates": max_candidates,
        "candidate_reduction_ratio": reduction_ratio,
        "total_true_pairs": total_true_pairs,
        "matched_true_pairs": matched_true_pairs,
        "covered_source1_entities_with_truth": covered_source1,
    }

    if output_dir is not None:
        out_path = Path(output_dir) / "blocking_evaluation.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(str(metrics).replace("'", '"'), encoding="utf-8")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate blocking quality for business entity resolution.")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    train_path = train_dir
    source1_df = read_tsv(train_path / "train_source1.tsv")
    source2_df = read_tsv(train_path / "train_source2.tsv")
    source3_df = read_tsv(train_path / "train_source3.tsv")
    ground_truth_df = read_tsv(args.ground_truth)
    metrics = evaluate_blocking(source1_df, source2_df, source3_df, ground_truth_df, args.output_dir)
    print(metrics)


if __name__ == "__main__":
    main()
