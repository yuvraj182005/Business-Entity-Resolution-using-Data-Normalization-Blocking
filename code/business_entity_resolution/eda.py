from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from .data_loader import load_source_files, save_summary_report
from .normalization import extract_pin_code


def analyze_name_quality(df: pd.DataFrame) -> dict:
    values = df.get("business_name", pd.Series(["" for _ in range(len(df))])).fillna("").astype(str)
    quality = {
        "empty_name_count": int((values.str.strip() == "").sum()),
        "avg_name_length": round(float(values.str.len().mean()), 2) if len(values) else 0.0,
        "max_name_length": int(values.str.len().max()) if len(values) else 0,
    }
    return quality


def analyze_address_quality(df: pd.DataFrame) -> dict:
    values = df.get("business_address", pd.Series(["" for _ in range(len(df))])).fillna("").astype(str)
    quality = {
        "empty_address_count": int((values.str.strip() == "").sum()),
        "avg_address_length": round(float(values.str.len().mean()), 2) if len(values) else 0.0,
        "max_address_length": int(values.str.len().max()) if len(values) else 0,
        "pin_extracted_count": int(values.map(lambda x: bool(extract_pin_code(x))).sum()),
    }
    return quality


def analyze_match_cardinality(ground_truth_df: pd.DataFrame) -> dict:
    if ground_truth_df.empty:
        return {"match_count_per_source1": {}}

    counts = Counter()
    for _, row in ground_truth_df.iterrows():
        source1_id = str(row.get("source1_entity_id", "")).strip()
        if not source1_id:
            continue
        matched_ids = [item.strip() for item in str(row.get("matched_entity_ids", "")).split(",") if item.strip()]
        counts[source1_id] = len(matched_ids)

    return {
        "match_count_per_source1": {k: int(v) for k, v in sorted(counts.items())},
        "num_entities_with_zero_matches": int(sum(1 for v in counts.values() if v == 0)),
        "num_entities_with_multiple_matches": int(sum(1 for v in counts.values() if v > 1)),
        "num_entities_with_matches": int(sum(1 for v in counts.values() if v > 0)),
    }


def run_eda(train_dir: str, test_dir: str, output_dir: str) -> dict:
    dataset = load_source_files(train_dir, test_dir)
    summary = {}
    for key, df in dataset.items():
        if key.startswith("train_") or key.startswith("test_"):
            summary[key] = {
                "rows": len(df),
                "columns": df.columns.tolist(),
                "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
                "duplicate_rows": int(df.duplicated().sum()),
                "country_distribution": (
                    df["country"].fillna("missing").astype(str).str.strip().str.lower().value_counts().to_dict()
                    if "country" in df.columns else {}
                ),
            }

    summary["train_source1_name_quality"] = analyze_name_quality(dataset["train_source1"])
    summary["train_source2_name_quality"] = analyze_name_quality(dataset["train_source2"])
    summary["train_source3_name_quality"] = analyze_name_quality(dataset["train_source3"])
    summary["train_source1_address_quality"] = analyze_address_quality(dataset["train_source1"])
    summary["train_source2_address_quality"] = analyze_address_quality(dataset["train_source2"])
    summary["train_source3_address_quality"] = analyze_address_quality(dataset["train_source3"])
    summary["match_cardinality"] = analyze_match_cardinality(dataset["train_ground_truth"])

    out_path = Path(output_dir) / "eda_summary.json"
    save_summary_report(summary, out_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run exploratory data analysis for the business entity resolution task.")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    run_eda(args.train_dir, args.test_dir, args.output_dir)


if __name__ == "__main__":
    main()
