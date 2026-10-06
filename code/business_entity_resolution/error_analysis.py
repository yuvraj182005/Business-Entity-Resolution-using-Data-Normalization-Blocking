from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import pandas as pd


def _coerce_truth_map(ground_truth_df: pd.DataFrame | None) -> Dict[str, set[str]]:
    if ground_truth_df is None or ground_truth_df.empty:
        return {}
    truth: Dict[str, set[str]] = {}
    for _, row in ground_truth_df.iterrows():
        source_id = str(row.get("source1_entity_id", row.get("entity_id", ""))).strip()
        if not source_id:
            continue
        values = row.get("matched_entity_ids", row.get("matched_entity_id", ""))
        truth[source_id] = {item.strip() for item in str(values).split(",") if item.strip()}
    return truth


def build_error_analysis(scored_rows: Iterable[Mapping[str, Any]], ground_truth_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Summarize false positives, false negatives, and review cases for debugging."""
    scored = list(scored_rows)
    truth = _coerce_truth_map(ground_truth_df)
    rows: List[Dict[str, Any]] = []
    for row in scored:
        source_id = str(row.get("source_entity_id", "")).strip()
        candidate_id = str(row.get("candidate_entity_id", "")).strip()
        truth_ids = truth.get(source_id, set())
        predicted = row.get("status", "unmatch")
        actual = "match" if candidate_id in truth_ids else "unmatch"
        if predicted == "match" and actual == "unmatch":
            rows.append({"error_type": "false_positive", **row})
        elif predicted != "match" and actual == "match":
            rows.append({"error_type": "false_negative", **row})
        elif predicted == "review":
            rows.append({"error_type": "review_case", **row})
    return pd.DataFrame(rows)


def write_error_analysis(output_path: str | Path, scored_rows: Iterable[Mapping[str, Any]], ground_truth_df: pd.DataFrame | None = None) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_error_analysis(scored_rows, ground_truth_df).to_csv(output, index=False)
