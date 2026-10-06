from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd

from .blocking import generate_candidate_pairs
from .data_loader import load_source_files, read_tsv
from .matching_features import generate_pair_features
from .match_scoring import classify_match_score, resolve_one_to_one_matches, score_candidate_pair

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _coerce_row_map(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    index = {}
    for _, row in df.iterrows():
        entity_id = str(row.get("entity_id", "")).strip()
        if not entity_id:
            entity_id = str(row.get("source_entity_id", "")).strip()
        if not entity_id:
            continue
        index[entity_id] = row.to_dict()
    return index


def build_scored_candidate_pairs(
    source1_df: pd.DataFrame,
    source2_df: pd.DataFrame,
    source3_df: pd.DataFrame,
    low_threshold: float = 0.45,
    high_threshold: float = 0.72,
) -> List[Dict[str, Any]]:
    candidate_map = generate_candidate_pairs(source1_df, source2_df, source3_df)
    rows: List[Dict[str, Any]] = []
    lookup = {**_coerce_row_map(source2_df), **_coerce_row_map(source3_df)}
    source1_lookup = _coerce_row_map(source1_df)
    for source_entity_id, candidate_ids in candidate_map.items():
        for candidate_id in sorted(set(candidate_ids)):
            left_row = source1_lookup.get(source_entity_id, {})
            right_row = lookup.get(candidate_id, {})
            if not left_row or not right_row:
                continue
            features = generate_pair_features(left_row, right_row)
            score = score_candidate_pair(features)
            status = classify_match_score(score, low_threshold, high_threshold)
            rows.append({
                "source_entity_id": source_entity_id,
                "candidate_entity_id": candidate_id,
                "score": score,
                "status": status,
                "confidence": "high" if status == "match" else "review" if status == "review" else "low",
                **features,
            })
    return rows


def build_resolved_entities(scored_rows: Iterable[Mapping[str, Any]], low_threshold: float = 0.45, high_threshold: float = 0.72) -> Dict[str, str]:
    return resolve_one_to_one_matches(list(scored_rows), high_threshold=high_threshold, low_threshold=low_threshold)


def build_truth_lookup(ground_truth_df: pd.DataFrame) -> Dict[str, set[str]]:
    lookup: Dict[str, set[str]] = {}
    if ground_truth_df is None or ground_truth_df.empty:
        return lookup
    for _, row in ground_truth_df.iterrows():
        source_id = str(row.get("source1_entity_id", row.get("entity_id", ""))).strip()
        if not source_id:
            continue
        raw_values = row.get("matched_entity_ids", row.get("matched_entity_id", ""))
        if pd.isna(raw_values):
            matched = set()
        else:
            matched = {item.strip() for item in str(raw_values).split(",") if item.strip()}
        lookup[source_id] = matched
    return lookup


def compute_match_metrics(scored_rows: Sequence[Mapping[str, Any]], ground_truth_df: pd.DataFrame | None = None) -> Dict[str, Any]:
    truth_lookup = build_truth_lookup(ground_truth_df) if ground_truth_df is not None else {}
    predictions = {}
    for row in scored_rows:
        source_id = str(row.get("source_entity_id", "")).strip()
        if not source_id:
            continue
        if row.get("status") == "match":
            predictions.setdefault(source_id, set()).add(str(row.get("candidate_entity_id", "")).strip())
    tp = 0
    fp = 0
    fn = 0
    for source_id, truth_ids in truth_lookup.items():
        predicted = predictions.get(source_id, set())
        tp += len(predicted & truth_ids)
        fp += len(predicted - truth_ids)
        fn += len(truth_ids - predicted)
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def run_pipeline(train_dir: str | Path, test_dir: str | Path, ground_truth_path: str | Path | None = None, output_dir: str | Path = "output", low_threshold: float = 0.45, high_threshold: float = 0.72) -> Dict[str, Any]:
    logger.info("Loading datasets")
    dataset = load_source_files(train_dir, test_dir)
    source1_df = dataset["train_source1"]
    source2_df = dataset["train_source2"]
    source3_df = dataset["train_source3"]

    logger.info("Generating candidate pairs")
    candidate_map = generate_candidate_pairs(source1_df, source2_df, source3_df)

    logger.info("Scoring candidate pairs")
    scored_rows = build_scored_candidate_pairs(source1_df, source2_df, source3_df, low_threshold=low_threshold, high_threshold=high_threshold)
    scored_df = pd.DataFrame(scored_rows)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(out_dir / "scored_pairs.csv", index=False)

    logger.info("Resolving one-to-one matches")
    resolved_map = build_resolved_entities(scored_rows, low_threshold=low_threshold, high_threshold=high_threshold)
    resolved_df = pd.DataFrame([
        {"source_id": source_id, "matched_entity_id": candidate_id} for source_id, candidate_id in resolved_map.items()
    ])
    resolved_df.to_csv(out_dir / "resolved_entities.csv", index=False)

    logger.info("Writing ambiguous and unmatched records")
    if scored_df.empty:
        unmatched = pd.DataFrame(columns=["source_entity_id", "reason"])
        ambiguous = pd.DataFrame(columns=["source_entity_id", "candidate_entity_id", "score"])
    else:
        unmatched = scored_df.loc[scored_df["status"] == "unmatch", ["source_entity_id", "candidate_entity_id", "score"]].copy()
        ambiguous = scored_df.loc[scored_df["status"] == "review", ["source_entity_id", "candidate_entity_id", "score"]].copy()
    unmatched.to_csv(out_dir / "unmatched_entities.csv", index=False)
    ambiguous.to_csv(out_dir / "ambiguous_matches.csv", index=False)

    result = {
        "source1_count": len(source1_df),
        "source2_count": len(source2_df),
        "source3_count": len(source3_df),
        "candidate_pairs": sum(len(v) for v in candidate_map.values()),
        "resolved_matches": len(resolved_map),
        "high_confidence_matches": int((scored_df["status"] == "match").sum()) if not scored_df.empty else 0,
        "review_cases": int((scored_df["status"] == "review").sum()) if not scored_df.empty else 0,
    }

    if ground_truth_path is not None:
        ground_truth_df = read_tsv(ground_truth_path)
        metrics = compute_match_metrics(scored_rows, ground_truth_df)
        result["evaluation_metrics"] = metrics
        with (out_dir / "evaluation_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)

    with (out_dir / "matching_report.csv").open("w", encoding="utf-8", newline="") as handle:
        scored_df.to_csv(handle, index=False)

    logger.info("Pipeline complete")
    return result
