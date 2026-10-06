from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

from .config import DEFAULT_MATCHING_WEIGHTS, DEFAULT_THRESHOLDS, MatchingConfig


def score_candidate_pair(feature_dict: Mapping[str, Any], weights: Optional[Mapping[str, float]] = None) -> float:
    """Compute a weighted overall match score from pairwise similarity features."""
    metrics = dict(feature_dict)
    weight_map = DEFAULT_MATCHING_WEIGHTS.copy() if weights is None else {**DEFAULT_MATCHING_WEIGHTS, **dict(weights)}

    total_weight = 0.0
    weighted_total = 0.0

    for key, weight in weight_map.items():
        if key == "name":
            value = float(metrics.get("name_similarity", 0.0))
        elif key == "address":
            value = float(metrics.get("address_similarity", 0.0))
        elif key == "location":
            value = float(metrics.get("location_similarity", 0.0))
        elif key == "identifier":
            value = float(metrics.get("identifier_similarity", 0.0))
        else:
            continue
        total_weight += weight
        weighted_total += value * weight

    if total_weight == 0:
        return 0.0
    return round(weighted_total / total_weight, 6)


def classify_match_score(score: float, low_threshold: float = None, high_threshold: float = None) -> str:
    """Return match/review/unmatch based on configured thresholds."""
    low = DEFAULT_THRESHOLDS["low"] if low_threshold is None else float(low_threshold)
    high = DEFAULT_THRESHOLDS["high"] if high_threshold is None else float(high_threshold)
    if score >= high:
        return "match"
    if score >= low:
        return "review"
    return "unmatch"


def resolve_one_to_one_matches(candidate_rows: Iterable[Mapping[str, Any]], high_threshold: float = 0.72, low_threshold: float = 0.45) -> Dict[str, str]:
    """Resolve candidate matches with a one-to-one policy, preferring the highest score."""
    ranked = sorted(
        candidate_rows,
        key=lambda row: (float(row.get("score", 0.0)), str(row.get("status", "unmatch")) == "match"),
        reverse=True,
    )
    resolved: Dict[str, str] = {}
    used_candidates: set[str] = set()
    for row in ranked:
        source_id = str(row.get("source_entity_id", "")).strip()
        cand_id = str(row.get("candidate_entity_id", "")).strip()
        if not source_id or not cand_id:
            continue
        if source_id in resolved:
            continue
        if cand_id in used_candidates:
            continue
        score = float(row.get("score", 0.0))
        status = str(row.get("status", classify_match_score(score, low_threshold, high_threshold)))
        if score < low_threshold and status != "match":
            continue
        resolved[source_id] = cand_id
        used_candidates.add(cand_id)
    return resolved
