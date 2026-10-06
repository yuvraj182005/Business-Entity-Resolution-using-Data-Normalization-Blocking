from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


DEFAULT_MATCHING_WEIGHTS: Dict[str, float] = {
    "name": 0.45,
    "address": 0.30,
    "location": 0.15,
    "identifier": 0.10,
}

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "low": 0.45,
    "high": 0.72,
}


@dataclass
class MatchingConfig:
    """Configuration object for the matching and classification stage."""

    weights: Dict[str, float] = None
    low_threshold: float = DEFAULT_THRESHOLDS["low"]
    high_threshold: float = DEFAULT_THRESHOLDS["high"]

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = DEFAULT_MATCHING_WEIGHTS.copy()

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "MatchingConfig":
        config = cls()
        if "weights" in values:
            config.weights = {**DEFAULT_MATCHING_WEIGHTS, **values["weights"]}
        if "low_threshold" in values:
            config.low_threshold = float(values["low_threshold"])
        if "high_threshold" in values:
            config.high_threshold = float(values["high_threshold"])
        return config
