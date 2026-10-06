from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class MatchingLogisticModel:
    """Small optional supervised classifier that uses the generated pairwise features."""

    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=2000, random_state=42)
        self.columns: List[str] = []

    def fit(self, feature_rows: Iterable[Dict[str, Any]], labels: Iterable[int]) -> None:
        rows = list(feature_rows)
        label_list = list(labels)
        if not rows or not label_list:
            raise ValueError("Feature rows and labels must be non-empty for model training.")
        self.columns = list(rows[0].keys())
        frame = pd.DataFrame(rows, columns=self.columns)
        self.model.fit(frame, label_list)

    def predict_proba(self, feature_rows: Iterable[Dict[str, Any]]) -> List[float]:
        if not self.columns:
            raise ValueError("The model has not been trained yet.")
        rows = list(feature_rows)
        frame = pd.DataFrame(rows, columns=self.columns)
        probabilities = self.model.predict_proba(frame)
        return probabilities[:, 1].tolist()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            import pickle

            pickle.dump({"model": self.model, "columns": self.columns}, handle)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("rb") as handle:
            import pickle

            payload = pickle.load(handle)
        self.model = payload["model"]
        self.columns = payload["columns"]


def train_logistic_matcher(feature_rows: Iterable[Dict[str, Any]], labels: Iterable[int]) -> MatchingLogisticModel:
    model = MatchingLogisticModel()
    model.fit(feature_rows, labels)
    return model


def split_training_data(feature_rows: List[Dict[str, Any]], labels: List[int], test_size: float = 0.2) -> tuple[list, list, list, list]:
    x_train, x_test, y_train, y_test = train_test_split(feature_rows, labels, test_size=test_size, random_state=42, stratify=labels)
    return x_train, x_test, y_train, y_test
