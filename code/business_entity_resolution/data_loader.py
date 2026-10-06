from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


def read_tsv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a TSV file with explicit tab separation and stable string typing."""
    default_kwargs = {
        "sep": "\t",
        "dtype": str,
        "keep_default_na": False,
        "low_memory": False,
    }
    default_kwargs.update(kwargs)
    return pd.read_csv(path, **default_kwargs)


def resolve_dataset_root(data_dir: str | Path) -> Path:
    """Resolve both dataset-root and direct train/test directory structures."""
    root = Path(data_dir)
    if not root.exists():
        return root

    if (root / "train").exists() and (root / "test").exists():
        return root

    if root.name in {"train", "test"} and (root.parent / "train").exists() and (root.parent / "test").exists():
        return root.parent

    for candidate in [
        root,
        root / "train",
        root / "test",
        root / "student_resource" / "dataset",
    ]:
        if candidate.exists() and (candidate / "train").exists() and (candidate / "test").exists():
            return candidate

    student_resource = root / "student_resource" / "dataset"
    if student_resource.exists() and (student_resource / "train").exists() and (student_resource / "test").exists():
        return student_resource

    return root


def load_source_files(train_dir: str | Path, test_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """Load all Person 1 source datasets and return them keyed by name."""
    train_path = resolve_dataset_root(train_dir)
    test_path = resolve_dataset_root(test_dir)

    train_root = train_path / "train" if (train_path / "train").exists() else train_path
    test_root = test_path / "test" if (test_path / "test").exists() else test_path

    dataset_map = {
        "train_source1": read_tsv(train_root / "train_source1.tsv"),
        "train_source2": read_tsv(train_root / "train_source2.tsv"),
        "train_source3": read_tsv(train_root / "train_source3.tsv"),
        "train_ground_truth": read_tsv(train_root / "train_ground_truth.tsv"),
        "test_source1": read_tsv(test_root / "test_source1.tsv"),
        "test_source2": read_tsv(test_root / "test_source2.tsv"),
        "test_source3": read_tsv(test_root / "test_source3.tsv"),
    }
    return dataset_map


def summarize_dataframe(df: pd.DataFrame, label: str) -> Dict[str, object]:
    """Compute dataset-level summary statistics for EDA and reporting."""
    summary = {
        "label": label,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_values": {
            col: int(df[col].isna().sum()) for col in df.columns
        },
        "duplicate_rows": int(df.duplicated().sum()),
    }

    if "country" in df.columns:
        summary["country_distribution"] = (
            df["country"].fillna("missing").astype(str).str.strip().str.lower().value_counts().to_dict()
        )

    return summary


def save_summary_report(summary: Dict[str, object], output_path: str | Path) -> None:
    """Persist a JSON summary for the EDA stage."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Check that all required columns are present in a DataFrame."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
