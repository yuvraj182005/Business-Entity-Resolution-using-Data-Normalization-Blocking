from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .normalization import (
    extract_pin_code,
    normalize_address,
    normalize_name,
    tokenize_address,
    tokenize_name,
)


def build_source_index(
    df,
    source_prefix: str,
    entity_col: str = "entity_id",
    name_col: str = "business_name",
    address_col: str = "business_address",
    country_col: str = "country",
) -> Dict[str, Dict[str, object]]:
    """Build a small per-row dictionary with normalized text fields for blocking."""
    index = {}
    for _, row in df.iterrows():
        entity_id = str(row.get(entity_col, "")).strip()
        if not entity_id:
            continue
        name_value = row.get(name_col, "") or ""
        address_value = row.get(address_col, "") or ""
        country_value = str(row.get(country_col, "") or "").strip()
        index[entity_id] = {
            "source": source_prefix,
            "name": name_value,
            "address": address_value,
            "country": country_value,
            "normalized_name": normalize_name(name_value),
            "normalized_address": normalize_address(address_value),
            "tokens_name": tokenize_name(name_value),
            "tokens_address": tokenize_address(address_value),
            "pin": extract_pin_code(address_value),
        }
    return index


def _deduplicate_pairs(pairs: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    pairs_list = []
    for left, right in pairs:
        key = (left, right)
        if key in seen:
            continue
        seen.add(key)
        pairs_list.append(key)
    return pairs_list


def _candidate_block(
    source1: Dict[str, Dict[str, object]],
    source2: Dict[str, Dict[str, object]],
    source3: Dict[str, Dict[str, object]],
    use_source2: bool = True,
    use_source3: bool = True,
) -> List[Tuple[str, str]]:
    all_pairs: List[Tuple[str, str]] = []
    all_sources = []
    if use_source2:
        all_sources.append(source2)
    if use_source3:
        all_sources.append(source3)

    for s1_id, s1_info in source1.items():
        s1_name = s1_info["normalized_name"]
        s1_tokens = s1_info["tokens_name"]
        s1_addr_tokens = s1_info["tokens_address"]
        s1_pin = s1_info["pin"]
        s1_country = str(s1_info.get("country", "")).strip().lower()

        for source in all_sources:
            for s_other_id, s_other_info in source.items():
                other_name = s_other_info["normalized_name"]
                other_tokens = s_other_info["tokens_name"]
                other_addr_tokens = s_other_info["tokens_address"]
                other_pin = s_other_info["pin"]
                other_country = str(s_other_info.get("country", "")).strip().lower()

                if s1_name and s1_name == other_name:
                    all_pairs.append((s1_id, s_other_id))
                    continue

                if s1_tokens and other_tokens:
                    prefix1 = " ".join(s1_tokens[:2])
                    prefix2 = " ".join(other_tokens[:2])
                    if prefix1 and prefix1 == prefix2:
                        all_pairs.append((s1_id, s_other_id))
                        continue

                    if set(s1_tokens) & set(other_tokens):
                        all_pairs.append((s1_id, s_other_id))
                        continue

                if s1_addr_tokens and s1_addr_tokens and set(s1_addr_tokens) & set(other_addr_tokens):
                    all_pairs.append((s1_id, s_other_id))
                    continue

                if s1_pin and s1_pin == other_pin:
                    all_pairs.append((s1_id, s_other_id))
                    continue

                if s1_country and s1_country == other_country:
                    all_pairs.append((s1_id, s_other_id))
                    continue

    return _deduplicate_pairs(all_pairs)


def generate_candidate_pairs(source1_df, source2_df, source3_df, output_path: str | Path | None = None):
    """Generate candidate pairs for Source 1 vs Source 2/3 using multiple blocking keys."""
    source1_index = build_source_index(source1_df, "source1")
    source2_index = build_source_index(source2_df, "source2")
    source3_index = build_source_index(source3_df, "source3")

    pairs = _candidate_block(source1_index, source2_index, source3_index)

    candidate_map: Dict[str, List[str]] = defaultdict(list)
    for s1_id, s_other_id in pairs:
        candidate_map[s1_id].append(s_other_id)

    for key in candidate_map:
        candidate_map[key] = sorted(set(candidate_map[key]))

    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for s1_id in sorted(source1_index):
            ids = candidate_map.get(s1_id, [])
            rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_ids": ",".join(ids),
            })
        import pandas as pd
        pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"]).to_csv(out_path, sep="\t", index=False)

    return candidate_map


def serialize_entity_ids(entity_ids: List[str]) -> str:
    """Serialize candidate ids as a comma-separated string for TSV output compatibility."""
    return ",".join(sorted(set(entity_ids)))
