from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from rapidfuzz import fuzz

from .normalization import normalize_address, normalize_name


def _as_text(value: Any) -> str:
    if value is None or value != value:
        return ""
    return str(value).strip()


def _normalize_for_similarity(value: Any) -> str:
    text = _as_text(value)
    if not text:
        return ""
    text = text.lower().replace("&", " and ")
    text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    text = " ".join(text.split())
    return text


def _pick_first_available(row: Dict[str, Any], candidates: Iterable[str]) -> str:
    for field in candidates:
        if field in row and row.get(field) is not None:
            return _as_text(row.get(field))
    return ""


def _compute_overlap(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    overlap = len(left_set & right_set) / max(1, len(left_set | right_set))
    return round(overlap, 6)


def _similarity_score(left: Any, right: Any, method: str = "token_sort") -> float:
    left_text = _normalize_for_similarity(left)
    right_text = _normalize_for_similarity(right)
    if not left_text and not right_text:
        return 1.0
    if not left_text or not right_text:
        return 0.0
    if left_text == right_text:
        return 1.0
    if method == "ratio":
        score = fuzz.ratio(left_text, right_text)
    elif method == "partial":
        score = fuzz.partial_ratio(left_text, right_text)
    elif method == "token_set":
        score = fuzz.token_set_ratio(left_text, right_text)
    else:
        score = fuzz.token_sort_ratio(left_text, right_text)
    return round(score / 100.0, 6)


def _exact_match(left: Any, right: Any) -> float:
    left_text = _normalize_for_similarity(left)
    right_text = _normalize_for_similarity(right)
    return 1.0 if left_text and left_text == right_text else 0.0


def _generate_identifier_similarity(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    field_groups = [
        ("phone", ["phone", "phone_number", "mobile", "telephone"]),
        ("email", ["email", "email_address"]),
        ("website", ["website", "url", "web_domain", "domain"]),
        ("registration", ["registration_number", "registration_no", "company_registration", "tax_id"]),
    ]
    scores = []
    for _, choices in field_groups:
        a = _pick_first_available(left, choices)
        b = _pick_first_available(right, choices)
        if not a or not b:
            continue
        if a.lower() == b.lower():
            scores.append(1.0)
        else:
            scores.append(_similarity_score(a, b, method="token_set"))
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 6)


def generate_pair_features(left_row: Dict[str, Any], right_row: Dict[str, Any]) -> Dict[str, float]:
    """Build deterministic pairwise features for a candidate source pair."""
    features: Dict[str, float] = {}

    name_left = _pick_first_available(left_row, ["business_name", "name", "company_name", "entity_name"])
    name_right = _pick_first_available(right_row, ["business_name", "name", "company_name", "entity_name"])
    if name_left or name_right:
        features["name_similarity"] = _similarity_score(normalize_name(name_left), normalize_name(name_right))
        features["name_exact_match"] = _exact_match(normalize_name(name_left), normalize_name(name_right))
        features["name_token_overlap"] = _compute_overlap(
            [token for token in normalize_name(name_left).split() if token],
            [token for token in normalize_name(name_right).split() if token],
        )
    else:
        features["name_similarity"] = 0.0
        features["name_exact_match"] = 0.0
        features["name_token_overlap"] = 0.0

    address_left = _pick_first_available(left_row, ["business_address", "address", "street_address", "office_address"])
    address_right = _pick_first_available(right_row, ["business_address", "address", "street_address", "office_address"])
    if address_left or address_right:
        features["address_similarity"] = _similarity_score(normalize_address(address_left), normalize_address(address_right))
        features["address_exact_match"] = _exact_match(normalize_address(address_left), normalize_address(address_right))
        features["address_token_overlap"] = _compute_overlap(
            [token for token in normalize_address(address_left).split() if token],
            [token for token in normalize_address(address_right).split() if token],
        )
    else:
        features["address_similarity"] = 0.0
        features["address_exact_match"] = 0.0
        features["address_token_overlap"] = 0.0

    country_left = _pick_first_available(left_row, ["country", "country_code"])
    country_right = _pick_first_available(right_row, ["country", "country_code"])
    features["country_match"] = 1.0 if country_left and country_right and country_left.lower() == country_right.lower() else 0.0

    city_left = _pick_first_available(left_row, ["city", "town", "district"])
    city_right = _pick_first_available(right_row, ["city", "town", "district"])
    features["city_match"] = 1.0 if city_left and city_right and city_left.lower() == city_right.lower() else 0.0

    state_left = _pick_first_available(left_row, ["state", "province", "region"])
    state_right = _pick_first_available(right_row, ["state", "province", "region"])
    features["state_match"] = 1.0 if state_left and state_right and state_left.lower() == state_right.lower() else 0.0

    pin_left = _pick_first_available(left_row, ["postal_code", "zip_code", "pin", "pincode", "postalcode"])
    pin_right = _pick_first_available(right_row, ["postal_code", "zip_code", "pin", "pincode", "postalcode"])
    features["pin_match"] = 1.0 if pin_left and pin_right and pin_left.lower() == pin_right.lower() else 0.0

    features["identifier_similarity"] = _generate_identifier_similarity(left_row, right_row)
    features["location_similarity"] = round(
        (
            features["country_match"]
            + features["city_match"]
            + features["state_match"]
            + features["pin_match"]
        )
        / max(1, sum(1 for value in [country_left, city_left, state_left, pin_left] if value))
        if any([country_left, city_left, state_left, pin_left])
        else 0.0,
        6,
    )
    return features


def map_candidate_pair_to_features(source1_row: Dict[str, Any], source2_row: Dict[str, Any]) -> Dict[str, float]:
    return generate_pair_features(source1_row, source2_row)
