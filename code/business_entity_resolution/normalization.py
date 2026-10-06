from __future__ import annotations

import re
import unicodedata
from typing import List


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("&", " and ")
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value


def normalize_name(name: str | None) -> str:
    """Normalize a business name while retaining useful words and numeric tokens."""
    value = _normalize_text(name)
    stopwords = {
        "the", "and", "of", "for", "in", "on", "at", "a", "an", "co", "company",
        "corp", "corporation", "inc", "llc", "ltd", "limited", "private", "pvt",
        "services", "service", "group", "solutions", "solutionss", "enterprise",
    }
    tokens = [tok for tok in value.split() if tok and tok not in stopwords]
    return " ".join(tokens)


def normalize_address(address: str | None) -> str:
    """Normalize an address while retaining numbers, locality names, and PIN-like patterns."""
    value = _normalize_text(address)
    abbreviations = {
        "street": "street",
        "st": "street",
        "road": "road",
        "rd": "road",
        "lane": "lane",
        "ln": "lane",
        "avenue": "avenue",
        "ave": "avenue",
        "boulevard": "boulevard",
        "blvd": "boulevard",
        "drive": "drive",
        "dr": "drive",
        "court": "court",
        "ct": "court",
        "sector": "sector",
        "sec": "sector",
        "district": "district",
        "block": "block",
        "blk": "block",
        "unit": "unit",
        "phase": "phase",
        "village": "village",
        "town": "town",
        "city": "city",
        "state": "state",
    }
    cleaned = []
    for token in value.split():
        if token in abbreviations:
            cleaned.append(abbreviations[token])
        else:
            cleaned.append(token)
    return " ".join(cleaned)


def tokenize_name(name: str | None) -> List[str]:
    """Tokenize a normalized business name."""
    return [tok for tok in normalize_name(name).split() if tok]


def tokenize_address(address: str | None) -> List[str]:
    """Tokenize a normalized address."""
    return [tok for tok in normalize_address(address).split() if tok]


def extract_numbers(value: str | None) -> List[str]:
    """Extract digit sequences from a string, preserving their order."""
    if value is None:
        return []
    return re.findall(r"\d+", str(value))


def extract_pin_code(value: str | None) -> str:
    """Extract the first 4- to 8-digit postal code-like sequence from a string."""
    numbers = extract_numbers(value)
    if not numbers:
        return ""
    for token in numbers:
        if 4 <= len(token) <= 8:
            return token
    return numbers[0]
