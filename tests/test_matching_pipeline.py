import pandas as pd

from code.business_entity_resolution.matching_features import generate_pair_features
from code.business_entity_resolution.match_scoring import (
    classify_match_score,
    resolve_one_to_one_matches,
    score_candidate_pair,
)


def test_generate_pair_features_for_available_columns():
    left = {
        'entity_id': 'A1',
        'business_name': 'Acme Logistics Ltd',
        'business_address': '12 Market Street, Bengaluru, Karnataka',
        'country': 'India',
        'city': 'Bengaluru',
        'state': 'Karnataka',
    }
    right = {
        'entity_id': 'B1',
        'business_name': 'Acme Logistics Limited',
        'business_address': '12 Market Street Bengaluru Karnataka',
        'country': 'India',
        'city': 'Bengaluru',
        'state': 'Karnataka',
    }

    features = generate_pair_features(left, right)

    assert 'name_similarity' in features
    assert 'address_similarity' in features
    assert features['name_similarity'] > 0.8
    assert features['address_similarity'] > 0.8
    assert features['country_match'] == 1.0
    assert features['city_match'] == 1.0


def test_score_candidate_pair_and_thresholds():
    left = {
        'entity_id': 'A1',
        'business_name': 'Acme Logistics',
        'business_address': '12 Main Road, Bangalore',
        'country': 'India',
    }
    right = {
        'entity_id': 'B1',
        'business_name': 'Acme Logistics',
        'business_address': '12 Main Road Bangalore',
        'country': 'India',
    }

    features = generate_pair_features(left, right)
    score = score_candidate_pair(features)

    assert score >= 0.75
    assert classify_match_score(score, 0.55, 0.75) == 'match'
    assert classify_match_score(0.80, 0.70, 0.85) == 'review'


def test_resolve_one_to_one_matches_prefers_best_score():
    candidate_pairs = [
        {'source_entity_id': 'A1', 'candidate_entity_id': 'B1', 'score': 0.91, 'status': 'match'},
        {'source_entity_id': 'A1', 'candidate_entity_id': 'B2', 'score': 0.88, 'status': 'match'},
        {'source_entity_id': 'A2', 'candidate_entity_id': 'B1', 'score': 0.84, 'status': 'match'},
    ]

    resolved = resolve_one_to_one_matches(candidate_pairs, high_threshold=0.8, low_threshold=0.6)

    assert resolved['A1'] == 'B1'
    assert 'B1' not in resolved.values() or list(resolved.values()).count('B1') == 1
