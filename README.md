# Business Entity Resolution

This project was developed as part of the Amazon ML Challenge. It implements an end-to-end business entity resolution workflow that starts with blocking, adds pairwise matching features, scores candidate pairs, applies configurable thresholds, and resolves one-to-one matches.

## Problem statement

Business entity resolution links records from multiple sources that refer to the same company or organization, even when names, addresses, and identifiers are noisy, abbreviated, or inconsistent. The challenge is to maintain high recall during candidate generation while limiting false positives when deciding final matches.

## Why entity resolution is difficult

- names can be shortened, reordered, or abbreviated
- addresses can vary by punctuation, spelling, and locality descriptors
- source systems may use different country/state conventions
- duplicate records and partial matches create ambiguity
- match cardinality can be one-to-one or one-to-many in practice

## Dataset structure

The repository expects tab-separated source files with columns such as:

- source entity id
- business name
- business address
- country
- city/state/locality fields when present
- optional phone, email, website, or registration fields

The challenge training data includes source 1, source 2, source 3, and a ground-truth mapping file.

## Pipeline architecture

1. Data loading
2. EDA and data quality profiling
3. Text normalization
4. Blocking and candidate generation
5. Pairwise feature generation
6. Weighted similarity scoring
7. Match classification and thresholding
8. One-to-one resolution
9. Evaluation and error analysis

## Data preprocessing

The project normalizes names and addresses using:

- Unicode normalization
- removal of punctuation and extra whitespace
- lowercasing
- consistent expansion for common abbreviations
- token-based matching for names and addresses
- postal-code extraction for PIN-like fields

## Normalization

The normalization layer is implemented in `code/business_entity_resolution/normalization.py`. It preserves useful business tokens such as district, street, town, and postal code fragments while reducing superficial noise from punctuation and formatting differences.

## Blocking

The blocking layer remains the conservative candidate-generation stage from the original Person 1 project. It combines several blocking keys such as:

- exact normalized name match
- shared name tokens
- address token overlap
- PIN/postal-code match
- country match

This keeps recall high while reducing the search space before matching.

## Candidate generation

Candidate pairs are generated in `code/business_entity_resolution/blocking.py` and written to output files such as `candidate_pairs.tsv` when needed. Each source 1 record retains a list of candidate source 2 / source 3 records that pass the blocking rules.

## Pairwise feature generation

The matching stage uses `code/business_entity_resolution/matching_features.py` to compute deterministic features for every candidate pair. Features include:

- normalized name similarity
- exact normalized name match
- address similarity
- token overlap
- country, city, state, and PIN comparisons
- identifier similarity for phone/email/website/registration-like fields when present

RapidFuzz is used for normalized string similarity because it is fast, deterministic, and appropriate for business-name and address matching.

## Matching and scoring

The scoring layer is in `code/business_entity_resolution/match_scoring.py` and uses a transparent weighted approach:

- name score is weighted most heavily
- address similarity is the next strongest signal
- location and identifier features provide additional evidence

The implementation is configurable through `code/business_entity_resolution/config.py`, which stores weights and thresholds rather than hard-coding them in the logic.

## Thresholding and match decision

The pipeline classifies each pair as:

- match: score >= high threshold
- review: low threshold <= score < high threshold
- unmatch: score < low threshold

This keeps the decision interpretable and reproducible.

## Final entity resolution

The one-to-one resolution step prefers the highest scoring valid candidate while preventing multiple source entities from matching the same target entity. The final output is written to `resolved_entities.csv` and an ambiguity report is saved separately.

## Evaluation

When ground truth is available, the pipeline computes:

- precision
- recall
- F1-score
- true positives / false positives / false negatives
- blocking metrics such as recall, average candidates per entity, and reduction ratio

This keeps blocking performance separate from final matching performance.

## Optional machine learning model

A lightweight logistic-regression matcher is included in `code/business_entity_resolution/ml_matcher.py` as an optional supervised alternative. It uses the same pairwise features, keeping the process explainable and easy to inspect.

## Project structure

- `code/business_entity_resolution/` — library modules for blocking, normalization, scoring, and pipeline execution
- `tests/` — regression tests and a small synthetic dataset generator
- `output/` — generated scored pairs and final resolved outputs
- `run_pipeline.py` — CLI entry point for the full process
- `Documentation_template.md` — report template used during challenge documentation
- `requirements.txt` — Python dependencies

## Installation

```bash
py -m pip install -r requirements.txt
```

## Usage

### Run the full pipeline on the synthetic sample dataset

```bash
py run_pipeline.py --train-dir tests/sample_dataset/train --test-dir tests/sample_dataset/test --ground-truth tests/sample_dataset/train/train_ground_truth.tsv --output-dir output
```

### Run the same approach on a challenge dataset

```bash
py run_pipeline.py --train-dir dataset/train --test-dir dataset/test --ground-truth dataset/train/train_ground_truth.tsv --output-dir output
```

## Example outputs

The pipeline writes files such as:

- `scored_pairs.csv`
- `resolved_entities.csv`
- `unmatched_entities.csv`
- `ambiguous_matches.csv`
- `evaluation_metrics.json`
- `matching_report.csv`

## Limitations

- the real challenge dataset is not committed to the repository
- blocking and matching performance depends on the dataset quality and column coverage
- some fields may be absent in some sources, so the system only generates features for columns that exist
- final match quality should be interpreted as internal evaluation unless challenge-level metrics are explicitly available

## Future improvements

- tune weights with labeled training data
- add more domain-specific similarity rules for postal codes and legal names
- add threshold sweep analysis using precision/recall curves
- add richer visual reporting in `output/plots/`

## Amazon ML Challenge context

This project was developed as part of the Amazon ML Challenge and focuses on practical engineering for entity resolution rather than claiming any competition result or placement.

- K.YUVRAJ SUNDARAM.
