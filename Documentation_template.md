# Business Entity Resolution Documentation Template

## 1. Dataset Overview

- Data source names: Source 1, Source 2, Source 3 and the train ground-truth mapping
- Number of rows per source: measured during loading and EDA
- Missing value counts: tracked per column during profiling
- Duplicate counts: checked at row level and by entity identifier where available
- Country distribution: computed from the normalized country field when present

## 2. Data Quality Findings

- Name quality summary: lengths, missing values, and normalization issues are reported in the EDA summary
- Address quality summary: address completeness, punctuation inconsistencies, and PIN presence are tracked
- Match cardinality summary: number of true matches per source entity and duplicate match risks are summarized

## 3. Normalization Strategy

- Name normalization steps: Unicode normalization, punctuation removal, lowercase conversion, whitespace cleanup, business keyword stripping, and token preservation
- Address normalization steps: punctuation standardization, abbreviation normalization, and token-based canonicalization
- PIN extraction details: the first 4- to 8-digit postal-like sequence is extracted and compared when present

## 4. Blocking Strategy

- Exact normalized name match
- Name prefix/token blocking
- Address token overlap blocking
- PIN-based blocking
- Country-based blocking

## 5. Blocking Evaluation

- Blocking recall: proportion of true matches retained in the generated candidate set
- Average candidates per Source 1 entity: candidate size per entity
- Maximum candidates: worst-case candidate number for a single source entity
- Candidate reduction ratio: reduction versus the full pair space

## 6. Risks and Recommendations

- Potential false positives: common names or addresses with high token overlap can over-generate candidates
- Potential false negatives: short or highly abbreviated names may be missed by blocking rules
- Recommended tuning actions: continue tuning name/address token logic and review high-risk review cases

## 7. Matching Features

- Exact normalized name match
- Fuzzy name similarity using RapidFuzz token-sort and ratio metrics
- Address similarity and token overlap
- Country, city, state, and PIN comparisons
- Identifier matching when phone, email, website, or registration fields are available

## 8. Similarity Scoring

A transparent weighted score is used to combine the available features. Name similarity carries the highest weight, followed by address similarity, then location and identifier evidence. The overall score is normalized to a 0-1 scale and kept configurable through `config.py`.

## 9. Match Classification

Each candidate pair is classified as:

- match if it exceeds the high threshold
- review if it sits between the low and high threshold values
- unmatch if it falls below the low threshold

This decision layer keeps the match policy clear and auditable.

## 10. Threshold Selection

Thresholds should be selected based on the available ground truth. A lower threshold increases recall but may introduce more false positives, while a higher threshold improves precision but can miss valid matches. The final threshold choice should reflect the business objective and the acceptable review burden.

## 11. Final Resolution

The final resolution stage applies a one-to-one candidate policy: the highest-scoring valid match is selected for each source entity while conflicts are flagged for review rather than silently dropped.

## 12. Matching Evaluation

If ground truth is available, evaluation should include:

- precision
- recall
- F1-score
- confusion matrix summary
- true positives, false positives, and false negatives

The project tracks these metrics separately from blocking metrics to ensure the candidate-generation stage is not confused with final matching quality.

## 13. Error Analysis

The repository includes an error-analysis layer to inspect:

- false positives
- false negatives
- ambiguous review cases
- duplicate or conflicting matches
- low-confidence results

This helps identify whether errors are caused by blocking, feature quality, or threshold calibration.
