from pathlib import Path

import pandas as pd


root = Path(__file__).resolve().parent / 'sample_dataset'
train = root / 'train'
test = root / 'test'
train.mkdir(parents=True, exist_ok=True)
test.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(Path(__file__).resolve().parent / 'synthetic_business_data.tsv', sep='\t', dtype=str, keep_default_na=False)
source1 = data.iloc[:5].copy().reset_index(drop=True)
source2 = data.iloc[5:10].copy().reset_index(drop=True)
source3 = data.iloc[5:10].copy().reset_index(drop=True)

source1.to_csv(train / 'train_source1.tsv', sep='\t', index=False)
source2.to_csv(train / 'train_source2.tsv', sep='\t', index=False)
source3.to_csv(train / 'train_source3.tsv', sep='\t', index=False)

truth = pd.DataFrame({
    'source1_entity_id': source1['entity_id'],
    'matched_entity_ids': ['B1', 'B2', 'B3', 'B4', 'B5'],
})
truth.to_csv(train / 'train_ground_truth.tsv', sep='\t', index=False)

source1.to_csv(test / 'test_source1.tsv', sep='\t', index=False)
source2.to_csv(test / 'test_source2.tsv', sep='\t', index=False)
source3.to_csv(test / 'test_source3.tsv', sep='\t', index=False)

print(f'Created synthetic sample dataset at {root}')
