from pathlib import Path
import pandas as pd

base = Path(r'C:\Users\K BHRAMEE SUNDARAM\Desktop\Amazon ML\student_resource\dataset')
for name in ['train_source1.tsv','train_source2.tsv','train_source3.tsv','train_ground_truth.tsv','test_source1.tsv','test_source2.tsv','test_source3.tsv']:
    p = base / 'train' / name if 'train_' in name else base / 'test' / name
    df = pd.read_csv(p, sep='\t', dtype=str, keep_default_na=False)
    print(f'\n--- {name} ---')
    print('shape:', df.shape)
    print('columns:', list(df.columns))
    print(df.head(3).to_string(index=False))
