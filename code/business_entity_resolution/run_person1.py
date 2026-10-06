from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from code.business_entity_resolution.data_loader import load_source_files
from code.business_entity_resolution.blocking import generate_candidate_pairs
from code.business_entity_resolution.eda import run_eda

base = Path(__file__).resolve().parents[2]
dataset_dir = base / 'student_resource' / 'dataset'
train_dir = dataset_dir / 'train'
test_dir = dataset_dir / 'test'
out_dir = base / 'output'
out_dir.mkdir(exist_ok=True)

datasets = load_source_files(train_dir, test_dir)
print('train_source1_rows', len(datasets['train_source1']))
print('train_source2_rows', len(datasets['train_source2']))
print('train_source3_rows', len(datasets['train_source3']))
print('ground_truth_rows', len(datasets['train_ground_truth']))

cand = generate_candidate_pairs(
    datasets['train_source1'],
    datasets['train_source2'],
    datasets['train_source3'],
    out_dir / 'candidate_pairs.tsv',
)
print('candidate_total_pairs', sum(len(v) for v in cand.values()))
print('sample_source1_entities', list(cand.keys())[:3])
print('sample_candidates', {k: cand[k][:3] for k in list(cand.keys())[:3]})

summary = run_eda(str(train_dir), str(test_dir), str(out_dir))
print('eda_summary_keys', list(summary.keys())[:5])
print('done')
