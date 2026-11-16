import sys,json,time
from pathlib import Path
from audit_semantics import audit
R=Path(__file__).resolve().parent
start=int(sys.argv[1]);out=R/'balanced_semantic_audits';out.mkdir(exist_ok=True)
for index in range(start,start+16,4):
    audit(R/f'dataset/train_balanced/worlds_{index:04d}.jsonl',out/f'{index:04d}.json')
