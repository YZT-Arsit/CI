"""Post-result TRAIN residual inspection; never changes primary scoring."""
import hashlib
import json
from pathlib import Path

R=Path(__file__).resolve().parent
readrows=lambda p:list(map(json.loads,p.read_text().splitlines()))
old=R.parent/'20261117_frozen_crossover'
train=readrows(old/'train_oracle.jsonl'); results={}; paths=[old/'train_oracle.jsonl']
for arm in ['teacher_sft','continue_original_lr','continue_lower_lr']:
    path=old/'oracle_train_teacher_sft.jsonl' if arm=='teacher_sft' else R/'repair'/arm/'train_fit.jsonl'
    paths.append(path); records=readrows(path);assert len(records)==len(train)==320
    bad=[]
    for case,row in zip(train,records):
        assert row['case_id']==case['case_id']
        actual=row if arm=='teacher_sft' else row['actor']
        best={case['candidates'][j]['action'] for j in case['optimal_indices']}
        if actual['selected'] not in best:
            bad.append({'case_id':case['case_id'],'role':case['observation']['role'],
                'safe_completion_exists':any(c['safe_completion'] for c in case['candidates']),
                'selected':actual['selected'],'full_position':'ABC'[actual['actions'].index('PUBLISH_FULL')],
                'raw':actual['raw'],'optimal_actions':sorted(best)})
    results[arm]=bad
result={'errors_by_arm':results,'same_error_case_ids':len({tuple(r['case_id'] for r in v) for v in results.values()})==1,
        'input_hashes':{p.relative_to(R.parents[1]).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'scope':'Post-result association on repeated TRAIN presentations. Isolated order intervention required before causal position attribution.',
        'new_model_calls':0,'whole_goal_complete':False}
(R/'RESIDUAL_TRAINING.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'counts':{k:len(v) for k,v in results.items()},'same_error_case_ids':result['same_error_case_ids']}))
