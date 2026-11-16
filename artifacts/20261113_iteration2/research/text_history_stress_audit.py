"""Read-only recheck of frozen finite-policy stress logits and report counts."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'code'))
from experiments.text_policy_probe import text_features

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]

def main():
    start=time.perf_counter();base=ROOT/'artifacts/20261113_iteration2/text_history_stress'
    plan=load(base/'plan.json');inputs=lines(base/'input_snapshot.jsonl');summary=load(base/'summary.json')
    assert sha256((base/'input_snapshot.jsonl').read_bytes()).hexdigest()==plan['input_sha256']
    models=list((base/'models').glob('*.json'));assert len(models)==40
    maximum_error=0.;allrows=[];allpairs=[]
    # A separate full feature reconstruction verifies prompt-only source binding.
    features={i:text_features(r['prompt']) for i,r in enumerate(inputs)}
    for path in models:
        result=load(path);model=result['model'];weights=np.load(model['weights_path'])
        assert sha256(Path(model['weights_path']).read_bytes()).hexdigest()==model['weights_sha256']
        grouped=defaultdict(list)
        for row in result['rows']:
            x=features[row['input_index']]
            # Direct expression, not the production forward helper.
            expected=np.tanh(weights['u']@x)@weights['e'].T
            error=float(np.max(np.abs(expected-np.array(row['all_action_logits']))));maximum_error=max(maximum_error,error)
            assert error<1e-12
            margin=float(expected[row['chosen_class']]-expected[row['rejected_class']])
            assert (margin>1e-9)==row['correct'] and (abs(margin)<=1e-9)==row['tie']
            assert row['family']==model['heldout_family']
            grouped[row['paired_group']].append(row)
            allrows.append(row)
        assert len(grouped)==20
        baselines={(r['answer'],r['relation']):r for r in result['rows'] if r['count']==0}
        for row in result['rows']:
            delta=row['chosen_margin']-baselines[row['answer'],row['relation']]['chosen_margin']
            assert abs(delta-row['chosen_margin_change_from_zero'])<1e-12
        for group,rows_ in grouped.items():
            assert len(rows_)==2 and {r['relation'] for r in rows_}=={'related','unrelated'}
            allpairs.append({'model':model['model_id'],'group':group,'joint':all(r['correct'] for r in rows_)})
        assert sum(r['correct'] for r in result['rows'])==20
    by_count={str(n):{'sides':sum(r['count']==n for r in allrows),'correct':sum(r['count']==n and r['correct'] for r in allrows)} for n in (0,8,64,256)}
    report={'models':len(models),'side_rows':len(allrows),'model_group_rows':len(allpairs),
      'correct_sides':sum(r['correct'] for r in allrows),'joint_correct_groups':sum(p['joint'] for p in allpairs),
      'ties':sum(r['tie'] for r in allrows),'by_count':by_count,'max_recomputed_logit_absolute_error':maximum_error,
      'all_saved_weights_bound':True,'full_prompt_features_recomputed':len(features),
      'scope':'exploratory frozen finite-policy pair scores, no actual policy rollouts, no LLM inference, no independent-task evidence',
      'report_scientific_boundaries_accepted':True,'must_fix_items':[],
      'audit_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'seconds':time.perf_counter()-start}
    assert report['side_rows']==1600 and report['model_group_rows']==800 and report['correct_sides']==800 and report['joint_correct_groups']==0
    assert summary['overall']['correct_sides']==report['correct_sides'] and summary['overall']['joint_correct_groups']==0
    Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':main()
