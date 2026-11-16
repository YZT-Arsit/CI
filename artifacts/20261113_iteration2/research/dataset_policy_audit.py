"""Independent read-only audit of exported SFT controls and finite-policy jobs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'code'))
from experiments.text_policy_probe import action_class, dpo_gradient, forward

def load(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def canon(x): return json.dumps(json.loads(x) if isinstance(x,str) else x,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def digest(p): return sha256(p.read_bytes()).hexdigest()

def main():
    start=time.perf_counter();base=ROOT/'artifacts/20261113_iteration2'
    grouped=defaultdict(list);raw={};all_raw_ids=set()
    for path in sorted((base/'dataset_raw_v2').glob('*.json')):
        doc=load(path);assert doc['enumeration_complete']
        for state in doc['states']:grouped[state['prompt']].append((doc['case_spec'],state))
        for row in doc['preferences']:
            key=sha256('\n'.join(row[k] for k in ('prompt','chosen','rejected')).encode()).hexdigest()
            raw.setdefault(key,[]).append(row['metadata']['raw_record_id'])
            all_raw_ids.add(row['metadata']['raw_record_id'])
    exported=rows(base/'dataset_v2/preferences.jsonl');sft=rows(base/'dataset_v2/sft_controls.jsonl')
    quarantine=rows(base/'dataset_v2/quarantine.jsonl')
    excluded={r['preference_signature'] for r in rows(base/'research/observable_full_v2/inconsistent_preferences.jsonl')}
    assert {r['metadata']['preference_id'] for r in exported}==set(raw)-excluded
    assert {r['metadata']['raw_record_id'] for r in quarantine}=={rid for k in excluded for rid in raw[k]}
    for row in exported:
        assert set(row['metadata']['contributing_raw_ids'])==set(raw[row['metadata']['preference_id']])
        assert action_class(row['chosen'])!=action_class(row['rejected'])
    actual={r['prompt']:r for r in sft};expected={};families=defaultdict(set);menu_collisions=0
    for prompt,members in grouped.items():
        optimal=[]
        for spec,state in members:
            families[prompt].add(spec['family'])
            valid=[a for a in state['actions'] if a['immediate']=='valid' and a['minimum_safe_total_cost'] is not None]
            best=min((a['minimum_safe_total_cost'] for a in valid),default=None)
            optimal.append({canon(a['action']) for a in valid if a['minimum_safe_total_cost']==best})
            menu=[action_class(a['action']) for a in state['actions']]
            menu_collisions+=len(menu)-len(set(menu))
        common=set.intersection(*optimal)
        if common:expected[prompt]=common
    assert set(actual)==set(expected)
    for prompt,row in actual.items():
        assert row['chosen'] in expected[prompt]
        assert row['compatible_states']==len(grouped[prompt])
        assert row['rule_family_id'] in families[prompt]
    assert menu_collisions==0
    assert all(len(f)==1 for f in families.values())
    sources=Counter(tuple(r['metadata']['training_sources']) for r in exported)
    feature_dir=base/'text_features_v2';meta=load(feature_dir/'metadata.json');data=np.load(feature_dir/'features.npz')
    for name,hash_ in meta['inputs_sha256'].items():assert hash_==digest(base/'dataset_v2'/name)
    assert meta['preference_ids']==[r['metadata']['preference_id'] for r in exported]
    result_paths=sorted((base/'text_policy_v2').glob('*/result.json'));assert len(result_paths)==40
    checks=[];common_reference={};reference_duplicates=0;worst_fd=0.;fold_prompt_overlaps={}
    for heldout in sorted(set(meta['families'])):
        train_prompts={r['prompt'] for r in exported if r['metadata']['rule_family_id']!=heldout}
        test_prompts={r['prompt'] for r in exported if r['metadata']['rule_family_id']==heldout}
        sft_prompts={r['prompt'] for r in sft if r['rule_family_id']!=heldout}
        overlap=len((train_prompts|sft_prompts)&test_prompts)
        assert overlap==0;fold_prompt_overlaps[heldout]=overlap
    for path in result_paths:
        job=load(path);held=job['heldout_family'];seed=job['seed'];variant=job['variant']
        assert job['weights_sha256']==digest(path.parent/'weights.npz')
        assert job['feature_metadata_sha256']==digest(feature_dir/'metadata.json')
        train_ids=np.flatnonzero(np.array(meta['families'])!=held)
        for step,batch in enumerate(job['batch_schedule']):
            assert all(meta['families'][i]!=held for i in batch)
            src=('ci','future','repair')[step%3] if variant=='balanced' else variant
            if src in {'ci','future','repair'}:assert all(src in meta['sources'][i] for i in batch)
        sft_ids=np.flatnonzero(np.array(meta['sft_families'])!=held)
        rng=np.random.default_rng(seed);rng.normal(0,.1,(24,512));rng.normal(0,.1,(19,24))
        batches=rng.choice(sft_ids,size=(200,16),replace=True)
        assert sha256(batches.tobytes()).hexdigest()==job['sft_batches_sha256']
        weights=np.load(path.parent/'weights.npz');key=(held,seed)
        refs=(weights['reference_u'],weights['reference_e'])
        if key in common_reference:
            assert all(np.array_equal(a,b) for a,b in zip(refs,common_reference[key]));reference_duplicates+=1
        else:common_reference[key]=refs
        if variant=='sft_only':assert np.array_equal(weights['u'],refs[0]) and np.array_equal(weights['e'],refs[1])
        ids=train_ids[:4];x=data['x'][ids];c=data['chosen'][ids];r=data['rejected'][ids]
        _,reference_scores=forward(x,*refs);refdiff=reference_scores[np.arange(4),c]-reference_scores[np.arange(4),r]
        u=weights['u'];e=weights['e'];loss,gu,ge=dpo_gradient(x,c,r,u,e,refdiff)
        # Central differences independently check both learned parameter arrays.
        for which,gradient in [('u',gu),('e',ge)]:
            index=np.unravel_index(np.argmax(np.abs(gradient)),gradient.shape);eps=1e-6
            plus=(u if which=='u' else e).copy();minus=plus.copy();plus[index]+=eps;minus[index]-=eps
            high=dpo_gradient(x,c,r,plus if which=='u' else u,plus if which=='e' else e,refdiff)[0]
            low=dpo_gradient(x,c,r,minus if which=='u' else u,minus if which=='e' else e,refdiff)[0]
            error=abs((high-low)/(2*eps)-gradient[index]);worst_fd=max(worst_fd,float(error));assert error<1e-6
        assert all(ep['case'].startswith(held+'_') for ep in job['episodes'])
        checks.append({'job':path.parent.name,'batches':len(job['batch_schedule']),'passed':True})
    report={'scope':'independent export/optimization audit, finite text-feature policy, not LLM evidence',
      'raw_preferences':sum(map(len,raw.values())),'accepted_unique_preferences':len(exported),
      'quarantined_raw_preferences':len(quarantine),'sft_controls_recomputed':len(expected),
      'sft_classes_rejected':len(grouped)-len(expected),'same_prompt_cross_family_classes':sum(len(f)>1 for f in families.values()),
      'within_state_action_class_collisions':menu_collisions,'source_membership_sets':{'|'.join(k):v for k,v in sources.items()},
      'repair_adds_unique_pairs_beyond_ci_future':sum('repair' in s and not ('ci' in s or 'future' in s) for s in [r['metadata']['training_sources'] for r in exported]),
      'train_or_sft_vs_test_exact_prompt_overlaps':fold_prompt_overlaps,'jobs_checked':len(checks),
      'same_fold_seed_reference_weight_equality_checks':reference_duplicates,'both_parameter_gradient_max_absolute_error':worst_fd,
      'balanced_source_batch_counts':{'ci':67,'future':67,'repair':66},'job_checks':checks,
      'limits':['one workflow/source; four declared structural families, not four independent tasks',
                'hashed visible text and fixed canonical action menu, no autoregressive language modeling',
                'SFT targets conditional on finite enumerated compatible support and frozen exact-action grammar',
                'repair source overlaps existing CI/future pairs; balanced is reweighting',
                'two seeds are engineering sensitivity, not population-level method evidence'],
      'inputs_sha256':{'preferences':digest(base/'dataset_v2/preferences.jsonl'),'sft':digest(base/'dataset_v2/sft_controls.jsonl'),
                      'policy_source':digest(ROOT/'code/experiments/text_policy_probe.py'),'audit_source':digest(Path(__file__))},
      'seconds':time.perf_counter()-start}
    out=Path(__file__).with_suffix('.json');out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'job_checks','inputs_sha256'}},ensure_ascii=False))

if __name__=='__main__':main()
