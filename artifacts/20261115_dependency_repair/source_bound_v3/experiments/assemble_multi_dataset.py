"""Apply independent observable-label gate and build conservative data groups."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path


def digest(text):return sha256(text.encode('utf-8')).hexdigest()
def signature(row):return digest('\n'.join(row[k] for k in ('prompt','chosen','rejected')))
def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def action_identity(value):return canonical(json.loads(value) if isinstance(value,str) else value)
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def dump(path,rows):path.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--audit',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    source=Path(a.input);audit=Path(a.audit);out=Path(a.output)
    if out.exists():raise FileExistsError('preserve prior dataset release')
    out.mkdir(parents=True)
    audit_summary=read(audit/'summary.json')
    excluded={r['preference_signature'] for r in
              (json.loads(x) for x in (audit/'inconsistent_preferences.jsonl').read_text(encoding='utf-8').splitlines())}
    assert excluded and audit_summary['strict_preferences_not_uniformly_supported']==len(excluded)
    rows=[];states=defaultdict(list);repair_keys=set();raw_dpr=Counter();inputs={}
    for path in sorted(source.glob('*.json')):
        inputs[path.name]=sha256(path.read_bytes()).hexdigest()
        assert audit_summary['input_sha256'][path.name]==inputs[path.name]
        case=read(path);rows.extend(case['preferences'])
        for st in case['states']:
            states[st['prompt']].append((case['case_spec'],st))
        for repair in case['repair_diagnostics']:
            raw_dpr[repair['status']]+=1
            if repair['preference']:repair_keys.add(signature(repair['preference']))
    merged={};quarantine=[];source_memberships=Counter()
    for row in rows:
        key=signature(row)
        if key in excluded:quarantine.append(row);continue
        if key not in merged:
            merged[key]=deepcopy(row)
            merged[key]['metadata']['contributing_raw_ids']=[]
            merged[key]['metadata']['contributing_case_ids']=[]
        meta=merged[key]['metadata']
        meta['contributing_raw_ids'].append(row['metadata']['raw_record_id'])
        meta['contributing_case_ids'].append(row['metadata']['case_spec']['case_id'])
    for key,row in merged.items():
        meta=row['metadata'];meta['preference_id']=key
        sources=set(meta['training_sources'])
        if key in repair_keys:sources.add('repair')
        meta['training_sources']=sorted(sources)
        meta['observation_class_id']=digest(row['prompt'])
        meta['observable_equivalence_gate']='uniform_strict_preference_in_enumerated_compatible_states'
        meta['observation_audit_sha256']=sha256((audit/'summary.json').read_bytes()).hexdigest()
        meta['primary_independent_holdout']=False
        source_memberships.update(sources)
    filtered=list(merged.values())
    orientations={(r['prompt'],r['chosen'],r['rejected']) for r in filtered}
    assert not any((p,b,c) in orientations for p,c,b in orientations)
    dump(out/'preferences.jsonl',filtered);dump(out/'quarantine.jsonl',quarantine)
    # Shared-observation SFT controls use only actions optimal in EVERY compatible
    # enumerated state, with no unsalvageable member. They are same-workflow controls.
    sft=[];sft_rejected=0
    for prompt,members in states.items():
        pools=[]
        for _,state in members:
            safe=[x for x in state['actions'] if x['immediate']=='valid' and x['minimum_safe_total_cost'] is not None]
            if not safe:pools.append(set());continue
            cost=min(x['minimum_safe_total_cost'] for x in safe)
            pools.append({action_identity(x['action']) for x in safe if x['minimum_safe_total_cost']==cost})
        common=set.intersection(*pools)
        if not common:sft_rejected+=1;continue
        spec,st=members[0]
        menu=[action_identity(x['action']) for x in st['actions']]
        target=sorted(common)[0]
        sft.append({'prompt':prompt,'menu':menu,'chosen':target,
                    'rule_family_id':spec['family'],'base_case_group_id':spec['family']+'_skeleton_v1',
                    'workflow_group_id':'shared_ci_workflow_v1','observation_class_id':digest(prompt),
                    'compatible_states':len(members),'target_rule':'common_minimum_safe_cost_action',
                    'human_verified':False,'evidence_class':'synthetic_structural_SFT_control'})
    dump(out/'sft_controls.jsonl',sft)
    families=sorted({r['metadata']['rule_family_id'] for r in filtered})
    folds=[]
    for heldout in families:
        train=[r['metadata']['preference_id'] for r in filtered if r['metadata']['rule_family_id']!=heldout]
        test=[r['metadata']['preference_id'] for r in filtered if r['metadata']['rule_family_id']==heldout]
        assert not set(train)&set(test)
        folds.append({'heldout_family':heldout,'train_preference_ids':train,'test_preference_ids':test,
                      'scope':'shared_workflow_synthetic_structural_transfer_NOT_task_holdout'})
    (out/'family_folds.json').write_text(json.dumps(folds,indent=2),encoding='utf-8')
    summary={'raw_pairs':len(rows),'raw_quarantined':len(quarantine),'unique_quarantined':len(excluded),
             'accepted_unique_pairs':len(filtered),'raw_duplicate_pairs':len(rows)-len(quarantine)-len(filtered),
             'accepted_source_memberships':dict(source_memberships),
             'accepted_by_family':dict(Counter(r['metadata']['rule_family_id'] for r in filtered)),
             'repair_diagnostic_statuses':dict(raw_dpr),'repair_unique_pairs':len(repair_keys),
             'repair_retained_pairs':sum('repair' in r['metadata']['training_sources'] for r in filtered),
             'sft_common_optimal_controls':len(sft),'sft_observation_classes_rejected':sft_rejected,
             'workflow_groups':1,'source_world_groups':1,'declared_rule_families':len(families),
             'base_case_skeletons':len(families),'primary_independent_holdout':False,
             'family_folds':4,'human_verified':False,'natural_language_semantics_verified':False,
             'inputs_sha256':inputs,'audit_summary_sha256':sha256((audit/'summary.json').read_bytes()).hexdigest(),
             'source_sha256':sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='inputs_sha256'},ensure_ascii=False))


if __name__=='__main__':main()
