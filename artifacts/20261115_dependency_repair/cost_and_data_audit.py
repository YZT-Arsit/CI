"""Task route/cost and complete-token diagnostics; no policy training."""
from pathlib import Path
from dataclasses import asdict
from hashlib import sha256
from copy import deepcopy
import argparse,json,sys

R=Path(__file__).resolve().parent;ROOT=R.parents[1];sys.path.insert(0,str(ROOT/'code'))
from experiments.task_evidence_dependency_probe import SOURCE
from kci.dynamic_tasks import DynamicTask,canonical


def routes(kind):
    records=[]
    for related in (False,True):
      for allow in (False,True):
        env=DynamicTask(kind,SOURCE,related=related,allow=allow,horizon=30)
        initial=env.decision_start();groups=[]
        for action in env.options(initial):
            after,audit=env.transition(initial,action);stack=[(after,[action])];terminals=[]
            while stack:
                s,path=stack.pop()
                if s.done:
                    if not s.success or s.violation:continue
                    assert s.payload_bytes==sum(len(canonical(m['content']).encode('utf-8')) for m in s.ledger)
                    client=next(m for m in s.ledger if m['recipient']=='Client')
                    ancestors=set()
                    def visit(ref):
                        if ref in ancestors:return
                        ancestors.add(ref)
                        if isinstance(ref,int):
                            for parent in next(m for m in s.ledger if m['event_id']==ref)['inputs']:visit(parent)
                    visit(client['event_id'])
                    sources=[a for a in s.access_log if a['access_id'] in ancestors]
                    required={'business_constraints','public_schedule'} if kind=='meeting' else {'company_id_map','company_gas_stations','company_locations'}
                    assert required.issubset({a['resource'] for a in sources})
                    terminals.append({'path':path,'steps':s.turn,'payload_bytes':s.payload_bytes,
                        'task_value_without_cost_or_deadline':[1,1,0],
                        'client_ancestor_sources':sources,'final_state_sha256':sha256(canonical(asdict(s)).encode()).hexdigest()})
                else:
                    for next_action in env.options(s):
                        nxt,check=env.transition(s,next_action)
                        if check['verdict']=='VALID':stack.append((nxt,path+[next_action]))
            assert terminals
            groups.append({'first_action':action,'safe_successful_routes':terminals,
                'min_steps':min(x['steps'] for x in terminals),'min_payload_bytes':min(x['payload_bytes'] for x in terminals),
                'cost_removed_best_value':[1,1,0]})
        choices={}
        for metric in ('min_steps','min_payload_bytes'):
            a,b=(x[metric] for x in groups);choices[metric]=0 if a<b else 1 if b<a else None
        choices['cost_removed']=None
        records.append({'kind':kind,'related':related,'allow':allow,'horizon':30,'branches':groups,
            'best_continuation_choices':choices,'scope':'exact finite legal continuation optimum; not a reference-policy mean'})
    out={'cases':records,'all_source_ancestor_checks_passed':True,
         'units':'UTF-8 canonical content bytes, excluding provenance envelope/network/compute; steps are synthetic API calls',
         'finding':'Both first actions admit safe success when deadlines and cost are removed; no strict initial task-value ordering then.'}
    (R/f'cost_routes_{kind}.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps([{'related':r['related'],'allow':r['allow'],'choices':r['best_continuation_choices'],
        'branches':[{k:g[k] for k in ('first_action','min_steps','min_payload_bytes')} for g in r['branches']]} for r in records]))


def tokens():
    from kci.training_data import PinnedTokenizerAssets,TrainingSequenceAdapter,PreferenceCorpusIndex,IGNORE_INDEX
    assets=PinnedTokenizerAssets.from_manifest(ROOT/'artifacts/20261112_cpu_pilot/tokenization/config.json')
    adapter=TrainingSequenceAdapter(assets);index=PreferenceCorpusIndex();rows=[]
    input_path=R/'training_steps_v2_final/all_preferences.jsonl'
    for text in input_path.read_text(encoding='utf-8').splitlines():
        row=json.loads(text);index.add(row);pair=adapter.encode_pair(row)
        assert row['metadata']['gpu_training_eligible'] is False
        dirty=deepcopy(row);dirty['metadata']['hidden_future_cost']='CANARY_799'
        twin=adapter.encode_pair(dirty);assert twin.prompt_ids==pair.prompt_ids and twin.chosen==pair.chosen and twin.rejected==pair.rejected
        for seq in (pair.chosen,pair.rejected):
            assert seq.labels.count(adapter.eos_token_id)==1
            assert all(x==IGNORE_INDEX for x in seq.labels[:seq.prompt_length])
            assert row['prompt'] in adapter.tokenizer.decode(list(seq.input_ids),skip_special_tokens=False)
        rows.append({'record_id':row['metadata']['record_id'],'required_tokens':pair.required_tokens,
            'metadata_noninterference':True,'gpu_training_eligible':False})
    out={'input_sha256':sha256(input_path.read_bytes()).hexdigest(),'rows':rows,'fully_encoded':len(rows),
         'min_tokens':min(x['required_tokens'] for x in rows),'max_tokens':max(x['required_tokens'] for x in rows),
         'truncation':False,'gpu_training_eligible':False,
         'adapter_training_eligible_means':'schema eligibility only; research/GPU readiness remains a separate false flag',
         'assets':adapter.asset_provenance}
    (R/'tokenization_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ('rows','assets')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode');a=p.parse_args()
    tokens() if a.mode=='tokens' else routes(a.mode)
