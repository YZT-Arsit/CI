"""Recompute structural/retention scores and replay structural actor episodes."""
import json,sys,re,hashlib
from pathlib import Path
from dataclasses import asdict
R=Path(__file__).resolve().parent;ROOT=R.parents[1];OLD=R.parent/'20261116_relation_solution';sys.path[:0]=[str(OLD),str(ROOT/'code')]
from audit_data import check
from kci.relation_language import prompt,seed_for
from kci.action_binding_metrics import pair_metrics
from experiments.relation_solution_data import environment

def run():
    structural=json.loads((R/'structural_cases/dataset.json').read_text());query_data=json.loads((R/'query_pilot/dataset.json').read_text());pool=json.loads((OLD/'entity_palettes.json').read_text())['fresh'];queries=0;episodes=0;decisions=0
    assert not [e for row in structural['states'] for e in check(row)]
    for folder,names in [('structural_cases',['query_independent','query_paired_margin']),('query_retention',['direct','query_warmup'])]:
        result=json.loads((R/folder/'RESULTS.json').read_text())
        for name in names:
            path=R/folder/(name+'.jsonl');seal=json.loads((R/folder/(name+'_SEALED.json')).read_text());assert hashlib.sha256(path.read_bytes()).hexdigest()==seal['raw_sha256'];rows=list(map(json.loads,path.read_text().splitlines()));assert len(rows)==96
            for r in rows:
                if folder=='structural_cases':
                    group,candidate=r['pair_id'].rsplit('_c',1);pair=[x for x in structural['states'] if x['base_group_id']==group];nonce=seed_for(group,7331)
                else:
                    depth,n,candidate=map(int,re.fullmatch(r'd(\d+)_g(\d+)_c(\d+)',r['pair_id']).groups());pair=query_data['dev'][str(depth)][n];nonce=seed_for(f'query_dev_{depth}_{n}',7331)
                row=pair[r['side']];candidate=row['candidates'][int(candidate)];text,_=prompt(row['observation'],pool,nonce,question='task',candidate=candidate['action']);text=text.replace('A yes; B no.','Reply ANSWER: followed by 1 if possible, otherwise 0.');assert r['prompt']==text and r['optimal_action']==str(candidate['safe_completion'])
                matches=re.findall(r'ANSWER:\s*([01])\b',r['raw']);selected=matches[0] if len(matches)==1 and r['seconds_batch']<9.5 else None;assert r['selected']==selected;queries+=1
            for summary in result['summaries']:
                if summary.get('model',summary.get('arm'))==name:assert pair_metrics([r for r in rows if r['depth']==summary['depth']])==summary.get('candidate_pairs',summary.get('pairs'))
    for name in ('actor_direct','actor_query_warmup'):
        path=R/'structural_cases'/(name+'.jsonl');seal=json.loads((R/'structural_cases'/(name+'_SEALED.json')).read_text());assert hashlib.sha256(path.read_bytes()).hexdigest()==seal['raw_sha256'];rows=list(map(json.loads,path.read_text().splitlines()));assert len(rows)==48
        for w,row in zip(structural['worlds'],rows):
            assert row['world_id']==w['world_id'];env=environment(w);state=env.initial();failed=False
            for phase,entry in enumerate(row['trace']):
                assert not state.done and not failed;nonce=seed_for(w['base_group_id'],phase);text,actions=prompt(env.observation(state),pool,nonce);assert entry['prompt']==text+'\nReply ANSWER:\nletter.' and entry['actions']==actions
                matches=re.findall(r'ANSWER:\s*([ABC])\b',entry['raw']);valid=len(matches)==1 and ord(matches[0])-65<len(actions) and entry['seconds_batch']<9.5;assert entry['format_valid']==valid
                if valid:
                    selected=actions[ord(matches[0])-65];assert selected==entry['selected'];state,audit=env.transition(state,selected);assert json.loads(json.dumps(audit))==entry['audit'];failed=audit['verdict']=='UNKNOWN'
                else:failed=True;assert entry['selected'] is None
                decisions+=1
            assert state.done or failed or len(row['trace'])==3;assert json.loads(json.dumps(asdict(state)))==row['final'];assert row['safe_success']==(state.success and not state.violation and not failed) and row['invalid']==failed and row['ci']==(not state.violation);episodes+=1
        reports=json.loads((R/'structural_cases/RESULTS.json').read_text())['summaries']
        for depth in (3,4,6):
            subset=[r for w,r in zip(structural['worlds'],rows) if len(w['providers'][0]['links'])==depth];report=next(s for s in reports if s['model']==name and s['depth']==depth)
            assert report['both_safe_success']==sum(a['safe_success'] and b['safe_success'] for a,b in zip(subset[::2],subset[1::2])) and report['safe_success']==sum(r['safe_success'] for r in subset)
    result={'query_outputs_checked':queries,'structural_actor_episodes_replayed':episodes,'structural_actor_decisions_replayed':decisions,'independent_structural_label_states':len(structural['states']),'errors':0};(R/'SECONDARY_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':run()
