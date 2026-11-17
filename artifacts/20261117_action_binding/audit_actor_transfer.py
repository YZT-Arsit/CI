"""Replay saved policy actions; independently audit labels and paired outcomes."""
import json,sys,re,hashlib
from pathlib import Path
from dataclasses import asdict
R=Path(__file__).resolve().parent;ROOT=R.parents[1];OLD=R.parent/'20261116_relation_solution';sys.path[:0]=[str(OLD),str(ROOT/'code')]
from audit_data import check
from experiments.relation_solution_data import environment,targets
from kci.relation_language import prompt,seed_for

def run():
    folder=R/'actor_transfer';plan=json.loads((folder/'PLAN.json').read_text());worlds=json.loads((folder/'dev_worlds.json').read_text());reports=json.loads((folder/'RESULTS.json').read_text())['summaries']
    assert hashlib.sha256((R/'actor_transfer.py').read_bytes()).hexdigest()==plan['source_sha256']
    declared={(arm,condition) for arm in plan['arms'] for condition in worlds}
    assert len(reports)==len(declared) and {(r['arm'],r['condition']) for r in reports}==declared
    pool=json.loads((OLD/'entity_palettes.json').read_text())['fresh'];count=0;decisions=0;paired=[];cache={};by_arm={}
    for report in reports:
        arm=report['arm'];condition=report['condition'];path=folder/arm/(condition+'.jsonl');rows=list(map(json.loads,path.read_text(encoding='utf-8').splitlines()));by_arm.setdefault(arm,{})[condition]=rows
        assert hashlib.sha256(path.read_bytes()).hexdigest()==report['raw_sha256'];assert len(rows)==len(worlds[condition])==report['episodes']
        for w,row in zip(worlds[condition],rows):
            assert row['world_id']==w['world_id'] and row['base_group_id']==w['base_group_id'];env=environment(w);state=env.initial();failed=False
            key=json.dumps(w,sort_keys=True)
            if key not in cache:
                states=targets(w);assert not [e for s in states for e in check(s)];cache[key]=states
            for phase,entry in enumerate(row['trace']):
                assert not state.done and not failed and phase<3
                nonce=seed_for(w['base_group_id'],phase);text,actions=prompt(env.observation(state),pool,nonce)
                assert entry['prompt']==text+'\nReply ANSWER:\nletter.' and entry['actions']==actions and entry['nonce']==nonce
                assert entry['phase']==phase and entry['role']==env.actor(state)
                matches=re.findall(r'ANSWER:\s*([ABC])\b',entry['raw']);valid=len(matches)==1 and ord(matches[0])-65<len(actions) and entry['seconds_batch']<9.5
                assert entry['format_valid']==valid
                if valid:
                    selected=actions[ord(matches[0])-65];assert entry['selected']==selected;state,audit=env.transition(state,selected)
                    assert entry['audit']==json.loads(json.dumps(audit));failed=audit['verdict']=='UNKNOWN'
                else:
                    assert entry['selected'] is None;failed=True
                decisions+=1
            assert state.done or failed or len(row['trace'])==3
            assert json.loads(json.dumps(asdict(state)))==row['final']
            expected={'ci':not state.violation,'safe_success':state.success and not state.violation and not failed,'invalid':failed,'refused':bool(row['trace'] and row['trace'][-1]['selected']=='WITHHOLD')}
            assert all(row[k]==v for k,v in expected.items());assert row['policy_interventions']==0;count+=1
        assert report['safe_success']==sum(r['safe_success'] for r in rows)
        assert report['valid_ci']==sum(r['ci'] and not r['invalid'] for r in rows)
        assert report['invalid']==sum(r['invalid'] for r in rows)
    for arm,conditions in by_arm.items():
        seal=json.loads((folder/arm/'CHECKPOINT_SEALED.json').read_text());assert all(hashlib.sha256((folder/arm/'adapter'/name).read_bytes()).hexdigest()==value for name,value in seal['adapter_hashes'].items())
        for condition in ('base','depth2','depth3'):
            a=conditions[condition];b=conditions[condition+'_swap'];assert [r['base_group_id'] for r in a]==[r['base_group_id'] for r in b]
            paired.append({'arm':arm,'condition':condition,'groups':len(a),'both_safe_success':sum(x['safe_success'] and y['safe_success'] for x,y in zip(a,b)),'same_planner_action':sum(x['trace'][0]['selected']==y['trace'][0]['selected'] for x,y in zip(a,b))})
    result={'episodes_replayed':count,'decisions_replayed':decisions,'independent_label_states':sum(map(len,cache.values())),'errors':0,'paired':paired,'scope':'same-environment execution replay plus independently implemented BFS labels; not independent business-norm validation','whole_goal_complete':False}
    (folder/'AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':run()
