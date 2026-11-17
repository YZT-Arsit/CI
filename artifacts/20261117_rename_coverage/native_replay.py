"""Native replay; copied auditor with explicit run.py source resolution only."""
import json,sys,re,hashlib
from pathlib import Path
from dataclasses import asdict
R=Path(__file__).resolve().parent;ROOT=R.parents[1];OLD=R.parent/'20261116_relation_solution';sys.path[:0]=[str(OLD),str(ROOT/'code')]
from audit_data import check
from kci.relation_language import prompt,seed_for
from experiments.relation_solution_data import environment,targets

def run(run_name='learned_evidence', worker_source='run.py'):
    folder=R/run_name;plan=json.loads((folder/'PLAN.json').read_text());worlds=json.loads((folder/'dev_worlds.json').read_text());reports=json.loads((folder/'RESULTS.json').read_text())['summaries'];pool=json.loads((R/'palettes.json').read_text())['new'];count=0;decisions=0;query_count=0;labels=0;stats={};by_arm={};initial_correct={};cache={};evidence_arms=plan.get('evidence_arms',['evidence'])
    assert hashlib.sha256((R/worker_source).read_bytes()).hexdigest()==plan['source_sha256'];assert len(reports)==len(plan['arms'])*len(worlds) and {(r['arm'],r['condition']) for r in reports}=={(a,c) for a in plan['arms'] for c in worlds}
    for report in reports:
        arm=report['arm'];condition=report['condition'];path=folder/arm/(condition+'.jsonl');rows=list(map(json.loads,path.read_text(encoding='utf-8').splitlines()));assert len(rows)==len(worlds[condition])==report['episodes'];assert hashlib.sha256(path.read_bytes()).hexdigest()==report['raw_sha256'];by_arm.setdefault(arm,{})[condition]=rows;initial_correct[(arm,condition)]=[]
        for w,row in zip(worlds[condition],rows):
            assert row['world_id']==w['world_id'] and row['base_group_id']==w['base_group_id'];env=environment(w);state=env.initial();failed=False;bad_query=0;key=json.dumps(w,sort_keys=True)
            if key not in cache:
                cases=targets(w);assert not [e for case in cases for e in check(case)];cache[key]=cases;labels+=len(cases)
            cases=cache[key]
            for phase,entry in enumerate(row['trace']):
                assert not state.done and not failed and phase<3;obs=env.observation(state);nonce=seed_for(w['base_group_id'],phase);text,actions=prompt(obs,pool,nonce);expected=next(case for case in cases if case['observation']==obs);estimates={}
                queries=entry['queries'];assert len(queries)==(len(obs['actions']) if arm in evidence_arms else 0)
                if arm in evidence_arms:assert [q['action'] for q in queries]==obs['actions']
                for q in queries:
                    qtext,_=prompt(obs,pool,nonce,question='task',candidate=q['action']);qtext=qtext.replace('A yes; B no.','Reply ANSWER: followed by 1 if possible, otherwise 0.');assert q['prompt']==qtext
                    matches=re.findall(r'ANSWER:\s*([01])\b',q['raw']);valid=len(matches)==1 and q['seconds_batch']<9.5;estimate=matches[0] if valid else 'unknown';assert q['valid']==valid and q['estimate']==estimate;estimates[q['action']]=estimate;bad_query+=int(not valid);query_count+=1
                    target=str(next(c['safe_completion'] for c in expected['candidates'] if c['action']==q['action']));stats_key=(arm+'_' if len(evidence_arms)>1 else '')+condition+'_'+obs['role'];s=stats.setdefault(stats_key,{'queries':0,'correct':0,'invalid':0});s['queries']+=1;s['correct']+=estimate==target;s['invalid']+=not valid
                if arm in evidence_arms:text+='Own estimates (1 means compliant completion remains possible): '+ '; '.join(chr(65+i)+'='+estimates[a] for i,a in enumerate(actions))+'.\n'
                assert entry['prompt']==text+'\nReply ANSWER:\nletter.' and entry['actions']==actions and entry['role']==obs['role'] and entry['phase']==phase
                matches=re.findall(r'ANSWER:\s*([ABC])\b',entry['raw']);valid=len(matches)==1 and ord(matches[0])-65<len(actions) and entry['seconds_batch']<9.5;assert entry['format_valid']==valid
                if valid:
                    selected=actions[ord(matches[0])-65];assert selected==entry['selected'];state,audit=env.transition(state,selected);assert json.loads(json.dumps(audit))==entry['audit'];failed=audit['verdict']=='UNKNOWN'
                else:selected=None;assert entry['selected'] is None;failed=True
                if phase==0:initial_correct[(arm,condition)].append(selected in [expected['candidates'][i]['action'] for i in expected['optimal_indices']])
                decisions+=1
            assert state.done or failed or len(row['trace'])==3;assert json.loads(json.dumps(asdict(state)))==row['final'];success=state.success and not state.violation and not failed
            assert row['safe_success']==success and row['strict_success']==(success and not bad_query) and row['invalid']==failed and row['invalid_queries']==bad_query and row['ci']==(not state.violation) and row['policy_interventions']==0;count+=1
        assert all(report[k]==sum(r[k] for r in rows) for k in ('safe_success','strict_success','invalid','invalid_queries'));assert report['valid_ci']==sum(r['ci'] and not r['invalid'] for r in rows)
    paired=[]
    for arm,conditions in by_arm.items():
        seal=json.loads((folder/arm/'CHECKPOINT_SEALED.json').read_text());assert all(hashlib.sha256((folder/arm/'adapter'/p).read_bytes()).hexdigest()==h for p,h in seal['adapter_hashes'].items())
        for condition in plan.get('paired_conditions',['base','depth2','depth3']):
            a=conditions[condition];b=conditions[condition+'_swap'];assert [r['base_group_id'] for r in a]==[r['base_group_id'] for r in b]
            paired.append({'arm':arm,'condition':condition,'groups':len(a),'both_strict_success':sum(x['strict_success'] and y['strict_success'] for x,y in zip(a,b)),'both_planner_correct':sum(x and y for x,y in zip(initial_correct[(arm,condition)],initial_correct[(arm,condition+'_swap')]))})
    result={'episodes_replayed':count,'decisions_replayed':decisions,'queries_reconstructed':query_count,'independent_label_states':labels,'paired':paired,'on_policy_query_accuracy_not_matched_across_arms':stats,'errors':0,'whole_goal_complete':False};(folder/'AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='on_policy_query_accuracy_not_matched_across_arms'}))

if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else 'learned_evidence')
