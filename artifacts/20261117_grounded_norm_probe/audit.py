"""Reconstruct source-scoped inputs and score all orders without changing UNKNOWN."""
import hashlib
import json
import re
import sys
from collections import Counter,defaultdict
from pathlib import Path

R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'20261117_native_agent_probe'))
from tokens import render,CONFIG
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
rows=lambda p:list(map(json.loads,p.read_text(encoding='utf-8').splitlines()))


def parse(raw, transport_ok=True):
    strict=re.fullmatch(r'\s*(PASS|FAIL|UNKNOWN)\s*',raw)
    secondary=re.fullmatch(r'\s*(?:ANSWER:\s*)?(PASS|FAIL|UNKNOWN)[.!]?\s*',raw,re.I)
    return {'strict':strict[1] if strict and transport_ok else None,
            'secondary':secondary[1].upper() if secondary and transport_ok else None}


def main():
    plan=read(R/'PLAN.json');run=read(R/'RUN_PLAN.json');seal=read(R/'RESPONSES_SEALED.json')
    assert run['plan_sha256']==sha(R/'PLAN.json') and not run['reads_labels'] and run['retries']==0
    assert all(sha(R/p)==h for p,h in plan['sources'].items())
    assert all(sha(R/p)==h for p,h in seal['files'].items())
    assert sha(R/'observations.jsonl')==plan['observations_sha256']
    assert not seal['labels_scored']
    assert sha(R/'contexts.json')==plan['contexts_sha256']
    assert sha(R.parent/'20261117_native_agent_probe/tokens.py')==plan['tokens_source_sha256']
    assert plan['tokenizer']=={k:CONFIG[k] for k in plan['tokenizer']}
    source=R.parent/'20261117_action_binding/policy_cases_v2'
    assert sha(source/'cases.jsonl')==plan['source_cases_sha256'] and sha(source/'labels.jsonl')==plan['source_labels_sha256']
    cases=rows(source/'cases.jsonl');labels={x['id']:x for x in rows(source/'labels.jsonl')}
    observations=rows(R/'observations.jsonl');responses=rows(R/'responses.jsonl') if (R/'responses.jsonl').exists() else []
    assert len(responses)==seal['completed_calls'] and len(observations)==84
    assert len(cases)==28 and len(responses)<=84
    assert [(o['case_id'],o['view']) for o in observations]==[(i,v) for i in range(28) for v in range(3)]
    assert [(r['case_id'],r['view']) for r in responses]==[(o['case_id'],o['view']) for o in observations[:len(responses)]]
    tasks={'internal_bug_discussion':'Discuss a customer-related bug internally.',
           'support_log_collaboration':'Collaborate on troubleshooting logs for customer support.',
           'local_log_reproduction':'Reproduce a customer log issue locally.',
           'local_database_reproduction':'Reproduce a customer database issue locally.'}
    contexts=read(R/'contexts.json')
    stats=defaultdict(Counter);scored=[]
    for obs,row in zip(observations,responses):
        assert (obs['case_id'],obs['view'])==(row['case_id'],row['view'])
        case=cases[obs['case_id']];assert case['id']==obs['source_case_id']
        assert contexts[case['clause_id']] in obs['prompt'] and case['question'] in obs['prompt']
        assert json.dumps(case['facts'],sort_keys=True) in obs['prompt']
        assert case['id'] not in obs['prompt'] and case['base_condition_group'] not in obs['prompt']
        rendered,tokens=render(obs['prompt']);assert rendered==obs['rendered_prompt'] and tokens==obs['prompt_tokens']
        expected=['PASS','FAIL','UNKNOWN'];expected=expected[obs['view']:]+expected[:obs['view']]
        assert obs['order']==expected and ', '.join(expected) in obs['prompt']
        exact=(contexts[case['clause_id']]+'\nTask: '+tasks[case['task_template']]+
               '\nHypothetical facts: '+json.dumps(case['facts'],sort_keys=True)+'\nQuestion: '+case['question']+
               '\nPASS means the named requirements are satisfied; FAIL means a named requirement is violated; '
               'UNKNOWN means missing information prevents either conclusion.\nAllowed answers: '+', '.join(expected)+
               '. Return exactly one allowed word, without explanation.')
        assert obs['prompt']==exact
        response=row['response'];raw=response['response']
        transport_ok=(row['seconds']<=25 and response.get('done') is True and response.get('done_reason')!='length'
                      and response.get('prompt_eval_count')==tokens and tokens+12+64<=2048)
        answers=parse(raw,transport_ok)
        truth=labels[case['id']]['label']
        for mode,answer in answers.items():
            stats[mode,truth].update(presentations=1,agree=int(answer==truth),unknown=int(answer=='UNKNOWN'),invalid=int(answer is None))
        scored.append({'case_id':obs['case_id'],'view':obs['view'],'group':case['base_condition_group'],
                       'clause':case['clause_id'],'label':truth,**answers})
    complete=seal['complete'] and len(responses)==84
    if responses:
        loaded=read(R/'LOADED_MODEL.json')
        active=next(row for row in loaded['models'] if row['name']==plan['model']['name'])
        assert active['digest']==run['model_digest']==plan['model']['digest'] and active['size_vram']==0
    metrics=[]
    for mode in ['strict','secondary']:
        by_case=defaultdict(list);by_group=defaultdict(list)
        for row in scored:by_case[row['case_id']].append(row);by_group[row['group']].append(row)
        metrics.append({'mode':mode,'labels':{label:dict(stats[mode,label]) for label in ['PASS','FAIL','UNKNOWN']},
            'all_three_agree':sum(len(v)==3 and all(r[mode]==r['label'] for r in v) for v in by_case.values()),
            'all_three_valid_same_choice':sum(len(v)==3 and v[0][mode] is not None and len({r[mode] for r in v})==1 for v in by_case.values()),
            'all_six_paired_task_agree':sum(len(v)==6 and all(r[mode]==r['label'] for r in v) for v in by_group.values()),
            'macro_agreement':sum(stats[mode,l]['agree']/stats[mode,l]['presentations'] if stats[mode,l]['presentations'] else 0 for l in ['PASS','FAIL','UNKNOWN'])/3})
    result={'completed':len(responses),'planned':84,'complete':complete,'metrics':metrics,
            'constant_baseline_agreement':{l:sum(r['label']==l for r in scored) for l in ['PASS','FAIL','UNKNOWN']},
            'max_seconds':max((r['seconds'] for r in responses),default=None),'seconds_sum':sum(r['seconds'] for r in responses),
            'release_verified':read(R/'RELEASE_CHECK.json')['model_absent'] if (R/'RELEASE_CHECK.json').exists() else False,'human_verified':False,
            'trained_policy_transfer':False,'executed_business_tasks':0,'whole_goal_complete':False}
    (R/'SCORED.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))
    (R/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    main()
