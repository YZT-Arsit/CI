"""Freeze source-context prompts without reading label values."""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

R=Path(__file__).resolve().parent
ROOT=R.parents[1]
sys.path.insert(0,str(R.parent/'20261117_native_agent_probe'))
from tokens import render, CONFIG

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R/'PLAN.json').exists()
source=R.parent/'20261117_action_binding/policy_cases_v2'
cases=list(map(json.loads,(source/'cases.jsonl').read_text().splitlines()))
contexts=json.loads((R/'contexts.json').read_text())
tasks={'internal_bug_discussion':'Discuss a customer-related bug internally.',
       'support_log_collaboration':'Collaborate on troubleshooting logs for customer support.',
       'local_log_reproduction':'Reproduce a customer log issue locally.',
       'local_database_reproduction':'Reproduce a customer database issue locally.'}
options=['PASS','FAIL','UNKNOWN']; records=[]
for i,case in enumerate(cases):
    for view in range(3):
        order=options[view:]+options[:view]
        prompt=(contexts[case['clause_id']]+'\nTask: '+tasks[case['task_template']]+
                '\nHypothetical facts: '+json.dumps(case['facts'],sort_keys=True)+
                '\nQuestion: '+case['question']+
                '\nPASS means the named requirements are satisfied; FAIL means a named requirement is violated; '
                'UNKNOWN means missing information prevents either conclusion.\nAllowed answers: '+', '.join(order)+
                '. Return exactly one allowed word, without explanation.')
        rendered,length=render(prompt)
        assert length+12+64<=2048
        records.append({'case_id':i,'source_case_id':case['id'],'view':view,'order':order,
                        'prompt':prompt,'rendered_prompt':rendered,'prompt_tokens':length})
(R/'observations.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
tags=json.load(urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=5))
model=next(row for row in tags['models'] if row['name']=='qwen2.5:7b-instruct-q4_K_M')
pin=json.loads((R.parent/'20261117_action_binding/SOURCE_PIN.json').read_text())
source_checks=[]
for item in pin['files'][:2]:
    cached=ROOT/'.cache/norm_probe_20261117'/Path(item['path']).name
    assert sha(cached)==item['sha256']
    source_checks.append({k:item[k] for k in ['path','sha256','permalink','revision']})
plan={'model':model,'cases':28,'presentations':84,'condition_groups':14,
      'source_files':source_checks,'source_cases_sha256':sha(source/'cases.jsonl'),
      'source_labels_sha256':sha(source/'labels.jsonl'),'contexts_sha256':sha(R/'contexts.json'),
      'observations_sha256':sha(R/'observations.jsonl'),'sources':{p.name:sha(p) for p in R.glob('*.py')},
      'tokenizer':{k:CONFIG[k] for k in ['official_repo','official_revision','tokenizer_files']},
      'tokens_source_sha256':sha(R.parent/'20261117_native_agent_probe/tokens.py'),
      'options':{'temperature':0,'seed':17,'num_ctx':2048,'num_predict':12,'num_gpu':0,'num_thread':4},
      'request_timeout_seconds':25,'max_prompt_tokens':max(row['prompt_tokens'] for row in records),
      'training_updates':0,'whole_goal_complete':False}
(R/'PLAN.json').write_text(json.dumps(plan,indent=2))
print(json.dumps({'presentations':84,'max_prompt_tokens':plan['max_prompt_tokens']}))
