"""Enumerate two finite adapted task contracts and quarantine observation conflicts."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from kci.dynamic_tasks import DynamicTask, canonical


def signature(x):return sha256(canonical(x).encode()).hexdigest()


def generate(root,source):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    cases=[];raw=[];states=[];repairs=[];initial=[]
    for kind in ('meeting','query'):
      for related in (False,True):
       for allow in (False,True):
        for horizon in ((6,9) if kind=='meeting' else (7,10)):
          e=DynamicTask(kind,source,related=related,allow=allow,horizon=horizon)
          case=f'{kind}_h{int(related)}_a{int(allow)}_t{horizon}'
          start=e.decision_start();queue=[(start,['DELIVER_HISTORY'])];seen=set();case_states=[]
          while queue:
            s,prefix=queue.pop();sid=signature({'contract':e.binding,'state':asdict(s)})
            if sid in seen or s.done:continue
            seen.add(sid);prompt=e.observation(s);actions=[]
            for action in e.options(s):
                n,audit=e.transition(s,action)
                future=e.exact(n) if audit['verdict']=='VALID' else None
                actions.append({'action':action,'audit':audit,'future':future})
                if audit['verdict']=='VALID':queue.append((n,prefix+[action]))
            strict=[]
            for a,b in combinations(actions,2):
                def rank(x):
                    ref=x['future'];cost=ref['min_total_cost'] if ref else None
                    return (x['audit']['verdict']=='VALID',cost is not None,-cost if cost is not None else 0)
                ka,kb=rank(a),rank(b)
                if ka==kb:continue
                good,bad=(a,b) if ka>kb else (b,a)
                if not good['future'] or good['future']['min_total_cost'] is None:continue
                level=('immediate_CI' if bad['audit']['verdict']=='VIOLATION' else
                       'future_reachability' if bad['future']['min_total_cost'] is None else 'future_cost')
                strict.append((good['action'],bad['action']))
                raw.append({'case_id':case,'state_id':sid,'prefix':prefix,'prompt':canonical(prompt),
                    'chosen':good['action'],'rejected':bad['action'],'level':level,
                    'proof':{'chosen':good,'rejected':bad},'workflow':kind,
                    'source_hashes':e.source_hashes,'human_verified':False,'norm_source':'explicit_synthetic_adaptation'})
            record={'case_id':case,'state_id':sid,'state':asdict(s),'prefix':prefix,
                    'prompt':canonical(prompt),'deprived_prompt':canonical(e.observation(s,hide_receipts=True)),
                    'actions':actions,'strict':strict,'workflow':kind}
            states.append(record);case_states.append(record)
            for bad in actions:
                if bad['audit']['verdict']=='VIOLATION' or (bad['future'] and bad['future']['min_total_cost'] is None):
                    alternatives=[x for x in actions if x['future'] and x['future']['min_total_cost'] is not None]
                    if alternatives:
                        good=min(alternatives,key=lambda x:x['future']['min_total_cost'])
                        repairs.append({'case_id':case,'state_id':sid,'prefix':prefix,
                           'failed_action':bad['action'],'repair_action':good['action'],
                           'prompt':canonical(prompt),'witness':good['future']['witness'],
                           'failure_scope':'immediate_violation_or_exact_unreachable_continuation'})
          first=next(r for r in case_states if r['state']['phase']=='choose')
          initial.append({'case_id':case,'workflow':kind,'related':related,'allow':allow,'horizon':horizon,
                          'prompt':first['prompt'],'actions':first['actions']})
          cases.append({'case_id':case,'contract':e.binding,'states':len(case_states)})
    classes=defaultdict(list);deprived=defaultdict(list)
    for s in states:classes[s['prompt']].append(s);deprived[s['deprived_prompt']].append(s)
    accepted={};quarantine=[]
    for r in raw:
        direction=(r['chosen'],r['rejected'])
        if not all(direction in map(tuple,s['strict']) for s in classes[r['prompt']]):quarantine.append(r);continue
        key=(r['prompt'],*direction)
        if key not in accepted:accepted[key]={**r,'contributing_states':[],'preference_id':signature(key)}
        accepted[key]['contributing_states'].append([r['case_id'],r['state_id']])
    history=[]
    for kind in ('meeting','query'):
      for horizon in ((6,9) if kind=='meeting' else (7,10)):
        pair=[r for r in initial if r['workflow']==kind and r['horizon']==horizon and not r['allow']]
        choices=[]
        for r in pair:
            feasible=[a for a in r['actions'] if a['future']['min_total_cost'] is not None]
            choices.append(min(feasible,key=lambda a:a['future']['min_total_cost'])['action'])
        assert len(set(choices))==2
        assert all(a['audit']['verdict']=='VALID' for r in pair for a in r['actions'])
        history.append({'workflow':kind,'horizon':horizon,'cases':[r['case_id'] for r in pair],
                        'chosen':choices,'both_current_actions_legal':True,
                        'intervention':'one actual delivered history record, same task/norm/actions'})
    conflict_classes=sum(len({canonical(s['strict']) for s in group})>1 for group in deprived.values())
    repair_keys={(r['prompt'],r['repair_action'],r['failed_action']) for r in repairs}
    independent=repair_keys-set(accepted)
    for name,records in [('states',states),('preferences_raw',raw),('preferences',list(accepted.values())),
                         ('quarantine',quarantine),('repairs',repairs),('history_pairs',history)]:
        (root/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    summary={'cases':cases,'raw_pairs':len(raw),'unique_pairs':len(accepted),'quarantined_rows':len(quarantine),
       'levels':dict(Counter(r['level'] for r in accepted.values())),'history_flip_groups':len(history),
       'history_flip_unique_preferences':sum(json.loads(r['prompt'])['phase']=='choose' and
            not r['case_id'].split('_a')[1].startswith('1') for r in accepted.values()),
       'workflow_contracts':2,'source_instances':2,'parameter_configs_not_independent_tasks':True,
       'states':len(states),'visible_prompt_classes':len(classes),'deprived_conflict_classes':conflict_classes,
       'repair_states':len(repairs),'unique_repair_pairs':len(repair_keys),'repair_pairs_not_in_main':len(independent),
       'human_verified':False,'real_llm_learning_established':False}
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('source');a=p.parse_args();generate(a.output,a.source)
