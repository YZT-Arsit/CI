"""Verify frozen probability records and independently recompute greedy gaps."""
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
rows=lambda p:list(map(json.loads,p.read_text().splitlines()))
plan=read(R/'LOSS_DIAGNOSTIC_PLAN.json');result=read(R/'LOSS_DIAGNOSTIC_RESULTS.json')
assert sha(Path(__file__))==plan['auditor_sha256'] and sha(R/'loss_diagnostic.py')==plan['source_sha256']
assert sha(R/'loss_probabilities.jsonl')==result['probabilities_sha256']
obs=rows(R.parent/'20261117_option_fallback/observations.jsonl')[:108]
labels=rows(R.parent/'20261117_option_fallback/labels.jsonl')[:108]
records=rows(R/'loss_probabilities.jsonl');assert len(records)==324
timings=[];summaries=[]
for arm in read(R/'PLAN.json')['arms']:
    data=[r for r in records if r['arm']==arm];assert len(data)==108
    free=rows(R/'repair'/arm/'train_orders.jsonl')[:108];stats=Counter()
    for i,(r,o,l,f) in enumerate(zip(data,obs,labels,free)):
        assert r['id']==o['id']==l['id']==f['id']
        ps=r['probabilities_ABC'];assert len(ps)==3 and all(math.isfinite(p) and 0<=p<=1 for p in ps) and sum(ps)<=1.000001
        good=[j for j,a in enumerate(o['actions']) if a in l['optimal_actions']];bad=[j for j in range(3) if j not in good]
        assert len(good)==2 and len(bad)==1
        total=sum(ps[j] for j in good);goodmax=max(ps[j] for j in good);badmax=max(ps[j] for j in bad)
        assert r['optimal_union_probability']==total and r['best_optimal_probability']==goodmax and r['best_nonoptimal_probability']==badmax
        chosen=r['greedy_action'];correct=chosen in l['optimal_actions'];unsafe=chosen is not None and not correct
        if chosen is not None:assert ps[o['actions'].index(chosen)]==max(ps)
        prefix=re.fullmatch(r'ANSWER:\n([ABC])\s*',f['raw']);agree=prefix is not None and o['actions'][ord(prefix[1])-65]==chosen
        assert r['optimal']==correct and r['union_beats_bad']==(total>badmax)
        assert r['free_output_prefix_matches']==(prefix is not None) and r['free_action_agrees']==agree
        stats.update(n=1,optimal=int(correct),unsafe_greedy=int(unsafe),union_wins_but_greedy_unsafe=int(total>badmax and unsafe),
            qualified_union_wins_but_greedy_unsafe=int(total>badmax and unsafe and agree),other_argmax=int(chosen is None),
            prefix_matches=int(prefix is not None),free_agrees=int(agree))
        if i%4==0:
            assert len({x['seconds_batch'] for x in data[i:i+4]})==1 and 0<r['seconds_batch']<12;timings.append(r['seconds_batch'])
    summaries.append({'arm':arm,**dict(stats),'all_cases_interpretation_eligible':stats['prefix_matches']==stats['free_agrees']==108})
assert result['summaries']==summaries and result['forwards']==len(timings)==81
assert result['seconds']==sum(timings) and result['max_seconds']==max(timings) and result['model_updates']==0
receipt={'records':len(records),'summaries':summaries,'errors':0,'source_sha256':sha(Path(__file__))}
(R/'LOSS_DIAGNOSTIC_AUDIT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
