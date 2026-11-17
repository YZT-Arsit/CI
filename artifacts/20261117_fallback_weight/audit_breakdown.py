"""Descriptive raw violation counts; does not replace the frozen pilot gate."""
import hashlib
import json
from pathlib import Path

R=Path(__file__).resolve().parent
plan=json.loads((R/'PLAN.json').read_text())
seal=json.loads((R/'SEALED.json').read_text())
assert all(hashlib.sha256((R/n).read_bytes()).hexdigest()==h for n,h in seal['files'].items())
summaries=json.loads((R/'repair/RESULTS.json').read_text())['summaries']
output=[]
for arm in plan['arms']:
    conditions=[]
    for report in summaries:
        if report['arm']!=arm:continue
        rows=[json.loads(s) for s in (R/'repair'/arm/(report['condition']+'.jsonl')).read_text().splitlines()]
        conditions.append({'condition':report['condition'],'episodes':len(rows),
            'strict_success':sum(r['strict_success'] for r in rows),'violations':sum(not r['ci'] for r in rows),
            'invalid_episodes':sum(r['invalid'] for r in rows),'invalid_queries':sum(r['invalid_queries'] for r in rows),
            'valid_ci':sum(r['ci'] and not r['invalid'] for r in rows)})
    totals={k:sum(c[k] for c in conditions) for k in conditions[0] if k!='condition'}
    output.append({'arm':arm,'totals':totals,'conditions':conditions})
result={'descriptive_only':True,'frozen_gate_unchanged':True,'arms':output}
(R/'RAW_VIOLATIONS.json').write_text(json.dumps(result,indent=2))
print(json.dumps([{'arm':r['arm'],**r['totals']} for r in output]))
