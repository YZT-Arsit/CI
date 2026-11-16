"""Verify the fixed constructor rejects inconsistent rendered source facts."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'code'))
from kci.multi_env import FAMILIES, MultiRuleCIEnv, build_case

rows = []
for family in FAMILIES:
    original = build_case(family)
    sources = {k:v for k,v in original.facts.items() if k in {*original.context_ids,'payload_0','notification_target'}}
    rule = json.loads(original.facts['task_rule'].value)
    for fid, fact in sources.items():
        changes = [('subject', replace(fact, subject=fact.subject+'_changed')),
                   ('text', replace(fact, text=fact.text+' Extra undeclared text.'))]
        if fid == 'notification_target':
            changes.append(('value', replace(fact, value='DifferentPerson')))
        if fid == 'payload_0':
            value = json.loads(fact.value)
            if isinstance(value, int):
                value += 1
            elif 'total' in value:
                value['total'] += 1
            else:
                value['members'].append('DifferentPerson')
            changes.append(('value', replace(fact, value=json.dumps(value, sort_keys=True))))
        for field, altered in changes:
            bad = dict(sources, **{fid:altered})
            try:
                MultiRuleCIEnv(family, bad, rule, world_id='inconsistent_source_recheck')
                rejected, reason = False, None
            except ValueError as error:
                rejected, reason = True, str(error)
            rows.append({'family':family,'fact_id':fid,'changed_field':field,'rejected':rejected,'reason':reason})
            assert rejected
result = {'checks':len(rows),'all_rejected':all(r['rejected'] for r in rows),'rows':rows,
          'source_sha256':sha256((ROOT/'code/kci/multi_env.py').read_bytes()).hexdigest(),
          'scope':'canonical typed-source consistency only; not validation of arbitrary natural-language semantics'}
out = Path(__file__).with_suffix('.json')
if out.exists():
    raise FileExistsError('new audit output required')
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
