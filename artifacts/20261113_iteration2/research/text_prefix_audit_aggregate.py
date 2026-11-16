from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path

base=Path(__file__).with_name('text_prefix_audit')
files=[base/(f+'.json') for f in ('identity_attribute','alias_chain_3','aggregate_difference','set_intersection')]
docs=[json.loads(p.read_text(encoding='utf-8')) for p in files]
totals=defaultdict(Counter);original=defaultdict(Counter);first=defaultdict(Counter)
for doc in docs:
    for key,value in doc['by_variant'].items():totals[key].update(value)
    for key,value in doc['original_initial_rollout_counts'].items():original[key].update(value)
    for key,value in doc['original_initial_A_action_counts'].items():first[key].update(value)
out={'family_workers':len(docs),'independent_prefix_states':sum(d['prefix_states_independently_enumerated'] for d in docs),
     'replayed_episodes':sum(d['replayed_episodes'] for d in docs),'replayed_policy_actions':sum(d['replayed_policy_actions'] for d in docs),
     'by_variant':dict(totals),'original_initial_rollout_counts':dict(original),'original_initial_A_action_counts':dict(first),
     'max_internal_worker_seconds':max(d['seconds'] for d in docs),'must_fix_items':[],
     'scope':'Post-hoc lawful-prefix coverage diagnostic within existing family-heldout synthetic source; no new independent test or learning-efficacy estimate',
     'input_sha256':{p.name:sha256(p.read_bytes()).hexdigest() for p in files},
     'aggregate_source_sha256':sha256(Path(__file__).read_bytes()).hexdigest()}
(base/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k not in {'input_sha256','by_variant','original_initial_rollout_counts','original_initial_A_action_counts'}}))
