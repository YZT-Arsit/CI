"""Pure static-source comparison; does not import/execute external agent code."""
from hashlib import sha256
import json
from pathlib import Path
import time

root=Path(__file__).resolve().parent
started=time.perf_counter()
raw=(root/'anchors.jsonl').read_bytes();anchors=[json.loads(line) for line in raw.decode('utf-8').splitlines()]
details=[]
for anchor in anchors[:2]:
    artifacts={a['id']:a for a in anchor['initial_materials']['initial_artifacts']}
    atomic={a['attribute_id']:a for a in anchor['atomic_labels']}
    for difference in anchor['source_structural_audit']['artifact_atomic_visibility_mismatches']:
        attribute=atomic[difference['attribute_id']];artifact=artifacts[difference['artifact_id']]
        details.append({'workflow':anchor['workflow_group_id'],**difference,
            'source_value_exact_substring_of_raw_artifact':attribute['source_value'] in artifact['content'],
            'extra_readers_marked_hidden':sorted(set(difference['artifact_readers_outside_atomic_visibility'])&
                                                set(attribute['source_visibility']['hidden_from']))})
source=root/'pisas_github/evaluation'
result={'anchor_sha256_unchanged':sha256(raw).hexdigest(),'schema_mismatched_attributes':len(details),
    'schema_mismatched_recipient_attribute_pairs':sum(len(d['extra_readers_marked_hidden']) for d in details),
    'exact_text_matching_attributes':sum(d['source_value_exact_substring_of_raw_artifact'] for d in details),
    'exact_text_matching_recipient_attribute_pairs':sum(len(d['extra_readers_marked_hidden']) for d in details
                                                        if d['source_value_exact_substring_of_raw_artifact']),
    'details':details,'source_sha256':{p.name:sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob('*.py'))},
    'native_default_loader':'artifact.visible_to selects recipients; whole artifact.content is appended and rendered',
    'native_evaluator':'attribute.hidden_from is used after model-based extraction/correctness judgment',
    'normative_reconciliation':'UNKNOWN; no precedence repair or exemption found in reviewed loader/render/judge path',
    'executed_external_code':False,'model_calls':0,'delivered_events_created':0,
    'seconds':time.perf_counter()-started}
(root/'visibility_clarification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in {'details','source_sha256'}}))
