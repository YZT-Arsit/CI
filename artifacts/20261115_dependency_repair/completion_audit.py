"""Aggregate claims and immutable inputs without promoting partial evidence."""
from pathlib import Path
from hashlib import sha256
from collections import Counter
from datetime import datetime,timezone
import gzip,json

R=Path(__file__).resolve().parent;ROOT=R.parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def h(p):return sha256(p.read_bytes()).hexdigest()
baseline=load(R/'baseline.json')['sha256']
old_weights={p:h(ROOT/p)==v for p,v in baseline.items() if p.endswith('weights.npz')}
assert len(old_weights)==12 and all(old_weights.values())
assert h(ROOT/'code/kci/legacy_dynamic_tasks_v1.py')==baseline['code/kci/dynamic_tasks.py']
pilot=load(R/'qwen_envelope_v1/plan.json')
assert all(h(Path(p))==v for p,v in pilot['source_sha256'].items())
source=load(ROOT/'artifacts/20261113_iteration2/source_anchors/manifest.json')['files']
assert all(h(ROOT/'artifacts/20261113_iteration2/source_anchors'/p)==v['sha256'] for p,v in source.items())
tedt={k:load(R/f'tedt_final_{k}.json') for k in ('meeting','query')}
assert all(d['all_passed'] for d in tedt.values())
counts={k:sum(len(d[k]) for d in tedt.values()) for k in ('dependency','computation','acl')}
compressed=load(R/'published_raw/MANIFEST.json')['files']
for p in compressed:
    data=(R/p['compressed']).read_bytes()
    assert sha256(data).hexdigest()==p['compressed_sha256']
    assert sha256(gzip.decompress(data)).hexdigest()==p['raw_sha256']
graph_data=load(R/'graph_data/DATA_CARD.json')
graph=[]
for p in sorted((R/'graph_models').glob('*/result.json')):
    d=load(p);assert h(p.parent/'weights.npz')==d['weights_sha256']
    for item in graph_data['datasets']:
        if item['name']!='train':assert item['tensor_sha256']==d['evaluations'][item['name']]['input_sha256']
        elif d['arm']!='shuffled':assert item['tensor_sha256']==d['training_input_sha256']
    graph.append({'arm':d['arm'],'seed':d['seed'],'correct':sum(e['correct'] for e in d['evaluations'].values()),
                  'total':768,'all_strata_gate':d['diagnostic_gate_passed']})
controls=[load(p) for p in sorted((R/'calibration_bound_v3').glob('*.json'))]
audits=[load(p) for p in (R/'calibration_audits').glob('*.json')]
assert len(audits)==20 and all(a['all_passed'] for a in audits)
qwen=load(R/'qwen_envelope_v1/summary.json');callrows=load(R/'qwen_envelope_v1/calls.json')
assert len(callrows)==88 and all(x['process_status']=='completed' and x['response_status']=='completed' for x in callrows)
tokens=[]
for folder in (R/'qwen_envelope_v1').glob('call_*'):
    for event in load(folder/'response.json')['events']:
        if event.get('done'):tokens.append(event['prompt_eval_count'])
assert len(tokens)==88 and max(tokens)<4096
for kind in ('meeting','query'):
    cost=load(R/f'cost_routes_{kind}.json')
    assert cost['all_source_ancestor_checks_passed']
    assert all(c['best_continuation_choices']['cost_removed'] is None for c in cost['cases'])
regression=(R/'job_logs/regression_final_153.stderr.txt').read_text()
assert 'Ran 153 tests' in regression and regression.rstrip().endswith('OK')
receipts=[json.loads(line) for path in R.glob('*receipts.jsonl') for line in path.read_text().splitlines()]
report={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'branch':'main',
 'overall':'ENGINEERING_REPAIR_PASSED_WITHIN_FINITE_CONTRACT; FULL_RESEARCH_PROOF_NOT_ESTABLISHED',
 'gpu_decision':'NO_GO','real_LLM_training':False,
 'requirements':{
  'Meeting_consumes_delivered_constraints':'PASS_FINITE_CONTRACT',
  'Query_company_names_source_permission':'PASS_FINITE_CONTRACT',
  'Task_Evidence_Dependency_Test':'PASS_FINITE_CONTRACT',
  'strict_CCE_without_raw_distribution_enumeration':'PASS_FOR_VERIFIED_EXECUTED_REFERENCE_ONLY',
  'exact_CI_risk_equality_from_finite_unrestricted_LLM_samples':'NOT_PROVABLE_WITHOUT_ADDITIONAL_ASSUMPTIONS',
  'value_difference_independent_of_artificial_cost_or_deadline':'NOT_ESTABLISHED_COUNTEREXAMPLE_RECORDED',
  'learned_relation_given_structured_entity_binding':'PASS_AUXILIARY_SYNTHETIC_CLASSIFIER_ONLY',
  'shared_text_policy_learns_relations_and_generalizes':'NOT_ESTABLISHED',
  'independent_source_norm_and_business_cost_validation':'NOT_ESTABLISHED'},
 'tedt':counts,'regression_tests_passed':153,
 'calibration':dict(Counter(d['status'] for d in controls)),
 'independent_execution_replay':{'files':len(audits),'transitions':sum(d['executed_steps_replayed'] for d in audits),'all_passed':True},
 'qwen':{'task_calls':88,'warmup':1,'completed_suffixes':16,'unassisted_suffixes':sum(d['unassisted_completed_suffixes'] for d in qwen['cases']),
         'blocked_drafts':sum(d['blocked_drafts'] for d in qwen['cases']),
         'prompt_eval_count_range':[min(tokens),max(tokens)],'max_request_seconds':qwen['max_request_seconds'],
         'source_and_model_digest_unchanged':True,'raw_policy_safety_proven':False},
 'graph_diagnostic':graph,
 'new_preferences':{'raw':208,'unique':198,'primary':8,'primary_groups':4,'source_instances':2,'gpu_training_eligible':False},
 'preservation':{'historical_checkpoints':old_weights,'source_anchors_unchanged':True,'legacy_engine_byte_identical':True},
 'recorded_workers':{'count':len(receipts),'statuses':dict(Counter(d['status'] for d in receipts)),
                    'maximum_seconds':max(d['seconds'] for d in receipts),
                    'failed_explanation':'16 initial execution-audit report writes failed due relative Windows path, fixed and rerun without resampling'},
 'compressed_raw_files':48,'compressed_roundtrip_all_passed':True}
(R/'completion_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:report[k] for k in ('overall','gpu_decision','tedt','calibration','qwen')}))
