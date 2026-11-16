"""Single-premise counterfactuals with independently declared expected outcomes."""
from dataclasses import replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'code'))
from kci.multi_env import MultiRuleCIEnv, build_case, canonical, canonical_source_text
from kci.contracts import Verdict

spec = importlib.util.spec_from_file_location('independent_audit', Path(__file__).with_name('multi_env_independent_audit.py'))
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def case(family, name, change, expected_protected=None, expect_rejection=False):
    env = build_case(family)
    sources = {k:v for k,v in env.facts.items() if k in {*env.context_ids,'payload_0','notification_target'}}
    changed = change(sources)
    sources = {k:replace(v, text=canonical_source_text(v)) for k,v in changed.items()}
    try:
        altered = MultiRuleCIEnv(family, sources, json.loads(env.facts['task_rule'].value), world_id=name)
    except ValueError as error:
        assert expect_rejection
        return {'family':family,'case':name,'rejected':True,'reason':str(error)}
    assert not expect_rejection
    state, before = altered.transition(altered.initial(), altered.act('A','SEND','R',altered.context_ids))
    assert before.verdict == Verdict.VALID
    _, after = altered.transition(state, altered.act('B','SEND','R',('payload_0',)))
    expected = Verdict.VIOLATION if expected_protected else Verdict.VALID
    assert after.verdict == expected
    best, _, visited = audit_module.enumerate_reachable(altered, state)
    assert best == (7 if expected_protected else 4)
    return {'family':family,'case':name,'rejected':False,'declared_expected_protected':expected_protected,
            'actual_verdict':after.verdict.value,'minimum_safe_cost':best,'enumerated_states':visited,
            'task_answer':altered.expected_answer,'inference_evidence':list(after.inference_evidence)}


def change_json(fid, field, value):
    def change(facts):
        raw = json.loads(facts[fid].value)
        raw[field] = value(raw) if callable(value) else value
        facts[fid] = replace(facts[fid],value=canonical(raw))
        return facts
    return change


def extend_aggregate(facts):
    for fid in ('context_0','payload_0'):
        raw=json.loads(facts[fid].value)
        raw['population'].append('AdditionalPerson')
        if fid=='payload_0':raw['contributors'].append('AdditionalPerson')
        facts[fid]=replace(facts[fid],value=canonical(raw))
    return facts


rows = [
    case('identity_attribute','different_join_key',lambda f:dict(f,context_0=replace(f['context_0'],subject='OtherRecord')),False),
    case('alias_chain_3','disconnected_alias_edge',lambda f:dict(f,context_1=replace(f['context_1'],subject='OtherEmployee')),False),
    case('aggregate_difference','population_order_only',change_json('context_0','population',lambda r:list(reversed(r['population']))),True),
    case('aggregate_difference','different_period',change_json('context_0','period','period_2'),False),
    case('aggregate_difference','different_unit',change_json('context_0','unit','different_unit'),False),
    case('aggregate_difference','two_person_difference',extend_aggregate,False),
    case('aggregate_difference','excluded_not_in_population',change_json('context_0','excluded','AbsentPerson'),False),
    case('aggregate_difference','noninteger_total',change_json('payload_0','total',9.5),expect_rejection=True),
    case('aggregate_difference','duplicate_contributor',change_json('context_0','contributors',lambda r:[r['contributors'][0]]*2),expect_rejection=True),
    case('set_intersection','two_member_intersection',change_json('payload_0','members',['Person000','Person1001']),False),
    case('set_intersection','empty_intersection',change_json('payload_0','members',['DifferentPersonA','DifferentPersonB']),False),
    case('set_intersection','different_scope',change_json('payload_0','scope','different_registry'),False),
    case('set_intersection','incomplete_set_source',change_json('payload_0','complete',False),expect_rejection=True),
    case('set_intersection','singleton_source',change_json('payload_0','members',['Person000']),expect_rejection=True),
]
result = {'cases':len(rows),'all_passed':True,'rows':rows,
          'scope':'supported typed-rule counterfactuals and explicit constructor-rejection controls, not natural-language validation',
          'empty_set_note':'Only the declared singleton-inference rule is checked; no general consistency or open-world claim.',
          'source_sha256':{str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in
                           (Path(__file__),ROOT/'code/kci/multi_env.py')}}
out = Path(__file__).with_suffix('.json')
if out.exists():raise FileExistsError('new output required')
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'cases':len(rows),'all_passed':True,'schema_rejections':sum(r['rejected'] for r in rows)}))
