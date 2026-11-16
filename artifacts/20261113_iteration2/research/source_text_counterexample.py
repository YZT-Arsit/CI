"""Minimal trusted-source text/value mismatch; only new audit files are written."""
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'code'))
from kci.multi_env import MultiRuleCIEnv, build_case
from kci.multi_candidates import MultiRuleCandidateGenerator
from kci.continuation import ContinuationEvaluator, ReferenceSearch, SearchBudget, certify_preference


def at_receiver(env):
    state, _ = env.transition(env.initial(), env.act('A', 'SKIP'))
    b_prompt = env.observation(state, 'B').as_prompt()
    state, audit = env.transition(state, env.act('B', 'SEND', 'R', ('payload_0',)))
    assert audit.verdict.value == 'valid'
    cce = ContinuationEvaluator(env, ReferenceSearch(env, MultiRuleCandidateGenerator(), SearchBudget(1000, 128)))
    yes = cce.evaluate(state, env.act('R', 'FINALIZE', answer='yes'))
    no = cce.evaluate(state, env.act('R', 'FINALIZE', answer='no'))
    preference = certify_preference(yes, no)
    return b_prompt, env.observation(state, 'R').as_prompt(), preference


original = build_case('identity_attribute', answer='yes')
facts = {k:v for k,v in original.facts.items() if k in {'context_0','payload_0','notification_target'}}
facts['payload_0'] = replace(facts['payload_0'], value='1')  # Leave displayed text "... 9." unchanged.
constructed = MultiRuleCIEnv('identity_attribute', facts, {'op':'threshold','threshold':7}, world_id='audit_mismatch')
before_b, before_r, pref_before = at_receiver(original)
after_b, after_r, pref_after = at_receiver(constructed)
assert before_b == after_b and before_r == after_r
assert pref_before.chosen.answer == 'yes' and pref_after.chosen.answer == 'no'
result = {
    'classification': 'constructor source-consistency defect outside factory-generated primary data',
    'original_payload': asdict(original.facts['payload_0']),
    'altered_payload': asdict(constructed.facts['payload_0']),
    'constructor_accepted_inconsistent_value_and_text': True,
    'same_B_prompt': before_b == after_b, 'same_R_prompt': before_r == after_r,
    'R_prompt_sha256': sha256(before_r.encode()).hexdigest(),
    'original_expected_answer': original.expected_answer,
    'altered_expected_answer': constructed.expected_answer,
    'original_preference': {'chosen':asdict(pref_before.chosen),'rejected':asdict(pref_before.rejected)},
    'altered_preference': {'chosen':asdict(pref_after.chosen),'rejected':asdict(pref_after.rejected)},
    'primary_factory_records_changed': False,
    'recommended_boundary': 'Reject inconsistent canonical source text at construction or restrict trusted API to verified factory source rendering. Schema validity alone is not semantic consistency.',
    'source_sha256': {str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in
                      (Path(__file__),ROOT/'code/kci/multi_env.py')}
}
output = Path(__file__).with_suffix('.json')
if output.exists():
    raise FileExistsError('new output required')
output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in {'source_sha256','original_preference','altered_preference'}}, ensure_ascii=False))
