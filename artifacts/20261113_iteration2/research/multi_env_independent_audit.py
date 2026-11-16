"""Independent finite-rule/action checks; run one family per <=9.5s worker."""
import argparse
from collections import deque
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code'))
from kci.contracts import Action, Verdict
from kci.continuation import ReferenceSearch, SearchBudget
from kci.multi_candidates import MultiRuleCandidateGenerator
from kci.multi_env import FAMILIES, build_case


def key(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def expected_inferences(env, ids):
    """Independent premise evaluation, without calling production infer/known_to."""
    facts = env.facts
    c = [facts[i] for i in env.context_ids if i in ids]
    p = facts['payload_0'] if 'payload_0' in ids else None
    found = set()
    if env.family == 'alias_chain_3' and len(c) == 2 and c[0].value == c[1].subject:
        found.add('person_case:' + key([c[0].subject, c[1].value]))
    if p is None or len(c) != len(env.context_ids):
        return found
    if env.family == 'identity_attribute':
        if c[0].subject == p.subject:
            found.add('protected:' + key([c[0].value, json.loads(p.value)]))
    elif env.family == 'alias_chain_3':
        # Build the explicit directed relation graph independently of nested joins.
        edges = {f.subject: f.value for f in c}
        node = c[0].subject
        for _ in range(2):
            node = edges.get(node)
        if node == p.subject:
            found.add('protected:' + key([c[0].subject, json.loads(p.value)]))
    elif env.family == 'aggregate_difference':
        other, total = json.loads(c[0].value), json.loads(p.value)
        remainder = set(total['contributors']) - set(other['contributors'])
        same_scope = c[0].subject == p.subject and all(other[k] == total[k] for k in ('period', 'unit'))
        exhaustive = (set(total['population']) == set(other['population']) == set(total['contributors'])
                      and set(other['contributors']) <= set(total['contributors']))
        if same_scope and exhaustive and remainder == {other['excluded']}:
            found.add('protected:' + key([other['excluded'], total['period'], total['unit'], total['total'] - other['total']]))
    else:
        left, right = json.loads(c[0].value), json.loads(p.value)
        matches = [person for person in set(left['members']) if person in set(right['members'])]
        if c[0].subject == p.subject and left['scope'] == right['scope'] and left['complete'] is True and right['complete'] is True and len(matches) == 1:
            found.add('protected:' + key([p.subject, matches[0]]))
    return found


def independent_actions(env, state):
    """Independent finite action grammar; excludes structurally invalid sends."""
    who = state.actor
    owned = state.local_fact_ids[who]
    actions = [Action(who, 'ABSTAIN')]
    if who == 'A':
        if state.finalized:
            actions += [Action(who, 'NOTIFY', env.target, answer=a) for a in ('yes', 'no')]
            ids = []
        else:
            actions += [Action(who, 'SKIP')]
            ids = sorted(set(env.context_ids) & owned)
        recipient = 'R'
    elif who == 'B':
        ids = sorted(owned & {'payload_0', 'decision_yes', 'decision_no'})
        recipient = 'R'
        if 'task_rule' not in owned and not state.pending_request:
            actions += [Action(who, 'REQUEST_RULES', 'R')]
        if {'payload_0', 'task_rule'} <= owned and not owned & {'decision_yes', 'decision_no'}:
            actions += [Action(who, 'COMPUTE_TASK')]
    else:
        ids = ['task_rule'] if state.pending_request and 'task_rule' in owned else []
        recipient = 'B'
        if {'payload_0', 'task_rule'} <= owned or owned & {'decision_yes', 'decision_no'}:
            actions += [Action(who, 'FINALIZE', answer=a) for a in ('yes', 'no')]
    for size in range(1, len(ids) + 1):
        actions += [env.act(who, 'SEND', recipient, part) for part in combinations(ids, size)]
    return actions


def enumerate_reachable(env, start):
    queue = deque([deepcopy(start)])
    visited, active, costs = set(), [], []
    generator = MultiRuleCandidateGenerator()
    while queue:
        state = queue.popleft()
        signature = repr(state.signature())
        if signature in visited:
            continue
        visited.add(signature)
        if len(visited) > 10000:
            raise AssertionError('independent enumeration incomplete')
        if state.done:
            if state.success and not state.any_violation:
                costs.append(state.cost)
            continue
        if state.turn >= state.max_turns:
            continue
        actions = independent_actions(env, state)
        batch = generator.generate(env.observation(state, state.actor), limit=128)
        assert batch.exhaustive and {a.key() for a in actions} == {a.key() for a in batch.actions}
        active.append(state)
        for action in actions:
            after, audit = env.transition(state, action)
            if audit.verdict == Verdict.VALID:
                queue.append(after)
    return min(costs) if costs else None, active, len(visited)


def run(family):
    started = time.perf_counter()
    closure_checks = 0
    rows, observation_classes = [], {}
    for answer in ('yes', 'no'):
        for op in (('membership',) if family == 'set_intersection' else ('threshold', 'parity')):
            for allow in (False, True):
                for horizon in (3, 6, 9):
                    env = build_case(family, answer, task_op=op, allow_protected=allow, max_turns=horizon)
                    assert env.expected_answer == answer
                    primitive = list(env.context_ids) + ['payload_0']
                    for size in range(len(primitive) + 1):
                        for subset in combinations(primitive, size):
                            assert set(env.infer(set(subset))) == expected_inferences(env, set(subset))
                            closure_checks += 1
                    best, states, visited = enumerate_reachable(env, env.initial())
                    expected = 4.0 if horizon >= 4 else None
                    assert best == expected
                    reference = ReferenceSearch(env, MultiRuleCandidateGenerator(), SearchBudget(10000, 128)).search(env.initial())
                    assert reference.best_success_cost == best and reference.status.value in {'exact_within_horizon', 'proven_unreachable_within_horizon'}
                    prefix, audit = env.transition(env.initial(), env.act('A', 'SEND', 'R', env.context_ids))
                    assert audit.verdict == Verdict.VALID
                    after_context, _, _ = enumerate_reachable(env, prefix)
                    expected_context = (4.0 if allow else 7.0) if horizon >= (4 if allow else 7) else None
                    assert after_context == expected_context
                    for state in states:
                        obs = env.observation(state, state.actor).as_prompt()
                        menu = independent_actions(env, state)
                        model_input = key([obs, sorted(a.as_json() for a in menu)])
                        for action in menu:
                            _, verdict = env.transition(state, action)
                            identity = (sha256(model_input.encode()).hexdigest(), action.as_json())
                            observation_classes.setdefault(identity, set()).add(verdict.verdict.value)
                    rows.append({'answer': answer, 'task_op': op, 'allow': allow, 'horizon': horizon, 'enumerated_states': visited,
                                 'initial_best_cost': best, 'after_all_context_best_cost': after_context})
    assert all(len(v) == 1 for v in observation_classes.values())
    env = build_case(family, max_turns=9)
    clean, _ = env.transition(env.initial(), env.act('A', 'SKIP'))
    hidden = deepcopy(clean)
    hidden.local_fact_ids['R'].update(env.context_ids)
    assert env.observation(clean, 'B').as_prompt() == env.observation(hidden, 'B').as_prompt()
    action = env.act('B', 'SEND', 'R', ('payload_0',))
    _, clean_label = env.transition(clean, action)
    _, hidden_label = env.transition(hidden, action)
    assert clean_label.verdict == Verdict.VALID and hidden_label.verdict == Verdict.VIOLATION
    files = [Path(__file__), ROOT/'code/kci/multi_env.py', ROOT/'code/kci/multi_candidates.py']
    return {'family': family, 'configurations': len(rows), 'rows': rows, 'closure_subset_checks': closure_checks,
            'same_observation_action_classes': len(observation_classes), 'primary_ci_label_conflicts': 0,
            'hidden_background_counterexample': {'same_B_prompt': True, 'clean_label': clean_label.verdict.value,
                'hidden_label': hidden_label.verdict.value, 'scope': 'injected background outside primary complete-information support; must not be exported'},
            'scope': 'independent finite action enumeration and primitive closure checks; transition engine reused; no natural-language or model claim',
            'future_equivalence_audit': 'not included in this first checker; pending data/label audit',
            'source_sha256': {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in files},
            'seconds': time.perf_counter()-started}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=FAMILIES, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('new audit output required')
    result = run(args.family)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('family','configurations','closure_subset_checks','primary_ci_label_conflicts','seconds')}))
