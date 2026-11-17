"""Integrity and unchanged small-fit inputs for the bit-weight repair."""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
ROOT = R.parents[1]
PARENT = R.parent / '20261117_small_fit'
sys.path.insert(0, str(PARENT))
from fit_data import verify as verify_parent, batches, schedule, cases, read, rows, sha


def verify_fixed_loss():
    original = ROOT / 'artifacts/20261117_teacher_binding/binding_loss.py'
    text = original.read_text(encoding='utf-8')
    old = '    bit = .5 * (qce[positive, -2].mean() + qce[~positive, -2].mean()) if positive.any() and (~positive).any() else qce[:, -2].mean()'
    assert text.count(old) == 1
    expected = text.replace(old, '    bit = qce[:, -2].mean()')
    expected = expected.replace('texts, answers, indices, labels = [], [], [], []', 'texts, answers, indices = [], [], []')
    expected = expected.replace("                labels.append(candidate['safe_completion'])\n", '')
    expected = expected.replace('    positive = torch.tensor(labels, dtype=torch.bool)\n', '')
    assert (R / 'fixed_loss.py').read_text(encoding='utf-8') == expected


def verify():
    verify_parent(); verify_fixed_loss()
    plan = read(R / 'PLAN.json'); seal = read(PARENT / 'SEALED.json')
    registration = read(R / 'PRECONTROL_REGISTRATION.json')
    assert sha(R / 'PRECONTROL_REGISTRATION.json') == plan['registration_sha256']
    assert all(sha(R / n) == h for n, h in registration['files'].items())
    assert seal['complete'] and sha(PARENT / 'SEALED.json') == plan['parent_seal_sha256']
    assert all(sha(PARENT / n) == h for n, h in seal['files'].items())
    assert all(sha(R / n) == h for n, h in plan['sources'].items())
    assert sha(R / 'PLAN.md') == plan['protocol_sha256']
    assert all(sha(ROOT / n) == h for n, h in plan['dependencies'].items())
    return plan


def build():
    assert not (R / 'PLAN.json').exists()
    verify_parent(); verify_fixed_loss()
    registration = read(R / 'PRECONTROL_REGISTRATION.json')
    assert not registration['parent_complete_at_registration']
    assert all(sha(R / n) == h for n, h in registration['files'].items())
    assert sha(PARENT / 'PLAN.json') == registration['parent_plan_sha256']
    parent = read(PARENT / 'PLAN.json'); seal = read(PARENT / 'SEALED.json'); results = read(PARENT / 'RESULTS.json')
    assert seal['complete'] and results['errors'] == 0
    assert all(sha(PARENT / n) == h for n, h in seal['files'].items())
    deps = parent['dependencies'].copy()
    for n in ['PLAN.json', 'RESULTS.json', 'batches.jsonl', 'observations.jsonl', 'labels.jsonl']:
        p = PARENT / n; deps[p.relative_to(ROOT).as_posix()] = sha(p)
    for p in PARENT.glob('*.py'): deps[p.relative_to(ROOT).as_posix()] = sha(p)
    plan = {'steps': 144, 'schedule': parent['schedule'], 'lr': parent['lr'], 'seed': parent['seed'],
            'checkpoint': parent['checkpoint'], 'parent_seal_sha256': sha(PARENT / 'SEALED.json'),
            'sources': {p.name: sha(p) for p in R.glob('*.py')}, 'dependencies': deps,
            'protocol_sha256': sha(R / 'PLAN.md'), 'registration_sha256': sha(R / 'PRECONTROL_REGISTRATION.json'),
            'arm': 'uniform_joint', 'control': 'small_fit/fit/joint',
            'new_responses': 480, 'reused_control_responses': 480, 'whole_goal_complete': False}
    (R / 'PLAN.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps({'new_updates': 144, 'new_responses': 480, 'reused_control_responses': 480, 'one_loss_term_changed': True}))


if __name__ == '__main__':
    build()
