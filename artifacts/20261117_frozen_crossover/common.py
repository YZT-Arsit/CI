"""Frozen CPU inference helpers shared by learned and explicitly oracle workers."""
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
ROOT = R.parents[1]
sys.path.insert(0, str(R.parent / '20261117_action_binding'))
from learned_evidence import *
R = Path(__file__).resolve().parent
ROOT = R.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((R / name).read_text())


def rows(name):
    return list(map(json.loads, (R / name).read_text().splitlines()))


def append(path, row):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row) + '\n')


def verify():
    plan = read('PLAN.json')
    for name, digest in plan['source_hashes'].items():
        assert sha(R / name) == digest
    for name, digest in plan['dependency_hashes'].items():
        assert sha(ROOT / name) == digest
    manifest = json.loads((ROOT / 'artifacts/20261116_relation_solution/lm_source_manifest.json').read_text())
    config = json.loads((ROOT / 'artifacts/20261116_relation_solution/environment.json').read_text())
    native = Path(config['native_lm_path'])
    assert native.name == manifest['revision']
    for row in manifest['files']:
        path = native / row['file']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    return plan


def load(arm, plan):
    path = ROOT / plan['checkpoints'][arm]['path']
    seal = plan['checkpoints'][arm]['seal']
    assert json.loads((path / 'CHECKPOINT_SEALED.json').read_text()) == seal
    assert all(sha(path / 'adapter' / name) == digest for name, digest in seal['adapter_hashes'].items())
    tokenizer, model = setup(17)
    set_peft_model_state_dict(model, load_file(str(path / 'adapter/adapter_model.safetensors')))
    model.eval()
    return tokenizer, model


def nonce(case):
    return seed_for(case['base_group_id'], {'Planner': 0, 'Source': 1, 'Worker': 2}[case['observation']['role']])


def actor_record(case, text, order, raw, seconds, estimates):
    found = re.findall(r'ANSWER:\s*([ABC])\b', raw)
    valid = len(found) == 1 and ord(found[0]) - 65 < len(order) and seconds < 9.5
    return {'case_id': case['case_id'], 'prompt': text, 'actions': order, 'raw': raw,
            'valid': valid, 'selected': order[ord(found[0]) - 65] if valid else None,
            'seconds_batch': seconds, 'input_estimates': estimates}


def seal(name, paths, **extra):
    result = {'complete': True, 'labels_scored': False,
              'files': {p.relative_to(R).as_posix(): sha(p) for p in paths}, **extra}
    (R / name).write_text(json.dumps(result, indent=2))

