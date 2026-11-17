import gc
import importlib.util
import json
import sys
from policy_data import R, ROOT, PARENT, read, sha, verify
spec = importlib.util.spec_from_file_location('join_runner_for_policy_transfer', PARENT / 'run_join.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner); w = runner.w
evaluate = sys.modules['common'].evaluate
assert evaluate.__code__.co_filename.endswith('20261117_action_binding/learned_evidence.py')
assert runner.R == PARENT


def main():
    plan = verify(); w.verify_model(); parent = read(PARENT / 'SEALED.json')
    assert parent['complete'] and parent['plan_sha256'] == sha(PARENT / 'PLAN.json')
    assert all(sha(PARENT / name) == digest for name, digest in parent['files'].items())
    study = R / 'study'; assert not study.exists(); study.mkdir(); parent_plan = read(PARENT / 'PLAN.json'); checkpoints = {}
    for arm, cfg in plan['arms'].items():
        source = PARENT / 'study' / arm
        checkpoints[arm] = parent_plan['checkpoints'][str(cfg['seed'])] if cfg['kind'] == 'initial' else {
            'path': source.relative_to(ROOT).as_posix(), 'seal': read(source / 'CHECKPOINT_SEALED.json')}
    runtime = {**plan, 'arms': list(plan['arms']), 'evidence_arms': list(plan['arms']), 'checkpoint_bindings': checkpoints,
               'source_sha256': sha(R / 'run_policy.py'), 'parent_seal_sha256': sha(PARENT / 'SEALED.json')}
    (study / 'PLAN.json').write_text(json.dumps(runtime, indent=2), encoding='utf-8')
    (study / 'dev_worlds.json').write_bytes((R / 'dev_worlds.json').read_bytes()); worlds = read(R / 'dev_worlds.json')
    pool = read(R / 'palettes.json')['new']; summaries = []
    for arm, cfg in plan['arms'].items():
        target = study / arm; target.mkdir(); tokenizer, model = w.load('start', {'checkpoints': {'start': checkpoints[arm]}})
        before = w.digest(model); assert before == read(PARENT / 'study' / arm / 'EVAL_IDENTITY.json')['before']
        for condition, cases in worlds.items():
            records = evaluate(tokenizer, model, cases, pool, True)
            path = target / (condition + '.jsonl'); path.write_text(''.join(json.dumps(v) + '\n' for v in records), encoding='utf-8')
            summaries.append({'arm': arm, 'condition': condition, 'episodes': len(records),
                **{k: sum(v[k] for v in records) for k in ['safe_success', 'strict_success', 'invalid', 'invalid_queries']},
                'valid_ci': sum(v['ci'] and not v['invalid'] for v in records), 'raw_sha256': sha(path)})
        after = w.digest(model); assert before == after
        (target / 'EVAL_IDENTITY.json').write_text(json.dumps({'before': before, 'after': after, 'checkpoint': checkpoints[arm], 'errors': 0}), encoding='utf-8')
        print(json.dumps({'arm': arm, 'episodes': 32}), flush=True); del model; gc.collect()
    (study / 'RESULTS.json').write_text(json.dumps({'summaries': summaries, 'whole_goal_complete': False}, indent=2), encoding='utf-8')
    (R / 'SEALED.json').write_text(json.dumps({'complete': True, 'plan_sha256': sha(R / 'PLAN.json'), 'parent_seal_sha256': sha(PARENT / 'SEALED.json'),
        'files': {p.relative_to(R).as_posix(): sha(p) for p in study.rglob('*') if p.is_file()}}, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
