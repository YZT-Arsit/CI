"""Evaluate the frozen earlier shallow reference on the same new DEV worlds."""
import shutil
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R.parent / '20261117_binding_factors'))
from repair import *
R = Path(__file__).resolve().parent


def main():
    out = R / 'reference'
    assert not (out / 'PLAN.json').exists()
    out.mkdir(exist_ok=True)
    original = R.parent / '20261117_binding_factors/repair/two_view_sft'
    seal = json.loads((original / 'CHECKPOINT_SEALED.json').read_text())
    frozen = json.loads((R / 'PLAN.json').read_text())
    assert hashlib.sha256((R / 'dev_worlds.json').read_bytes()).hexdigest() == frozen['dev_sha256']
    assert all(hashlib.sha256((original / 'adapter' / name).read_bytes()).hexdigest() == digest for name, digest in seal['adapter_hashes'].items())
    plan = {'arms': ['earlier_sft'], 'evidence_arms': ['earlier_sft'], 'paired_conditions': frozen['paired_conditions'],
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'primary_plan_sha256': hashlib.sha256((R / 'PLAN.json').read_bytes()).hexdigest(),
            'dev_sha256': frozen['dev_sha256'], 'checkpoint_hashes': seal['adapter_hashes'],
            'new_optimizer_updates': 0, 'scope': 'Exploratory additional fixed-checkpoint comparator; same primary DEV, not a new independent validation.'}
    (out / 'PLAN.json').write_text(json.dumps(plan, indent=2))
    shutil.copy2(R / 'dev_worlds.json', out / 'dev_worlds.json')
    target = out / 'earlier_sft'
    target.mkdir()
    shutil.copytree(original / 'adapter', target / 'adapter')
    shutil.copy2(original / 'CHECKPOINT_SEALED.json', target / 'CHECKPOINT_SEALED.json')
    (target / 'ORIGIN.json').write_text(json.dumps({'reused_checkpoint': original.relative_to(R.parents[1]).as_posix(), 'new_optimizer_updates': 0}))
    tokenizer, model = setup(17)
    set_peft_model_state_dict(model, load_file(str(target / 'adapter/adapter_model.safetensors')))
    model.eval()
    pool = palette(tokenizer)['fresh']
    summaries = []
    for condition, worlds in json.loads((out / 'dev_worlds.json').read_text()).items():
        rows = evaluate(tokenizer, model, worlds, pool, True)
        path = target / (condition + '.jsonl')
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        summary = {'arm': 'earlier_sft', 'condition': condition, 'episodes': len(rows),
                   **{k: sum(r[k] for r in rows) for k in ['safe_success', 'strict_success', 'invalid', 'invalid_queries']},
                   'valid_ci': sum(r['ci'] and not r['invalid'] for r in rows),
                   'raw_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    (out / 'RESULTS.json').write_text(json.dumps({'summaries': summaries, 'whole_goal_complete': False}, indent=2))


if __name__ == '__main__':
    main()
