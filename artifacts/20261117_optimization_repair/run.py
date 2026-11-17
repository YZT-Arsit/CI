"""Continue identical joint SFT batches at two rates, with fixed free-policy DEV."""
import gc
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

R = Path(__file__).resolve().parent
ROOT = R.parents[1]
sys.path[:0] = [str(R.parent / '20261117_frozen_crossover'), str(R.parent / '20261117_teacher_binding'),
                str(R.parent / '20261117_sampler_repair')]
from common import verify, load, torch, palette, generate, evaluate
from binding_loss import loss_for_views
from sampling import buckets
from render import query, actor


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def append(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row)+'\n')


def train_fit(tokenizer, model, target, pool):
    cases = list(map(json.loads, (R.parent / '20261117_frozen_crossover/train_oracle.jsonl').read_text().splitlines()))
    for start in range(0, len(cases), 4):
        batch = cases[start:start+4]; texts = []; indices = []; records = {}
        for case in batch:
            records[case['case_id']] = []
            for action in case['observation']['actions']:
                texts.append(query(case['observation'], pool, case['seeds'], action))
                indices.append((case['case_id'], action))
        raws, seconds = generate(tokenizer, model, texts)
        for (cid, action), text, raw in zip(indices, texts, raws):
            matches = re.findall(r'ANSWER:\s*([01])\b', raw)
            valid = len(matches)==1 and seconds < 9.5
            records[cid].append({'action': action, 'prompt': text, 'raw': raw, 'seconds_batch': seconds,
                                 'valid': valid, 'estimate': matches[0] if valid else 'unknown'})
        inputs, orders, bits = [], [], []
        for case in batch:
            truth = {c['action']: str(c['safe_completion']) for c in case['candidates']}
            text, order = actor(case['observation'], pool, case['seeds'], truth)
            inputs.append(text); orders.append(order); bits.append(truth)
        raws, seconds = generate(tokenizer, model, inputs)
        for case, text, order, truth, raw in zip(batch, inputs, orders, bits, raws):
            matches = re.findall(r'ANSWER:\s*([ABC])\b', raw)
            valid = len(matches)==1 and ord(matches[0])-65<len(order) and seconds < 9.5
            append(target / 'train_fit.jsonl', {'case_id': case['case_id'], 'queries': records[case['case_id']],
                'actor': {'prompt': text, 'actions': order, 'input_estimates': truth, 'raw': raw,
                          'valid': valid, 'selected': order[ord(matches[0])-65] if valid else None, 'seconds_batch': seconds}})


def main():
    plan = json.loads((R / 'PLAN.json').read_text())
    assert not (R / 'repair/PLAN.json').exists()
    assert all(sha(R / p)==h for p,h in plan['source_hashes'].items())
    assert all(sha(ROOT / p)==h for p,h in plan['dependencies'].items())
    assert sha(R / 'dev_worlds.json') == plan['dev_sha256']
    diagnostic = json.loads((R.parent / '20261117_optimization_probe/RESULTS.json').read_text())
    assert diagnostic['errors'] == 0 and diagnostic['persistent_model_updates'] == 0
    cross = verify()
    assert sha(R.parent / '20261117_frozen_crossover/PLAN.json') == plan['cross_plan_sha256']
    assert cross['checkpoints'] == plan['checkpoints']
    folder = R / 'repair'; folder.mkdir()
    (folder / 'PLAN.json').write_text(json.dumps(plan, indent=2))
    shutil.copy2(R / 'dev_worlds.json', folder / 'dev_worlds.json')
    worlds = json.loads((R / 'dev_worlds.json').read_text()); data = buckets(); summaries = []
    for arm in plan['arms']:
        target = folder / arm; target.mkdir()
        origin = arm if arm in ['teacher_sft', 'earlier_sft'] else 'teacher_sft'
        tokenizer, model = load(origin, cross); pools = palette(tokenizer)
        if arm in plan['rates']:
            model.train()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=plan['rates'][arm], weight_decay=0.)
            tokens = 0
            for step, batch in enumerate(plan['batches']):
                def timeout():
                    (target / 'TIMEOUT.json').write_text(json.dumps({'step':step, 'limit':9.5}))
                    os._exit(124)
                timer = threading.Timer(9.5, timeout); timer.daemon=True; timer.start(); start=time.perf_counter()
                loss, parts, count, predicted, supplied = loss_for_views(tokenizer, model,
                    data[tuple(batch['key'])][batch['index']], pools['train'], batch['views'], step, 0., teacher=True)
                optimizer.zero_grad(); loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
                optimizer.step(); timer.cancel(); tokens += count
                row = {'step':step+1, 'role':batch['key'][0], 'lr':plan['rates'][arm], 'seconds':time.perf_counter()-start,
                       'input_tokens':count, 'gradient_norm_before_clip':norm,
                       **{k:float(v.detach()) for k,v in parts.items()},
                       'predicted_estimates':predicted, 'actor_input_estimates':supplied}
                append(target / 'training.jsonl', row)
                if (step+1)%20==0:
                    print(json.dumps({'arm':arm,'step':step+1,'query_loss':row['query_loss'],'actor_loss':row['actor_loss']}),flush=True)
            model.save_pretrained(target / 'adapter')
            (target / 'CHECKPOINT_SEALED.json').write_text(json.dumps({'adapter_hashes':{p.name:sha(p) for p in (target/'adapter').iterdir() if p.is_file()},
                'steps':80,'input_tokens':tokens,'initial_adapter_hashes':plan['checkpoints']['teacher_sft']['seal']['adapter_hashes']},indent=2))
            del optimizer
        else:
            source = ROOT / plan['checkpoints'][origin]['path']
            shutil.copytree(source / 'adapter', target / 'adapter')
            shutil.copy2(source / 'CHECKPOINT_SEALED.json', target / 'CHECKPOINT_SEALED.json')
        model.eval()
        for condition, ws in worlds.items():
            rows = evaluate(tokenizer,model,ws,pools['fresh'],True)
            path = target / (condition+'.jsonl'); path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            report = {'arm':arm,'condition':condition,'episodes':len(rows),
                **{k:sum(r[k] for r in rows) for k in ['safe_success','strict_success','invalid','invalid_queries']},
                'valid_ci':sum(r['ci'] and not r['invalid'] for r in rows),'raw_sha256':sha(path)}
            summaries.append(report)
            print(json.dumps(report),flush=True)
        if arm in plan['rates']:
            train_fit(tokenizer,model,target,pools['train'])
        del model; gc.collect()
    (folder / 'RESULTS.json').write_text(json.dumps({'summaries':summaries,'whole_goal_complete':False},indent=2))
    (R / 'SEALED.json').write_text(json.dumps({'complete':True,'files':{p.relative_to(R).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()},
                                            'whole_goal_complete':False},indent=2))


if __name__ == '__main__':
    main()
