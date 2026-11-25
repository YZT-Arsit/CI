"""Run the sealed V9 pilot using a single role-shared adapter per arm."""
import argparse
from collections import Counter, defaultdict
import gc
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from baseline import base, start
from v9_experiment_policy import Policy, digest, dump, evaluate, loadgz
from v9_policy import semantic_logps
from v9_schedule import schedule


def coefficients(ex):
    weights = Counter()
    for k, sign in (("pos", 1), ("neg", -1)):
        for tag, act in ex[k]:
            weights[tag, act] += sign / len(ex[k])
    by_tag = defaultdict(dict)
    for (tag, act), w in weights.items():
        if w:
            by_tag[tag][act] = w
    return by_tag


def action_scores(policy, prompts, domain, tag, actions):
    r = prompts[tag]
    if domain == "neg":
        return policy.balanced(r)[[r["actions"].index(a) for a in actions]]
    return semantic_logps(policy.net, policy.tok, r["prompt"], actions)


def term(policy, prompts, domain, tag, weights):
    scores = action_scores(policy, prompts, domain, tag, list(weights))
    return (scores * scores.new_tensor(list(weights.values()))).sum()


def ce_backward(policy, prompts, domain, path):
    total = 0.
    for tag, act in path:
        tokens = len(policy.tok.encode(act, add_special_tokens=False)) + 1 if domain == "proof" else 1
        loss = -action_scores(policy, prompts, domain, tag, [act])[0] / (len(path) * tokens)
        (loss / 2).backward()
        total += float(loss.detach())
    return total


def train(arm, seed):
    sealed = Path(f"schedule_{seed}.json")
    gs = loadgz(Path("graphs.gz"))
    ps = loadgz(Path("prompts.gz"))
    freeze = json.loads(Path("freeze.json").read_text())
    assert all(digest(Path(k)) == v for k, v in freeze["hashes"].items())
    if not sealed.exists():
        init = {d: json.loads(Path(d+"_initial.json").read_text()) for d in ("proof", "neg")}
        assert all(x["weights"] == freeze["initial"] for x in init.values())
        dump(sealed, schedule(gs, init, seed))
    records = json.loads(sealed.read_text())["arms"][arm]
    output = Path(f"{arm}_{seed}")
    output.mkdir(exist_ok=True)
    assert not (output/"result.json").exists(), "Completed arm already exists"
    if not records:
        dump(output/"result.json", {"status":"NO_ELIGIBLE_NATIVE_FAILURE", "updates":0})
        return
    torch.manual_seed(seed)
    domain = arm.split("_")[0]
    p = Policy(base, start, trainable=True)
    assert p.version == freeze["initial"]
    p.net.train()
    for layer in p.net.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = 0.
    params = [t for t in p.net.parameters() if t.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=2e-5, weight_decay=0.)
    # Reference margin uses the identical train-mode forward path with dropout disabled.
    ref = {}
    with torch.no_grad():
        for ex in records:
            for t, ws in coefficients(ex).items() if ex["neg"] else []:
                k = (t, tuple(ws.items()))
                if k not in ref:
                    ref[k] = float(term(p, ps, domain, t, ws))
    with (output/"updates.jsonl").open("w", buffering=1) as log:
        for i, ex in enumerate(records):
            optimizer.zero_grad(set_to_none=True)
            common_loss = ce_backward(p, ps, domain, ex["common"])
            ref_margin = margin = None
            if ex["neg"]:
                terms = coefficients(ex)
                ref_margin = sum(ref[t, tuple(ws.items())] for t, ws in terms.items())
                with torch.no_grad():
                    margin = sum(float(term(p, ps, domain, t, ws)) for t, ws in terms.items())
                delta = torch.tensor(.2 * (margin - ref_margin))
                intervention = float(F.softplus(-delta))
                # Differentiate the logistic loss at the measured margin; replay one role observation at a time.
                multiplier = -.1 * float(torch.sigmoid(-delta))
                for t, ws in terms.items():
                    (multiplier * term(p, ps, domain, t, ws)).backward()
            else:
                intervention = ce_backward(p, ps, domain, ex["pos"])
            norm = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
            optimizer.step()
            log.write(json.dumps(dict(update=i+1, **ex, common_ce=common_loss, intervention=intervention,
                                      loss=(common_loss+intervention)/2, ref_margin=ref_margin,
                                      margin=margin, grad_norm=norm, schedule=digest(sealed)))+"\n")
            if (i+1)%8==0: print(arm, seed, i+1, flush=True)
    p.net.save_pretrained(output/"adapter")
    del p, optimizer, params, layer
    gc.collect()
    torch.cuda.empty_cache()
    restored = Policy(base, output/"adapter")
    assert restored.version != freeze["initial"]
    result = evaluate(restored, [g for g in gs if g["domain"] == domain and g["split"]=="dev"], ps)
    result.update(arm=arm, seed=seed, schedule=digest(sealed), updates=len(records), start=freeze["initial"])
    dump(output/"result.json", result)
    print("RELOADED_AND_EVALUATED", arm, seed, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("arm")
    cli.add_argument("seed", type=int)
    args = cli.parse_args()
    train(args.arm, args.seed)

