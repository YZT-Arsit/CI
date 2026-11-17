"""Change only semantic actor-loss weights; query and format losses unchanged."""
import random
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R.parent / '20261117_binding_factors'))
from two_view_loss import torch, answer_ids, forward_targets, response, semantic_log_probs, js
from render import query, actor
from weighting import weights


def loss_for_views(tokenizer, model, pair, pool, views, step, weight, teacher=False, multiplier=1.):
    assert len(pair) == len(views) == 2
    letters = answer_ids(tokenizer)
    binary_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in (0, 1)]
    binary = dict(zip(binary_ids, ['0', '1']))
    texts, answers, indices, labels = [], [], [], []
    for view, seeds in enumerate(views):
        for side, row in enumerate(pair):
            for candidate in row['candidates']:
                texts.append(query(row['observation'], pool, seeds, candidate['action']))
                answers.append('ANSWER:\n' + str(candidate['safe_completion']))
                indices.append((view, side, candidate['action']))
                labels.append(candidate['safe_completion'])
    qlogits, qce, qtokens = forward_targets(tokenizer, model, texts, answers)
    estimates = [[{}, {}], [{}, {}]]
    for (view, side, action), value in zip(indices, qlogits[:, -2].detach().argmax(-1).tolist()):
        estimates[view][side][action] = binary.get(value, 'unknown')
    positive = torch.tensor(labels, dtype=torch.bool)
    bit = .5 * (qce[positive, -2].mean() + qce[~positive, -2].mean()) if positive.any() and (~positive).any() else qce[:, -2].mean()
    qloss = bit + .1 * torch.cat([qce[:, :-2], qce[:, -1:]], 1).mean()
    half = len(indices) // 2
    assert [(s, a) for v, s, a in indices[:half]] == [(s, a) for v, s, a in indices[half:]]
    qprob = semantic_log_probs(qlogits[:, -2], binary_ids)
    qjs = js(qprob[:half], qprob[half:]).mean()
    actor_estimates = [[{c['action']: str(c['safe_completion']) for c in row['candidates']}
                        for row in pair] for _ in views] if teacher else estimates
    texts, answers, goodsets = [], [], []
    for view, seeds in enumerate(views):
        for side, row in enumerate(pair):
            text, order = actor(row['observation'], pool, seeds, actor_estimates[view][side])
            answer, good = response(row, order, pool, seeds['names'], 'plain', random.Random(step * 2 + side))
            texts.append(text)
            answers.append(answer)
            goodsets.append(good)
    logits, ce, atokens = forward_targets(tokenizer, model, texts, answers)
    actor_weights = torch.tensor(weights(pair, multiplier) * len(views), device=logits.device)
    aloss = (actor_weights * torch.stack([torch.logsumexp(logits[i, -2], 0) - torch.logsumexp(logits[i, -2, letters[good]], 0)
                        for i, good in enumerate(goodsets)])).mean() + .1 * torch.cat([ce[:, :-2], ce[:, -1:]], 1).mean()
    ajs = logits.sum() * 0.
    total = qloss + aloss + weight * (qjs + ajs)
    return total, {'query_loss': qloss, 'actor_loss': aloss, 'query_js': qjs, 'actor_js': ajs}, qtokens + atokens, estimates, actor_estimates
