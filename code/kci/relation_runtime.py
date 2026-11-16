"""Frozen structured reference policy; not a replacement for the LLM research goal."""
import random
import torch
from .relation_solution import encode,collate

def choose_action(model,observation,order_seed):
    """Learned scores only. The 64D/256-fact bounds never truncate or mask actions."""
    actions=list(observation['actions']);random.Random(order_seed).shuffle(actions)
    inputs=collate([encode(observation,a,salt=811,width=64,max_facts=256) for a in actions])
    with torch.inference_mode():
        logits=model(*inputs);scores=model.score(logits)
    selected=actions[int(scores.argmax())]
    return selected,{'actions':actions,'logits':logits.tolist(),'scores':scores.tolist(),
        'entity_width':64,'max_facts':256,'oracle_mask':False,'fallback':False}
