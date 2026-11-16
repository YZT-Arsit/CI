"""Candidate-independent symbol vocabulary and semantics-preserving augmentation."""
import hashlib,random
import numpy as np
from kci.receipt_policy import candidate_view,CHANNELS


def token_encode(obs,action,*,fresh=False,max_tokens=192,shuffle_seed=0):
    subject,edges,flags=candidate_view(obs,action)
    inventory={subject}
    for provider in obs['contract']['providers']:
        inventory.add(provider['ticket'])
        for a,b in provider['links']:inventory.update((a,b))
    for m in obs['delivered']:
        if m['recipient']!='R':continue
        for name in ('identities','links','histories'):
            for a,b in m['content'].get(name,[]):inventory.update((a,b))
    nodes=sorted(inventory);seed=int(hashlib.sha256(('|'.join(nodes)+str(shuffle_seed)).encode()).hexdigest()[:12],16)
    rng=random.Random(seed);pool=list(range(128 if fresh else 64,192 if fresh else 128));rng.shuffle(pool)
    if len(nodes)>len(pool):raise ValueError('symbol budget')
    mapping=dict(zip(nodes,pool));triples=[]
    for channel,name in enumerate(CHANNELS):
        triples += [[30+channel,mapping[a],mapping[b]] for a,b in edges[name]]
    rng.shuffle(triples)
    tokens=[1,mapping[subject]]+[3+2*i+int(flag) for i,flag in enumerate(flags)]
    tokens += [v for triple in triples for v in triple]
    if len(tokens)>max_tokens:raise ValueError('no truncation')
    return np.array(tokens+[0]*(max_tokens-len(tokens)),np.int64)


def augment_pairs(batch,rng):
    # Batch order is [chosen_0..chosen_n, rejected_0..rejected_n]. A shared
    # per-pair bijection preserves cross-action identity. Triple permutation
    # changes presentation only and never alters the relation facts.
    result=batch.copy();half=len(batch)//2
    for i in range(half):
        permutation=rng.permutation(np.arange(64,128))
        for j in (i,i+half):
            row=result[j];mask=(row>=64)&(row<128);row[mask]=permutation[row[mask]-64]
            count=(np.count_nonzero(row)-12)//3
            row[12:12+3*count]=row[12:12+3*count].reshape(count,3)[rng.permutation(count)].reshape(-1)
    return result
