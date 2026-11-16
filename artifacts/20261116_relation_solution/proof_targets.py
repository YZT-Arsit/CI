"""Offline supervised targets; forbidden from inference imports/use."""
from collections import deque
from kci.relation_language import vocabulary
from kci.relation_solution import facts_and_flags

def supporting_path(obs,action):
    facts,_=facts_and_flags(obs,action);subject=obs['subject']
    starts=[b for t,a,b in facts if t==0 and a==subject]
    links=[(a,b) for t,a,b in facts if t in (1,2)]
    history={}
    for t,a,b in facts:
        if t==3:history.setdefault(a,[]).append(b)
    queue=deque((x,[subject,x]) for x in sorted(starts));visited=set()
    while queue:
        node,path=queue.popleft()
        if node in visited:continue
        visited.add(node)
        if node in history:return path+[sorted(history[node])[0]]
        queue.extend((b,path+[b]) for a,b in sorted(links) if a==node)
    return []

def response(row,actions,palette,nonce,arm,rng):
    good=[actions.index(row['candidates'][i]['action']) for i in row['optimal_indices']]
    choice=rng.choice(good);pieces=[]
    if arm=='trace':
        names=vocabulary(row['observation'],palette,nonce);lookup={c['action']:c for c in row['candidates']}
        for i,action in enumerate(actions):
            c=lookup[action];path=supporting_path(row['observation'],action)
            text=' > '.join(names[x] for x in path) if path else 'none'
            pieces.append(f"{chr(65+i)} SAFE {c['immediate_safe']} TASK {c['safe_completion']} PATH {text};")
    pieces.append('ANSWER:\n'+chr(65+choice))
    return '\n'.join(pieces),good
