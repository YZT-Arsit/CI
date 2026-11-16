"""Separate chain depth, distractors, branching before reading test predictions."""
import json,sys
from pathlib import Path

R=Path(__file__).resolve().parent;ROOT=R.parents[1];sys.path.insert(0,str(ROOT/'code'))
from experiments.receipt_dataset import make_world
from kci.dynamic_tasks import canonical

for condition,base in [('depth_only',90000),('noise_only',100000),('branching',110000)]:
    folder=R/'dataset'/condition;folder.mkdir(exist_ok=True);worlds=[]
    for index in range(64):
        world=make_world(base+index,('meeting','query')[index%2],
            depth=3 if condition=='depth_only' else 1+(index//2)%2,
            noise=4 if condition=='noise_only' else (index//4)%3)
        if condition=='branching':
            for j,provider in enumerate(world['providers']):
                provider['links'].append([provider['ticket'],f'branch{base+index}_{j}'])
        worlds.append(world)
    path=folder/'worlds_0000.jsonl'
    if path.exists():raise FileExistsError('fixed controls already exist')
    path.write_text(''.join(canonical(w)+'\n' for w in worlds),encoding='utf-8')
print(json.dumps({'additional_worlds':192,'conditions':['depth_only','noise_only','branching'],'predictions_seen':False}))
