"""Guarantee six value-reversing pairs per role/depth while retaining anchors."""
import collections
import random
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'20261117_query_coverage'))
from training_data import schedule as original_schedule,matched_groups


def schedule():
    data=matched_groups();batches=original_schedule();rng=random.Random(42117);used=collections.Counter()
    for step,b in enumerate(batches):
        round_index=step//3;depth=b['mixed_depth'];occurrence=round_index//3
        forced=occurrence%2==0 or (depth==3 and occurrence==9)
        b['forced_contrast']=forced
        if not forced:continue
        role=b['key'][0];kind=['meeting','query'][used[role,depth]%2];eligible=[]
        for key,groups in sorted(data.items()):
            if key[0]!=role or key[1] or not key[2]:continue
            for entry in groups:
                if entry['factors'][0]!=kind:continue
                a,c=entry['depths'][depth]
                if any(x['safe_completion']!=y['safe_completion'] for x,y in zip(a['candidates'],c['candidates'])):eligible.append((key,entry))
        key,entry=rng.choice(eligible);used[role,depth]+=1
        b.update(key=list(key),factors=entry['factors'],pairs={str(d):p for d,p in entry['depths'].items()})
    assert len(used)==9 and set(used.values())=={6}
    return batches
