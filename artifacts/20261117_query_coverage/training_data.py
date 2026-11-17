"""Align existing TRAIN history pairs across depths by non-depth factors."""
import collections
import itertools
import random
from probe_data import R,rows

DATA=R.parent/'20261116_relation_solution/dataset/train'


def matched_groups():
    worlds={w['world_id']:w for p in sorted(DATA.glob('worlds_*.jsonl')) for w in rows(p)}
    groups={}
    for p in sorted(DATA.glob('states_*.jsonl')):
        for row in rows(p):
            obs=row['observation']
            groups.setdefault((row['base_group_id'],obs['role'],obs['committed_ticket']),[]).append(row)
    grid=list(itertools.product(('meeting','query'),(1,2,3),(0,2),('one','none','both'),(False,True),(False,True),(False,True)))
    matched={}
    for pair in groups.values():
        pair.sort(key=lambda r:r['world_id']);assert len(pair)==2
        w=worlds[pair[0]['world_id']];kind,depth,noise,mode,allow,required,delivered=grid[w['factor_cell']]
        obs=pair[0]['observation'];replica=int(w['base_group_id'].split('_')[1])-1000000-2*w['factor_cell']
        committed=next((i for i,p in enumerate(w['providers']) if p['ticket']==obs['committed_ticket']),-1)
        key=(kind,noise,mode,allow,required,delivered,replica,obs['role'],committed)
        assert depth==len(w['providers'][0]['links']) and replica in (0,1)
        assert depth not in matched.setdefault(key,{})
        matched[key][depth]=pair
    buckets=collections.defaultdict(list)
    for key,depths in sorted(matched.items()):
        assert set(depths)=={1,2,3}
        row_signature=lambda r:[(c['immediate_safe'],c['safe_completion'],c['value']) for c in r['candidates']]
        for pair in depths.values():pair.sort(key=row_signature)
        signature=lambda pair:[row_signature(r) for r in pair]
        assert signature(depths[1])==signature(depths[2])==signature(depths[3]), (key,signature(depths[1]),signature(depths[2]),signature(depths[3]))
        pair=depths[1];obs=pair[0]['observation'];c=obs['contract']
        bucket=(obs['role'],c['permission'],c['binding_required'],any(v['safe_completion'] for r in pair for v in r['candidates']),any(not v['immediate_safe'] for r in pair for v in r['candidates']))
        buckets[bucket].append({'factors':list(key),'depths':depths})
    return buckets


def schedule():
    data=matched_groups();rng=random.Random(39117);batches=[]
    for step in range(96):
        role=['Planner','Source','Worker'][step%3];keys=sorted(k for k in data if k[0]==role)
        key=keys[(step//3)%len(keys)];entry=data[key][rng.randrange(len(data[key]))]
        views=[{k:rng.randrange(10**9) for k in ['names','facts','plans','links','options']} for _ in range(2)]
        batches.append({'step':step,'key':list(key),'factors':entry['factors'],'views':views,
            'mixed_depth':1+(step//3)%3,'pairs':{str(d):pair for d,pair in entry['depths'].items()}})
    return batches


if __name__=='__main__':
    data=matched_groups();batches=schedule()
    print({'matched_factor_groups':sum(map(len,data.values())),'condition_buckets':len(data),
        'batches':len(batches),'mixed_role_depth_counts':dict(collections.Counter((b['key'][0],b['mixed_depth']) for b in batches))})
