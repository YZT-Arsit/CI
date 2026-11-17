"""Freeze matched training depth pairs and fresh evaluation evidence before model calls."""
import collections
import json
from probe_data import R,ROOT,OLD,read,rows,sha,verify_prior
from training_data import schedule,DATA
from eval_worlds import dev_worlds,targets,check,strict_reversal
from render import actor,query
from kci.relation_language import seed_for


def write_rows(name,data):
    (R/name).write_text(''.join(json.dumps(r)+'\n' for r in data),encoding='utf-8')


def main():
    assert not (R/'PLAN.json').exists();verify_prior();batches=schedule();worlds,paired=dev_worlds()
    used={r['world_id'] for p in DATA.glob('worlds_*.jsonl') for r in rows(p)}
    used|={w['world_id'] for ws in read(OLD/'dev_worlds.json').values() for w in ws}
    assert not used & {w['world_id'] for ws in worlds.values() for w in ws}
    pool=read(R.parent/'20261116_relation_solution/entity_palettes.json')['fresh'];qobs=[];qlabels=[];unique={}
    for condition,ws in worlds.items():
        for index,w in enumerate(ws):
            for si,case in enumerate(targets(w)):
                assert not check(case);role=case['observation']['role'];phase={'Planner':0,'Source':1,'Worker':2}[role]
                nonce=seed_for(w['base_group_id'],phase);seeds={k:nonce for k in ['names','facts','plans','links','options']}
                for ci,c in enumerate(case['candidates']):
                    identifier=f'{condition}_{index}_{si}_{ci}'
                    qobs.append({'id':identifier,'prompt':query(case['observation'],pool,seeds,c['action'])})
                    qlabels.append({'id':identifier,'condition':condition,'world_index':index,'state_index':si,'candidate_index':ci,'target':c['safe_completion'],'role':role})
                if role=='Worker':unique[sha_observation(case)]=case
    for condition in paired:
        assert all(strict_reversal([targets(a)[0],targets(b)[0]]) for a,b in zip(worlds[condition],worlds[condition+'_swap']))
    obs=[];labels=[]
    for zero in [True,False]:
        selected=sorted((cid,c) for cid,c in unique.items() if (not any(x['safe_completion'] for x in c['candidates']))==zero)[:12]
        assert len(selected)==12
        for cid,case in selected:
            seeds={k:410000000+int(cid[:6],16) for k in ['names','facts','plans','links','options']};perms={}
            bits={c['action']:str(c['safe_completion']) for c in case['candidates']}
            for n in range(1000):
                ss={**seeds,'options':n};text,order=actor(case['observation'],pool,ss,bits);perms.setdefault(tuple(order),(ss,text))
                if len(perms)==6:break
            assert len(perms)==6
            for n,(order,(ss,text)) in enumerate(sorted(perms.items())):
                identifier=cid+'_'+str(n);obs.append({'id':identifier,'case_id':cid,'group':'zero' if zero else 'positive','prompt':text,'actions':list(order),'seeds':ss})
                labels.append({'id':identifier,'case':case})
    write_rows('train_batches.jsonl',batches);write_rows('query_observations.jsonl',qobs);write_rows('query_labels.jsonl',qlabels)
    write_rows('fresh_observations.jsonl',obs);write_rows('fresh_labels.jsonl',labels)
    (R/'dev_worlds.json').write_text(json.dumps(worlds),encoding='utf-8')
    cp=OLD/'repair/interleaved';deps=dict(read(OLD/'PLAN.json')['dependencies'])
    for name in ['20261117_fallback_order/correction_loss.py','20261117_fallback_order/correction.jsonl','20261117_fallback_order/PLAN.json','20261117_fallback_order/SEALED.json','20261117_teacher_binding/replay.py']:
        deps['artifacts/'+name]=sha(R.parent/name)
    for path in DATA.glob('*.jsonl'):deps[path.relative_to(ROOT).as_posix()]=sha(path)
    frozen_sources=['training_data.py','eval_worlds.py','build.py','run.py','audit.py','probe_data.py']
    data_names=['train_batches.jsonl','query_observations.jsonl','query_labels.jsonl','fresh_observations.jsonl','fresh_labels.jsonl','dev_worlds.json','PLAN.md']
    plan={'arms':['one_hop','mixed_depth','initial','earlier'],'evidence_arms':['one_hop','mixed_depth','initial','earlier'],
        'checkpoint':{'path':cp.relative_to(ROOT).as_posix(),'seal':read(cp/'CHECKPOINT_SEALED.json')},
        'earlier_checkpoint':{'path':(OLD/'repair/initial').relative_to(ROOT).as_posix(),'seal':read(OLD/'repair/initial/CHECKPOINT_SEALED.json')},
        'lr':.0002,'seed':17,'updates_per_arm':123,'joint_batches':96,'correction_batches':27,'update_guard_seconds':15,
        'schedules':[[kind,n] for n in range(96) for kind in (['joint','correct'] if n<81 and n%3==2 else ['joint'])],
        'paired_conditions':paired,'source_sha256':sha(R/'run.py'),'dependencies':deps,
        'source_hashes':{n:sha(R/n) for n in frozen_sources},'data_hashes':{n:sha(R/n) for n in data_names},
        'probe_seal_sha256':sha(R/'PROBE_SEALED.json'),'query_count':len(qobs),'episodes_per_arm':sum(map(len,worlds.values())),
        'whole_goal_complete':False,'oracle_at_runtime':False}
    for item in plan['schedules']:
        if item[0]=='correct':item[1]=(item[1]+1)//3-1
    assert len(plan['schedules'])==123
    (R/'PLAN.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print(json.dumps({'episodes_per_arm':plan['episodes_per_arm'],'fixed_queries_per_arm':len(qobs),'new_updates':246,
        'role_depth_counts':{str(k):v for k,v in collections.Counter((b['key'][0],b['mixed_depth']) for b in batches).items()}}))


def sha_observation(case):
    import hashlib
    return hashlib.sha256(json.dumps(case['observation'],sort_keys=True).encode()).hexdigest()


if __name__=='__main__':main()
