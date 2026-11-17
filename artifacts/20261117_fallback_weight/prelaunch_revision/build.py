"""Freeze a single-factor weight experiment and unseen diagnostic states."""
import hashlib
import json
import sys
from pathlib import Path
from worlds import dev_worlds, targets, check, strict_reversal, DATA
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'20261117_binding_factors'))
from render import actor
from weighting import weights

R=Path(__file__).resolve().parent; ROOT=R.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
rows=lambda p:list(map(json.loads,p.read_text().splitlines()))


def main():
    assert not (R/'PLAN.json').exists()
    old=read(R.parent/'20261117_optimization_repair/PLAN.json')
    worlds,paired=dev_worlds()
    used={r['world_id'] for p in DATA.glob('states_*.jsonl') for r in rows(p)}
    used|={w['world_id'] for ws in read(R.parent/'20261117_optimization_repair/dev_worlds.json').values() for w in ws}
    assert not used & {w['world_id'] for ws in worlds.values() for w in ws}
    unique={}
    for ws in worlds.values():
        for w in ws:
            for case in targets(w):
                assert not check(case)
                if case['observation']['role']=='Worker':
                    cid=hashlib.sha256(json.dumps(case['observation'],sort_keys=True).encode()).hexdigest()
                    unique[cid]=case
    for name in paired:
        assert all(strict_reversal([targets(a)[0],targets(b)[0]]) for a,b in zip(worlds[name],worlds[name+'_swap']))
    train=rows(R.parent/'20261117_frozen_crossover/train_oracle.jsonl')
    assert len(train)==320 and abs(sum(weights(train,4))-320)<1e-8
    pool=read(R.parent/'20261116_relation_solution/entity_palettes.json')['fresh']
    obs=[];labels=[]
    for zero in (True,False):
        selected=sorted((cid,c) for cid,c in unique.items() if (not any(x['safe_completion'] for x in c['candidates']))==zero)[:12]
        assert len(selected)==12
        for cid,case in selected:
            seeds={k:370000000+int(cid[:6],16) for k in ['names','facts','links','plans','options']}
            bits={c['action']:str(c['safe_completion']) for c in case['candidates']};perms={}
            for seed in range(1000):
                ss={**seeds,'options':seed};text,order=actor(case['observation'],pool,ss,bits)
                perms.setdefault(tuple(order),(ss,text))
                if len(perms)==6:break
            assert len(perms)==6
            for n,(order,(ss,text)) in enumerate(sorted(perms.items())):
                identifier=cid+'_'+str(n)
                obs.append({'id':identifier,'case_id':cid,'group':'zero' if zero else 'positive','prompt':text,'actions':list(order),'seeds':ss})
                labels.append({'id':identifier,'case':case})
    (R/'dev_worlds.json').write_text(json.dumps(worlds))
    for name,data in [('fresh_observations',obs),('fresh_labels',labels)]:
        (R/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in data))
    checkpoint=R.parent/'20261117_optimization_repair/repair/continue_lower_lr'
    deps=dict(old['dependencies'])
    for name in ['20261117_optimization_repair/PLAN.json','20261117_option_fallback/observations.jsonl','20261117_option_fallback/labels.jsonl',
                 '20261117_frozen_crossover/PLAN.json','20261117_frozen_crossover/common.py','20261117_teacher_binding/binding_loss.py']:
        deps['artifacts/'+name]=sha(R.parent/name)
    plan={'arms':['uniform','weighted','initial'],'evidence_arms':['uniform','weighted','initial'],
          'multipliers':{'uniform':1.,'weighted':4.},'lr':.0002,'updates_per_arm':80,'batches':old['batches'],
          'paired_conditions':paired,'checkpoint':{'path':checkpoint.relative_to(ROOT).as_posix(),'seal':read(checkpoint/'CHECKPOINT_SEALED.json')},
          'dependencies':deps,'source_hashes':{p.name:sha(p) for p in R.glob('*.py')},'source_sha256':sha(R/'run.py'),
          'data_hashes':{n:sha(R/n) for n in ['dev_worlds.json','fresh_observations.jsonl','fresh_labels.jsonl','PLAN.md']},
          'whole_goal_complete':False,'oracle_at_runtime':False,'seed':17,'train_zero_actor_presentations':18,'train_actor_presentations':320}
    (R/'PLAN.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps({'episodes_per_arm':sum(map(len,worlds.values())),'fresh_oracle_prompts':len(obs),'new_updates':160}))


if __name__=='__main__':main()
