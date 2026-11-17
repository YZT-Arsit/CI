"""Freeze a single-factor weight experiment and unseen diagnostic states."""
import hashlib
import json
import sys
from pathlib import Path
from worlds import dev_worlds, targets, check, strict_reversal, DATA
from experiments.receipt_dataset import SOURCE
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'20261117_binding_factors'))
from render import actor
from schedules import make_schedule

R=Path(__file__).resolve().parent; ROOT=R.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
rows=lambda p:list(map(json.loads,p.read_text().splitlines()))


def main():
    assert not (R/'PLAN.json').exists()
    old=read(R.parent/'20261117_fallback_weight/PLAN.json')
    worlds,paired=dev_worlds()
    used={r['world_id'] for p in DATA.glob('states_*.jsonl') for r in rows(p)}
    for previous in ['20261117_optimization_repair','20261117_fallback_weight']:
        used|={w['world_id'] for ws in read(R.parent/previous/'dev_worlds.json').values() for w in ws}
    assert not used & {w['world_id'] for ws in worlds.values() for w in ws}
    unique={};source_assets={}
    for ws in worlds.values():
        for w in ws:
            for case in targets(w):
                assert not check(case)
                for name,digest in case['source_hashes'].items():
                    path=SOURCE/name;assert sha(path)==digest
                    source_assets[path.relative_to(ROOT).as_posix()]=digest
                if case['observation']['role']=='Worker':
                    cid=hashlib.sha256(json.dumps(case['observation'],sort_keys=True).encode()).hexdigest()
                    unique[cid]=case
    for name in paired:
        assert all(strict_reversal([targets(a)[0],targets(b)[0]]) for a,b in zip(worlds[name],worlds[name+'_swap']))
    train=rows(R.parent/'20261117_frozen_crossover/train_oracle.jsonl')
    assert len(train)==320
    pool=read(R.parent/'20261116_relation_solution/entity_palettes.json')['fresh']
    obs=[];labels=[]
    for zero in (True,False):
        selected=sorted((cid,c) for cid,c in unique.items() if (not any(x['safe_completion'] for x in c['candidates']))==zero)[:12]
        assert len(selected)==12
        for cid,case in selected:
            seeds={k:390000000+int(cid[:6],16) for k in ['names','facts','links','plans','options']}
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
    deps=dict(old['dependencies']);deps.update(source_assets)
    for folder in [ROOT/'code/kci',ROOT/'code/experiments']:
        deps.update({p.relative_to(ROOT).as_posix():sha(p) for p in folder.glob('*.py')})
    for name in ['20261117_optimization_repair/PLAN.json','20261117_option_fallback/observations.jsonl','20261117_option_fallback/labels.jsonl',
                 '20261117_fallback_weight/PLAN.json','20261117_fallback_weight/dev_worlds.json',
                 '20261117_option_fallback/responses.jsonl','20261117_action_binding/build_structural_cases.py',
                 '20261117_action_binding/build_degree_controls_v2.py','20261116_relation_solution/audit_data.py',
                 '20261117_frozen_crossover/PLAN.json','20261117_frozen_crossover/common.py','20261117_teacher_binding/binding_loss.py']:
        deps['artifacts/'+name]=sha(R.parent/name)
    oldobs=rows(R.parent/'20261117_option_fallback/observations.jsonl')
    oldlabels=rows(R.parent/'20261117_option_fallback/labels.jsonl')
    correction=[]
    for obs,label in zip(oldobs,oldlabels):
        if obs['group']!='zero':continue
        assert obs['id']==label['id']
        good=[i for i,a in enumerate(obs['actions']) if a in label['optimal_actions']]
        assert len(good)==2
        choice=good[int(hashlib.sha256(obs['id'].encode()).hexdigest()[:8],16)%2]
        correction.append({'id':obs['id'],'prompt':obs['prompt'],'actions':obs['actions'],
                           'good_indices':good,'answer':'ANSWER:\n'+'ABC'[choice]})
    assert len(correction)==108
    (R/'correction.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in correction))
    arms=['correct_then_replay','replay_then_correct','interleaved','initial']
    plan={'arms':arms,'evidence_arms':arms,'schedules':{a:make_schedule(a) for a in arms if a!='initial'},
          'milestones':{'correct_then_replay':27,'replay_then_correct':80},
          'lr':.0002,'updates_per_arm':107,'batches':old['batches'],'correction_batches':27,
          'paired_conditions':paired,'checkpoint':old['checkpoint'],
          'dependencies':deps,'source_hashes':{p.name:sha(p) for p in R.glob('*.py')},'source_sha256':sha(R/'run.py'),
          'data_hashes':{n:sha(R/n) for n in ['dev_worlds.json','fresh_observations.jsonl','fresh_labels.jsonl','correction.jsonl','PLAN.md']},
          'whole_goal_complete':False,'oracle_at_runtime':False,'seed':17}
    (R/'PLAN.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps({'episodes_per_arm':sum(map(len,worlds.values())),'fresh_oracle_prompts':144,'new_updates':321}))


if __name__=='__main__':main()
