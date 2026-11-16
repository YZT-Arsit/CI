"""Recover fixed missing world IDs after timeouts, without resampling worlds."""
import json
from pathlib import Path

R=Path(__file__).resolve().parent
counts={'train':256,'dev':32,'test':128,'larger':64,'optional':64,'authorized':64,'undelivered':64,'both':32}
jobs=[];summary={}
for split,count in counts.items():
    folder=R/'dataset'/split;covered=set()
    for wpath in folder.glob('worlds_*.jsonl'):
        ppath=wpath.with_name(wpath.name.replace('worlds_','preferences_'))
        if not ppath.exists():
            quarantine=R/'partial_data'/split;quarantine.mkdir(parents=True,exist_ok=True)
            wpath.rename(quarantine/wpath.name);continue
        worlds=[json.loads(s) for s in wpath.read_text().splitlines()]
        records=[json.loads(s) for s in ppath.read_text().splitlines()]
        assert len(worlds)==16 and {r['world_id'] for r in records}.issubset({w['world_id'] for w in worlds})
        start=int(wpath.stem.split('_')[-1]);covered.update(range(start,start+16))
    missing=sorted(set(range(count))-covered);summary[split]={'complete_worlds':len(covered),'missing_worlds':len(missing)}
    for start in missing[::4]:
        assert all(i in missing for i in range(start,start+4))
        jobs.append({'id':f'recover_{split}_{start:04d}','args':['-m','experiments.receipt_dataset',
            '../artifacts/20261115_semantic_repair/dataset',split,str(start),'4']})
(R/'recovery_jobs.json').write_text(json.dumps(jobs,indent=2))
(R/'recovery_plan.json').write_text(json.dumps({'coverage':summary,'jobs':len(jobs),'same_fixed_world_ids':True},indent=2))
print(json.dumps({'coverage':summary,'jobs':len(jobs)}))
