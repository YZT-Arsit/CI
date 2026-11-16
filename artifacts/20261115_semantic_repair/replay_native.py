import sys,json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];sys.path.insert(0,str(ROOT/'code'))
from kci.native_result_tasks import NativeResultTask
from kci.dynamic_tasks import canonical

name=sys.argv[1];total=0
for file in sorted((R/'native_evaluation'/name).glob('*.jsonl')):
    split='native_base' if file.stem.endswith('_fresh') else file.stem
    worlds={w['world_id']:w for p in (R/'dataset'/split).glob('worlds_*.jsonl') for w in map(json.loads,p.read_text().splitlines())}
    for line in file.read_text().splitlines():
        record=json.loads(line);w=worlds[record['world_id']];env=NativeResultTask(w,ROOT/w['source_root_relative']);s=env.initial()
        for step in record['trace']:
            assert sha256(canonical(env.observation(s)).encode()).hexdigest()==step['observation_sha256']
            assert set(step['actions'])==set(env.actions(s))
            assert step['selected']==step['actions'][max(range(len(step['scores'])),key=lambda i:step['scores'][i])]
            s,audit=env.transition(s,step['selected']);assert canonical(audit)==canonical(step['audit'])
        assert canonical(asdict(s))==canonical(record['final'])
        if s.success:assert s.ledger[-1]['content']=={'type':'native_result','result':env.expected}
        total+=1
out=R/'native_replay_audits';out.mkdir(exist_ok=True)
report={'model':name,'episodes':total,'all_passed':True,'argmax_observation_audit_final_state_verified':True}
(out/(name+'.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report))
