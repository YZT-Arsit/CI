"""Replay saved learned actions, independently recheck final task/CI outcome."""
import sys,json
from hashlib import sha256
from dataclasses import asdict
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];sys.path.insert(0,str(ROOT/'code'))
from kci.receipt_tasks import ReceiptTask
from kci.ordering_receipt_tasks import OrderingReceiptTask
from kci.dynamic_tasks import canonical
from experiments.receipt_dataset import SOURCE,rename

def run(version,model,condition):
    base=condition.replace('_renamed','').replace('_fresh_tokens','')
    worlds={w['world_id']:w for p in (R/'dataset'/base).glob('worlds_*.jsonl') for w in map(json.loads,p.read_text().splitlines())}
    records=[json.loads(x) for x in (R/f'evaluation_v{version}_fast'/model/(condition+'.jsonl')).read_text().splitlines()]
    for record in records:
        w=worlds[record['world_id']]
        if condition.endswith('_renamed'):w=rename(w,771)
        cls=OrderingReceiptTask if w.get('kernel_variant')=='ordering' else ReceiptTask
        env=cls(w,ROOT/w['source_root_relative'] if 'source_root_relative' in w else SOURCE);s=env.initial()
        for step in record['trace']:
            assert step['role']==env.actor(s)
            assert sha256(canonical(env.observation(s)).encode()).hexdigest()==step['observation_sha256']
            assert set(step['actions'])==set(env.actions(s))
            assert step['selected']==step['actions'][max(range(len(step['scores'])),key=lambda i:step['scores'][i])]
            s,audit=env.transition(s,step['selected']);assert canonical(audit)==canonical(step['audit'])
        assert s.done and s.success==record['success'] and (not s.violation)==record['ci']
        assert canonical(asdict(s))==canonical(record['final'])
        # Every successful receipt must match native computational truth.
        if s.success:
            output=s.ledger[-1]['content']['result']
            if w['kind']=='meeting':assert output==env.kernel.utility['oracle_schedule']
            else:
                from audit_semantics import independent_task_truth
                assert output==independent_task_truth(env)
    out=R/'replay_audits';out.mkdir(exist_ok=True)
    report={'version':version,'model':model,'condition':condition,'episodes':len(records),'all_passed':True,'argmax_verified':True,'native_task_truth_verified':True,'interventions':0}
    (out/f'v{version}_{model}_{condition}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':run(*sys.argv[1:])
