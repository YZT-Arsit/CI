"""Keep immediate safety dominance even when safe task completion is impossible."""
import sys,json,itertools,copy
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];sys.path.insert(0,str(ROOT/'code'))
from experiments.receipt_dataset import SOURCE,observations
from kci.receipt_tasks import ReceiptTask
from kci.dynamic_tasks import canonical

def run(start):
    out=R/'dataset/train_safety';out.mkdir(exist_ok=True);new=[]
    source=R/'dataset/train_balanced';key=f'{start:04d}'
    worlds=[json.loads(x) for x in (source/f'worlds_{key}.jsonl').read_text().splitlines()]
    rows=[json.loads(x) for x in (source/f'preferences_{key}.jsonl').read_text().splitlines()]
    for row in rows:row['supervision']='future' if row['level']=='future' else 'task_utility'
    for w in worlds:
        env=ReceiptTask(w,SOURCE)
        for s in observations(env):
            for a,b in itertools.combinations(env.actions(s),2):
                proof=env.preference(s,a,b)
                if proof['chosen'] is not None or set(proof['immediate'])!={'VALID','VIOLATION'}:continue
                chosen=proof['immediate'].index('VALID');assert proof['values'][chosen]>proof['values'][1-chosen]
                pair=[a,b]
                fixed={**proof,'status':'PREFERENCE','chosen':chosen,'scope':'immediate CI dominance; no successful-continuation claim'}
                new.append({'world_id':w['world_id'],'level':'immediate','supervision':'immediate_ci',
                    'observation':env.observation(s),'chosen':pair[chosen],'rejected':pair[1-chosen],
                    'proof':fixed,'source_hashes':env.kernel.source_hashes,'gpu_large_scale_eligible':False})
    for name,data in [('worlds',worlds),('preferences',rows+new)]:
        path=out/f'{name}_{key}.jsonl'
        if path.exists():raise FileExistsError(path)
        path.write_text(''.join(canonical(x)+'\n' for x in data))
    print(json.dumps({'worlds':len(worlds),'original_pairs':len(rows),'new_immediate_ci_pairs':len(new)}))
if __name__=='__main__':run(int(sys.argv[1]))
