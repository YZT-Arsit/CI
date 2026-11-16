"""Frozen V5 transfer to native-output joins; no policy fallback or new training."""
import sys,json,time,threading,os,copy
from pathlib import Path
from hashlib import sha256
R=Path(__file__).resolve().parent;ROOT=R.parents[1];sys.path.insert(0,str(ROOT/'code'))
import evaluate_fast_v5 as runner
from kci.native_result_tasks import NativeResultTask
from kci.dynamic_tasks import canonical
torch=runner.torch
templates={}
def native_env(world,source):
    key=(world['kind'],str(source))
    if key not in templates:templates[key]=NativeResultTask(world,source)
    env=copy.deepcopy(templates[key]);env.world=copy.deepcopy(world);env.binding['world']=copy.deepcopy(world);return env
runner.build_env=native_env

def run():
    seal=R/'models_v5/CHECKPOINTS_SEALED.json';models=json.loads(seal.read_text())['models']
    out=R/'native_evaluation';out.mkdir(exist_ok=True)
    if (out/'plan.json').exists():raise FileExistsError('native evaluation exists')
    sources=json.loads((R/'source_qualification_v2.json').read_text())
    (out/'plan.json').write_text(json.dumps({'checkpoint_seal':sha256(seal.read_bytes()).hexdigest(),
        'no_extra_task_receipt':True,'no_model_retraining':True,'no_fallback':True,
        'source_hashes':{str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'code/kci/native_result_tasks.py',R/'NATIVE_RESULT_PLAN.md']}},indent=2))
    results=[]
    for report in models:
        kind,arm,seed=report['kind'],report['arm'],report['seed'];name=f'{kind}_{arm}_{seed}'
        model=runner.GraphPolicy(relations=arm!='no_relations') if kind=='graph' else runner.TokenPolicy()
        checkpoint=R/'models_v5'/name/'latest.pt';assert sha256(checkpoint.read_bytes()).hexdigest()==report['checkpoint_sha256']
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model']);model.eval()
        folder=out/name;folder.mkdir();conditions=[]
        for split,fresh in [('native_base',False),('native_swapped',False)]+([('native_base',True)] if kind=='token' else []):
            worlds=[json.loads(x) for p in sorted((R/'dataset'/split).glob('worlds_*.jsonl')) for x in p.read_text().splitlines()]
            rows=[];timings=[]
            for offset in range(0,len(worlds),8):
                timer=threading.Timer(9.5,lambda:os._exit(124));timer.daemon=True;timer.start();start=time.perf_counter()
                try:rows.extend(runner.execute(model,kind,w,fresh=fresh) for w in worlds[offset:offset+8])
                finally:timer.cancel()
                timings.append(time.perf_counter()-start)
            label=split+('_fresh' if fresh else '')
            (folder/(label+'.jsonl')).write_text(''.join(canonical(r)+'\n' for r in rows))
            conditions.append({'condition':label,'episodes':len(rows),'ci':sum(r['ci'] for r in rows),'safe_success':sum(r['safe_success'] for r in rows),'chunk_seconds':timings})
        result={'model':name,'kind':kind,'arm':arm,'seed':seed,'conditions':conditions};results.append(result)
    (out/'EVALUATION_SEALED.json').write_text(json.dumps({'models':results,'no_extra_native_task_requirement':True},indent=2))
    print(json.dumps({'models':len(results),'sealed':True}))
if __name__=='__main__':run()
