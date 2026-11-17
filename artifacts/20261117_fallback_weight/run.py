"""Matched weight-only continuation with unchanged autonomous policy inference."""
import gc
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

R=Path(__file__).resolve().parent;ROOT=R.parents[1]
sys.path[:0]=[str(R.parent/'20261117_frozen_crossover'),str(R.parent/'20261117_teacher_binding'),str(R.parent/'20261117_sampler_repair')]
from common import verify,load,torch,palette,generate,evaluate
from sampling import buckets
from binding_loss import loss_for_views as original_loss
sys.path.insert(0,str(R))
from weighted_loss import loss_for_views
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
rows=lambda p:list(map(json.loads,p.read_text().splitlines()))


def append(path,row):
    with path.open('a') as f:f.write(json.dumps(row)+'\n')


def main():
    plan=read(R/'PLAN.json');folder=R/'repair';assert not folder.exists()
    assert all(sha(R/n)==h for n,h in {**plan['source_hashes'],**plan['data_hashes']}.items())
    assert all(sha(ROOT/n)==h for n,h in plan['dependencies'].items())
    verify();folder.mkdir();(folder/'PLAN.json').write_text(json.dumps(plan,indent=2))
    shutil.copy2(R/'dev_worlds.json',folder/'dev_worlds.json')
    data=buckets();worlds=read(R/'dev_worlds.json');summaries=[]
    for arm in plan['arms']:
        target=folder/arm;target.mkdir()
        tokenizer,model=load('start',{'checkpoints':{'start':plan['checkpoint']}});pools=palette(tokenizer)
        if arm=='uniform':
            mixed=next(b for b in plan['batches'] if b['key'][0]=='Worker' and
                       len({not any(c['safe_completion'] for c in r['candidates']) for r in data[tuple(b['key'])][b['index']]})==2)
            pair=data[tuple(mixed['key'])][mixed['index']];start=time.perf_counter()
            with torch.no_grad():
                a=original_loss(tokenizer,model,pair,pools['train'],mixed['views'],0,0.,teacher=True)
                b=loss_for_views(tokenizer,model,pair,pools['train'],mixed['views'],0,0.,teacher=True,multiplier=1.)
            deltas={k:float((a[1][k]-b[1][k]).abs()) for k in a[1]}
            assert all(v<1e-6 for v in deltas.values()) and a[2:]==b[2:]
            (R/'IDENTITY_CHECK.json').write_text(json.dumps({'deltas':deltas,'seconds':time.perf_counter()-start,'errors':0},indent=2))
            del a,b
        if arm in plan['multipliers']:
            model.train();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=plan['lr'],weight_decay=0.)
            tokens=0
            for step,batch in enumerate(plan['batches']):
                def timeout():
                    (target/'TIMEOUT.json').write_text(json.dumps({'step':step,'limit':12.}));os._exit(124)
                timer=threading.Timer(12.,timeout);timer.daemon=True;timer.start();start=time.perf_counter()
                loss,parts,count,predicted,supplied=loss_for_views(tokenizer,model,data[tuple(batch['key'])][batch['index']],
                    pools['train'],batch['views'],step,0.,teacher=True,multiplier=plan['multipliers'][arm])
                optimizer.zero_grad();loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
                optimizer.step();timer.cancel();tokens+=count
                append(target/'training.jsonl',{'step':step+1,'role':batch['key'][0],'seconds':time.perf_counter()-start,'input_tokens':count,
                    'gradient_norm':norm,'multiplier':plan['multipliers'][arm],**{k:float(v.detach()) for k,v in parts.items()},
                    'predicted_estimates':predicted,'actor_input_estimates':supplied})
                if (step+1)%20==0:print(json.dumps({'arm':arm,'step':step+1}),flush=True)
            model.save_pretrained(target/'adapter')
            (target/'CHECKPOINT_SEALED.json').write_text(json.dumps({'adapter_hashes':{p.name:sha(p) for p in (target/'adapter').iterdir() if p.is_file()},
                'steps':80,'input_tokens':tokens,'initial_adapter_hashes':plan['checkpoint']['seal']['adapter_hashes']},indent=2));del optimizer
        else:
            source=ROOT/plan['checkpoint']['path'];shutil.copytree(source/'adapter',target/'adapter');shutil.copy2(source/'CHECKPOINT_SEALED.json',target/'CHECKPOINT_SEALED.json')
        model.eval()
        for condition,ws in worlds.items():
            result=evaluate(tokenizer,model,ws,pools['fresh'],True);path=target/(condition+'.jsonl')
            path.write_text(''.join(json.dumps(r)+'\n' for r in result))
            summary={'arm':arm,'condition':condition,'episodes':len(result),
                     **{k:sum(r[k] for r in result) for k in ['safe_success','strict_success','invalid','invalid_queries']},
                     'valid_ci':sum(r['ci'] and not r['invalid'] for r in result),'raw_sha256':sha(path)}
            summaries.append(summary);print(json.dumps(summary),flush=True)
        for name,path in [('train',R.parent/'20261117_option_fallback/observations.jsonl'),('fresh',R/'fresh_observations.jsonl')]:
            cases=rows(path)
            for start in range(0,len(cases),4):
                batch=cases[start:start+4];raws,seconds=generate(tokenizer,model,[c['prompt'] for c in batch])
                for case,raw in zip(batch,raws):append(target/(name+'_orders.jsonl'),{'id':case['id'],'raw':raw,'seconds_batch':seconds})
            print(json.dumps({'arm':arm,'oracle_diagnostic':name,'responses':len(cases)}),flush=True)
        del model;gc.collect()
    (folder/'RESULTS.json').write_text(json.dumps({'summaries':summaries,'whole_goal_complete':False},indent=2))
    (R/'SEALED.json').write_text(json.dumps({'complete':True,'files':{p.relative_to(R).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()},
                                         'whole_goal_complete':False},indent=2))


if __name__=='__main__':main()
