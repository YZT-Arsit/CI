"""Order-only intervention, with actual free-policy evaluation after training."""
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
from binding_loss import loss_for_views
from correction_loss import correction_loss
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
rows=lambda p:list(map(json.loads,p.read_text().splitlines()))


def append(path,row):
    with path.open('a') as f:f.write(json.dumps(row)+'\n')


def digests(model,optimizer):
    params=hashlib.sha256();state=hashlib.sha256()
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        params.update(name.encode());params.update(p.detach().numpy().tobytes())
        if optimizer is not None:
            state.update(name.encode())
            for key,value in sorted(optimizer.state[p].items()):
                state.update(key.encode());state.update(value.detach().numpy().tobytes())
    return {'trainable_parameters':params.hexdigest(),'optimizer':state.hexdigest()}


def save(model,target,step,tokens,initial):
    model.save_pretrained(target/'adapter')
    (target/'CHECKPOINT_SEALED.json').write_text(json.dumps({'adapter_hashes':{p.name:sha(p) for p in (target/'adapter').iterdir() if p.is_file()},
        'steps':step,'input_tokens':tokens,'initial_adapter_hashes':initial},indent=2))


def diagnostic(tokenizer,model,target,optimizer=None):
    before=digests(model,optimizer);model.eval()
    for name,path in [('train',R.parent/'20261117_option_fallback/observations.jsonl'),('fresh',R/'fresh_observations.jsonl')]:
        cases=rows(path)
        for start in range(0,len(cases),4):
            batch=cases[start:start+4];raws,seconds=generate(tokenizer,model,[r['prompt'] for r in batch])
            for case,raw in zip(batch,raws):append(target/(name+'_orders.jsonl'),{'id':case['id'],'raw':raw,'seconds_batch':seconds})
    after=digests(model,optimizer);assert before==after
    (target/'EVAL_IDENTITY.json').write_text(json.dumps({'before':before,'after':after,'errors':0},indent=2))


def main():
    plan=read(R/'PLAN.json');folder=R/'repair';assert not folder.exists()
    assert all(sha(R/n)==h for n,h in {**plan['source_hashes'],**plan['data_hashes']}.items())
    assert all(sha(ROOT/n)==h for n,h in plan['dependencies'].items());verify()
    folder.mkdir();(folder/'PLAN.json').write_text(json.dumps(plan,indent=2));shutil.copy2(R/'dev_worlds.json',folder/'dev_worlds.json')
    data=buckets();correction=rows(R/'correction.jsonl');worlds=read(R/'dev_worlds.json');summaries=[]
    for arm in plan['arms']:
        target=folder/arm;target.mkdir();tokenizer,model=load('start',{'checkpoints':{'start':plan['checkpoint']}});pools=palette(tokenizer)
        if arm in plan['schedules']:
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=plan['lr'],weight_decay=0.)
            initial=digests(model,optimizer);(target/'INITIAL_STATE.json').write_text(json.dumps(initial,indent=2));tokens=0
            for index,(kind,number) in enumerate(plan['schedules'][arm]):
                model.train()
                def timeout():
                    (target/'TIMEOUT.json').write_text(json.dumps({'step':index+1,'limit':12}));os._exit(124)
                timer=threading.Timer(12,timeout);timer.daemon=True;timer.start();start=time.perf_counter()
                if kind=='correct':
                    batch=correction[number*4:number*4+4]
                    loss,parts,count=correction_loss(tokenizer,model,batch);extra={'ids':[r['id'] for r in batch]}
                else:
                    batch=plan['batches'][number]
                    loss,parts,count,predicted,supplied=loss_for_views(tokenizer,model,data[tuple(batch['key'])][batch['index']],
                        pools['train'],batch['views'],number,0.,teacher=True)
                    parts={k:float(v.detach()) for k,v in parts.items()};extra={'actor_input_estimates':supplied,'predicted_estimates':predicted}
                optimizer.zero_grad();loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
                optimizer.step();timer.cancel();tokens+=count
                opt_steps={int(s['step']) for s in optimizer.state.values() if 'step' in s};assert opt_steps=={index+1}
                append(target/'training.jsonl',{'step':index+1,'kind':kind,'batch_index':number,'seconds':time.perf_counter()-start,
                    'input_tokens':count,'gradient_norm':norm,'optimizer_step':index+1,'loss':float(loss.detach()),'parts':parts,**extra})
                if (index+1)%20==0:print(json.dumps({'arm':arm,'step':index+1}),flush=True)
                if index+1==plan['milestones'].get(arm):
                    mid=target/'midpoint';mid.mkdir();save(model,mid,index+1,tokens,plan['checkpoint']['seal']['adapter_hashes'])
                    diagnostic(tokenizer,model,mid,optimizer);print(json.dumps({'arm':arm,'milestone':index+1,'diagnostic_done':True}),flush=True)
            save(model,target,107,tokens,plan['checkpoint']['seal']['adapter_hashes']);diagnostic(tokenizer,model,target,optimizer);del optimizer
        else:
            source=ROOT/plan['checkpoint']['path'];shutil.copytree(source/'adapter',target/'adapter');shutil.copy2(source/'CHECKPOINT_SEALED.json',target/'CHECKPOINT_SEALED.json')
            diagnostic(tokenizer,model,target)
        model.eval()
        for condition,ws in worlds.items():
            result=evaluate(tokenizer,model,ws,pools['fresh'],True);path=target/(condition+'.jsonl');path.write_text(''.join(json.dumps(r)+'\n' for r in result))
            summary={'arm':arm,'condition':condition,'episodes':len(result),**{k:sum(r[k] for r in result) for k in ['safe_success','strict_success','invalid','invalid_queries']},
                'valid_ci':sum(r['ci'] and not r['invalid'] for r in result),'raw_sha256':sha(path)}
            summaries.append(summary);print(json.dumps(summary),flush=True)
        del model;gc.collect()
    (folder/'RESULTS.json').write_text(json.dumps({'summaries':summaries,'whole_goal_complete':False},indent=2))
    (R/'SEALED.json').write_text(json.dumps({'complete':True,'files':{p.relative_to(R).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()},'whole_goal_complete':False},indent=2))


if __name__=='__main__':main()
