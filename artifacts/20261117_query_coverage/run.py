"""Matched depth supervision with unchanged losses and fully autonomous evaluation."""
import gc
import json
import os
import shutil
import sys
import threading
import time
from probe_data import R,ROOT,OLD,read,rows,sha,verify_prior
sys.path[:0]=[str(R.parent/'20261117_frozen_crossover'),str(R.parent/'20261117_teacher_binding'),str(OLD)]
from common import verify,load,torch,palette,generate,evaluate
from binding_loss import loss_for_views
from correction_loss import correction_loss


def append(path,row):
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')


def digest(model):
    import hashlib
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad:h.update(name.encode());h.update(p.detach().numpy().tobytes())
    return h.hexdigest()


def main():
    plan=read(R/'PLAN.json');folder=R/'repair';assert not folder.exists();verify_prior();verify()
    assert all(sha(R/n)==h for n,h in {**plan['source_hashes'],**plan['data_hashes']}.items())
    assert all(sha(ROOT/n)==h for n,h in plan['dependencies'].items())
    assert sha(R/'PROBE_SEALED.json')==plan['probe_seal_sha256']
    folder.mkdir();shutil.copy2(R/'PLAN.json',folder/'PLAN.json');shutil.copy2(R/'dev_worlds.json',folder/'dev_worlds.json')
    batches=rows(R/'train_batches.jsonl');correction=rows(OLD/'correction.jsonl');worlds=read(R/'dev_worlds.json');summaries=[]
    for arm in plan['arms']:
        cp=plan['earlier_checkpoint'] if arm=='earlier' else plan['checkpoint']
        target=folder/arm;target.mkdir();tokenizer,model=load('start',{'checkpoints':{'start':cp}});pools=palette(tokenizer)
        (target/'INITIAL_STATE.json').write_text(json.dumps({'trainable_sha256':digest(model),'initial_checkpoint':cp['path']}))
        if arm in ['one_hop','mixed_depth']:
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=plan['lr'],weight_decay=0.);tokens=0
            for step,(kind,n) in enumerate(plan['schedules']):
                model.train()
                def timeout():
                    (target/'TIMEOUT.json').write_text(json.dumps({'step':step+1,'seconds':15}));os._exit(124)
                timer=threading.Timer(15,timeout);timer.daemon=True;timer.start();start=time.perf_counter()
                if kind=='joint':
                    b=batches[n];depth=1 if arm=='one_hop' else b['mixed_depth'];pair=b['pairs'][str(depth)]
                    loss,parts,count,predicted,supplied=loss_for_views(tokenizer,model,pair,pools['train'],b['views'],n,0.,teacher=True)
                    parts={k:float(v.detach()) for k,v in parts.items()};extra={'depth':depth,'key':b['key'],'actor_input_estimates':supplied,'predicted_estimates':predicted}
                else:
                    batch=correction[n*4:n*4+4];loss,parts,count=correction_loss(tokenizer,model,batch);extra={'ids':[r['id'] for r in batch]}
                opt.zero_grad();loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step();timer.cancel()
                assert {int(s['step']) for s in opt.state.values() if 'step' in s}=={step+1};tokens+=count
                append(target/'training.jsonl',{'step':step+1,'kind':kind,'batch_index':n,'loss':float(loss.detach()),'parts':parts,'gradient_norm':norm,
                    'seconds':time.perf_counter()-start,'input_tokens':count,'optimizer_step':step+1,**extra})
                if (step+1)%20==0:print(json.dumps({'arm':arm,'step':step+1,'seconds':time.perf_counter()-start}),flush=True)
            model.save_pretrained(target/'adapter')
            (target/'CHECKPOINT_SEALED.json').write_text(json.dumps({'adapter_hashes':{p.name:sha(p) for p in (target/'adapter').iterdir() if p.is_file()},
                'steps':len(plan['schedules']),'input_tokens':tokens,'initial_adapter_hashes':cp['seal']['adapter_hashes']},indent=2));del opt
        else:
            shutil.copytree(ROOT/cp['path']/'adapter',target/'adapter');shutil.copy2(ROOT/cp['path']/'CHECKPOINT_SEALED.json',target/'CHECKPOINT_SEALED.json')
        before=digest(model);model.eval()
        for name in ['query','fresh']:
            cases=rows(R/(name+'_observations.jsonl'))
            for start in range(0,len(cases),4):
                batch=cases[start:start+4];raws,seconds=generate(tokenizer,model,[c['prompt'] for c in batch])
                for c,raw in zip(batch,raws):append(target/(name+'_raw.jsonl'),{'id':c['id'],'raw':raw,'seconds_batch':seconds})
        for condition,ws in worlds.items():
            result=evaluate(tokenizer,model,ws,pools['fresh'],True);path=target/(condition+'.jsonl');path.write_text(''.join(json.dumps(r)+'\n' for r in result))
            summary={'arm':arm,'condition':condition,'episodes':len(result),**{k:sum(r[k] for r in result) for k in ['safe_success','strict_success','invalid','invalid_queries']},
                'valid_ci':sum(r['ci'] and not r['invalid'] for r in result),'raw_sha256':sha(path)}
            summaries.append(summary);print(json.dumps(summary),flush=True)
        after=digest(model);assert before==after
        (target/'EVAL_IDENTITY.json').write_text(json.dumps({'before':before,'after':after,'errors':0}))
        del model;gc.collect()
    (folder/'RESULTS.json').write_text(json.dumps({'summaries':summaries,'whole_goal_complete':False},indent=2))
    (R/'SEALED.json').write_text(json.dumps({'complete':True,'files':{p.relative_to(R).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()},'whole_goal_complete':False},indent=2))


if __name__=='__main__':main()
