"""Frozen-weight numerical representation repair on new confirmatory worlds."""
from lm_common import *
import hashlib,time,numpy as np
from dataclasses import asdict
from experiments.receipt_dataset import make_world
from experiments.relation_solution_data import environment
from kci.relation_solution import LocalRulePolicy,encode,collate
from kci.dynamic_tasks import canonical
from train_structured import short

def execute(model,w,width):
    envcase=environment(w);s=envcase.initial();trace=[]
    while not s.done:
        obs=envcase.observation(s);actions=list(obs['actions']);random.Random(seed_for(w['base_group_id'],len(trace))).shuffle(actions)
        with torch.inference_mode():logits=model(*collate([encode(obs,a,salt=811,width=width,max_facts=256) for a in actions]));scores=model.score(logits)
        action=actions[int(scores.argmax())];s,audit=envcase.transition(s,action)
        trace.append({'role':obs['role'],'selected':action,'actions':actions,'scores':scores.tolist(),'audit':audit})
    return {'world_id':w['world_id'],'ci':not s.violation,'safe_success':s.success and not s.violation,'refused':trace[-1]['selected']=='WITHHOLD','trace':trace,'final':asdict(s)}

def run():
    out=R/'representation_repair';out.mkdir(exist_ok=True);assert not (out/'plan.json').exists()
    plan={'widths':[16,64],'max_facts':256,'depths':[2,10,32],'noise':[0,32,64,128],'worlds_per_cell':16,'seeds':[17,29,41],
        'new_world_seed_start':40000000,'model_weights':'all existing local checkpoints unchanged, trained width16','hypothesis':'higher random-code dimension concentrates non-identical distances; same learned mean-distance matcher, no equality oracle','selection':'width64 fixed from identified 16D collision issue; no sweep','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'plan.json').write_text(json.dumps(plan,indent=2));summaries=[];worlds=[]
    for depth in plan['depths']:
        for noise in plan['noise']:
            for j in range(16):
                w=make_world(40000000+depth*10000+noise*100+j,depth=depth,noise=noise);w['base_group_id']=w['world_id'];worlds.append(w)
    (out/'worlds.jsonl').write_text(''.join(canonical(w)+'\n' for w in worlds))
    for seed in plan['seeds']:
        path=R/'structured_models'/f'local_{seed}'/'model.pt';before=hashlib.sha256(path.read_bytes()).hexdigest();model=LocalRulePolicy();model.load_state_dict(torch.load(path,weights_only=True));model.eval()
        for width in plan['widths']:
            rows=[]
            for depth in plan['depths']:
                for noise in plan['noise']:
                    ledger=[];cell=[]
                    for j in range(16):
                        w=next(w for w in worlds if w['world_id']==f'meeting_{40000000+depth*10000+noise*100+j}')
                        row=short(lambda:execute(model,w,width),ledger,'episode');row.update(depth=depth,noise=noise);cell.append(row)
                    rows+=cell;summary={'seed':seed,'width':width,'depth':depth,'noise':noise,'episodes':16,'safe_success':sum(r['safe_success'] for r in cell),'ci':sum(r['ci'] for r in cell),'max_seconds':max(r['seconds'] for r in ledger)};summaries.append(summary);print(json.dumps(summary),flush=True)
            (out/f'local_{seed}_width{width}.jsonl').write_text(''.join(canonical(r)+'\n' for r in rows))
        assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    (out/'RESULTS.json').write_text(json.dumps({'summaries':summaries,'weights_unchanged':True,'scope':'structured diagnostic only; no language-model repair claim'},indent=2))

if __name__=='__main__':run()
