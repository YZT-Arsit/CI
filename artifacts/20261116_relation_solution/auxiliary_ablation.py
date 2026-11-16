"""Post-hoc explanatory ablation: learn the local core directly from action labels."""
from lm_common import *
import numpy as np,hashlib,time
from kci.relation_solution import LocalRulePolicy,encode,collate
from experiments.receipt_dataset import make_world
from representation_repair import execute
from train_structured import short

def run():
    out=R/'auxiliary_ablation';out.mkdir(exist_ok=True);assert not (out/'plan.json').exists()
    plan={'seeds':[17,29,41],'updates':2400,'batch':64,'lr':.01,'width':64,'max_facts':256,'primitive_teacher':False,
        'data':'same dense executed train labels','evaluation':'new fixed 50M seed worlds; depths2/10/32, noise0/128,16 worlds per cell',
        'purpose':'post-hoc explanatory ablation, not main-gate selection','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'plan.json').write_text(json.dumps(plan,indent=2));data=[];targets=[];buckets={}
    for row in load_states('train'):
        for c in row['candidates']:
            e=encode(row['observation'],c['action'],width=64,max_facts=40,salt=811);idx=len(data);data.append(e);targets.append([c['immediate_safe'],c['safe_completion']]);buckets.setdefault((row['observation']['role'],*targets[-1]),[]).append(idx)
    labels=torch.tensor(targets,dtype=torch.float32);keys=sorted(buckets);reports=[]
    for seed in plan['seeds']:
        torch.manual_seed(seed);rng=random.Random(seed);model=LocalRulePolicy();opt=torch.optim.Adam(model.parameters(),lr=.01);ledger=[];losses=[]
        for step in range(2400):
            def update():
                ids=[rng.choice(buckets[rng.choice(keys)]) for _ in range(64)];pred=model(*collate([data[i] for i in ids]));loss=torch.nn.functional.binary_cross_entropy_with_logits(pred,labels[ids]);opt.zero_grad();loss.backward();opt.step();return float(loss.detach())
            losses.append(short(update,ledger,'update'))
        model.eval();path=out/f'model_{seed}.pt';torch.save(model.state_dict(),path);summaries=[];rows=[]
        for depth in (2,10,32):
            for noise in (0,128):
                cell=[]
                for j in range(16):
                    w=make_world(50000000+depth*10000+noise*100+j,depth=depth,noise=noise);w['base_group_id']=w['world_id']
                    row=short(lambda:execute(model,w,64),ledger,'episode');row.update(depth=depth,noise=noise);cell.append(row)
                rows+=cell;summaries.append({'depth':depth,'noise':noise,'episodes':16,'safe_success':sum(r['safe_success'] for r in cell),'ci':sum(r['ci'] for r in cell)})
        (out/f'episodes_{seed}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(out/f'training_{seed}.json').write_text(json.dumps({'losses':losses,'timings':ledger},indent=2))
        report={'seed':seed,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'summaries':summaries,'final_loss':losses[-1]};reports.append(report);print(json.dumps(report),flush=True)
    (out/'RESULTS.json').write_text(json.dumps({'models':reports,'scope':'supplemental explanatory ablation'},indent=2))

if __name__=='__main__':run()
