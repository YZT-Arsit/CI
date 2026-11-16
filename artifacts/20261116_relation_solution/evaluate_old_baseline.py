"""Unchanged old V5 checkpoints on the new grouped worlds; no retraining."""
from lm_common import *
import numpy as np,hashlib
from dataclasses import asdict
from kci.receipt_policy import GraphPolicy,graph_encode
from experiments.relation_solution_data import environment
from kci.dynamic_tasks import canonical
from train_structured import short

def execute(model,w):
    env=environment(w);s=env.initial();trace=[]
    while not s.done:
        obs=env.observation(s);actions=list(obs['actions']);random.Random(seed_for(w['base_group_id'],len(trace))).shuffle(actions)
        enc=[graph_encode(obs,a) for a in actions]
        with torch.no_grad():scores=model(*[torch.tensor(np.array([x[i] for x in enc])) for i in range(2)])
        a=actions[int(scores.argmax())];s,audit=env.transition(s,a);trace.append({'role':obs['role'],'actions':actions,'selected':a,'scores':scores.tolist(),'audit':audit})
    return {'world_id':w['world_id'],'ci':not s.violation,'safe_success':s.success and not s.violation,'refused':trace[-1]['selected']=='WITHHOLD','trace':trace,'final':asdict(s),'policy_interventions':0}

def run():
    old=ROOT/'artifacts/20261115_semantic_repair/models_v5';seal=json.loads((old/'CHECKPOINTS_SEALED.json').read_text());out=R/'old_baseline_evaluation';out.mkdir(exist_ok=True);assert not (out/'plan.json').exists()
    splits=['test','test_swap','depth4','depth4_swap','depth6','depth6_swap','depth10','depth10_swap','noise','optional','authorized','draft','both','broken','reverse','branch','cycle','external_complete','external_complete_swap']
    (out/'plan.json').write_text(json.dumps({'models':'old graph_combined 17/29/41 unchanged','conditions':splits,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2));summaries=[]
    for seed in (17,29,41):
        path=old/f'graph_combined_{seed}'/'latest.pt';entry=next(r for r in seal['models'] if r['kind']=='graph' and r['arm']=='combined' and r['seed']==seed);assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['checkpoint_sha256']
        model=GraphPolicy();model.load_state_dict(torch.load(path,weights_only=False)['model']);model.eval();folder=out/f'old_{seed}';folder.mkdir()
        for split in splits:
            ws=[json.loads(l) for p in sorted((R/'dataset'/split).glob('worlds_*.jsonl')) for l in p.read_text().splitlines()];ledger=[];rows=[short(lambda:execute(model,w),ledger,'episode') for w in ws]
            (folder/(split+'.jsonl')).write_text(''.join(canonical(r)+'\n' for r in rows),encoding='utf-8')
            summary={'model':folder.name,'condition':split,'episodes':len(rows),**{k:sum(r[k] for r in rows) for k in ('ci','safe_success','refused')}};summaries.append(summary);print(json.dumps(summary),flush=True)
    (out/'EVALUATION_SEALED.json').write_text(json.dumps({'summaries':summaries},indent=2))

if __name__=='__main__':run()
