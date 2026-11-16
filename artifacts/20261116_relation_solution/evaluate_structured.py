"""Unmasked structured-policy execution on all 64 worlds per stratum."""
from lm_common import *
import time,hashlib,numpy as np
from dataclasses import asdict
from kci.relation_solution import LocalRulePolicy,DenseGraphPolicy,encode,collate,graph_encode
from experiments.relation_solution_data import environment
from kci.dynamic_tasks import canonical
from train_structured import short
CONDITIONS=['test','test_swap','depth4','depth4_swap','depth6','depth6_swap','depth10','depth10_swap','noise','optional','authorized','draft','both','broken','reverse','branch','cycle','external','external_swap','test_fresh_vectors']

def execute(model,arm,world,salt):
    env=environment(world);s=env.initial();trace=[]
    while not s.done:
        obs=env.observation(s);actions=list(obs['actions']);random.Random(seed_for(world['base_group_id'],len(trace))).shuffle(actions)
        with torch.no_grad():
            if arm=='dense_graph':
                enc=[graph_encode(obs,a) for a in actions];logits=model(*[torch.tensor(np.array([x[i] for x in enc])) for i in range(2)])
            else:logits=model(*collate([encode(obs,a,salt=salt) for a in actions]))
            score=torch.sigmoid(logits[:,0]) if arm=='safety_only' else model.score(logits)
        choice=int(torch.argmax(score));action=actions[choice];s,audit=env.transition(s,action)
        if audit['verdict']=='UNKNOWN':raise AssertionError(audit)
        trace.append({'role':obs['role'],'selected':action,'actions':actions,'logits':logits.tolist(),'scores':score.tolist(),'audit':audit})
    return {'world_id':world['world_id'],'ci':not s.violation,'safe_success':s.success and not s.violation,'refused':trace[-1]['selected']=='WITHHOLD','trace':trace,'final':asdict(s),'policy_interventions':0}

def run():
    torch.set_num_threads(2);out=R/'structured_evaluation';out.mkdir(exist_ok=True)
    plan=out/'plan.json'
    if plan.exists():raise FileExistsError('immutable evaluation')
    plan.write_text(json.dumps({'conditions':CONDITIONS,'models':15,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'oracle_masks':False},indent=2))
    seal=R/'structured_models/CHECKPOINTS_SEALED.json'
    while not seal.exists():time.sleep(2)
    summaries=[]
    for report in json.loads(seal.read_text())['models']:
        arm=report['arm'];path=R/'structured_models'/report['model']/'model.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()==report['checkpoint_sha256']
        model=DenseGraphPolicy() if arm=='dense_graph' else LocalRulePolicy(no_binding=arm=='no_binding');model.load_state_dict(torch.load(path,weights_only=True));model.eval()
        folder=out/report['model'];folder.mkdir(exist_ok=True)
        for split in CONDITIONS:
            base='test' if split=='test_fresh_vectors' else split
            ws=[json.loads(l) for p in sorted((R/'dataset'/base).glob('worlds_*.jsonl')) for l in p.read_text().splitlines()];rows=[];ledger=[]
            for w in ws:rows.append(short(lambda:execute(model,arm,w,990 if split=='test_fresh_vectors' else 811),ledger,'episode'))
            (folder/(split+'.jsonl')).write_text(''.join(canonical(r)+'\n' for r in rows),encoding='utf-8')
            summary={'model':report['model'],'condition':split,'episodes':len(rows),**{k:sum(r[k] for r in rows) for k in ('ci','safe_success','refused')},'max_episode_seconds':max([r['seconds'] for r in ledger],default=0)}
            summaries.append(summary);print(json.dumps(summary),flush=True)
    (out/'EVALUATION_SEALED.json').write_text(json.dumps({'summaries':summaries},indent=2))

if __name__=='__main__':run()
