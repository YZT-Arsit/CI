"""Post-result oracle-prefix diagnostics only; no training or new primary gate."""
import json,time,copy,threading,os
from pathlib import Path
import evaluate_fast_v5 as ev
from experiments.receipt_dataset import make_world,SOURCE
R=Path(__file__).resolve().parent;torch=ev.torch;np=ev.np

def choose(model,kind,env,s):
    obs=env.observation(s);actions=list(env.actions(s))
    with torch.no_grad():
        if kind=='graph':
            enc=[ev.graph_encode(obs,a) for a in actions]
            scores=model(torch.tensor(np.array([x for x,a in enc])),torch.tensor(np.array([a for x,a in enc])))
        else:scores=model(torch.tensor(np.array([ev.token_encode(obs,a) for a in actions])))
    return actions[int(torch.argmax(scores))]

def run():
    assert (R/'RESULTS_OPENED.json').exists();out=[]
    for m in json.loads((R/'models_v5/CHECKPOINTS_SEALED.json').read_text())['models']:
        if m['arm']!='combined':continue
        kind=m['kind'];name=f"{kind}_combined_{m['seed']}";model=ev.GraphPolicy() if kind=='graph' else ev.TokenPolicy()
        model.load_state_dict(torch.load(R/'models_v5'/name/'latest.pt',map_location='cpu',weights_only=False)['model']);model.eval()
        for depth in range(1,7):
            timer=threading.Timer(9.5,lambda:os._exit(124));timer.daemon=True;timer.start();start=time.perf_counter()
            row={'model':name,'depth':depth,'planner_correct':0,'planner_unique':0,'worker_ci':0,'worker_total':0,'worker_safe_success':0,'worker_viable':0}
            for noise in (0,4):
                for mode in ('one','both'):
                    for i in range(2):
                        world=make_world(710000+depth*100+noise*10+i,'meeting',depth=depth,noise=noise,mode=mode)
                        env=ev.build_env(world,SOURCE);s=env.initial();proof=env.preference(s,*env.actions(s))
                        selected=choose(model,kind,env,s)
                        if proof['chosen'] is not None:
                            row['planner_unique']+=1;row['planner_correct']+=selected==env.actions(s)[proof['chosen']]
                        for a in env.actions(s):
                            p,_=env.transition(s,a);p,_=env.transition(p,'DELIVER_INPUTS')
                            viable=env.exact(p)['safe_success'];after,_=env.transition(p,choose(model,kind,env,p))
                            row['worker_total']+=1;row['worker_ci']+=not after.violation
                            row['worker_viable']+=viable;row['worker_safe_success']+=after.success and not after.violation
            timer.cancel();row['seconds']=time.perf_counter()-start;out.append(row)
    (R/'posthoc_depth_diagnostic.json').write_text(json.dumps({'exploratory_after_test_opening':True,'oracle_prefixes_not_unassisted_policy_episodes':True,'further_training':False,'rows':out},indent=2))
    print(json.dumps({'rows':len(out),'done':True}))
if __name__=='__main__':run()
