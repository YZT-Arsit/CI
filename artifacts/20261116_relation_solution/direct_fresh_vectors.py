"""Complete the fresh-vector control declared in the direct-learning plan."""
from lm_common import *
from kci.relation_solution import LocalRulePolicy,encode,collate
from experiments.relation_solution_data import environment
from kci.dynamic_tasks import canonical
from dataclasses import asdict
import time,hashlib

def run():
    out=R/'direct_fresh_vectors';out.mkdir(exist_ok=True);assert not (out/'RESULTS.json').exists()
    worlds=[json.loads(l) for l in (R/'direct_learning_evaluation/test_worlds.jsonl').read_text().splitlines()];summaries=[]
    for seed in (17,29,41):
        model=LocalRulePolicy();model.load_state_dict(torch.load(R/'auxiliary_ablation'/f'model_{seed}.pt',weights_only=True));model.eval();rows=[]
        for w in worlds:
            e=environment(w);s=e.initial();trace=[];t=time.perf_counter()
            while not s.done:
                obs=e.observation(s);actions=list(obs['actions']);random.Random(seed_for(w['base_group_id'],len(trace))).shuffle(actions)
                with torch.inference_mode():scores=model.score(model(*collate([encode(obs,a,salt=990,width=64,max_facts=256) for a in actions])))
                action=actions[int(scores.argmax())];s,audit=e.transition(s,action);trace.append({'selected':action,'audit':audit})
            rows.append({'world_id':w['world_id'],'ci':not s.violation,'safe_success':s.success and not s.violation,'trace':trace,'final':asdict(s),'seconds':time.perf_counter()-t})
        (out/f'direct_{seed}.jsonl').write_text(''.join(canonical(r)+'\n' for r in rows));summaries.append({'seed':seed,'episodes':64,'ci':sum(r['ci'] for r in rows),'safe_success':sum(r['safe_success'] for r in rows)})
    (out/'RESULTS.json').write_text(json.dumps({'salt':990,'width':64,'summaries':summaries,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2));print(json.dumps(summaries))

if __name__=='__main__':run()
