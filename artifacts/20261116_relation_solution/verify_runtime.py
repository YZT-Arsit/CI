"""Verify the reusable runtime against every new 64D confirmatory trajectory."""
from lm_common import *
import time
from kci.relation_solution import LocalRulePolicy
from kci.relation_runtime import choose_action
from experiments.relation_solution_data import environment
from kci.dynamic_tasks import canonical
from dataclasses import asdict

def run():
    folder=R/'representation_repair';ws={w['world_id']:w for w in map(json.loads,(folder/'worlds.jsonl').read_text().splitlines())};episodes=0;errors=[];maximum=0
    for seed in (17,29,41):
        model=LocalRulePolicy();model.load_state_dict(torch.load(R/'structured_models'/f'local_{seed}'/'model.pt',weights_only=True));model.eval()
        for row in map(json.loads,(folder/f'local_{seed}_width64.jsonl').read_text().splitlines()):
            t=time.perf_counter();w=ws[row['world_id']];e=environment(w);s=e.initial()
            for i,trace in enumerate(row['trace']):
                action,scores=choose_action(model,e.observation(s),seed_for(w['base_group_id'],i))
                if action!=trace['selected']:errors.append([seed,w['world_id'],i,'action'])
                s,audit=e.transition(s,action)
                if canonical(audit)!=canonical(trace['audit']):errors.append([seed,w['world_id'],i,'audit'])
            if asdict(s)!=row['final']:errors.append([seed,w['world_id'],'final'])
            episodes+=1;maximum=max(maximum,time.perf_counter()-t)
    result={'episodes':episodes,'errors':errors,'max_episode_seconds':maximum,'fixed_representation':{'width':64,'max_facts':256}}
    (R/'RUNTIME_VERIFIED.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));assert not errors

if __name__=='__main__':run()
