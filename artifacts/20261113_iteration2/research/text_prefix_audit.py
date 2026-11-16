"""Independently replay one family's frozen-policy prefix stress per short worker."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'code'))
from experiments.multi_dataset import case_env
from experiments.text_policy_probe import action_class,text_features
from kci.contracts import Action,Verdict
from kci.multi_candidates import MultiRuleCandidateGenerator
from multi_env_independent_audit import enumerate_reachable

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--family',required=True);a=p.parse_args();start=time.perf_counter()
    base=ROOT/'artifacts/20261113_iteration2';raw={};starts={};optimal={};gen=MultiRuleCandidateGenerator()
    for path in sorted((base/'dataset_raw_v2').glob(a.family+'_*.json')):
        doc=load(path);raw[doc['case_spec']['case_id']]=(path,doc)
        env=case_env(doc['case_spec'])
        for branch in ('skip','context'):
            action=env.act('A','SKIP') if branch=='skip' else env.act('A','SEND','R',env.context_ids)
            state,check=env.transition(env.initial(),action)
            assert check.verdict==Verdict.VALID and state.actor=='B' and state.turn==1 and state.cost==1
            cost,_,_=enumerate_reachable(env,state)
            starts[doc['case_spec']['case_id'],branch]=(env,state,action)
            optimal[doc['case_spec']['case_id'],branch]=cost
    by_variant=defaultdict(Counter);first_actions=defaultdict(Counter);original_counts=defaultdict(Counter)
    count=0;replayed_actions=0;source_hashes={}
    for path in sorted((base/'text_prefix_stress').glob(a.family+'_*.json')):
        result=load(path);variant=result['variant'];modeldir=base/'text_policy_v2'/path.stem
        original=load(modeldir/'result.json');weights=np.load(modeldir/'weights.npz')
        assert sha(modeldir/'weights.npz')==result['model_weights_sha256']==original['weights_sha256']
        assert result['source_sha256']==sha(ROOT/'code/experiments/text_prefix_probe.py')
        source_hashes[path.name]=sha(path)
        assert {(r['case_id'],r['intervention']) for r in result['rows']}==set(starts)
        assert len(result['rows'])==len(starts)
        for filename,expected in result['input_sha256'].items():assert sha(base/'dataset_raw_v2'/filename)==expected
        for row in result['rows']:
            key=(row['case_id'],row['intervention']);env,initial,first=starts[key];state=deepcopy(initial)
            assert json.loads(row['prefix_action'])==json.loads(first.as_json())
            assert row['optimal_total_cost']==optimal[key]
            assert row['feasible']==(optimal[key] is not None)
            nonvalid=None
            for index,step in enumerate(row['actions']):
                assert not state.done and nonvalid is None
                obs=env.observation(state,state.actor);menu=gen.generate(obs,12).actions
                x=text_features(obs.as_prompt());scores=np.tanh(weights['u']@x)@weights['e'].T
                ids=[action_class(act.__dict__) for act in menu]
                chosen=menu[int(np.argmax(scores[ids]))]
                saved=Action.from_mapping(json.loads(step['action']))
                assert chosen.key()==saved.key()
                state,check=env.transition(state,saved);assert check.verdict.value==step['verdict']
                if check.verdict!=Verdict.VALID:nonvalid=check.verdict
                replayed_actions+=1
            expected_status=('VIOLATION' if nonvalid==Verdict.VIOLATION else 'UNKNOWN') if nonvalid else ('SAFE_SUCCESS' if state.done and state.success and not state.any_violation else 'TASK_FAILURE')
            assert expected_status==row['status'] and state.cost==row['final_cost']
            assert nonvalid is not None or state.done
            scope='feasible' if row['feasible'] else 'infeasible'
            by_variant[variant]['episodes']+=1;by_variant[variant][scope]+=1
            by_variant[variant][row['status']]+=1;by_variant[variant][scope+'_'+row['status']]+=1
            by_variant[variant][row['intervention']+'_'+row['status']]+=1
            count+=1
        assert dict(Counter(r['status'] for r in result['rows']))==result['counts']
        for episode in original['episodes']:
            original_counts[variant][episode['status']]+=1
            first_actions[variant][json.loads(episode['actions'][0]['action'])['kind']]+=1
    report={'family':a.family,'source_cases':len(raw),'prefix_states_independently_enumerated':len(starts),
      'prefixes_all_valid_at_B_turn1_cost1':True,'replayed_episodes':count,'replayed_policy_actions':replayed_actions,
      'by_variant':{k:dict(v) for k,v in by_variant.items()},'original_initial_rollout_counts':{k:dict(v) for k,v in original_counts.items()},
      'original_initial_A_action_counts':{k:dict(v) for k,v in first_actions.items()},'source_results_sha256':source_hashes,
      'scope':'exploratory intervention in existing heldout synthetic cases; no new independent test or training effect estimate',
      'audit_sha256':sha(Path(__file__)),'seconds':time.perf_counter()-start}
    out=base/'research/text_prefix_audit';out.mkdir(exist_ok=True)
    (out/(a.family+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='source_results_sha256'}))

if __name__=='__main__':main()
