"""Executed-reference CCE checks; raw proposal behavior remains separately visible."""
from dataclasses import asdict
from collections import Counter
from pathlib import Path
import argparse,json,random
from experiments.task_evidence_dependency_probe import SOURCE
from kci.dynamic_tasks import DynamicTask
from kci.execution_envelope import VerifiedExecutionEnvelope
from kci.stratified import ComparisonPlan
from kci.stream_cce import compare_rollouts


def run(kind,related,mode,cost_metric,n,output):
    env=DynamicTask(kind,SOURCE,related=related,horizon=12)
    start=env.decision_start()
    identity={'name':'seeded_proposal_control','version':1,'seed_rule':'common random integer; arbitrary malformed/abort/unsafe proposals included'}
    gateway=VerifiedExecutionEnvelope(env,start,env.options(start),identity,mode=mode,cost_metric=cost_metric)
    def propose(obs,seed,step,side):
        # This deliberately includes bad drafts. The upper proof does not rely
        # on this finite choice list or its sampling probabilities.
        choices=list(obs['actions'])+['not a tool action',{'body':'undeclared message'}]
        return random.Random(seed+step*73).choice(choices)
    plan=ComparisonPlan(n=n,delta=.01/16,cost_bound=max(1,gateway.max_cost),seed=20261115)
    result=compare_rollouts(lambda seed:gateway.sample_pair(propose,seed),plan,gateway.binding,execution_envelope=gateway)
    counts=Counter()
    for sample in gateway.samples.values():
        for side in sample['execution']:
            for step in side['trace']:
                counts['drafts']+=1;counts['blocked_drafts']+=step['blocked']
    result.update(kind=kind,related=related,mode=mode,cost_metric=cost_metric,
       execution_counts=dict(counts),execution_samples=gateway.samples,
       proof={'verified_nodes':len(gateway.nodes),'transition_graph_sha256':gateway._graph_hash,
              'unconstrained_LLM_distribution_enumerated':False,
              'scope':'structural guarantee only for executed reference inside this finite tool world'},
       optimal_route_diagnostic=[{'action':a,'steps':env.exact(env.transition(start,a)[0])} for a in env.options(start)])
    Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('kind','related','mode','cost_metric','status','level','chosen','execution_counts')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind');p.add_argument('output')
    p.add_argument('--related',type=int,default=1);p.add_argument('--mode',default='safe_success')
    p.add_argument('--metric',default='steps');p.add_argument('--n',type=int,default=64)
    a=p.parse_args();run(a.kind,bool(a.related),a.mode,a.metric,a.n,a.output)
