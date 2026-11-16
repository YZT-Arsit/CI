"""Actual transition cross-calibration of a frozen observation-only reference."""
from dataclasses import asdict
from hashlib import sha256
import argparse,json,random
from pathlib import Path
from kci.dynamic_tasks import DynamicTask,canonical
from kci.dynamic_reference import DynamicReference,rollout
from kci.stratified import Trial,ComparisonPlan,FiniteRolloutPopulation
from kci.stream_cce import compare_rollouts


def run(output,source,kind,related,allow,horizon,mode,n,certificate):
    env=DynamicTask(kind,source,related=related,allow=allow,horizon=horizon)
    start=env.decision_start();actions=env.options(start);policy=DynamicReference(mode)
    binding={'environment':env.binding,'state':asdict(start),'actions':actions,
             'reference':{'mode':mode,'version':policy.version},
             'code_sha256':{name:sha256((Path(__file__).resolve().parents[1]/'kci'/name).read_bytes()).hexdigest()
                            for name in ('dynamic_tasks.py','dynamic_reference.py','stream_cce.py')}}
    seeds={}
    for seed in range(100):seeds.setdefault(random.Random(seed).randrange(2),seed)
    cache={};traces={}
    def sample(seed):
        latent=random.Random(seed).randrange(2) if mode=='mixed' else 0
        if latent not in cache:
            trials=[];proof=[]
            for action in actions:
                state,audit=env.transition(start,action);assert audit['verdict']=='VALID'
                final,trace=rollout(env,state,policy,seed)
                known=all(x['audit']['verdict']!='UNKNOWN' for x in trace)
                trials.append(Trial(not final.violation if known else None,final.success if known else None,final.turn))
                proof.append({'first_action':action,'suffix':trace,'final':asdict(final)})
            cache[latent]=tuple(trials);traces[latent]=proof
        return cache[latent]
    support=[sample(seeds[i]) for i in (0,1)] if mode=='mixed' else [sample(0)]
    population=FiniteRolloutPopulation(support,binding)
    # 16 scenarios * 3 reference modes * 2 sample sizes * 2 proof conditions =192.
    # All tests together receive total confidence error budget .01.
    plan=ComparisonPlan(n=n,delta=.01/192,cost_bound=16,seed=17)
    report=compare_rollouts(sample,plan,binding,upper_support=population if certificate else None)
    exact=[env.exact(env.transition(start,a)[0]) for a in actions]
    safe=lambda t:int(t.ci and t.success)
    means=[(sum(int(pair[i].ci) for pair in support)/len(support),
            sum(safe(pair[i]) for pair in support)/len(support),
            -sum(pair[i].cost for pair in support)/len(support)) for i in (0,1)]
    # Cost tier meaningful here only if all supported outcomes are safe successes.
    if not all(safe(t) for p in support for t in p):means=[v[:2] for v in means]
    target=0 if means[0]>means[1] else 1 if means[1]>means[0] else None
    report.update(kind=kind,related=related,allow=allow,horizon=horizon,reference_mode=mode,
                  optimal_safe_continuations=exact,exact_reference_values=means,
                  exact_reference_choice=target,agrees_if_certified=report['chosen'] is None or report['chosen']==target,
                  support_proof={'scope':'one fixed latent coin for entire rollout; deterministic observation policy otherwise',
                                 'population_sha256':population.sha256,'representative_traces':traces},
                  cache_scope='fixed source/state/actions/reference fingerprint; repeated deterministic latent outcomes only',
                  assumptions=['correct canonical audits','PRNG seed draws treated as independent',
                               'frozen reference has exactly the declared finite latent support'])
    Path(output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','level','chosen','exact_reference_choice','agrees_if_certified')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('source');p.add_argument('kind')
    p.add_argument('--related',type=int,required=True);p.add_argument('--allow',type=int,required=True)
    p.add_argument('--horizon',type=int,required=True);p.add_argument('--mode',required=True)
    p.add_argument('--n',type=int,required=True);p.add_argument('--certificate',type=int,required=True)
    a=p.parse_args();run(a.output,a.source,a.kind,bool(a.related),bool(a.allow),a.horizon,a.mode,a.n,bool(a.certificate))
