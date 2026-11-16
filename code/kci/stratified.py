"""Strict hierarchical CCE for a frozen finite rollout population.

Upper-level certificates inspect all outcomes in the declared finite support.
Lower-level means are estimated from IID uniform support draws. This is a
calibration bridge, NOT a scalable certificate for arbitrary LLM policies.
No empirical zero is promoted into exact equality, and no tolerance changes
the CI > safe success > cost objective.
"""
from dataclasses import dataclass, asdict
from hashlib import sha256
from math import isfinite, log, sqrt
import json
import random
from statistics import mean


@dataclass(frozen=True)
class Trial:
    ci: bool | None
    success: bool | None
    cost: float


@dataclass(frozen=True)
class ComparisonPlan:
    n: int = 1024
    delta: float = .01
    cost_bound: float = 16.
    seed: int = 17

    def __post_init__(self):
        if type(self.n) is not int or self.n < 2: raise ValueError('n >= 2 required')
        if not 0 < self.delta < 1: raise ValueError('delta out of range')
        if not isfinite(self.cost_bound) or self.cost_bound <= 0: raise ValueError('finite cost bound required')


@dataclass(frozen=True, init=False)
class FiniteRolloutPopulation:
    """Uniform frozen support of paired counterfactual rollout outcomes.

    Caller must produce these outcomes from independently audited transitions;
    this class verifies data and statistical contracts, not rollout semantics.
    Binding records the environment/state/actions/reference identity.
    """
    pairs: tuple
    binding: str
    sha256: str
    def __init__(self, pairs, binding):
        object.__setattr__(self,'pairs',tuple(tuple(pair) for pair in pairs))
        if not self.pairs or any(len(p) != 2 for p in self.pairs): raise ValueError('nonempty paired support')
        for pair in self.pairs:
            for value in pair:
                if not isinstance(value, Trial): raise ValueError('Trial required')
                if value.ci is not None and type(value.ci) is not bool: raise ValueError('invalid CI')
                if value.success is not None and type(value.success) is not bool: raise ValueError('invalid success')
                if not isfinite(value.cost) or value.cost < 0: raise ValueError('invalid cost')
        object.__setattr__(self,'binding',json.dumps(binding, sort_keys=True, ensure_ascii=False))
        object.__setattr__(self,'sha256',sha256(json.dumps({'binding':self.binding,
            'pairs':[[asdict(x) for x in pair] for pair in self.pairs]},sort_keys=True).encode()).hexdigest())

    def upper_certificate(self):
        known = all(t.ci is not None and t.success is not None for p in self.pairs for t in p)
        safe = lambda t: t.ci and t.success
        # Sum equality is exact for this uniform population; no float tolerances.
        ci_equal = known and sum(int(a.ci)-int(b.ci) for a,b in self.pairs) == 0
        success_equal = known and sum(int(safe(a))-int(safe(b)) for a,b in self.pairs) == 0
        all_success = known and all(safe(t) for p in self.pairs for t in p)
        return {'population_sha256':self.sha256,'support_size':len(self.pairs),
                'known':known,'ci_equal':ci_equal,'safe_success_equal':success_equal,
                'all_success':all_success,'scope':'exact_declared_uniform_finite_reference_support'}

    def compare(self, plan: ComparisonPlan, *, use_upper_certificate=False):
        if any(t.cost > plan.cost_bound for p in self.pairs for t in p):
            raise ValueError('cost exceeds predeclared bound')
        rng=random.Random(plan.seed)
        indices=[rng.randrange(len(self.pairs)) for _ in range(plan.n)]
        sampled=[self.pairs[i] for i in indices]
        cert=self.upper_certificate() if use_upper_certificate else None
        report={'status':'UNKNOWN','chosen':None,'level':None,'plan':asdict(plan),
                'population_sha256':self.sha256,'binding':json.loads(self.binding),
                'upper_certificate':cert,'sample_indices':indices,'intervals':{},
                'scope':'fixed_reference_not_optimality','reason':'unresolved priority'}
        if any(t.ci is None or t.success is None for p in self.pairs for t in p):
            report['reason']='unresolved outcome in declared support';return report
        # Three simultaneous intervals: each paired normalized difference in [-1,1].
        radius=sqrt(2*log(6/plan.delta)/plan.n)
        safe=lambda t: bool(t.ci and t.success)
        ci=mean(int(a.ci)-int(b.ci) for a,b in sampled)
        success=mean(int(safe(a))-int(safe(b)) for a,b in sampled)
        cost=mean((b.cost-a.cost)/plan.cost_bound for a,b in sampled)
        for key,value in [('ci',ci),('safe_success',success),('cost_normalized',cost)]:
            report['intervals'][key]=[max(-1.,value-radius),min(1.,value+radius)]
        level=None;advantage=ci
        if abs(ci)>radius and not (cert and cert['ci_equal']):level='ci'
        elif cert and cert['ci_equal']:
            advantage=success
            if abs(success)>radius and not cert['safe_success_equal']:level='safe_success'
            # Conditional-cost inference only when every supported rollout succeeds.
            # Mixed successful/failed branches remain UNKNOWN at the cost tier.
            elif cert['safe_success_equal'] and cert['all_success']:
                advantage=cost
                if abs(cost)>radius:level='completion_cost'
        if level:
            side=0 if advantage>0 else 1
            if not any(safe(p[side]) for p in sampled):
                report['reason']='no verified safe-success witness';return report
            report.update(status='STATISTICAL_PREFERENCE',chosen=side,level=level,
                          reason='simultaneous bounded interval plus required exact upper equivalence')
        return report
