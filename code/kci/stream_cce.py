"""Non-exhaustive, fixed-n rollout comparison under strict CI priority.

Exact upper equivalence is optional and bound to a verified finite reference
support. Without it, lower tiers remain UNKNOWN. This deliberately does not
claim finite samples can prove exact equality for an arbitrary LLM reference.
"""
from math import log, sqrt
from statistics import mean
from dataclasses import asdict
import json
from .stratified import ComparisonPlan, FiniteRolloutPopulation, Trial


def compare_rollouts(sample_pair, plan:ComparisonPlan, binding, *, upper_support=None):
    bound=json.dumps(binding,sort_keys=True,ensure_ascii=False)
    certificate=None
    if upper_support is not None:
        if not isinstance(upper_support,FiniteRolloutPopulation) or upper_support.binding!=bound:
            raise ValueError('upper proof belongs to another state/actions/reference')
        if any(t.cost>plan.cost_bound for p in upper_support.pairs for t in p):raise ValueError('bound violated by proof support')
        certificate=upper_support.upper_certificate()
    evidence=[]
    for index in range(plan.n):
        seed=plan.seed+104729*index
        pair=tuple(sample_pair(seed))
        if len(pair)!=2 or any(not isinstance(t,Trial) for t in pair):raise ValueError('paired audited Trial required')
        for t in pair:
            if t.ci is not None and type(t.ci) is not bool:raise ValueError('CI label invalid')
            if t.success is not None and type(t.success) is not bool:raise ValueError('success label invalid')
            if not 0<=t.cost<=plan.cost_bound:raise ValueError('cost outside predeclared bound')
        if upper_support is not None and pair not in upper_support.pairs:
            raise ValueError('observed outcome outside certified frozen support')
        evidence.append({'seed':seed,'a':asdict(pair[0]),'b':asdict(pair[1])})
    result={'status':'UNKNOWN','level':None,'chosen':None,'plan':asdict(plan),'binding':binding,
            'upper_certificate':certificate,'sample_evidence':evidence,'intervals':{},
            'scope':'frozen_reference_distribution_NOT_optimality','reason':'unproven higher-priority equivalence'}
    if any(v[k] is None for e in evidence for v in (e['a'],e['b']) for k in ('ci','success')):
        result['reason']='unresolved sampled outcome';return result
    if certificate and not certificate['known']:
        result['reason']='unresolved certified support';return result
    safe=lambda t:t['ci'] and t['success']
    radius=sqrt(2*log(6/plan.delta)/plan.n)
    values={'ci':mean(int(e['a']['ci'])-int(e['b']['ci']) for e in evidence),
            'safe_success':mean(int(safe(e['a']))-int(safe(e['b'])) for e in evidence),
            'completion_cost':mean((e['b']['cost']-e['a']['cost'])/plan.cost_bound for e in evidence)}
    result['intervals']={key:[max(-1.,v-radius),min(1.,v+radius)] for key,v in values.items()}
    eligible=[] if certificate and certificate['ci_equal'] else ['ci']
    if certificate and certificate['ci_equal']:
        if not certificate['safe_success_equal']:eligible.append('safe_success')
        if certificate['safe_success_equal'] and certificate['all_success']:eligible.append('completion_cost')
    for level in eligible:
        value=values[level]
        if abs(value)<=radius:continue
        chosen=0 if value>0 else 1;side='a' if chosen==0 else 'b'
        if not any(safe(e[side]) for e in evidence):
            result['reason']='no verified safe-success witness';return result
        result.update(status='STATISTICAL_PREFERENCE',level=level,chosen=chosen,
                      reason='simultaneous interval; exact upper equivalence when needed')
        return result
    return result
