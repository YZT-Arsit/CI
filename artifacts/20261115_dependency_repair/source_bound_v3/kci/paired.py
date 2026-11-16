"""Conservative empirical comparator for NON-exhaustive reference policies.

This estimates outcomes under a frozen, permission-constrained policy, not the
global optimum Q_safe*. A hard preference requires a confidence-supported CI
advantage AND a safe-success witness for the chosen branch. Finite samples do
not prove equality of the higher-priority CI/task outcomes, so task and cost
differences remain diagnostics. Exact cost preferences use ReferenceSearch.
Independence across seeds and correct outcome auditing remain assumptions.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import sqrt,log
from statistics import mean
from .contracts import Action, Preference, State, Trace, TraceStep, Verdict
from .continuation import ContinuationEvaluator,DeterministicPolicy,SeededPolicy
from .rewards import outcome_from_trace

@dataclass(frozen=True)
class PairedSampleReport:
    status: str
    n: int
    delta: float
    success_diff_mean: float|None
    success_diff_radius: float|None
    conditional_cost_diff_mean: float|None
    conditional_cost_diff_radius: float|None
    reference_policy_version: str
    preference: Preference|None
    reason: str
    ci_diff_mean: float|None = None
    ci_diff_radius: float|None = None
    ci_known_pairs: int = 0
    success_known_pairs: int = 0
    cost_pairs: int = 0
    sample_evidence: tuple[dict,...] = ()

class PairedReferenceComparator:
    def __init__(self,env,policy:DeterministicPolicy|SeededPolicy,n:int=64,delta:float=0.05,seed_base:int=71):
        if n<2:raise ValueError('n must be >=2')
        if not 0<delta<1:raise ValueError('delta out of range')
        self.env=env;self.policy=policy;self.n=n;self.delta=delta;self.seed_base=seed_base

    def _full_trace(self, state:State,action:Action,seed:int)->Trace:
        next_state,audit=self.env.transition(state,action)
        st=TraceStep(state,action,audit,next_state)
        if audit.verdict!=Verdict.VALID:return Trace((st,))
        runner=ContinuationEvaluator(self.env,reference=None,actual_policy=self.policy)
        suffix=runner.rollout_actual(next_state,seed=seed) if not next_state.done else Trace(())
        return Trace((st,)+suffix.steps)

    def compare(self,state:State,a:Action,b:Action)->PairedSampleReport:
        if a.key()==b.key() or a.actor!=b.actor or a.actor!=state.actor:
            raise ValueError('same state/actor, distinct actions required')
        sa,aa=self.env.transition(state,a);sb,ab=self.env.transition(state,b)
        if aa.verdict!=Verdict.VALID or ab.verdict!=Verdict.VALID:
            return PairedSampleReport('ineligible',0,self.delta,None,None,None,None,
                             self.policy.version,None,'current actions must both be CI-valid')
        ci_diffs=[];ds=[];dc=[];samples=[];witnesses={}
        for j in range(self.n):
            seed=self.seed_base+104729*j   # same external randomness across a and b
            ta=self._full_trace(state,a,seed);tb=self._full_trace(state,b,seed)
            oa=outcome_from_trace(ta);ob=outcome_from_trace(tb)
            sa=oa.safe_success;sb=ob.safe_success
            samples.append({'seed':seed,'a_ci':oa.ci_compliant,'b_ci':ob.ci_compliant,
                            'a_task':oa.task_success,'b_task':ob.task_success,
                            'a_safe_success':sa,'b_safe_success':sb,
                            'a_cost':oa.execution_cost,'b_cost':ob.execution_cost})
            if oa.ci_compliant is not None and ob.ci_compliant is not None:
                ci_diffs.append(int(oa.ci_compliant)-int(ob.ci_compliant))
            if sa is not None and sb is not None:ds.append(int(sa)-int(sb))
            if sa is True:witnesses.setdefault('a',(seed,ta))
            if sb is True:witnesses.setdefault('b',(seed,tb))
            if sa is True and sb is True:
                dc.append(ob.execution_cost-oa.execution_cost)
        # Two simultaneous Hoeffding intervals for paired CI and safe success.
        # Each difference is in [-1,1]. Do not estimate from a selected known-only
        # subset when other outcomes are unresolved.
        rad=sqrt(2*log(4/self.delta)/self.n)
        ci_mean=mean(ci_diffs) if len(ci_diffs)==self.n else None
        d=mean(ds) if len(ds)==self.n else None
        diff_cost=mean(dc) if dc else None
        # Costs have arbitrary environment units; max_turns is not a cost bound.
        # This conditional sample mean has no asserted concentration interval.
        chosen=rejected=None;side=None
        reason='no proven upper-level equivalence; success and cost differences are diagnostics only'
        if ci_mean is None:
            reason='some full-trajectory CI outcomes are unresolved'
        elif ci_mean>rad:chosen,rejected,side=a,b,'a'
        elif ci_mean < -rad:chosen,rejected,side=b,a,'b'
        if chosen is not None and side not in witnesses:
            chosen=rejected=None
            reason='CI advantage lacks a verified safe-success witness; refusal is not certified'
        preference=None
        if chosen is not None:
            reason='CI-compliance probability advantage with a verified safe-success witness'
            obs=self.env.observation(state,state.actor)
            witness_seed,witness=witnesses[side]
            preference=Preference(obs,chosen,rejected,'future_empirical_reference',
                    ContinuationEvaluator.state_id(state),{
                       'certified':True,
                       'evidence_type':'paired_ci_hoeffding_advantage_with_safe_success_witness',
                       'epistemic_status':'statistical_estimate_NOT_optimality_proof',
                       'reference_policy_version':self.policy.version,
                       'environment_version':self.env.version,
                       'norm_version':state.norm_version,'oracle_version':state.oracle_version,
                       'horizon':state.max_turns,'samples':self.n,'delta':self.delta,
                       'sample_seeds':[s['seed'] for s in samples],
                       'seed_step_stride':10007,'seed_pair_stride':104729,
                       'ci_diff_mean':ci_mean,'ci_diff_radius':rad,
                       'success_diff_mean':d,'success_diff_radius':rad if d is not None else None,
                       'cost_diff_mean':diff_cost,'cost_diff_radius':None,
                       'cost_units':'environment_execution_cost','cost_pairs':len(dc),
                       'witness_seed':witness_seed,
                       'witness_actions':[st.action.as_json() for st in witness.steps],
                       'sample_outcomes':samples,
                       'comparison_scope':'fixed_reference_policy_and_declared_horizon',
                       'assumptions':['independent_seed_draws','correct_transition_and_outcome_audits'],
                       'evidence_note':reason,
                    })
        status='estimated_preference' if preference else ('unknown' if ci_mean is None else 'inconclusive')
        return PairedSampleReport(status,self.n,self.delta,d,rad if d is not None else None,
                      diff_cost,None,self.policy.version,preference,reason,
                      ci_mean,rad if ci_mean is not None else None,len(ci_diffs),len(ds),len(dc),tuple(samples))
