"""Auditable reward evidence; no arbitrary per-history reward constants."""
from __future__ import annotations
from dataclasses import dataclass
from statistics import mean
from .contracts import Trace,Verdict

@dataclass(frozen=True)
class Outcome:
    ci_compliant: bool|None
    task_success: bool|None
    execution_cost: float
    reason: str
    @property
    def safe_success(self)->bool|None:
        # A known violation already disproves safe success, even if rollout stops
        # before task completion. Preserve the unresolved task label separately.
        if self.ci_compliant is False or self.task_success is False:return False
        if self.ci_compliant is None or self.task_success is None:return None
        return True
    def lex_key(self):
        # Only compare values when both CI and task success are verified.
        if self.ci_compliant is None or self.task_success is None:
            raise ValueError('Unknown labels cannot produce a strict reward ranking')
        return (int(self.ci_compliant),int(self.task_success),
                -self.execution_cost if self.ci_compliant and self.task_success else 0)

@dataclass(frozen=True)
class RolloutSummary:
    n: int
    confirmed_ci_violation_rate: float
    verified_safe_success_rate: float
    mean_cost_on_safe_success: float|None
    unknown_rate: float
    version: str


def outcome_from_trace(trace:Trace)->Outcome:
    if not trace.steps and trace.start_state is None:return Outcome(None,None,0.0,'empty rollout')
    flags=[st.audit.verdict for st in trace.steps]
    state=trace.final
    violation=state.any_violation or Verdict.VIOLATION in flags or any(
        st.before.any_violation or st.after.any_violation for st in trace.steps)
    if Verdict.UNKNOWN in flags or Verdict.INVALID_ACTION in flags:
        return Outcome(False if violation else None,None,state.cost,'unresolved action or semantic verifier')
    if not state.done and not violation:
        return Outcome(None,None,state.cost,'partial trace without terminal outcome')
    return Outcome(not violation,bool(state.success) if state.done else None,state.cost,
                   'verifiable simulator state')


def summarize_rollouts(traces:list[Trace]|tuple[Trace,...],policy_version:str)->RolloutSummary:
    outcomes=[outcome_from_trace(t) for t in traces]
    n=len(outcomes)
    if n==0:return RolloutSummary(0,0.0,0.0,None,0.0,policy_version)
    costs=[o.execution_cost for o in outcomes if o.ci_compliant is True and o.task_success is True]
    return RolloutSummary(n,
        sum(o.ci_compliant is False for o in outcomes)/n,
        len(costs)/n,
        mean(costs) if costs else None,
        sum(o.ci_compliant is None or o.task_success is None for o in outcomes)/n,
        policy_version)
