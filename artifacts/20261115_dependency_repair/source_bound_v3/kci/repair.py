"""DPR: repair only decision states backed by an executable verified alternative."""
from __future__ import annotations
from .contracts import (Action, CandidateGenerator, Preference, RepairResult, RefStatus, Trace,
                        Verdict)
from .continuation import ContinuationEvaluator,certify_preference

class DownstreamPolicyRepair:
    def __init__(self,evaluator:ContinuationEvaluator,generator:CandidateGenerator):
        self.cce=evaluator;self.generator=generator

    def repair(self,failed:Trace)->RepairResult:
        if not failed.steps:return RepairResult('unresolved',None,'empty trace')
        steps=failed.steps
        if steps[-1].after.success and failed.is_safe():
            return RepairResult('no_repair',None,'trajectory already successful and CI-compliant')
        first_problem=next((i for i,st in enumerate(steps)
                           if st.audit.verdict==Verdict.VIOLATION),None)
        inspected=steps[:first_problem] if first_problem is not None else steps
        if any(st.audit.verdict!=Verdict.VALID for st in inspected):
            return RepairResult('unresolved',None,'failure prefix contains unresolved or invalid actions')
        if first_problem is None and (not steps[-1].after.done or steps[-1].after.any_violation):
            return RepairResult('unresolved',None,'requires a confirmed violation or verified terminal task failure')
        # Once a disclosure violation occurred, later steps cannot undo it;
        # only inspect lawful prefixes up to the first violation.
        upper=first_problem if first_problem is not None else len(steps)-1
        for k in range(upper,-1,-1):
            st=steps[k]
            if st.before.any_violation:continue
            orig=self.cce.evaluate(st.before,st.action)
            batch=self.generator.generate(self.cce.env.observation(st.before,st.action.actor))
            for alt in batch.actions:
                if alt.key()==st.action.key():continue
                alt_report=self.cce.evaluate(st.before,alt)
                pair=certify_preference(alt_report,orig)
                if pair is not None and pair.chosen.key()==alt.key():
                    return RepairResult('verified_repair',pair,
                       'same-state alternate has certified CI/continuation improvement',k)
        return RepairResult('unresolved',None,'no certified same-state corrective pair')
