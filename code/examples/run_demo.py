"""Reproducible interface-only demo. No LLM calls, no private source exports."""
import json
from pathlib import Path
from kci import (ToyCIEnv,ToyCandidateGenerator,ReferenceSearch,SearchBudget,
                 ContinuationEvaluator,DeterministicPolicy,DownstreamPolicyRepair,
                 certify_preference,export_verified_preferences,summarize_rollouts)
from kci.contracts import Trace,TraceStep,RefStatus,Verdict


def bad_current_policy(obs):
    env=ToyCIEnv()
    if obs.actor=='A':
        if obs.awaiting_notification:return env.act('A','NOTIFY',recipient='Alice',answer='yes')
        return env.act('A','SEND','R',('identity',))
    if obs.actor=='B':
        local={x.id for x in obs.local_facts}
        if 'eligible' in local:return env.act('B','SEND','R',('eligible',))
        if 'rule' in local:return env.act('B','COMPUTE_ELIGIBILITY')
        return env.act('B','SEND','R',('band',))  # deliberately ignores recipient history
    if obs.actor=='R':
        if obs.pending_request:return env.act('R','SEND','B',('rule',))
        return env.act('R','FINALIZE',answer='yes')
    raise ValueError(obs.actor)


def rollout_from_start(env,policy):
    steps=[];st=env.initial()
    while not st.done:
        a=policy(env.observation(st,st.actor))
        ns,aud=env.transition(st,a)
        steps.append(TraceStep(st,a,aud,ns))
        if aud.verdict!=Verdict.VALID:break
        st=ns
    return Trace(tuple(steps))


def main():
    env=ToyCIEnv()
    generator=ToyCandidateGenerator()
    search=ReferenceSearch(env,generator,SearchBudget(max_nodes=3000,max_actions=12))
    current=DeterministicPolicy(bad_current_policy,'bad_policy_frozen_demo_v1')
    cce=ContinuationEvaluator(env,search,current)
    s0=env.initial()
    skip=cce.evaluate(s0,env.act('A','SKIP'),actual_rollouts=1)
    send=cce.evaluate(s0,env.act('A','SEND','R',('identity',)),actual_rollouts=1)
    preference=certify_preference(skip,send)
    trace=rollout_from_start(env,bad_current_policy)
    dpr=DownstreamPolicyRepair(cce,generator)
    repair=dpr.repair(trace)
    assert skip.reference.status==RefStatus.EXACT
    assert send.reference.status==RefStatus.EXACT
    assert preference is not None and preference.chosen.kind=='SKIP'
    assert trace.steps[-1].audit.verdict==Verdict.VIOLATION
    assert repair.preference is not None and repair.failed_step==1
    actual_skip=summarize_rollouts(skip.actual_traces,'bad_policy_frozen_demo_v1')
    actual_send=summarize_rollouts(send.actual_traces,'bad_policy_frozen_demo_v1')
    preferences=[preference,repair.preference]
    records=export_verified_preferences(preferences)
    output=Path(__file__).resolve().parents[1]/'data'/'verified_demo_preferences.jsonl'
    output.write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in records),encoding='utf-8')
    print(json.dumps({
        'A_SKIP':{'reference_status':skip.reference.status.value,'safe_success_cost':skip.reference.best_success_cost},
        'A_SEND_IDENTITY':{'reference_status':send.reference.status.value,'safe_success_cost':send.reference.best_success_cost},
        'A_long_term_preference':[preference.chosen.kind,preference.rejected.kind],
        'actual_policy_after_A_SKIP':{'safe_success_rate':actual_skip.verified_safe_success_rate,
                                      'ci_violation_rate':actual_skip.confirmed_ci_violation_rate},
        'actual_policy_after_A_SEND':{'safe_success_rate':actual_send.verified_safe_success_rate,
                                      'ci_violation_rate':actual_send.confirmed_ci_violation_rate},
        'actual_bad_trace':[st.action.actor+':'+st.action.kind for st in trace.steps],
        'failure_audit':trace.steps[-1].audit.inference_evidence,
        'DPR':{'repair_at_step_index':repair.failed_step,'actor':repair.preference.observation.actor,
               'chosen':repair.preference.chosen.kind,'rejected':repair.preference.rejected.kind},
        'verified_dpo_records':len(records),
        'output':str(output),
    },ensure_ascii=False,indent=2))

if __name__=='__main__':main()
