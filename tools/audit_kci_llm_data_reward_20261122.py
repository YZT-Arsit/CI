"""2026-11-22 source-backed LLM readiness: data composition and reward consistency.

Read-only against frozen KCI native environment and saved CPU trained models.
Explicitly audit whether scalar terminal reward preferences coincide with the
research's intended CI->Task->Cost lexicographic objectives; no model training,
hidden label mutation or new data generation.
"""
from __future__ import annotations
from pathlib import Path
from copy import deepcopy
from collections import Counter,defaultdict
from dataclasses import asdict
import json,sys,math,statistics
import torch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
import compare_kci_chain_methods_cpu_action_interaction as T
from chain_shared_actor_cpu_action_interaction import (
    valid_decision_candidates,baseline_automatic_action,
)
from kci_exact_policy_continuation_cpu import (
    environment_public, FrozenPolicyValue, enumerate_decisions, terminal_vector,
)
from run_kci_role_local_credit_cpu import load_base
from run_kci_downstream_residual_cpu import RepairedDownstream, CalibratedAfterRepair
from kci.diverse_ci_branch_env import digest

torch.set_num_threads(2)
OUT=ROOT/"artifacts"/"20261122_kci_data_reward_alignment_audit_cpu_v1"

def saved_actor(seed,kind):
    base,_=load_base(seed)
    if kind=="base":return base.eval()
    file=ROOT/"artifacts"/"20261121_kci_native_repair_mc_bridge_cpu_v1"/"CHECKPOINTS"/("KCI_DPR_BCCE_ACCE_seed"+str(seed)+".pt")
    state=torch.load(file,weights_only=True,map_location="cpu")
    repaired=RepairedDownstream(base).eval()
    repaired.R.load_state_dict(state["R"])
    if kind=="R_corrected":return repaired.eval()
    repaired.B.load_state_dict(state["B"])
    if kind=="B_corrected":return repaired.eval()
    if kind=="all_roles_trained":
        all_actor=CalibratedAfterRepair(repaired).eval()
        all_actor.A.load_state_dict(state["A"])
        return all_actor.eval()
    raise ValueError(kind)

def all_nonabort_terminals(env):
    todo=[env.initial()]
    seen=set()
    terminals=[]
    while todo:
        st=todo.pop()
        h=digest(asdict(st))
        if h in seen:continue
        seen.add(h)
        if st.native.done:
            reward,success,violation=terminal_vector(env,st)
            terminals.append({"reward":reward,"success":bool(success),
                              "violation":bool(violation),"steps":st.native.steps,
                              "aborted":bool(st.native.aborted)})
            continue
        ob=env.observation(st)
        actions=valid_decision_candidates(ob,env.options(st))
        if len(actions)>=2:
            candidates=[a for a in actions if a not in ("ABORT","ABORT_CHAIN")]
        else:
            choice=baseline_automatic_action(env,st)
            candidates=[choice] if choice else []
        for a in candidates:
            nxt,au=env.transition(st,a)
            assert au["verdict"] in ("VALID","VIOLATION"),(au,a)
            todo.append(nxt)
    return terminals,len(seen)

def classify_pair(qA,qB,cost,safety_eps=1e-3,success_eps=1e-3):
    # Each tuple is (scalar expected terminal reward, P(safe completed), P(CI violation)).
    r0,s0,v0=qA
    r1,s1,v1=qB
    # Sequence is safety first, task second, then conditional cost of successful episodes.
    if abs(v0-v1)>safety_eps:
        lex=0 if v0<v1 else 1
        primary="CI_violation_probability"
    elif abs(s0-s1)>success_eps:
        lex=0 if s0>s1 else 1
        primary="task_success_probability"
    else:
        lex=0 if r0>=r1 else 1
        primary="cost_or_negligible"
    scalar=(0 if r0>r1+1e-7 else 1 if r1>r0+1e-7 else -1)
    # Only count a conflict if Q preference is measurable.
    return {
        "scalar_index":scalar,"priority_index":lex,"priority_reason":primary,
        "conflict":scalar>=0 and scalar!=lex,
        "risk_of_scalar_best":(qA if scalar==0 else qB)[2] if scalar>=0 else None,
        "risk_of_other":(qB if scalar==0 else qA)[2] if scalar>=0 else None,
        "reward_gap":r0-r1,
        "success_gap":s0-s1,
        "violation_gap":v0-v1,
        "matched_reward_diff_component":{
            "difference_in_success_probability":s0-s1,
            "difference_in_violation_probability":-(v0-v1),
            "difference_in_success_step_cost":(r0-r1)-(s0-s1)+(v0-v1),
        }
    }

def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    data_report={}
    terminal_stats=[]
    for split,rows in T.GROUP.items():
        data_report[split]={
            "n_worlds":len(rows),
            "n_unique_task_roots":len(set(w["root"] for w in rows)),
            "graph_families":dict(Counter(w["graph"] for w in rows)),
            "businesses":dict(Counter(w["business"] for w in rows)),
            "package_swap":dict(Counter(str(w["swapped_package_names"]) for w in rows)),
            "source_mode":dict(Counter(w["source_mode"] for w in rows)),
            "roots":dict(Counter(w["root"] for w in rows)),
        }
        for w in rows:
            e=environment_public(w)
            terminals,states=all_nonabort_terminals(e)
            st={"split":split,"world":w["world"],"graph":w["graph"],
                "business":w["business"],"step_cost":e.spec.cost_per_step,
                "native_budget":e.native.budget,"num_native_states":states,
                "native_terminal_states":len(terminals),
                "successful_terminals":sum(x["success"] for x in terminals),
                "violating_terminals":sum(x["violation"] for x in terminals),
                "safe_incomplete_terminals":sum(not x["success"] and not x["violation"] for x in terminals),
                "successful_return_min":min((x["reward"] for x in terminals if x["success"]),default=None),
                "successful_return_max":max((x["reward"] for x in terminals if x["success"]),default=None),
                "negative_success_count":sum(x["success"] and x["reward"]<0 for x in terminals),
                "negative_success_min_steps":min((x["steps"] for x in terminals if x["success"] and x["reward"]<0),default=None),
               }
            terminal_stats.append(st)
    print("DATA_COMPOSITION",json.dumps(data_report),flush=True)
    print("TERMINAL_WARNING",{
        "native_worlds":len(terminal_stats),
        "negative_success_worlds":sum(x["negative_success_count"]>0 for x in terminal_stats),
        "negative_success_terminal_states":sum(x["negative_success_count"] for x in terminal_stats),
        "min_success_reward":min(x["successful_return_min"] for x in terminal_stats if x["successful_return_min"] is not None),
        "max_native_budget":max(x["native_budget"] for x in terminal_stats),
        "max_native_steps_in_reachable":max(x["num_native_states"] for x in terminal_stats),
    },flush=True)
    all_recs=[]
    for seed in (7,1234,2025):
        for kind in ("base","R_corrected","B_corrected","all_roles_trained"):
            actor=saved_actor(seed,kind)
            for split,rows in T.GROUP.items():
                for w in rows:
                    e=environment_public(w)
                    states,aud=enumerate_decisions(e)
                    calc=FrozenPolicyValue(e,actor)
                    for p in states:
                        if p["phase"] not in ("A_SEND","preflight","release"):continue
                        a,b=p["actions"]
                        verdicts=p["verdicts"]
                        if verdicts[a]!="VALID" or verdicts[b]!="VALID":continue
                        qa=calc.q(p["state"],a)
                        qb=calc.q(p["state"],b)
                        decision=classify_pair(qa,qb,e.spec.cost_per_step)
                        all_recs.append({"seed":seed,"policy":kind,"split":split,
                                         "world":w["world"],"graph":w["graph"],
                                         "business":w["business"],"phase":p["phase"],
                                         "action_a":a,"action_b":b,
                                         "a":qa,"b":qb,
                                         **decision})
            print("AUDITED",seed,kind,flush=True)
    summary={}
    for kind in ("base","R_corrected","B_corrected","all_roles_trained"):
        summary[kind]={}
        for phase in ("A_SEND","preflight","release"):
            group=[x for x in all_recs if x["policy"]==kind and x["phase"]==phase]
            strict=[x for x in group if x["scalar_index"]!=-1]
            conflict=[x for x in strict if x["conflict"]]
            by_stage=Counter(x["split"] for x in group)
            summary[kind][phase]={
                "all_legal_fork_comparisons":len(group),
                "strict_scalar_rewards":len(strict),
                "priority_conflicts_eps_0p001":len(conflict),
                "increased_CI_risk_over_priority_eps_0p001":sum(x["priority_reason"]=="CI_violation_probability" for x in conflict),
                "by_split":dict(by_stage),
                "priority_driver":dict(Counter(x["priority_reason"] for x in group)),
                "conflict_examples":[
                    {k:x[k] for k in ("seed","split","world","phase","action_a","action_b",
                                     "reward_gap","success_gap","violation_gap","priority_reason")}
                    for x in conflict[:6]],
            }
    doc={
        "status":"SOURCE_AUDITED_CURRENT_NATIVE_DATA_AND_CI_REWARD",
        "source":"current V3.1 dynamic A->R->B->C + saved KCI 20261121 models",
        "no_optimiser_or_weights_modified":True,
        "primary_score":"1 - step_cost * steps if safe_success, -1 if CI_violation, else 0",
        "stated_priority":"CI compliance then successful task then cost",
        "independent_root_count":len(set(w["root"] for rows in T.GROUP.values() for w in rows)),
        "data":data_report,"terminal_worlds":terminal_stats,"summary":summary,
        "full_policy_legal_action_comparisons":all_recs,
        "inference": "Pairwise priority ranking is a diagnostic with fixed 1e-3 tolerances, not normative proof of acceptable stochastic risk."
    }
    path=OUT/"AUDIT.json"
    path.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("AUDIT_SUMMARY",json.dumps(summary,ensure_ascii=False),flush=True)
    print("AUDIT_FILE",path,"RECORDS",len(all_recs),flush=True)

if __name__=="__main__":
    main()
