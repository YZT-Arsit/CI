"""Actually execute full native policies on same 105 P0 test contexts.

Unlike the first-action counterfactual study, this script does not
automatically use an oracle continuation after a heuristic's first choice.
Each strong baseline follows a concrete executable rule, and the simulator
determines its final privacy, business success and step cost.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter,defaultdict
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
sys.path.insert(0,str(ROOT/"tools"))
from kci.diverse_ci_branch_env import task_catalog,BASE_FACTS,FINAL,digest
from kci.diverse_ci_long_horizon_v3 import LongHorizonFlipCIEnv
from solve_longterm_flip_v3 import enter_preflight,terminal_return

ROOTDIR=ROOT/"artifacts"/"20261119_composition_signals"/"KCI_V3_TRUE_FLIPS_DATASET_01"
def load(name):
    return [json.loads(x) for x in (ROOTDIR/name).read_text(encoding="utf-8").splitlines() if x.strip()]

def predict_early_action(env,state,policy):
    history=env.history
    if policy in ("always_open","business_only"):
        return "OPEN_PRIVATE_CHANNEL"
    if policy in ("always_skip","myopic_immediate"):
        return "SKIP_PRIVATE_CHANNEL"
    if policy=="entity_only":
        match=any(m["prior"] and m["fact"].startswith(env.task.id+"::")
                  for m in state.ledger if m["recipient"]=="R")
        return "OPEN_PRIVATE_CHANNEL" if match else "SKIP_PRIVATE_CHANNEL"
    if policy=="signal_only":
        match=any(m["prior"] and m["fact"].split("::")[-1] in BASE_FACTS
                  for m in state.ledger if m["recipient"]=="R")
        return "OPEN_PRIVATE_CHANNEL" if match else "SKIP_PRIVATE_CHANNEL"
    if policy=="public_rule_lookahead":
        future=__import__("copy").deepcopy(state)
        # A deterministic full-set proof check for the MANDATORY task facts
        # which are all available in the legitimate public task contract.
        for fact in env.required_facts:
            qualified=env.task.id+"::"+fact
            future.delivered["R"].append(qualified)
        dangerous=FINAL in env.knowledge(future,"R") and not env._recipient_authorized(future)
        return "OPEN_PRIVATE_CHANNEL" if dangerous else "SKIP_PRIVATE_CHANNEL"
    raise ValueError(policy)

def execute(env,policy):
    pre,route,obs=enter_preflight(env)
    assert pre is not None
    first=predict_early_action(env,pre,policy)
    st,a=env.transition(pre,first)
    assert a["verdict"]=="VALID"
    trace=[first]
    for fact in env.required_facts:
        action="READ:"+__import__("kci.diverse_ci_branch_env",fromlist=["SLOT_LABELS"]).SLOT_LABELS[fact]
        st,a=env.transition(st,action)
        assert a["verdict"]=="VALID",(policy,env.task.id,action,a)
        trace.append(action)
        possible,check=env.transition(st,"SEND_RAW_R")
        if check["verdict"]=="VIOLATION":
            if "VERIFY_PRIVATE_R" in env.options(st):
                st,a=env.transition(st,"VERIFY_PRIVATE_R")
                assert a["verdict"]=="VALID"
                trace.append("VERIFY_PRIVATE_R")
                st,a=env.transition(st,"SEND_CERT_R")
                assert a["verdict"]=="VALID"
                trace.append("SEND_CERT_R")
            else:
                st,a=env.transition(st,"ABORT")
                assert a["verdict"]=="VALID" and st.done
                trace.append("ABORT")
                break
        elif check["verdict"]=="VALID":
            st=possible
            trace.append("SEND_RAW_R")
        else:raise AssertionError((policy,env.task.id,"source invalid",check))
    if not st.done:
        assert "COMMIT_RESULT" in env.options(st)
        st,a=env.transition(st,"COMMIT_RESULT")
        assert a["verdict"]=="VALID",(policy,env.task.id,"commit",a)
        trace.append("COMMIT_RESULT")
    return {
        "policy":policy,
        "actual_actions":trace,
        "first_action":first,
        "success":st.success,
        "violation":st.violation,
        "done":st.done,
        "steps":st.steps,
        "actual_native_return":terminal_return(st),
        "business_result":st.generated_result,
        "last_delivered_R":list(st.delivered["R"]),
    }

def main():
    out=ROOTDIR/"CPU_ACTUAL_POLICY_BASELINES.json"
    if out.exists():raise FileExistsError(out)
    target=load("EXACT_QSTAR_PREFERENCES_EVAL_ONLY.jsonl")
    chosen=[x for x in target if x["world_id"].endswith(
        ("__linked","__unrelated","__same_entity_decoy"))]
    assert len(chosen)==105
    tasks={x.id:x for x in task_catalog()}
    policies=("always_open","always_skip","business_only",
              "entity_only","signal_only","myopic_immediate",
              "public_rule_lookahead")
    results=[]
    counts=defaultdict(lambda:Counter())
    for rec in chosen:
        world=rec["world_id"]
        task_id,condition,hist=world.split("__")
        task=tasks[task_id]
        env=LongHorizonFlipCIEnv(task,hist,condition,False)
        optimum=max(rec["native_Qstar_open"],rec["native_Qstar_skip"])
        for policy in policies:
            r=execute(env,policy)
            assert r["actual_native_return"]<=optimum+1e-6
            counts[policy]["episodes"]+=1
            counts[policy]["correct_early_action"]+=int(r["first_action"]==rec["preferred_action"])
            counts[policy]["safe_success"]+=int(r["success"] and not r["violation"])
            counts[policy]["violation"]+=int(r["violation"])
            counts[policy]["safe_incomplete"]+=int(not r["success"] and not r["violation"])
            counts[policy]["total_return_milli"]+=round(r["actual_native_return"]*1000)
            counts[policy]["total_regret_milli"]+=round((optimum-r["actual_native_return"])*1000)
            results.append({
                "world_id":world,"task":task_id,"split":task.root_split,
                "policy":policy,"oracle_best_return":optimum,
                **r})
    summary={
        p:{
            "contexts":v["episodes"],
            "correct_first_action":v["correct_early_action"],
            "safe_task_success":v["safe_success"],
            "privacy_violations":v["violation"],
            "safe_incomplete":v["safe_incomplete"],
            "mean_actual_native_return":round(v["total_return_milli"]/1000/v["episodes"],6),
            "cumulative_regret_vs_native_oracle":round(v["total_regret_milli"]/1000,6)
        } for p,v in counts.items()}
    assert all(s["contexts"]==105 for s in summary.values())
    assert summary["public_rule_lookahead"]["correct_first_action"]==105
    assert summary["always_open"]["safe_task_success"]==105
    assert summary["always_skip"]["safe_task_success"]==70
    assert summary["myopic_immediate"]["privacy_violations"]==0
    report={
        "status":"REAL_NATIVE_FULL_POLICY_REPLAY_COMPLETE",
        "contexts":105,"policies":len(policies),
        "episodes":len(results),
        "source_of_business_and_privacy_outcomes":"actual simulator transitions, NOT counterfactual oracle after baseline first choice",
        "strong_policy_methods":{
            "always_open":"always chooses costly preflight protection",
            "always_skip":"never chooses preflight protection",
            "business_only":"chooses protection for every business condition without seeing R history",
            "entity_only":"uses only whether prior R facts mention the current task entity",
            "signal_only":"uses only whether prior R facts mention any declared source signal type",
            "myopic_immediate":"does not plan at preflight, but may query exact next-step privacy risk; aborts if privacy release impossible",
            "public_rule_lookahead":"uses declared knowledge rules, actual R message history and required evidence set to plan early preflight",
        },
        "results":summary,
        "episodes_evidence":results,
        "no_LLM_or_API":True,
        "human_social_norm_certified":False,
    }
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":report["status"],"episodes":len(results),
        "results":summary},ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__":main()
