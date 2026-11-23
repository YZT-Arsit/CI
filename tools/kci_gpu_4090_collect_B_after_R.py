"""Qwen3-4B own-policy TRAIN-only B-CCE after frozen SourceAgent LoRA repair.

At each real B state, two immediately legal B choices are forced. Remaining
steps are produced by the SAME pretrained Qwen3-4B with SourceAgent adapter
only at release. The resulting CI, safe task success and reward are verified
through native execution; no Qstar reference teachers are consulted.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
from hashlib import sha256
from collections import Counter
import argparse,json,sys,time,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from evaluate_real_4090_R import Policy,preflight
from chain_shared_actor_cpu_action_interaction import baseline_automatic_action,valid_decision_candidates
from kci_exact_policy_continuation_cpu import environment_public
from compare_kci_chain_methods_cpu_action_interaction import GROUP
from kci_llm_text_surface import render_role_text
from kci.diverse_ci_branch_env import digest

def continuation(env,initial,forced,model):
    st,a=env.transition(initial,forced)
    assert a["verdict"]=="VALID"
    trace=[{"phase":"preflight","action":forced,"native_verdict":a["verdict"]}]
    for step in range(60):
        if st.native.done:break
        obs=env.observation(st)
        avail=list(env.options(st))
        actions=valid_decision_candidates(obs,avail)
        if len(actions)>1:
            action,probs=model.decision(obs,"trained_R")
            assert action in actions
            t="Qwen3-4B-frozen-after-source-repair"
        else:
            action=baseline_automatic_action(env,st)
            probs=None;t="common_native_tool"
        if action is None:raise RuntimeError("No legal native action")
        nxt,au=env.transition(st,action)
        trace.append({"phase":obs.get("stage",obs.get("current_stage")),
                      "role":obs.get("actor",obs.get("role")),"action":action,
                      "native_verdict":au["verdict"],"source":t})
        if au["verdict"] not in ("VALID","VIOLATION"):
            raise RuntimeError(("Unverified native action",au))
        st=nxt
    safe=bool(st.native.done and st.native.success and not st.native.violation)
    violating=bool(st.native.violation)
    val=round(1-env.spec.cost_per_step*st.native.steps,6) if safe else (-1. if violating else 0.)
    return {"safe_success":safe,"CI_violation":violating,"reward":val,
            "native_steps":st.native.steps,"trace":trace,"done":bool(st.native.done)}

def label(open_result,skip_result):
    if not open_result["done"] or not skip_result["done"]:
        return None,"unfinished_branch"
    if open_result["CI_violation"]!=skip_result["CI_violation"]:
        return ("OPEN_PRIVATE_CHANNEL" if not open_result["CI_violation"] else "SKIP_PRIVATE_CHANNEL"),"CI_verified"
    if open_result["safe_success"]!=skip_result["safe_success"]:
        return ("OPEN_PRIVATE_CHANNEL" if open_result["safe_success"] else "SKIP_PRIVATE_CHANNEL"),"safe_task_success"
    if open_result["safe_success"] and skip_result["safe_success"]:
        d=open_result["reward"]-skip_result["reward"]
        if abs(d)>.01:return ("OPEN_PRIVATE_CHANNEL" if d>0 else "SKIP_PRIVATE_CHANNEL"),"completion_cost"
    return None,"equal_outcomes"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",required=True)
    ap.add_argument("--source-adapter",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    model=Policy(args.base,args.source_adapter)
    rows=GROUP["train"]
    all_records=[];pairs=[];start=time.monotonic()
    for j,w in enumerate(rows):
        env=environment_public(w)
        s0=env.initial()
        s0,_=env.transition(s0,"A_READ_VERIFIED_PACKAGES")
        a_choice,a_prob=model.decision(env.observation(s0),"base")
        if not a_choice.startswith("A_SEND_PACKAGE"):
            raise AssertionError(("Model declined to send at A",w["world"],a_choice))
        s=preflight(env,a_choice)
        obs=env.observation(s)
        assert "OPEN_PRIVATE_CHANNEL" in env.options(s)
        assert "SKIP_PRIVATE_CHANNEL" in env.options(s)
        x=continuation(env,s,"OPEN_PRIVATE_CHANNEL",model)
        y=continuation(env,s,"SKIP_PRIVATE_CHANNEL",model)
        chosen,reason=label(x,y)
        p={"world":w["world"],"root":w["root"],"A_model_action":a_choice,
           "B_state_hash":digest(asdict(s)),"B_observation_hash":digest(obs),
           "OPEN_result":x,"SKIP_result":y,"pref_action":chosen,"reason":reason}
        all_records.append(p)
        with (output/"TRAIN_FORKS.jsonl").open("a",encoding="utf8") as f:f.write(json.dumps(p)+"\n")
        if chosen:
            reject=("SKIP_PRIVATE_CHANNEL" if chosen=="OPEN_PRIVATE_CHANNEL"
                    else "OPEN_PRIVATE_CHANNEL")
            acts=obs["legal_actions"]
            pref={"split":"train","role":"Coordinator","stage":"preflight",
                  "source":"frozen_Qwen3_4B_real_native_CCE_after_SourceAgent_repair",
                  "evidence_type":"actual_same_frozen_LM_continuation_2_legal_actions",
                  "world":w["world"],"root":w["root"],
                  "state_hash":digest(asdict(s)),
                  "observation_hash":digest(obs),
                  "chosen_action":chosen,"rejected_action":reject,
                  "chosen_letter":chr(65+acts.index(chosen)),
                  "rejected_letter":chr(65+acts.index(reject)),
                  "all_executable_actions":acts,
                  "prompt":render_role_text(obs),
                  "native_evidence":{"OPEN_PRIVATE_CHANNEL":x,
                                     "SKIP_PRIVATE_CHANNEL":y},
                  "preference_reason":reason,
                  "not_Qstar_oracle_teacher":True,
                  "sample_count_per_action":1}
            pairs.append(pref)
            with (output/"DPO_ROLE_B_TRAIN.jsonl").open("a",encoding="utf8") as f:
                f.write(json.dumps(pref)+"\n")
        print("OWN_POLICY_B_FORK",j+1,len(rows),w["world"],a_choice,
              "OPEN",x["reward"],"SKIP",y["reward"],"LABEL",chosen,reason,
              "S",round(time.monotonic()-start,1),flush=True)
    summary={"status":"REAL_QWEN3_4B_ROLE_STAGED_B_CCE_FROM_OWN_FROZEN_POLICY",
        "train_worlds":len(rows),"train_roots":sorted(set(w["root"] for w in rows)),
        "n_counterfactual_full_episodes":2*len(rows),
        "n_accepted_pairs":len(pairs),
        "labels":dict(Counter(p["chosen_action"] for p in pairs)),
        "reasons":dict(Counter(p["preference_reason"] for p in pairs)),
        "both_safe_success":sum(x["OPEN_result"]["safe_success"] and x["SKIP_result"]["safe_success"] for x in all_records),
        "source_adapter":args.source_adapter,"same_frozen_LM_for_both_branches":True,
        "no_test_label_for_train":True,
        "model_forward_calls":model.model_forward_calls,
        "elapsed":round(time.monotonic()-start,2)}
    (output/"SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf8")
    print("FINAL_B_CCE",json.dumps(summary),flush=True)
if __name__=="__main__":main()
