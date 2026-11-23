"""Real Qwen3-4B KCI A-stage own-policy counterfactual rollout collector.

Freeze one HF pretrained 4B; activate learned SourceAgent adapter ONLY during
release. At the SAME A_SEND state, force two locally legal source disclosures;
run K actual stochastic continuation trajectories per action and adjudicate
CI-first task outcomes. No Qstar or hidden native oracle teacher.

Outputs independent native trajectory evidence plus TRAIN-only role-A DPO pairs.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
from hashlib import sha256
from collections import Counter
import argparse,sys,json,time,random
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from compare_kci_chain_methods_cpu_action_interaction import GROUP
from kci_gpu_4090_eval_R import Policy
from kci_exact_policy_continuation_cpu import environment_public
from chain_shared_actor_cpu_action_interaction import baseline_automatic_action,valid_decision_candidates
from kci_llm_text_surface import render_role_text
from kci_ci_first_preference_gate import classify_ci_first
from kci.diverse_ci_branch_env import digest

def run_suffix(env,prefix,forced,model,rng):
    state,verdict=env.transition(prefix,forced)
    if verdict["verdict"]!="VALID":
        raise AssertionError((forced,verdict))
    trace=[{"stage":"A_SEND","action":forced,"role":"RecordsAgent",
            "verdict":verdict["verdict"],"forced_counterfactual":True,
            "parent_sha256":digest(asdict(prefix)),
            "state_sha256":digest(asdict(state)),
            "new_receipts":len(state.native.delivered["R"])-len(prefix.native.delivered["R"])}]
    sample_logprob=0.0
    for step in range(60):
        if state.native.done:break
        obs=env.observation(state)
        options=list(env.options(state))
        alternatives=valid_decision_candidates(obs,options)
        if len(alternatives)>1:
            greedy,probs=model.decision(obs,"trained_R")
            # The model provides an actual normalized action distribution;
            # sample under that distribution, not a scripted safety planner.
            assert set(probs)==set(options)
            weights=np.asarray([probs[a] for a in options],dtype=np.float64)
            weights=weights/max(weights.sum(),1e-30)
            choice=options[int(rng.choice(len(options),p=weights))]
            sample_logprob+=float(np.log(max(probs[choice],1e-30)))
            mode="Qwen3-4B-categorical-frozen"
        else:
            choice=baseline_automatic_action(env,state)
            probs=None;mode="common-scripted-tool"
        if choice not in options:raise AssertionError(("INVALID_ACTION",choice,options))
        previous=digest(asdict(state))
        before_receipts=len(state.native.delivered["R"])
        new,audit=env.transition(state,choice)
        trace.append({
            "stage":obs.get("stage",obs.get("current_stage")),
            "role":obs.get("actor",obs.get("role")),
            "action":choice,"source":mode,"verdict":audit["verdict"],
            "forced_counterfactual":False,
            "parent_sha256":previous,
            "state_sha256":digest(asdict(new)),
            "new_receipts":len(new.native.delivered["R"])-before_receipts})
        if audit["verdict"] not in ("VALID","VIOLATION"):
            raise AssertionError(("NATIVE_INVALID",audit))
        state=new
    if not state.native.done:raise AssertionError("NOT_TERMINAL_AFTER_NATIVE_HORIZON")
    success=bool(state.native.success and not state.native.violation)
    violation=bool(state.native.violation)
    ret=round(1-env.spec.cost_per_step*state.native.steps,6) if success else(-1. if violation else 0.)
    return {"reward":ret,"safe_success":success,"CI_violation":violation,
            "native_steps":state.native.steps,"terminal_sha256":digest(asdict(state)),
            "trace":trace,"model_logprob":sample_logprob}

def prepare(env):
    s=env.initial()
    s,verdict=env.transition(s,"A_READ_VERIFIED_PACKAGES")
    assert verdict["verdict"]=="VALID"
    obs=env.observation(s)
    assert obs["stage"]=="A_SEND"
    choices=("A_SEND_PACKAGE_A","A_SEND_PACKAGE_B")
    for c in choices:
        nxt,verdict=env.transition(s,c)
        assert verdict["verdict"]=="VALID"
    return s,obs,choices

def scalar_preference(samples):
    diffs=[a["reward"]-b["reward"] for a,b in samples]
    mean=float(np.mean(diffs))
    positive=sum(x>1e-8 for x in diffs)
    negative=sum(x< -1e-8 for x in diffs)
    if abs(mean)<.01 or max(positive,negative)/len(samples)<.75:
        return 0,"uncertain_scalar"
    return (1 if mean>0 else -1),"scalar_expected_reward"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",required=True)
    ap.add_argument("--R-adapter",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--splits",nargs="+",default=["train"])
    ap.add_argument("--K",type=int,default=4)
    ap.add_argument("--seed",type=int,default=20261123)
    args=ap.parse_args()
    if args.K<1 or args.K>64:raise ValueError("Invalid suffix sample count")
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    actor=Policy(args.base,args.R_adapter)
    records=[];accepted=[];start=time.monotonic()
    for split in args.splits:
        assert split in GROUP
        for w in GROUP[split]:
            env=environment_public(w)
            state,obs,actions=prepare(env)
            branch={act:[] for act in actions}
            samples=[]
            for i in range(args.K):
                paired_seed=int(sha256((w["world"]+":"+str(args.seed)+":"+str(i)).encode()).hexdigest()[:13],16)
                # Common-random-number streams, independent instances per branch.
                one=run_suffix(env,state,actions[0],actor,np.random.default_rng(paired_seed))
                two=run_suffix(env,state,actions[1],actor,np.random.default_rng(paired_seed))
                branch[actions[0]].append(one)
                branch[actions[1]].append(two)
                samples.append((one,two))
            vec=[((x["reward"],float(x["safe_success"]),float(x["CI_violation"])),
                  (y["reward"],float(y["safe_success"]),float(y["CI_violation"])))
                 for x,y in samples]
            comparison=classify_ci_first(obs,actions,vec,gap_min=.01,consensus=.75)
            raw_sign,raw_reason=scalar_preference(samples)
            outrec={
                "split":split,"world":w["world"],"root":w["root"],
                "graph":w["graph"],"source_mode":w["source_mode"],
                "A_state_hash":digest(asdict(state)),
                "A_observation_hash":digest(obs),
                "legal_A_options":list(actions),
                "K":args.K,"reference_policy":"Qwen3-4B-frozen+SourceAgent-R-LoRA",
                "branches":branch,
                "sampled_scalar_sign":raw_sign,"sampled_scalar_reason":raw_reason,
                "CI_first_accepted":comparison.accepted,
                "CI_first_direction":comparison.direction,
                "CI_first_reason":comparison.reason,
                "CI_first_gap":comparison.gap,
                "public_inference_hazards":[comparison.risky_potential_A,comparison.risky_potential_B],
                "violation_counts":[comparison.observed_violations_A,comparison.observed_violations_B],
                "safe_success_counts":[comparison.observed_safe_success_A,comparison.observed_safe_success_B],
            }
            records.append(outrec)
            with (out/"NATIVE_A_FORKS.jsonl").open("a",encoding="utf8") as f:
                f.write(json.dumps(outrec,ensure_ascii=False)+"\n")
            if comparison.accepted and split=="train":
                chosen=actions[0] if comparison.direction==1 else actions[1]
                rejected=actions[1] if comparison.direction==1 else actions[0]
                all_=obs["legal_actions"]
                pair={
                  "split":"train","role":"RecordsAgent","stage":"A_SEND",
                  "source":"native_Qwen3-4B_own_Qpi_with_R_repaired",
                  "evidence_type":"K_sampled_real_frozen_policy_native_CI_first",
                  "world":w["world"],"root":w["root"],
                  "state_hash":digest(asdict(state)),
                  "observation_hash":digest(obs),
                  "all_executable_actions":list(all_),
                  "chosen_action":chosen,"rejected_action":rejected,
                  "chosen_letter":chr(65+all_.index(chosen)),
                  "rejected_letter":chr(65+all_.index(rejected)),
                  "prompt":render_role_text(obs),
                  "native_evidence":{a:{"n":args.K,
                    "CI_violations":sum(x["CI_violation"] for x in branch[a]),
                    "safe_successes":sum(x["safe_success"] for x in branch[a]),
                    "mean_reward":float(np.mean([x["reward"] for x in branch[a]]))}
                     for a in actions},
                  "preference_reason":comparison.reason,
                  "not_Qstar_oracle_teacher":True,
                  "sample_count_per_action":args.K,
                }
                accepted.append(pair)
                with (out/"DPO_ROLE_A_TRAIN.jsonl").open("a",encoding="utf8") as f:
                    f.write(json.dumps(pair,ensure_ascii=False)+"\n")
            print("QWEN4B_A_FORK",split,w["world"],
                  "A",[(a,round(float(np.mean([x["reward"] for x in branch[a]])),5),
                        sum(x["safe_success"] for x in branch[a]),sum(x["CI_violation"] for x in branch[a]))
                       for a in actions],
                  "SCALAR",raw_sign,"CI_FIRST",comparison.direction if comparison.accepted else "ABSTAIN",
                  comparison.reason,"elapsed",round(time.monotonic()-start,1),flush=True)
    summary={"status":"REAL_QWEN3_4B_A_SAME_STATE_K_SAMPLED_OWN_POLICY_COUNTERFACTUALS",
            "splits":args.splits,"worlds":len(records),"K":args.K,
            "n_native_suffixes":sum(2*args.K for _ in records),
            "train_preferences":len(accepted),
            "train_action_counts":dict(Counter(x["chosen_action"] for x in accepted)),
            "train_roots":sorted(set(x["root"] for x in accepted)),
            "train_preference_reasons":dict(Counter(x["preference_reason"] for x in accepted)),
            "train_R_adapter_only_in_release":True,
            "original_native_verifier_unchanged":True,
            "no_ideal_Qstar_in_teacher":True,
            "eval_splits_not_used_as_DPO_teacher":True,
            "gpu_model_forward_calls":actor.model_forward_calls,
            "model_prompt_tokens":actor.tokens,
            "runtime_seconds":round(time.monotonic()-start,2)}
    (out/"SUMMARY.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("FINAL_A_CCE",json.dumps(summary),flush=True)
if __name__=="__main__":
    main()
