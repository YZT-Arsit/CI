"""KCI-Align CPU: executable local downstream repair -> B CCE -> A CCE.

The scaled-transfer mechanism uses ONLY:
 - Same state native verdicts for R release (RAW causes VIOLATION vs VERIFY valid).
 - K=4 *actual frozen-policy* continuation samples for B and A legal actions.
No ideal Qstar label or full expert trace is used to form any new preference.

One shared frozen base with role-specific small trainable heads (analog to
role-routed LoRA adapters). The first R update cannot change A or B logits,
the B update cannot change R and A, and A cannot change any downstream logits.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from copy import deepcopy
from pathlib import Path
from hashlib import sha256
import sys,json,random

import torch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
import compare_kci_chain_methods_cpu_action_interaction as T
from chain_shared_actor_cpu_action_interaction import score_actions
from kci_exact_policy_continuation_cpu import environment_public,enumerate_decisions
from run_kci_downstream_residual_cpu import (
    RepairedDownstream,CalibratedAfterRepair
)
from run_kci_role_local_credit_cpu import load_base
from run_kci_cpu_policy_conditioned_gate import greedy_eval
from run_kci_budgeted_counterfactual_cpu import (
    FrozenNativeSampler,sample_pair_stream,classify
)

torch.set_num_threads(2)
OUT=ROOT/"artifacts"/"20261121_kci_native_repair_mc_bridge_cpu_v1"
SEEDS=(7,1234,2025)
K=4
R_UPDATES=80
B_UPDATES=80
A_UPDATES=80
LR=.03


def immediate_downstream_repairs():
    pairs=[]
    metrics=Counter()
    for w in T.GROUP["train"]:
        env=environment_public(w)
        positions,_=enumerate_decisions(env)
        for p in positions:
            if p["phase"]!="release":continue
            actions=p["actions"]
            if set(p["verdicts"].values())=={"VIOLATION","VALID"}:
                valid=next(a for a in actions if p["verdicts"][a]=="VALID")
                unsafe=next(a for a in actions if p["verdicts"][a]=="VIOLATION")
                assert valid=="VERIFY_PRIVATE_R" and unsafe=="SEND_RAW_R"
                pairs.append(T.point(p["obs"],valid,rejected=unsafe,weight=1.,
                                     origin="executed_same_state_CI_violation_repair"))
                metrics["verified_direct_violation_pairs"]+=1
            elif set(p["verdicts"].values())=={"VALID"}:
                metrics["both_release_actions_currently_legal"]+=1
            else:
                raise AssertionError(p["verdicts"])
    assert len(pairs)>0
    return pairs,dict(metrics)


def fit_local_repair(base,records):
    model=RepairedDownstream(base).eval()
    for p in model.B.parameters():p.requires_grad_(False)
    cache=T.feature_cache(records)
    initial=T.ref_diff(model,cache)
    optimizer=torch.optim.Adam(model.R.parameters(),lr=LR)
    for i in range(R_UPDATES):
        optimizer.zero_grad()
        loss=T.supervised_loss(model,cache,"dpo",initial)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.R.parameters(),3)
        optimizer.step()
    model.eval()
    for p in model.R.parameters():p.requires_grad_(False)
    return model


def native_mc_pairs(policy,phase,k,seed):
    pairs=[]
    resource=Counter()
    documents=[]
    for w in T.GROUP["train"]:
        env=environment_public(w)
        pts,_=enumerate_decisions(env)
        if phase=="A_SEND":
            # Exactly one source-tool origin with two immediately legal packages.
            pts=[p for p in pts if p["phase"]=="A_SEND"]
        else:
            pts=[p for p in pts if p["phase"]==phase]
        sampler=FrozenNativeSampler(env,policy)
        for p in pts:
            a,b=p["actions"]
            if not all(p["verdicts"][action]=="VALID" for action in (a,b)):
                resource["nonlegal_candidates_skipped"]+=1
                continue
            two=[]
            for action in (a,b):
                st,ver=env.transition(p["state"],action)
                assert ver["verdict"]=="VALID"
                two.append(st)
            forkseed=seed*331+int(sha256((w["world"]+phase+p["hash"]).encode()).hexdigest()[:10],16)
            sampled=sample_pair_stream(env,two,sampler,k,seed=forkseed,paired=True)
            verdict=classify(sampled,k,gap=.010,consensus=.75)
            resource["native_candidate_states"]+=1
            resource["native_branch_episodes"]+=2*k
            if verdict["accepted"]:
                good=a if verdict["direction"]>0 else b
                bad=b if verdict["direction"]>0 else a
                assert good!=bad
                pairs.append(T.point(
                    p["obs"],good,rejected=bad,
                    weight=max(.01,abs(verdict["mean_gap"])),
                    origin="same_state_frozen_Qpi_mc_"+phase))
                resource["accepted_native_pairs"]+=1
            else:resource["abstained_uncertain_pairs"]+=1
            documents.append({
                "world":w["world"],"phase":phase,
                "state_hash":p["hash"],"legal_actions":[a,b],
                "mean_gap":verdict["mean_gap"],
                "accepted":verdict["accepted"],
                "agreement":verdict["agreement"],
                "chosen":(good if verdict["accepted"] else None),
            })
        resource["policy_forward_states"]+=sampler.policy_queries
        resource["native_actual_steps"]+=sampler.native_steps
    return pairs,dict(resource),documents


def fit_B(policy,pairs):
    assert pairs,"No B-CCE evidence was generated"
    model=policy
    for p in model.B.parameters():p.requires_grad_(True)
    cache=T.feature_cache(pairs)
    ref=T.ref_diff(model,cache)
    opt=torch.optim.Adam(model.B.parameters(),lr=LR)
    for i in range(B_UPDATES):
        opt.zero_grad()
        loss=T.supervised_loss(model,cache,"dpo",ref)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.B.parameters(),3)
        opt.step()
    model.eval()
    for p in model.B.parameters():p.requires_grad_(False)
    return model


def fit_A(policy,pairs):
    assert pairs,"No A-CCE evidence was generated"
    net=CalibratedAfterRepair(policy).eval()
    cache=T.feature_cache(pairs)
    ref=T.ref_diff(net,cache)
    opt=torch.optim.Adam(net.A.parameters(),lr=LR)
    for i in range(A_UPDATES):
        opt.zero_grad()
        loss=T.supervised_loss(net,cache,"dpo",ref)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.A.parameters(),3)
        opt.step()
    return net.eval()


def measure(net):
    return {split:{k:v for k,v in greedy_eval(net,T.GROUP[split]).items()
                   if k!="per_world"}
            for split in ("dev","exploratory_test")}


def verify_role_isolation(base,after_R,after_B,after_A):
    n=Counter()
    for row in T.GROUP["train"][:5]:
        env=environment_public(row)
        states,_=enumerate_decisions(env)
        for p in states:
            options=list(env.options(p["state"]))
            with torch.no_grad():
                l0=score_actions(base,p["obs"],options)
                l1=score_actions(after_R,p["obs"],options)
                l2=score_actions(after_B,p["obs"],options)
                l3=score_actions(after_A,p["obs"],options)
            if p["phase"]=="A_SEND":
                assert torch.equal(l0,l1)
                assert torch.equal(l1,l2)
                n["A_logit_isolated_until_A_training"]+=1
            if p["phase"]=="preflight":
                assert torch.equal(l0,l1)
                assert torch.equal(l2,l3)
                n["B_logit_isolated_R_then_A"]+=1
            if p["phase"]=="release":
                assert torch.equal(l1,l2)
                assert torch.equal(l2,l3)
                n["release_logits_unchanged_B_then_A"]+=1
    return dict(n)


def train_one(seed,repair_pairs):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    base,base_path=load_base(seed)
    before=measure(base)
    after_R=fit_local_repair(base,repair_pairs)
    metrics_R=measure(after_R)
    b_pairs,b_cost,b_docs=native_mc_pairs(after_R,"preflight",K,seed)
    if not b_pairs:
        after_B=after_R
    else:
        # Store true R stage state before updating B; role-specific head.
        after_B=fit_B(deepcopy(after_R),b_pairs)
    metrics_B=measure(after_B)
    a_pairs,a_cost,a_docs=native_mc_pairs(after_B,"A_SEND",K,seed+7307)
    after_A=fit_A(after_B,a_pairs)
    metrics_A=measure(after_A)
    isolate=verify_role_isolation(base,after_R,after_B,after_A)
    ck=OUT/"CHECKPOINTS"/("KCI_DPR_BCCE_ACCE_seed"+str(seed)+".pt")
    torch.save({"seed":seed,"R":after_A.base.R.state_dict(),
                "B":after_A.base.B.state_dict(),"A":after_A.A.state_dict(),
                "base_sha256":sha256(base_path.read_bytes()).hexdigest(),
                "training": "native immediate-release correction DPO; B and A K4 policy continuation DPO"},ck)
    print("REPAIR_MC",seed,
          "DPR",len(repair_pairs),"B_pairs",len(b_pairs),"A_pairs",len(a_pairs),
          "base_R",round(before["exploratory_test"]["mean_policy_expected_return"],5),
          "R_only",round(metrics_R["exploratory_test"]["mean_policy_expected_return"],5),
          "after_B",round(metrics_B["exploratory_test"]["mean_policy_expected_return"],5),
          "after_A",round(metrics_A["exploratory_test"]["mean_policy_expected_return"],5),
          "A_opt",metrics_A["exploratory_test"]["A_strict_optimal"],
          "safe",metrics_A["exploratory_test"]["greedy_safe_success"],flush=True)
    return {"seed":seed,"source_base":base_path.name,"saved":ck.name,
            "sha256":sha256(ck.read_bytes()).hexdigest(),
            "verified_downstream_repair_examples":len(repair_pairs),
            "B_policy_sampling":b_cost,"A_policy_sampling":a_cost,
            "B_evidence":b_docs,"A_evidence":a_docs,
            "role_isolation":isolate,
            "before":before,"R_after_local_repair":metrics_R,
            "B_after_Qpi_education":metrics_B,
            "A_after_Qpi_education":metrics_A}


def main():
    if OUT.exists():raise FileExistsError(OUT)
    (OUT/"CHECKPOINTS").mkdir(parents=True)
    repair_pairs,repair_counts=immediate_downstream_repairs()
    results=[train_one(seed,repair_pairs) for seed in SEEDS]
    summary={}
    for split in ("dev","exploratory_test"):
        summary[split]={}
        for stage in ("before","R_after_local_repair","B_after_Qpi_education","A_after_Qpi_education"):
            summary[split][stage]={
                k:[r[stage][split][k] for r in results]
                for k in ("mean_policy_expected_return","mean_policy_expected_CI_violation",
                          "greedy_safe_success","A_strict_optimal","mean_A_reference_regret")
            }
    report={"status":"REAL_NATIVE_ROLE_ISOLATED_DPR_BCCE_ACCE_CPU_CHAIN",
            "source":"unchanged native 16train/4dev/8seen test worlds",
            "all_sources_synthetic":True,"no_hidden_native_Qstar_or_Qpi_train_labels":True,
            "DPR_validated_by_actual_same_state_verdicts_not_generated_failure_trajectory":True,
            "B_A_Qpi_only_MC_K4_with_pairwise_agreement":True,
            "not_trained_LLM":True,"test_is_not_blind":True,
            "seeds":SEEDS,"K_per_action":K,
            "initial_repair_data":repair_counts,
            "summary":summary,"results":results}
    (OUT/"RESULTS.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("FINAL",json.dumps(summary["exploratory_test"],ensure_ascii=False),flush=True)
    print("OUT",OUT/"RESULTS.json",flush=True)


if __name__=="__main__":
    main()
