"""KCI 2026-11-21: replace overconfident A logits, freeze downstream exactly.

Motivation (observed): base SFT-CF is 14/14 train but 2/6 exploratory A
and role-local additive residual cannot overcome large incorrect OOD logits.
This is a controlled architectural feasibility experiment, NOT ICML evidence.
"""
from __future__ import annotations

from pathlib import Path
from collections import defaultdict,Counter
import sys,random,json,hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci_exact_policy_continuation_cpu import environment_public,enumerate_decisions
from chain_shared_actor_cpu_action_interaction import (
    ACTIONS,STAGES,score_actions,
)
import compare_kci_chain_methods_cpu_action_interaction as T
from run_kci_role_local_credit_cpu import load_base,train_forks
from run_kci_cpu_policy_conditioned_gate import greedy_eval

torch.set_num_threads(2)
OUT=ROOT/"artifacts"/"20261121_kci_calibrated_upstream_cpu_v1"
SEEDS=(7,1234,2025)
METHODS=("A_label_SFT","A_Qstar_pair_DPO","A_Qpi_pair_DPO","A_agreement_pair_DPO")
LR=.03; UPDATES=80

class CalibratedAActor(nn.Module):
    """Raw A package score replaced, never additive; others EXACT base logits."""
    def __init__(self,base):
        super().__init__()
        self.base=base.eval()
        for p in self.base.parameters():p.requires_grad_(False)
        self.A_score=nn.Linear(5,1,bias=False)
        nn.init.normal_(self.A_score.weight,mean=0,std=.025)

    def public_A_features(self,x):
        risk=self.base.infer(
            torch.maximum(x["bundle"],x["req"]),x["prem"],x["head"]).detach()
        setup=x["extra"][:,2]*5.0
        return torch.stack([
            x["extra"][:,5],                # actual source bundle step count / 6
            risk*(setup-1.0)/6.0,           # public setup overhead
            torch.zeros_like(risk),         # avoid oracle #attestation feature
            risk,                          # public-rule implication
            x["extra"][:,1]*6.0/4.0,        # required evidence count
        ],dim=-1)

    def forward(self,x):
        with torch.no_grad():baseline=self.base(x)
        A=(x["action"][:,ACTIONS["A_SEND_PACKAGE_A"]] +
           x["action"][:,ACTIONS["A_SEND_PACKAGE_B"]])
        proposed=self.A_score(self.public_A_features(x)).squeeze(-1)
        # This is a routed head, not a residual. No downstream coupling.
        return torch.where(A>.5,proposed,baseline)


def pairs_for_training(forks,mode):
    pairs=[]
    for row in forks:
        qstar=row["gstar"]
        qpi=row["gpi"]
        if abs(qstar)<1e-8:continue
        if mode=="A_Qpi_pair_DPO":gap=qpi
        elif mode=="A_agreement_pair_DPO":
            if qstar*qpi<=0:continue
            gap=qstar
        else:gap=qstar
        if abs(gap)<=0.004:continue
        pos="A_SEND_PACKAGE_A" if gap>0 else "A_SEND_PACKAGE_B"
        neg="A_SEND_PACKAGE_B" if gap>0 else "A_SEND_PACKAGE_A"
        pairs.append(T.point(row["obs"],pos,rejected=neg,
                             weight=abs(gap),origin=mode))
    return pairs


def check_downstream_frozen(net,base):
    checked=0
    for row in T.GROUP["train"][:5]:
        e=environment_public(row);pts,_=enumerate_decisions(e)
        for p in pts:
            if p["phase"]=="A_SEND":continue
            choices=list(e.options(p["state"]))
            with torch.no_grad():
                old=score_actions(base,p["obs"],choices)
                new=score_actions(net,p["obs"],choices)
            assert torch.equal(old,new),p["phase"]
            checked+=1
    return checked


def fit_one(method,seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    base,base_path=load_base(seed)
    actor=CalibratedAActor(base)
    train_fork,teachers=train_forks(base)
    examples=pairs_for_training(train_fork,method)
    assert examples
    data=T.feature_cache(examples)
    with torch.no_grad():
        raw_scores=actor(data["features"])
        # Build the DPO reference from this SAME initial random head.
        ref_scores=T.ref_diff(actor,data)
    optimizer=torch.optim.Adam(actor.A_score.parameters(),lr=LR)
    checkpoints=[]
    for i in range(UPDATES):
        optimizer.zero_grad()
        if method=="A_label_SFT":
            loss=T.supervised_loss(actor,data)
        else:
            loss=T.supervised_loss(actor,data,"dpo",ref_scores)
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(actor.A_score.parameters(),3)
        optimizer.step()
        if i+1 in (1,10,20,40,80):
            checkpoints.append({"updates":i+1,"loss":float(loss.detach()),
                                "gradient_norm":float(grad)})
    downstream_states=check_downstream_frozen(actor,base)
    evaluation={name:greedy_eval(actor,rows)
                for name,rows in T.GROUP.items() if name!="train"}
    path=OUT/"CHECKPOINTS"/(method+"_seed"+str(seed)+".pt")
    torch.save({"A_only_head":actor.A_score.state_dict(),
                "method":method,"seed":seed,
                "base_sha256":hashlib.sha256(base_path.read_bytes()).hexdigest(),
                "input_only_public_rules_costs":True},path)
    result={
        "method":method,"seed":seed,"train_A_pairs":len(examples),
        "strict_train":sum(abs(v["gstar"])>1e-8 for v in train_fork),
        "train_Qstar_Qpi_disagreements":sum(t["Qstar_Qpi_disagree"] for t in teachers),
        "training_curve":checkpoints,
        "held_frozen_downstream_states":downstream_states,
        "A_head_weights":actor.A_score.weight.detach().view(-1).tolist(),
        "eval":evaluation,
        "saved_checkpoint":path.name,
        "checksum_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    print("CALIBRATED",method,seed,
          "TRAIN",len(examples),
          "DEV",evaluation["dev"]["greedy_safe_success"],
          "TEST",evaluation["exploratory_test"]["greedy_safe_success"],
          "A",evaluation["exploratory_test"]["A_strict_optimal"],
          "ExpR",round(evaluation["exploratory_test"]["mean_policy_expected_return"],4),
          "Regret",round(evaluation["exploratory_test"]["mean_A_reference_regret"],4),
          flush=True)
    return result


def main():
    if OUT.exists():raise FileExistsError(OUT)
    (OUT/"CHECKPOINTS").mkdir(parents=True)
    all_results=[fit_one(m,s) for m in METHODS for s in SEEDS]
    summary={}
    for mode in METHODS:
        summary[mode]={}
        for split in ("dev","exploratory_test"):
            runs=[r for r in all_results if r["method"]==mode]
            summary[mode][split]={
                k:[r["eval"][split][k] for r in runs]
                for k in ("greedy_safe_success","greedy_CI_violations",
                          "A_strict_optimal","mean_A_reference_regret",
                          "mean_policy_expected_return",
                          "mean_policy_expected_CI_violation",
                          "mean_greedy_return")
            }
    doc={
        "status":"CPU_UPSTREAM_ROLE_ROUTED_VALUE_HEAD_EXPERIMENT",
        "base":"SFT-CF shared native actor; frozen all downstream logits",
        "training_A_strict_worlds":14,
        "three_seed_matched":True,
        "input_is_public_rules_cost_contract_only":True,
        "full_attestation_handcrafted_feature_disabled":True,
        "no_future_Q_labels_in_model_features":True,
        "no_true_blind_test":True,
        "not_LLM":True,
        "summary":summary,
        "records":all_results,
    }
    (OUT/"RESULTS.json").write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("FINAL_CALIBRATED",json.dumps(summary,ensure_ascii=False),flush=True)
    print("SAVED",(OUT/"RESULTS.json"),flush=True)


if __name__=="__main__":
    main()
