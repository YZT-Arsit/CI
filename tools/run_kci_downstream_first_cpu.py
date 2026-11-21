"""KCI CPU staged role-factorized experiment: downstream repair then upstream CCE.

Key controls:
  - Frozen original shared SFT-CF backbone, SAME 16 training worlds.
  - B and source release train from existing counterfactual expert actions only.
  - Upstream A train from EXACT frozen repaired-policy Qpi or native Qstar.
  - No hidden labels passed to actor, no eval graph optimization, no LLM/GPU.
  - This is a causal mechanism prototype, not a claimed SOTA algorithm.
"""
from __future__ import annotations
from pathlib import Path
from copy import deepcopy
import sys,json,random,hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
import compare_kci_chain_methods_cpu_action_interaction as T
from chain_shared_actor_cpu_action_interaction import (
    SharedHornActor,ACTIONS,score_actions,
)
from kci_exact_policy_continuation_cpu import (
    environment_public,FrozenPolicyValue,enumerate_decisions,
)
from run_kci_cpu_policy_conditioned_gate import (
    build_common_inputs,greedy_eval,
)
from run_kci_role_local_credit_cpu import load_base,train_forks
from run_kci_calibrated_upstream_cpu import pairs_for_training

torch.set_num_threads(2)
OUT=ROOT/"artifacts"/"20261121_kci_downstream_first_cpu_v1"
SEEDS=(7,1234,2025)
UPDATES_B=120
UPDATES_A=80
METHODS=("A_Qstar_after_downstream","A_Qpi_after_downstream")
LR=.03

class RepairedDownstream(nn.Module):
    """Source and preflight-only learnable logits; A routes remain frozen base."""
    def __init__(self,base):
        super().__init__()
        self.base=base.eval()
        for p in self.base.parameters():p.requires_grad_(False)
        self.B=nn.Linear(6,1,bias=False)
        self.R=nn.Linear(6,1,bias=False)
        nn.init.zeros_(self.B.weight)
        nn.init.zeros_(self.R.weight)

    def infer(self,*args):
        return self.base.infer(*args)

    def forward(self,x):
        with torch.no_grad():orig=self.base(x)
        future=torch.maximum(x["known"],x["req"])
        riskB=self.base.infer(future,x["prem"],x["head"]).detach()
        now=torch.maximum(x["known"],x["src"])
        riskR=self.base.infer(now,x["prem"],x["head"]).detach()
        a=x["action"]
        open_=a[:,ACTIONS["OPEN_PRIVATE_CHANNEL"]]
        skip=a[:,ACTIONS["SKIP_PRIVATE_CHANNEL"]]
        raw=a[:,ACTIONS["SEND_RAW_R"]]
        verify=a[:,ACTIONS["VERIFY_PRIVATE_R"]]
        setup=x["extra"][:,2]
        bp=torch.stack((open_,skip,open_*riskB,skip*riskB,
                        open_*setup,skip*setup),dim=-1)
        rp=torch.stack((raw,verify,raw*riskR,verify*riskR,
                        raw*x["extra"][:,12],verify*x["extra"][:,12]),dim=-1)
        bs=self.B(bp).squeeze(-1)
        rs=self.R(rp).squeeze(-1)
        value=torch.where((open_+skip)>.5,bs,orig)
        value=torch.where((raw+verify)>.5,rs,value)
        return value


class CalibratedAfterRepair(nn.Module):
    """Replace upstream package logits, strictly preserve repaired downstream."""
    def __init__(self,repaired):
        super().__init__()
        self.base=repaired.eval()
        for p in self.base.parameters():p.requires_grad_(False)
        self.A=nn.Linear(5,1,bias=False)
        nn.init.normal_(self.A.weight,mean=0,std=.025)

    def forward(self,x):
        with torch.no_grad():
            logits=self.base(x)
            risk=self.base.infer(torch.maximum(x["bundle"],x["req"]),
                                 x["prem"],x["head"])
        setup=x["extra"][:,2]*5.
        features=torch.stack((
            x["extra"][:,5],risk*(setup-1)/6,
            torch.zeros_like(risk),risk,x["extra"][:,1]*6/4,
        ),dim=-1)
        pkg=(x["action"][:,ACTIONS["A_SEND_PACKAGE_A"]] +
             x["action"][:,ACTIONS["A_SEND_PACKAGE_B"]])
        return torch.where(pkg>.5,self.A(features).squeeze(-1),logits)


def train_downstream(seed,expanded):
    base,path=load_base(seed)
    net=RepairedDownstream(base)
    active=[p for p in expanded
            if p["obs"].get("current_stage") in ("preflight","release")]
    assert len(active)>40
    cache=T.feature_cache(active)
    opt=torch.optim.Adam([*net.B.parameters(),*net.R.parameters()],lr=LR)
    curve=[]
    for step in range(UPDATES_B):
        opt.zero_grad()
        loss=T.supervised_loss(net,cache,"sft")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_([*net.B.parameters(),*net.R.parameters()],3)
        opt.step()
        if step+1 in (1,30,60,120):
            curve.append({"step":step+1,"loss":float(loss.detach()),
                          "gradient_norm":float(norm)})
    for row in T.GROUP["train"][:4]:
        e=environment_public(row);pts,_=enumerate_decisions(e)
        for p in pts:
            if p["phase"]!="A_SEND":continue
            actions=list(e.options(p["state"]))
            with torch.no_grad():
                old=score_actions(base,p["obs"],actions)
                new=score_actions(net,p["obs"],actions)
            assert torch.equal(old,new)
    return net,base,curve,len(active),path


def get_A_data(repaired):
    rows=[]
    for w in T.GROUP["train"]:
        e=environment_public(w)
        st,a=e.transition(e.initial(),"A_READ_VERIFIED_PACKAGES")
        assert a["verdict"]=="VALID"
        obs=e.observation(st)
        vals=FrozenPolicyValue(e,repaired)
        piA=vals.q(st,"A_SEND_PACKAGE_A")[0]
        piB=vals.q(st,"A_SEND_PACKAGE_B")[0]
        gstar=(w["upstream_A_values_given_exact_B_continuation"]["PACKAGE_A"]-
               w["upstream_A_values_given_exact_B_continuation"]["PACKAGE_B"])
        rows.append({
            "obs":obs,"gstar":gstar,"gpi":piA-piB,
            "world":w["world"]
        })
    return rows


def fit_A(repaired,rows,seed,method):
    torch.manual_seed(seed)
    policy=CalibratedAfterRepair(repaired)
    pairs=pairs_for_training(rows,"A_Qpi_pair_DPO" if "Qpi" in method else "A_Qstar_pair_DPO")
    assert pairs
    data=T.feature_cache(pairs)
    reference=T.ref_diff(policy,data)
    opt=torch.optim.Adam(policy.A.parameters(),lr=LR)
    loss_log=[]
    for step in range(UPDATES_A):
        opt.zero_grad()
        loss=T.supervised_loss(policy,data,"dpo",reference)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(policy.A.parameters(),3)
        opt.step()
        if step+1 in (1,20,40,80):
            loss_log.append({"step":step+1,"loss":float(loss.detach()),
                             "gradient_norm":float(norm)})
    # All B, source and C scores must equal the frozen repaired downstream.
    checked=0
    for w in T.GROUP["train"][:5]:
        e=environment_public(w);pts,_=enumerate_decisions(e)
        for p in pts:
            if p["phase"]=="A_SEND":continue
            actions=list(e.options(p["state"]))
            with torch.no_grad():
                x=score_actions(repaired,p["obs"],actions)
                y=score_actions(policy,p["obs"],actions)
            assert torch.equal(x,y)
            checked+=1
    evals={k:greedy_eval(policy,g)
           for k,g in T.GROUP.items() if k!="train"}
    return policy,evals,loss_log,checked,len(pairs)


def train_one(seed):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    _,cf,_,_=build_common_inputs()
    repaired,base,curve,num_B,source=train_downstream(seed,cf)
    before={k:greedy_eval(base,v) for k,v in T.GROUP.items() if k!="train"}
    downstream={k:greedy_eval(repaired,v) for k,v in T.GROUP.items() if k!="train"}
    pairs=get_A_data(repaired)
    first_disagree=sum(p["gstar"]*p["gpi"]<0 for p in pairs if abs(p["gstar"])>1e-8)
    print("DOWNSTREAM_ONLY",seed,
          "test_safe",downstream["exploratory_test"]["greedy_safe_success"],
          "test_E_R",round(downstream["exploratory_test"]["mean_policy_expected_return"],4),
          "Qstar_Qpi_conflicts",first_disagree,flush=True)
    records=[]
    for method in METHODS:
        a,ev,logs,checked,n=fit_A(repaired,pairs,seed,method)
        chk=OUT/"CHECKPOINTS"/(method+"_seed"+str(seed)+".pt")
        torch.save({"B":repaired.B.state_dict(),"R":repaired.R.state_dict(),
                    "A":a.A.state_dict(),"seed":seed,"method":method,
                    "source_hash":hashlib.sha256(source.read_bytes()).hexdigest()},chk)
        r={"seed":seed,"method":method,"num_A_pairs":n,
           "downstream_frozen_states":checked,
           "loss_A":logs,"eval":ev,
           "checkpoint_name":chk.name,
           "sha256":hashlib.sha256(chk.read_bytes()).hexdigest()}
        records.append(r)
        print("AFTER_DOWNSTREAM",method,seed,
              "TEST",ev["exploratory_test"]["greedy_safe_success"],
              "A",ev["exploratory_test"]["A_strict_optimal"],
              "E_R",round(ev["exploratory_test"]["mean_policy_expected_return"],4),
              "Regret",round(ev["exploratory_test"]["mean_A_reference_regret"],4),
              flush=True)
    return {"seed":seed,"downstream_label_samples":num_B,
            "downstream_curve":curve,
            "train_A_Qstar_Qpi_conflicts_after_repair":first_disagree,
            "before":before,"downstream_repair_only":downstream,
            "trained_A":records}


def main():
    if OUT.exists():raise FileExistsError(OUT)
    (OUT/"CHECKPOINTS").mkdir(parents=True)
    results=[train_one(seed) for seed in SEEDS]
    doc={
        "status":"CPU_DOWNSTREAM_FIRST_THEN_UPSTREAM_REAL_NATIVE_EXPERIMENT",
        "source_fixed":"existing 16 train 4 dev 8 developer-seen eval",
        "downstream_learner":"role-gated CE trained on original CF expert states",
        "upstream_learner":"DPO with exact frozen repaired-policy Qpi vs native Qstar",
        "current_method":"staged conditional/role-factorized mechanism feasibility, not original CCE/DPR LLM",
        "has_unrestricted_agent_autonomy":False,
        "Qpi_train_only_from_actual_policies":True,
        "paper_grade_blind_test":False,
        "seeds":SEEDS,
        "results":results}
    (OUT/"RESULTS.json").write_text(json.dumps(doc,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    for run in results:
        for item in run["trained_A"]:
            ev=item["eval"]["exploratory_test"]
            print("SUMMARY",run["seed"],item["method"],
                  ev["greedy_safe_success"],ev["A_strict_optimal"],
                  ev["mean_policy_expected_return"],flush=True)
    print("SAVED",OUT/"RESULTS.json",flush=True)


if __name__=="__main__":
    main()
