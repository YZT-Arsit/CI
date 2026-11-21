"""KCI CPU role-local value-credit gate: preserve aligned downstream policy.

The baseline shared A->R->B->C actor is frozen. A learnable, role-gated residual
head modifies ONLY RecordsAgent/A_SEND logits. Compare the same train inputs
under direct Qstar labels, Qstar DPO, frozen-baseline Qpi DPO and disagreement-
filtered DPO. This is a mechanism experiment, NOT a new natural-language LLM.
"""
from __future__ import annotations

from collections import defaultdict, Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
import compare_kci_chain_methods_cpu_action_interaction as T
from chain_shared_actor_cpu_action_interaction import (
    SharedHornActor, STAGES, score_actions, run_episode,
)
from kci_exact_policy_continuation_cpu import (
    environment_public, FrozenPolicyValue, enumerate_decisions,
)
from kci.diverse_ci_branch_env import digest
from run_kci_cpu_policy_conditioned_gate import greedy_eval

OUT=ROOT/"artifacts"/"20261121_kci_role_local_credit_cpu_v1"
SOURCE=ROOT/"artifacts"/"20261121_kci_policy_conditioned_cpu_v1"/"CHECKPOINTS"
SEEDS=(7,1234,2025)
METHODS=("A_label_CE","A_Qstar_DPO","A_Qpi_DPO","A_agreement_DPO")
UPDATES=140
LR=.035
BETA=3.0
torch.set_num_threads(2)

class UpstreamResidualActor(nn.Module):
    """Role-gated correction: downstream logits are BITWISE UNCHANGED."""
    def __init__(self,base):
        super().__init__()
        self.base=base.eval()
        for t in self.base.parameters():
            t.requires_grad_(False)
        self.a_head=nn.Linear(10,1,bias=False)
        nn.init.zeros_(self.a_head.weight)

    def features(self,x):
        with torch.no_grad():
            visible=torch.maximum(x["known"],torch.maximum(x["bundle"],x["req"]))
            inferred=self.base.infer(visible,x["prem"],x["head"])
        package_cost=x["extra"][:,5]
        setup=x["extra"][:,2]
        per_step=x["extra"][:,3]
        nrequired=x["extra"][:,1]
        free=x["extra"][:,4]
        return torch.stack([
            inferred,
            package_cost,
            setup,
            per_step,
            nrequired,
            free,
            inferred*setup,
            inferred*per_step,
            inferred*nrequired,
            torch.ones_like(inferred),
        ],dim=-1)

    def forward(self,x):
        with torch.no_grad():
            original=self.base(x)
        A=x["stage"][:,STAGES.index("A_SEND")]
        return original+8.0*A*self.a_head(self.features(x)).squeeze(-1)


def load_base(seed):
    path=SOURCE/("sft_counterfactual_coverage_seed"+str(seed)+".pt")
    assert path.exists(),path
    frozen=SharedHornActor().eval()
    frozen.load_state_dict(torch.load(path,map_location="cpu",
                                     weights_only=True)["model_state_dict"])
    return frozen,path


def train_forks(base):
    """One actual upstream A state per native training world."""
    all_points=[]
    teacher_rows=[]
    for row in T.GROUP["train"]:
        e=environment_public(row)
        st,audit=e.transition(e.initial(),"A_READ_VERIFIED_PACKAGES")
        assert audit["verdict"]=="VALID"
        obs=e.observation(st)
        assert "privacy_resource_contract" in obs
        opts=tuple(a for a in e.options(st) if a.startswith("A_SEND_PACKAGE"))
        assert len(opts)==2
        frozen=FrozenPolicyValue(e,base)
        qpi={a.removeprefix("A_SEND_"):frozen.q(st,a)[0] for a in opts}
        oracle=row["upstream_A_values_given_exact_B_continuation"]
        gap_star=oracle["PACKAGE_A"]-oracle["PACKAGE_B"]
        gap_pi=qpi["PACKAGE_A"]-qpi["PACKAGE_B"]
        point={"obs":obs,"options":tuple(e.options(st)),
               "gstar":gap_star,"gpi":gap_pi,"world":row["world"],
               "source_mode":row["source_mode"],"case_hash":digest(asdict(st))}
        all_points.append(point)
        teacher_rows.append({"world":row["world"],
                             "gap_Qstar":gap_star,"gap_Qpi":gap_pi,
                             "strict_Qstar":abs(gap_star)>1e-8,
                             "Qstar_Qpi_disagree":gap_star*gap_pi<0})
    return all_points,teacher_rows


def active_pairs(rows,mode):
    data=[]
    for row in rows:
        if abs(row["gstar"])<=1e-8:continue
        sign=(row["gpi"] if mode=="A_Qpi_DPO" else row["gstar"])
        if mode=="A_agreement_DPO" and row["gstar"]*row["gpi"]<=0:
            continue
        if abs(sign)<=0.004:continue
        best="A_SEND_PACKAGE_A" if sign>0 else "A_SEND_PACKAGE_B"
        wrong="A_SEND_PACKAGE_B" if sign>0 else "A_SEND_PACKAGE_A"
        data.append(T.point(
            row["obs"],best,weight=max(.004,abs(sign)),rejected=wrong,
            origin=mode))
    return data


def downstream_unchanged(base,adapted):
    seen=0
    for row in T.GROUP["train"][:3]:
        e=environment_public(row)
        pts,_=enumerate_decisions(e)
        for p in pts:
            if p["phase"]=="A_SEND":continue
            opts=list(e.options(p["state"]))
            with torch.no_grad():
                a=score_actions(base,p["obs"],opts)
                b=score_actions(adapted,p["obs"],opts)
            assert torch.equal(a,b),(row["world"],p["phase"],a,b)
            seen+=1
    return seen


def run(seed,mode):
    np.random.seed(seed);random.seed(seed);torch.manual_seed(seed)
    base,source=load_base(seed)
    actor=UpstreamResidualActor(base)
    records,teachers=train_forks(base)
    pairs=active_pairs(records,mode)
    assert pairs,mode
    data=T.feature_cache(pairs)
    ref=T.ref_diff(base,data)
    # For CE, the targets are the same precise native Qstar labels as DPO.
    before={s:greedy_eval(actor,r) for s,r in T.GROUP.items() if s!="train"}
    optimizer=torch.optim.Adam(actor.a_head.parameters(),lr=LR)
    logs=[]
    for step in range(UPDATES):
        optimizer.zero_grad()
        if mode=="A_label_CE":
            loss=T.supervised_loss(actor,data,"sft")
        else:
            loss=T.supervised_loss(actor,data,"dpo",ref)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(actor.a_head.parameters(),3.0)
        optimizer.step()
        if step+1 in (1,20,40,80,140):
            logs.append({"update":step+1,"loss":float(loss.detach()),
                         "gradient_norm":float(norm)})
    isolation=downstream_unchanged(base,actor)
    after={s:greedy_eval(actor,r) for s,r in T.GROUP.items() if s!="train"}
    folder=OUT/"CHECKPOINTS"
    file=folder/(mode+"_seed"+str(seed)+".pt")
    torch.save({"A_residual_only":actor.a_head.state_dict(),
                "seed":seed,"method":mode,
                "base_checkpoint_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                "downstream_base_frozen":True,"updates":UPDATES},file)
    r={
        "mode":mode,"seed":seed,"training_A_pairs":len(pairs),
        "train_CCE_teacher_Qpi_disagreements":sum(x["Qstar_Qpi_disagree"] for x in teachers),
        "teacher_rows":teachers,"loss_log":logs,
        "downstream_logits_unchanged_states":isolation,
        "before":before,"after":after,
        "saved_head":file.name,
        "saved_head_sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
        "new_head_weights":actor.a_head.weight.detach().view(-1).tolist(),
        "reference_base":"SFT_counterfactual_coverage",
    }
    print("ROLE_LOCAL",mode,seed,
          "TRAIN",len(pairs),
          "DEV",after["dev"]["greedy_safe_success"],
          "TEST",after["exploratory_test"]["greedy_safe_success"],
          "A",after["exploratory_test"]["A_strict_optimal"],
          "E_R",round(after["exploratory_test"]["mean_policy_expected_return"],5),
          "A_regret",round(after["exploratory_test"]["mean_A_reference_regret"],5),
          flush=True)
    return r


def main():
    if OUT.exists():raise FileExistsError(OUT)
    (OUT/"CHECKPOINTS").mkdir(parents=True)
    results=[run(seed,mode) for mode in METHODS for seed in SEEDS]
    summary={}
    for mode in METHODS:
        summary[mode]={}
        for split in ("dev","exploratory_test"):
            subset=[x for x in results if x["mode"]==mode]
            summary[mode][split]={
                key:[r["after"][split][key] for r in subset]
                for key in ("greedy_safe_success","greedy_CI_violations",
                            "A_strict_optimal","mean_A_reference_regret",
                            "mean_policy_expected_return",
                            "mean_policy_expected_CI_violation")
            }
    report={
        "status":"REAL_CPU_ROLE_LOCAL_CREDIT_TRAINING_COMPLETED",
        "no_GPU_or_LLM":True,
        "same_shared_frozen_SFT_CF_base":True,
        "changes_only_A_logits_and_leaves_downstream_exactly_frozen":True,
        "train_graphs":4,"dev_graphs":1,"exploratory_test_graphs":2,
        "exploratory_test_seen_during_development":True,
        "not_an_independent_benchmark":True,
        "all_training_A_supervision_from_train_worlds_only":True,
        "teacher_source":"Qstar from native reference or exact frozen base Qpi",
        "method_list":METHODS,
        "seeds":SEEDS,
        "train_updates":UPDATES,
        "summary":summary,
        "results":results,
    }
    (OUT/"RESULTS.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("FINAL_ROLE_SUMMARY",json.dumps(summary,ensure_ascii=False),flush=True)
    print("REPORT",OUT/"RESULTS.json",flush=True)


if __name__=="__main__":
    main()
