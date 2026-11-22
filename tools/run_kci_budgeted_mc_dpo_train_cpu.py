"""Can finite-budget sampled CCE actually TRAIN the scalable role-routed policy?

Reuse the *actual sampled* Qpi evidence from 2026-11-21 sampled-fork audit,
never exact Qpi/Qstar for constructing sampled training preferences. Train an
A-only DPO head on identical original 16 train states, after existing B/source
residual correction. Compare independent vs common-random-number paired sampling
at K=2, 8, 32, holding all other parameters, reference and states fixed.

Small CPU trainability pilot, not LLM transfer evidence.
"""
from __future__ import annotations
from collections import defaultdict, Counter
from pathlib import Path
from hashlib import sha256
import json
import sys
import random

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
import compare_kci_chain_methods_cpu_action_interaction as T
from run_kci_downstream_residual_cpu import RepairedDownstream,CalibratedAfterRepair
from run_kci_role_local_credit_cpu import load_base
from run_kci_cpu_policy_conditioned_gate import greedy_eval
from kci_exact_policy_continuation_cpu import environment_public
from chain_shared_actor_cpu_action_interaction import score_actions
from run_kci_budgeted_counterfactual_cpu import OUT as SAMPLED_OUT

torch.set_num_threads(2)
OUT=ROOT/"artifacts"/"20261121_kci_budgeted_training_cpu_v1"
SEEDS=(7,1234,2025)
KS=(2,8,16)
MODES=("paired_CRN","independent")
REPS=(0,1)
UPDATES=80
LR=.03
REFERENCE=ROOT/"artifacts"/"20261121_kci_downstream_residual_cpu_v1"/"CHECKPOINTS"


def train_context(seed):
    base,base_path=load_base(seed)
    downstream=RepairedDownstream(base).eval()
    ck=torch.load(REFERENCE/("A_Qpi_after_downstream_seed"+str(seed)+".pt"),
                  map_location="cpu",weights_only=True)
    downstream.B.load_state_dict(ck["B"])
    downstream.R.load_state_dict(ck["R"])
    return downstream


def state_for_world(world):
    e=environment_public(world)
    state,audit=e.transition(e.initial(),"A_READ_VERIFIED_PACKAGES")
    assert audit["verdict"]=="VALID"
    return e.observation(state)


def pair_from_sample(x,lookup):
    if not x["accepted"]:
        return None
    obs=lookup[x["world"]]
    options=obs["legal_actions"]
    assert "A_SEND_PACKAGE_A" in options and "A_SEND_PACKAGE_B" in options
    sign=x["direction"]
    better="A_SEND_PACKAGE_A" if sign>0 else "A_SEND_PACKAGE_B"
    worse="A_SEND_PACKAGE_B" if sign>0 else "A_SEND_PACKAGE_A"
    return T.point(obs,better,rejected=worse,
                   weight=max(.01,abs(float(x["mean_gap"]))),
                   origin="sampled_frozen_policy_Qpi_"+("paired" if x["paired"] else "independent"))


def fit(seed,mode,k,rep,records):
    random.seed(seed); np.random.seed(seed);torch.manual_seed(seed)
    downstream=train_context(seed)
    actor=CalibratedAfterRepair(downstream)
    train_lookup={w["world"]:state_for_world(w) for w in T.GROUP["train"]}
    data=[x for x in records if x["version"]=="downstream_repaired"
          and x["split"]=="train" and x["seed"]==seed
          and x["K"]==k and x["replicate"]==rep
          and x["paired"]==(mode=="paired_CRN")]
    assert len(data)==len(T.GROUP["train"])
    pairs=[p for x in data if (p:=pair_from_sample(x,train_lookup)) is not None]
    if not pairs:
        return {"seed":seed,"method":mode,"K":k,"rep":rep,
                "status":"NO_RELIABLE_SAMPLED_SUPERVISION",
                "num_accepted":0}
    features=T.feature_cache(pairs)
    ref=T.ref_diff(actor,features)
    optimizer=torch.optim.Adam(actor.A.parameters(),lr=LR)
    grad_norms=[]
    for step in range(UPDATES):
        optimizer.zero_grad()
        loss=T.supervised_loss(actor,features,"dpo",ref)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(actor.A.parameters(),3)
        optimizer.step()
        if step+1 in (1,20,40,80):
            grad_norms.append({"step":step+1,"loss":float(loss.detach()),
                               "norm":float(norm)})
    score={}
    for split in ("dev","exploratory_test"):
        e=greedy_eval(actor,T.GROUP[split])
        score[split]={key:value for key,value in e.items() if key!="per_world"}
    # TRAIN-only information for post-hoc sample-signal audit, never for training.
    mislabels=sum(x["wrong_accepted"] for x in data)
    measured_accepted=sum(x["accepted"] for x in data)
    return {"seed":seed,"method":mode,"K":k,"rep":rep,
            "status":"TRAINED_FROZEN_ROLE_ROUTED_ACTOR",
            "num_accepted":len(pairs),
            "accepted_teacher_records":measured_accepted,
            "teacher_wrong_accepted_against_exact_Qpi_AUDIT_ONLY":mislabels,
            "training_80_step_grad":grad_norms,
            "eval":score}


def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    source=SAMPLED_OUT/"RESULTS.json"
    assert source.exists(),source
    raw=json.loads(source.read_text(encoding="utf8"))
    assert raw["status"]=="CPU_SAMPLE_BASED_POLICY_CONDITIONED_CREDIT_GATE"
    sampled=raw["full_records"]
    results=[]
    for k in KS:
        for mode in MODES:
            for seed in SEEDS:
                for rep in REPS:
                    r=fit(seed,mode,k,rep,sampled)
                    results.append(r)
                    if r["status"].startswith("TRAINED"):
                        e=r["eval"]["exploratory_test"]
                        print("SAMPLED_DPO",k,mode,seed,rep,
                              "pairs",r["num_accepted"],"errors",r["teacher_wrong_accepted_against_exact_Qpi_AUDIT_ONLY"],
                              "A",e["A_strict_optimal"],"safe",e["greedy_safe_success"],
                              "E_R",round(e["mean_policy_expected_return"],6),
                              "CIprob",round(e["mean_policy_expected_CI_violation"],6),
                              flush=True)
                    else:print("NO_SIGNAL",k,mode,seed,rep,flush=True)
    summary=defaultdict(dict)
    for k in KS:
        for mode in MODES:
            subset=[r for r in results if r["K"]==k and r["method"]==mode
                    and r["status"].startswith("TRAINED")]
            label=mode+"__K"+str(k)
            summary[label]={
                "runs":len(subset),
                "train_sampled_pair_count":[r["num_accepted"] for r in subset],
                "training_teacher_wrong_pairs":[r["teacher_wrong_accepted_against_exact_Qpi_AUDIT_ONLY"] for r in subset],
                "test_A_strict_optimal":[r["eval"]["exploratory_test"]["A_strict_optimal"] for r in subset],
                "test_greedy_safe":[r["eval"]["exploratory_test"]["greedy_safe_success"] for r in subset],
                "test_expected_return":[r["eval"]["exploratory_test"]["mean_policy_expected_return"] for r in subset],
                "test_expected_ci_probability":[r["eval"]["exploratory_test"]["mean_policy_expected_CI_violation"] for r in subset],
            }
    payload={
        "status":"CPU_MC_CCE_ACTUAL_TRAINING_GATE",
        "train_16_worlds_from_original_native_exec":True,
        "train_sufficiency": "two legally valid A choices, frozen repaired downstream, only sampled Qpi teacher",
        "no_exact_Qpi_Qstar_training_teacher":True,
        "no_test_labels_or_data_leaked_into_training":True,
        "sample_source_SHA256":sha256(source.read_bytes()).hexdigest(),
        "teacher_filter_mean_gap":.010,
        "teacher_filter_paired_consensus":.75,
        "sample_budgets":KS,"modes":MODES,
        "sample_repeats":REPS,
        "seeds":SEEDS,
        "not_blind_test":True,
        "not_LLM_or_GPU":True,
        "summary":dict(summary),"per_run":results,
    }
    (OUT/"RESULTS.json").write_text(
        json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("SUMMARY",json.dumps(payload["summary"],ensure_ascii=False),flush=True)
    print("RESULT_FILE",OUT/"RESULTS.json",flush=True)


if __name__=="__main__":
    main()
