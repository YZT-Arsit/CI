"""Adaptive CPU proof: use 2/4/8/16 *paired* frozen-policy suffixes only as needed.

This is an exploratory, engineering transfer-feasibility test. Uses sampled
returns as DPO labels without any native Qstar/Qpi oracle target. Exact Qpi
appears only in a separate audit column.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter,defaultdict
from hashlib import sha256
import random,json,sys

import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
import compare_kci_chain_methods_cpu_action_interaction as T
from kci_exact_policy_continuation_cpu import environment_public
from run_kci_downstream_residual_cpu import CalibratedAfterRepair
from run_kci_budgeted_mc_dpo_train_cpu import train_context,state_for_world,pair_from_sample
from run_kci_cpu_policy_conditioned_gate import greedy_eval
from run_kci_budgeted_counterfactual_cpu import OUT as SAMPLED_OUT

torch.set_num_threads(2)
SEEDS=(7,1234,2025)
REPS=(0,1,2)
OUT=ROOT/"artifacts"/"20261121_kci_adaptive_sampled_cee_cpu_v1"
SCHEDULE=[
    (2,.05,1.0),
    (4,.03,.75),
    (8,.015,.875),
    (16,.01,.75),
]


def choose_adaptive(table):
    for k,margin,agree in SCHEDULE:
        r=table[k]
        if abs(r["mean_gap"])>=margin and r["agreement"]>=agree and (
            r["raw_direction"] != 0 and
            r["raw_direction"]==(1 if r["positive_pairs"]>r["negative_pairs"]
                                   else -1 if r["negative_pairs"]>r["positive_pairs"] else 0)):
            x=dict(r)
            x["accepted"]=True
            x["direction"]=x["raw_direction"]
            x["spent_K"]=k
            return x
    x=dict(table[16])
    x["accepted"]=False
    x["direction"]=0
    x["spent_K"]=16
    return x


def train_one(seed,rep,raw):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    downstream=train_context(seed)
    policy=CalibratedAfterRepair(downstream)
    obs_lookup={w["world"]:state_for_world(w) for w in T.GROUP["train"]}
    per_case={}
    samples=[r for r in raw if r["seed"]==seed and r["replicate"]==rep
             and r["split"]=="train" and r["paired"] is True
             and r["version"]=="downstream_repaired"]
    assert len(samples)==16*4
    for x in samples:
        per_case.setdefault(x["world"],{})[x["K"]]=x
    selected=[choose_adaptive(per_case[w["world"]]) for w in T.GROUP["train"]]
    points=[x for s in selected if (x:=pair_from_sample(s,obs_lookup)) is not None]
    assert points
    cache=T.feature_cache(points)
    ref=T.ref_diff(policy,cache)
    opt=torch.optim.Adam(policy.A.parameters(),lr=.03)
    grad=[]
    for step in range(80):
        opt.zero_grad()
        loss=T.supervised_loss(policy,cache,"dpo",ref)
        loss.backward()
        n=torch.nn.utils.clip_grad_norm_(policy.A.parameters(),3)
        opt.step()
        if step+1 in (1,20,40,80):
            grad.append({"step":step+1,"loss":float(loss.detach()),"norm":float(n)})
    ev={s:greedy_eval(policy,w) for s,w in T.GROUP.items() if s!="train"}
    stats={
        "cases_seen":16,
        "accepted":len(points),
        "wrong_accepted_exact_Qpi_AUDIT_ONLY":sum(x["wrong_accepted"] for x in selected),
        "episodes_spent":sum(2*x["spent_K"] for x in selected),
        "episodes_fixed_K16":16*2*16,
        "mean_K":float(np.mean([x["spent_K"] for x in selected])),
        "K_histogram":dict(Counter(str(x["spent_K"]) for x in selected)),
        "accepted_case_ids":[x["world"] for x in selected if x["accepted"]],
        "abstained_case_ids":[x["world"] for x in selected if not x["accepted"]],
    }
    print("ADAPTIVE",seed,rep,
          "teacher",stats["accepted"],"extra_episodes",stats["episodes_spent"],
          "meanK",round(stats["mean_K"],2),
          "wrong",stats["wrong_accepted_exact_Qpi_AUDIT_ONLY"],
          "A",ev["exploratory_test"]["A_strict_optimal"],
          "safe",ev["exploratory_test"]["greedy_safe_success"],
          "E_R",round(ev["exploratory_test"]["mean_policy_expected_return"],5),
          flush=True)
    return {"seed":seed,"replicate":rep,"budget":stats,
            "train_grad":grad,
            "dev":{k:v for k,v in ev["dev"].items() if k!="per_world"},
            "exploratory_test":{k:v for k,v in ev["exploratory_test"].items() if k!="per_world"}}


def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    source=SAMPLED_OUT/"RESULTS.json"
    payload=json.loads(source.read_text(encoding="utf8"))
    results=[train_one(seed,rep,payload["full_records"])
             for seed in SEEDS for rep in REPS]
    summary={
        "mean_total_episodes":float(np.mean([r["budget"]["episodes_spent"] for r in results])),
        "mean_K":float(np.mean([r["budget"]["mean_K"] for r in results])),
        "mean_train_pairs_accepted":float(np.mean([r["budget"]["accepted"] for r in results])),
        "total_wrong_accepted":sum(r["budget"]["wrong_accepted_exact_Qpi_AUDIT_ONLY"] for r in results),
        "test_A_optimal":[r["exploratory_test"]["A_strict_optimal"] for r in results],
        "test_safe_success":[r["exploratory_test"]["greedy_safe_success"] for r in results],
        "test_expected_returns":[r["exploratory_test"]["mean_policy_expected_return"] for r in results],
        "test_CI_probabilities":[r["exploratory_test"]["mean_policy_expected_CI_violation"] for r in results],
    }
    results_json={
        "status":"ADAPTIVE_SAMPLED_POLICY_CCE_CPU_REAL_TRAINING",
        "source_sampled_SHA256":sha256(source.read_bytes()).hexdigest(),
        "small_resource_only":True,
        "paired_random_tail_rollouts_not_independent":True,
        "no_exact_oracle_teacher_in_training":True,
        "adaptive_gate_exploratory_not_preregistered":True,
        "no_blind_test_or_LLM":True,
        "schedule":SCHEDULE,
        "summary":summary,"runs":results,
    }
    (OUT/"RESULTS.json").write_text(json.dumps(results_json,indent=2,
                                ensure_ascii=False)+"\n",encoding="utf8")
    print("SUMMARY",json.dumps(summary,ensure_ascii=False),flush=True)
    print("RESULT_FILE",OUT/"RESULTS.json",flush=True)


if __name__=="__main__":
    main()
