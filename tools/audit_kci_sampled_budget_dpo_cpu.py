"""Deterministic re-training and native replay of tiny-budget KCI sampled DPO.

Reload frozen downstream policies, reconstruct exactly the K=2 paired sampled
train preference pairs (replicate 0) WITHOUT any oracle future value, repeat
the A-only DPO update, reload saved checkpoints and check complete native
trace/soft-policy returns on the dev and developer-seen test worlds.
"""
from __future__ import annotations
from pathlib import Path
from hashlib import sha256
import json
import random
import sys
import torch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
import compare_kci_chain_methods_cpu_action_interaction as T
from run_kci_budgeted_mc_dpo_train_cpu import train_context,state_for_world,pair_from_sample
from run_kci_downstream_residual_cpu import CalibratedAfterRepair
from run_kci_budgeted_counterfactual_cpu import OUT as SAMPLED_OUT
from kci_exact_policy_continuation_cpu import environment_public,enumerate_decisions
from chain_shared_actor_cpu_action_interaction import run_episode,score_actions
from run_kci_cpu_policy_conditioned_gate import greedy_eval

torch.set_num_threads(2)
OUT=ROOT/"artifacts"/"20261121_kci_budgeted_dpo_saved_audit_v1"
METHOD="paired_CRN"
K=2
REP=0
SEEDS=(7,1234,2025)


def train_and_recheck(seed,records):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    downstream=train_context(seed)
    net=CalibratedAfterRepair(downstream)
    obs_lookup={w["world"]:state_for_world(w) for w in T.GROUP["train"]}
    data=[x for x in records if x["version"]=="downstream_repaired"
          and x["split"]=="train" and x["seed"]==seed
          and x["K"]==K and x["replicate"]==REP and x["paired"]]
    assert len(data)==16
    pairs=[p for item in data if (p:=pair_from_sample(item,obs_lookup)) is not None]
    assert len(pairs)==14
    cache=T.feature_cache(pairs)
    initial_ref=T.ref_diff(net,cache)
    optimizer=torch.optim.Adam(net.A.parameters(),lr=.03)
    for step in range(80):
        optimizer.zero_grad()
        loss=T.supervised_loss(net,cache,"dpo",initial_ref)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.A.parameters(),3)
        optimizer.step()
    reference_results={split:greedy_eval(net,rows)
                       for split,rows in T.GROUP.items() if split!="train"}
    file=OUT/"CHECKPOINTS"/("K2_paired_seed"+str(seed)+".pt")
    torch.save({"seed":seed,"head":net.A.state_dict(),
                "method":"sampled_K2_CRN_same_state_DPO",
                "training_label_source":"only four frozen continuation suffixes per A fork",
                "teacher_paired_records":len(pairs)},file)
    saved=torch.load(file,map_location="cpu",weights_only=True)
    net2=CalibratedAfterRepair(downstream)
    net2.A.load_state_dict(saved["head"])
    iso=0
    for row in T.GROUP["train"][:5]:
        env=environment_public(row)
        positions,_=enumerate_decisions(env)
        for pos in positions:
            if pos["phase"]=="A_SEND":continue
            acts=list(env.options(pos["state"]))
            with torch.no_grad():
                old=score_actions(downstream,pos["obs"],acts)
                new=score_actions(net2,pos["obs"],acts)
            assert torch.equal(old,new)
            iso+=1
    for split in ("dev","exploratory_test"):
        recomputed=greedy_eval(net2,T.GROUP[split])
        for key in ("greedy_safe_success","greedy_CI_violations",
                    "A_strict_optimal","mean_A_reference_regret",
                    "mean_policy_expected_return",
                    "mean_policy_expected_CI_violation","mean_greedy_return"):
            a=reference_results[split][key]
            b=recomputed[key]
            assert abs(a-b)<1e-7,(split,key,a,b)
        for row in T.GROUP[split]:
            ep=run_episode(environment_public(row),net2,greedy=True)
            assert ep["invalid"] is False
    print("SAVED_REPLAY",seed,"accepted",len(pairs),"frozen_states",iso,
          "A",reference_results["exploratory_test"]["A_strict_optimal"],
          "safe",reference_results["exploratory_test"]["greedy_safe_success"],
          "expected",reference_results["exploratory_test"]["mean_policy_expected_return"],flush=True)
    return {"seed":seed,"training_pairs":len(pairs),
            "A_only_downstream_logits_exactly_unchanged_points":iso,
            "sha256":sha256(file.read_bytes()).hexdigest(),
            "filename":file.name,
            "dev":{k:v for k,v in reference_results["dev"].items() if k!="per_world"},
            "exploratory_test":{k:v for k,v in reference_results["exploratory_test"].items() if k!="per_world"}}


def main():
    if OUT.exists():raise FileExistsError(OUT)
    (OUT/"CHECKPOINTS").mkdir(parents=True)
    src=SAMPLED_OUT/"RESULTS.json"
    records=json.loads(src.read_text(encoding="utf8"))["full_records"]
    runs=[train_and_recheck(s,records) for s in SEEDS]
    doc={"status":"SAMPLED_K2_DPO_CPU_SAVED_POLICY_INDEPENDENT_REPLAY_PASS",
         "training_source_SHA256":sha256(src.read_bytes()).hexdigest(),
         "no_ideal_Qstar_or_exact_Qpi_in_training_labels":True,
         "all_three_reloaded_policies_replayed":True,
         "original_native_A_delivery_and_downstream_transition":True,
         "A_training_does_not_change_downstream_logits":True,
         "not_blind_test":True,
         "not_4B_LLM_training":True,"seeds":SEEDS,"runs":runs}
    (OUT/"AUDIT.json").write_text(json.dumps(doc,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("AUDIT_SAVED",OUT/"AUDIT.json",flush=True)


if __name__=="__main__":
    main()
