r"""Four-way frozen-task CPU learning comparison: SFT, Qstar DPO, Qpi DPO, Terminal PG.

Frozen original 28 A->R->B->C worlds, 4/1/2 independent RULE families.
Per seed all four methods start from SAME saved SFT checkpoint (4471 params).
Equal 120 Adam optimizer updates, same native decision interface, same
public-rule actor representation. Same expanded native expert observations
for SFT and DPO auxiliary loss. DPO prefers matched actual A/B states:
- Qstar pairs use exact native best successors on TRAIN cases only.
- Qpi pairs use exact FROZEN reference-policy continuations on TRAIN only.
These preferences are not interchangeable. Terminal RL sees ONLY rewards of
actual sampled native policy episodes, not unselected actions / Qstar.
All source data and V3.1 benchmark files are READ ONLY. No GPU.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter,defaultdict
from hashlib import sha256
from copy import deepcopy
import json,sys,time,random
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from chain_shared_actor_cpu_action_interaction import (
 SharedHornActor,run_episode,score_actions,environment,digest)
import compare_kci_chain_methods_cpu_action_interaction as common
from kci_exact_frozen_qpi_cpu import (
 GROUP,SEEDS,CKPT,DATA,load_frozen_policy,ExactQpi,prepare_A)
BASE=ROOT/"artifacts"/"20261121_policy_gap_pilot_01"
TEACHER=BASE/"TRAIN_ONLY_QPI_CREDIT_FROM_SFT_REFERENCE"
OUT=BASE/"FOUR_WAY_CPU_TRAINED_POLICY_RESULTS_03"
if OUT.exists():raise FileExistsError(OUT)
OUT.mkdir()
(OUT/"MODEL_WEIGHTS").mkdir()
METHODS=("action_sft","qstar_dpo","frozen_qpi_dpo","terminal_reinforce")
UPDATES=120
LR=common.LR
MIN_GAP=1e-4
assert len(GROUP["train"])==16 and len(GROUP["dev"])==4 and len(GROUP["exploratory_test"])==8
common_expert,expanded_sft,qstar_all,ref_counts=common.build_expert_dataset(GROUP["train"])
cf_cache=common.feature_cache(expanded_sft)
# Four groups start from the same PRETRAINED expert actor checkpoint per seed.
# Plain Action SFT is given the expanded same-state oracle-label baseline,
# not an artificially weak single-path coverage.
assert len(expanded_sft)>len(common_expert)
def reference_prefs(seed):
    rows=[json.loads(r) for r in (TEACHER/f"TRAIN_ONLY_SFT_REFERENCE_QPI_SEED{seed}.jsonl").read_text(
        encoding="utf-8").splitlines() if r.strip()]
    assert len(rows)==48 and all(r["split"]=="train" for r in rows)
    qstar_lookup={p["obs_sha256"]:p for p in qstar_all}
    qpi=[];qstar=[];audit=Counter()
    for t in rows:
        opts=t["options"];qpi_vals=t["qpi"];qstar_vals=t["qstar_train_only_eval"]
        assert len(opts)==len(qpi_vals)==len(qstar_vals)==2
        obs=t["model_actor_observation"]
        if abs(qstar_vals[0]-qstar_vals[1])<=MIN_GAP:
            audit["train_Qstar_ties_excluded_from_both"]+=1
            continue
        h=digest(obs)
        assert h in qstar_lookup,(seed,t["world"],t["type"],"original label state missing")
        baseline=qstar_lookup[h]
        assert set(opts)<=set(baseline["options"]) # ABORT remains legal but is not a Qpi teacher pair
        # Remove native Qstar ties only to MATCH the two teacher training
        # state sets. Qpi sign always calculated exclusively from Qpi values.
        stargap=qstar_vals[0]-qstar_vals[1]
        pigap=qpi_vals[0]-qpi_vals[1]
        if abs(stargap)<=MIN_GAP:
            audit["train_Qstar_ties_excluded_from_both"]+=1
            continue
        if abs(pigap)<=MIN_GAP:
            audit["low_policy_gap_excluded_from_both"]+=1
            continue
        qa=opts[0] if pigap>0 else opts[1]
        qb=opts[1] if pigap>0 else opts[0]
        pa=common.point(obs,qa,weight=abs(pigap),
                        rejected=qb,origin="FROZEN_SFT_Qpi_NATIVE_CONTINUATION_TRAIN_ONLY")
        qa_star=opts[0] if stargap>0 else opts[1]
        qb_star=opts[1] if stargap>0 else opts[0]
        sb=common.point(obs,qa_star,weight=abs(stargap),
                        rejected=qb_star,origin="EXACT_NATIVE_Qstar_TRAIN_ONLY")
        assert h==pa["obs_sha256"]==sb["obs_sha256"]
        assert pa["actor"]==sb["actor"]
        qpi.append(pa);qstar.append(sb)
        audit["same_actual_train_A_or_B_states"]+=1
        audit["role_"+pa["actor"]]+=1
        audit["Qstar_vs_Qpi_label_conflict"]+=int(qa!=qa_star)
    assert len(qpi)==len(qstar)>0
    assert audit["Qstar_vs_Qpi_label_conflict"]>0
    assert all(x["obs_sha256"]==y["obs_sha256"] for x,y in zip(qpi,qstar))
    return qstar,qpi,dict(audit)

def clone_checkpoint(seed):
    model,reference_meta=load_frozen_policy("action_sft",seed)
    assert sum(p.numel() for p in model.parameters())==4471
    for parameter in model.parameters():parameter.requires_grad_(True)
    return model,reference_meta

def train_one(seed,method,star,pi,teacher_audit):
    model,meta=clone_checkpoint(seed)
    np.random.seed(seed+271828)
    random.seed(seed+271828)
    torch.manual_seed(seed+271828)
    reference=deepcopy(model).eval()
    for p in reference.parameters():p.requires_grad_(False)
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    cache_star=common.feature_cache(star)
    cache_pi=common.feature_cache(pi)
    ref_star=common.ref_diff(reference,cache_star)
    ref_pi=common.ref_diff(reference,cache_pi)
    rng=np.random.default_rng(seed+2027)
    cost=Counter()
    updates=[]
    start=time.perf_counter()
    for step in range(UPDATES):
        model.train()
        if method=="action_sft":
            loss=common.optimize_sft(model,cf_cache,opt)
        elif method in ("qstar_dpo","frozen_qpi_dpo"):
            cache=cache_star if method=="qstar_dpo" else cache_pi
            ref_delta=ref_star if method=="qstar_dpo" else ref_pi
            opt.zero_grad()
            loss_tensor=(common.supervised_loss(model,cache,"dpo",ref_delta)
                +0.35*common.supervised_loss(model,cf_cache,"sft"))
            assert torch.isfinite(loss_tensor)
            loss_tensor.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),3)
            opt.step()
            loss=float(loss_tensor.detach())
        elif method=="terminal_reinforce":
            episodes=[]
            for row in GROUP["train"]:
                ep=run_episode(environment(row),model,rng,greedy=False,collect_logprob=True)
                assert not ep["invalid"]
                episodes.append(ep)
                cost["native_training_episodes"]+=1
                cost["native_training_steps"]+=len(ep["actions"])
                cost["native_model_actions"]+=ep["n_policy_choices"]
                cost["CI_violations_observed_during_training"]+=int(ep["violation"])
            returns=np.array([e["return"] for e in episodes],np.float32)
            baseline=float(returns.mean())
            opt.zero_grad()
            terms=[]
            for ep in episodes:
                if ep["logp"]:
                    terms.append(-(float(ep["return"])-baseline)*
                                 torch.stack(ep["logp"]).sum())
            if terms:
                l=torch.stack(terms).mean()
                assert torch.isfinite(l)
                l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),3)
                opt.step()
                loss=float(l.detach())
            else:
                loss=0.0
                # Preserve EXACT update count; no feedback available.
                cost["zero_nontrivial_action_grad_updates"]+=1
        else:raise ValueError(method)
        if step in (0,19,39,79,119):
            updates.append({"updates":step+1,"loss":round(loss,7)})
    model.eval()
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    out=OUT/"MODEL_WEIGHTS"/f"{method}_seed{seed}.pt"
    torch.save({
        "method":method,"seed":seed,
        "model_state_dict":model.state_dict(),
        "frozen_SFT_start_checkpoint_sha256":meta["checkpoint_sha256"],
        "source_dataset_sha256":meta["source_data_sha256"],
        "train_graph_families":sorted({r["graph"] for r in GROUP["train"]}),
        "train_optimizer_steps":UPDATES,
        "no_dev_or_test_labels_used_in_training":True,
        "Qpi_training_teacher_checkpoint_sha256":(
             meta["checkpoint_sha256"] if method=="frozen_qpi_dpo" else None),
    },out)
    return model,{
        "method":method,"seed":seed,
        "initial_reference_checkpoint":meta,
        "teacher_pair_audit":teacher_audit,
        "actual_training_cost":dict(cost),
        "optimizer_updates":UPDATES,"training_wall_seconds":round(time.perf_counter()-start,3),
        "model_checkpoint":str(out),
        "model_checkpoint_sha256":sha256(out.read_bytes()).hexdigest(),
        "training_loss_selected_points":updates,
        "feedback_type":{
          "action_sft":"SAME_NATIVE_COUNTERFACTUAL_ACTOR_STATES_OPTIMAL_ACTION_LABELS",
          "qstar_dpo":"EXACT_NATIVE_Qstar_PAIRWISE_TRAIN_ONLY",
          "frozen_qpi_dpo":"EXACT_FROZEN_REFERENCE_POLICY_Qpi_PAIRWISE_TRAIN_ONLY",
          "terminal_reinforce":"CURRENT_ONPOLICY_NATIVE_TERMINAL_REWARD_SAMPLED_ONLY"
        }[method]
    }

results=[]
for seed in SEEDS:
    star,pi,audit=reference_prefs(seed)
    print("TRAIN_REFERENCE",seed,len(star),audit,flush=True)
    for method in METHODS:
        trained,rec=train_one(seed,method,star,pi,audit)
        # Do not inspect heldout scores to modify training: report only.
        for split in ("train","dev","exploratory_test"):
            summary,traces=common.evaluate_model(trained,GROUP[split],per_episode=True)
            rec[split]={"actual_native_greedy":summary,
                        "actual_native_actions":traces}
        results.append(rec)
        with (OUT/"TRAINING_EVALUATIONS.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(rec,ensure_ascii=False,separators=(",",":"),sort_keys=True)+"\n")
        print("ACTUAL_NATIVE",method,seed,
          "train",rec["train"]["actual_native_greedy"]["safe_success"],
          "dev",rec["dev"]["actual_native_greedy"]["safe_success"],
          "test",rec["exploratory_test"]["actual_native_greedy"]["safe_success"],
          "leaks",rec["exploratory_test"]["actual_native_greedy"]["ci_violations"],
          "Aoptimal",rec["exploratory_test"]["actual_native_greedy"]["A_optimal_action_given_eval_truth"],
          "time",rec["training_wall_seconds"],flush=True)
assert len(results)==len(METHODS)*len(SEEDS)
summary={
   "status":"FOUR_WAY_SHARED_CPU_AGENT_REAL_NATIVE_TRAINING_COMPLETE",
   "methods":METHODS,"seeds":SEEDS,
   "source_graph_families":{"train":sorted({r["graph"] for r in GROUP["train"]}),
       "dev":sorted({r["graph"] for r in GROUP["dev"]}),
       "exploratory_test":sorted({r["graph"] for r in GROUP["exploratory_test"]})},
   "train_worlds":16,"dev_worlds":4,"exploratory_test_worlds":8,
   "initial_checkpoint_same_within_seed_for_four_methods":True,
   "optimizer":"Adam","lr":LR,"updates_each":UPDATES,
   "SFT_equal_expanded_counterfactual_actor_state_coverage":len(expanded_sft),
   "train_PREFERENCE_State_count_per_seed":len(star),
   "train_Qpi_reference_is_original_frozen_SFT_before_updates":True,
   "train_Qpi_from_exact_native_πref_not_Qstar":True,
   "test_Qstar_labels_never_used_to_construct_training_pairs":True,
   "terminal_rl_only_selected_action_native_reward":True,
   "checkpoint_count":len(results),
   "frozen_source_world_dataset_sha256":sha256((DATA/"CHAIN_CAUSAL_PAIRS_EVAL_ONLY.jsonl").read_bytes()).hexdigest(),
   "environment_source_modified":False,
   "not_Qwen_training":True,
   "results":[{k:v for k,v in x.items() if k not in ("train","dev","exploratory_test")}
              for x in results],
}
(OUT/"TRAINING_SUMMARY.json").write_text(json.dumps(
    summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("FINAL_TRAINING",json.dumps({"status":summary["status"],
        "all_runs":len(results),"sample_count":summary["train_PREFERENCE_State_count_per_seed"],
        "methods":METHODS},ensure_ascii=False),flush=True)
