r"""DECISIVE V3.1 ablation: plain optimal-action BCE vs Q*-margin weighting.

Same frozen 72 training / 18 dev / 36 exploratory knowledge-topology heldout,
same observation adapter, same NeuralRule 4 parameters, identical seeded init,
Adam optimizer, and 160 full-batch updates per condition. We measure actual
parameter deltas and convergence checkpoint curves, not just terminal labels.

Current 'long-term value supervision' is margin-weighted BCE, NOT DPO nor
a numerical Q regression. This experiment asks whether the Q* MAGNITUDES
help beyond an oracle-derived 0/1 action label. A shuffled-margin negative
control randomly reassigns weights, preserving the label and weight histogram.

No model-weight fine-tuning of Qwen or full multi-agent RL is claimed.
"""
from __future__ import annotations
from pathlib import Path
from hashlib import sha256
import json,random,sys,time
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
sys.path.insert(0,str(ROOT/"tools"))
from train_v31_cpu_policy_probe import (
    get_rows,pack,compute_native_terminal_returns,
    NeuralRule,evaluate,DATA as DATASET)
OUTPUT=ROOT/"artifacts"/"20261120_v31"/"P0_LABEL_VS_QMARGIN_BCE_20261120"
if OUTPUT.exists():raise FileExistsError("Refuse overwrite "+str(OUTPUT))
ROWS=get_rows()
native=compute_native_terminal_returns(ROWS)
SETS={s:[r for r in ROWS if r["split"]==s]
      for s in ("train","dev","exploratory_test")}
assert [len(SETS[k]) for k in SETS]==[72,18,36]
FEATS={s:pack(SETS[s]) for s in SETS}
TARGET=torch.tensor([r["y"] for r in SETS["train"]],dtype=torch.float32)
MARGIN=torch.tensor([r["margin"] for r in SETS["train"]],dtype=torch.float32)
assert len(MARGIN)==72
assert all(r["Qstar_open"]!=r["Qstar_skip"] for r in ROWS)
assert all((r["Qstar_open"]>r["Qstar_skip"]) == bool(r["y"]) for r in ROWS)
SEEDS=(7,1234,2025,42,999)
METHODS=("label_BCE","Qmargin_weighted_BCE","shuffled_Qmargin_BCE")
CHECKPOINTS=(0,1,2,3,5,8,12,20,30,40,60,80,100,120,160)
N_EPOCHS=160
LR=0.03
results=[]
for seed in SEEDS:
    for method in METHODS:
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        net=NeuralRule()
        init={k:v.detach().clone() for k,v in net.state_dict().items()}
        opt=torch.optim.Adam(net.parameters(),lr=LR)
        shuffle=np.random.default_rng(seed+314159).permutation(len(MARGIN))
        shuffled=MARGIN[torch.as_tensor(shuffle,dtype=torch.long)]
        if method=="label_BCE":
            weights=torch.ones_like(MARGIN)
        elif method=="Qmargin_weighted_BCE":
            weights=0.75+0.25*MARGIN/MARGIN.mean()
        else:
            weights=0.75+0.25*shuffled/shuffled.mean()
        assert len(weights)==len(TARGET)
        assert abs(float(weights.mean())-1.0)<1e-6
        snapshots=[]
        start=time.monotonic()
        for epoch in range(N_EPOCHS+1):
            if epoch in CHECKPOINTS:
                stat={s:evaluate(net,FEATS[s],SETS[s]) for s in SETS}
                snapshots.append({
                    "completed_updates":epoch,
                    "train":stat["train"],
                    "dev":stat["dev"],
                    "exploratory_test":stat["exploratory_test"],
                    "elapsed_training_seconds":round(time.monotonic()-start,4),
                })
            if epoch==N_EPOCHS:break
            net.train();opt.zero_grad()
            logits=net(FEATS["train"])
            loss=(F.binary_cross_entropy_with_logits(
                logits,TARGET,reduction="none")*weights).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),max_norm=5)
            opt.step()
        delta=sum(float((v-init[k]).abs().sum().item())
                  for k,v in net.state_dict().items())
        assert delta>1e-6
        end=snapshots[-1]
        results.append({
            "seed":seed,"condition":method,
            "data_train_hash":sha256((DATASET/"MODEL_OBSERVATIONS.jsonl").read_bytes()).hexdigest(),
            "training_rows":72,"dev_rows":18,"test_rows":36,
            "model_architecture":"NeuralRule same four trainable params with public rule-incidence adapter",
            "updates":160,"optimizer":"Adam","lr":LR,
            "input_signature":sum(x.numel() for x in FEATS["train"].values()),
            "init_seed_shared_across_methods":True,
            "weight_mean":round(float(weights.mean()),8),
            "weight_min":round(float(weights.min()),6),
            "weight_max":round(float(weights.max()),6),
            "changed_parameters_l1":round(delta,6),
            "checkpoints":snapshots,
            "total_train_wall_seconds":round(time.monotonic()-start,4),
            "value_origin":"Exact native Qstar is used ONLY for pair weighting in Qmargin method",
            "not_Qpi_or_LLM":True,
        })
        print("ABLATION",seed,method,"test",end["exploratory_test"]["accuracy"],
              "flip",end["exploratory_test"]["paired_flip_accuracy"],
              "first_checkpoint_test100",next((x["completed_updates"] for x in snapshots
                if x["exploratory_test"]["accuracy"]==1.0),None),
              "SECONDS",round(time.monotonic()-start,3),flush=True)
aggregate={}
for method in METHODS:
    m=[r for r in results if r["condition"]==method]
    aggregate[method]={}
    for split in ("train","dev","exploratory_test"):
        accuracy=[r["checkpoints"][-1][split]["accuracy"] for r in m]
        flip=[r["checkpoints"][-1][split]["paired_flip_accuracy"] for r in m]
        regret=[r["checkpoints"][-1][split]["mean_native_regret_vs_Qstar"] for r in m]
        aggregate[method][split]={
            "mean_accuracy":round(float(np.mean(accuracy)),6),
            "sample_sd_accuracy":round(float(np.std(accuracy,ddof=1)),6),
            "mean_flip_accuracy":round(float(np.mean(flip)),6),
            "mean_regret":round(float(np.mean(regret)),6),
            "per_seed_accuracies":accuracy
        }
    aggregate[method]["learning_speed"]={
        "first_checkpoint_all_36_test_correct_each_seed":[
            next((c["completed_updates"] for c in r["checkpoints"]
                 if c["exploratory_test"]["accuracy"]==1.0),None)
            for r in m],
        "first_checkpoint_all_dev_correct_each_seed":[
            next((c["completed_updates"] for c in r["checkpoints"]
                 if c["dev"]["accuracy"]==1.0),None)
            for r in m],
        "wall_seconds":[r["total_train_wall_seconds"] for r in m],
    }
# Direct paired comparison across seeds: no tuning based on exploratory test
paired=[{
    "seed":seed,
    "accuracy_delta_Qmargin_minus_BCE":round(
        next(r for r in results if r["seed"]==seed and r["condition"]=="Qmargin_weighted_BCE")
            ["checkpoints"][-1]["exploratory_test"]["accuracy"]-
        next(r for r in results if r["seed"]==seed and r["condition"]=="label_BCE")
            ["checkpoints"][-1]["exploratory_test"]["accuracy"],6),
    "regret_delta_Qmargin_minus_BCE":round(
        next(r for r in results if r["seed"]==seed and r["condition"]=="Qmargin_weighted_BCE")
            ["checkpoints"][-1]["exploratory_test"]["mean_native_regret_vs_Qstar"]-
        next(r for r in results if r["seed"]==seed and r["condition"]=="label_BCE")
            ["checkpoints"][-1]["exploratory_test"]["mean_native_regret_vs_Qstar"],6),
} for seed in SEEDS]
report={
    "status":"ACTUAL_MATCHED_CPU_ABLATION_COMPLETED",
    "research_question":"Do Qstar VALUE MAGNITUDES add incremental benefit over gold optimal action labels?",
    "neural_model":"NeuralRule 4 trainable scalars; explicit Horn-rule message-passing inductive bias",
    "fixed_cases":"KCI V3.1 original 63 mechanisms, balanced 126 states; graph-family split",
    "fixed_seed_set":SEEDS,
    "methods":METHODS,
    "epochs_each":N_EPOCHS,
    "training_examples":72,
    "optimizer_and_lr_equal":True,
    "same_seed_init":True,
    "train_labels_equal_across_conditions":True,
    "only_difference":("per-instance BCE weight: 1.0 vs "
        "0.75+0.25*abs(Qstar open-skip)/mean margin vs shuffled weight"),
    "magnitude_labels_generated_with_CPU_native_Qstar":True,
    "cost_of_acquiring_optimal_action_labels_not_free":True,
    "shuffled_control_preserves_all_label_classes":True,
    "immediate_CI_training":"In preflight both actions immediately legal; cannot distinguish correct longterm action",
    "terminal_bandit_comparison_file":"CPU_GATE_B_NEURAL_01/RESULTS.json",
    "seeds":len(SEEDS),
    "aggregate":aggregate,
    "paired_effects":paired,
    "traces":results,
    "no_Qwen_weight_training":True,
    "true_blind_test":False,
    "no_paid_API":True,
    "scientific_conclusion":"To be determined from measured differences, not assumed.",
}
OUTPUT.mkdir(parents=True)
(OUTPUT/"ABLATION_RESULTS.json").write_text(json.dumps(
    report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("SUMMARY",json.dumps({
    "status":report["status"],
    "aggregate":aggregate,
    "paired_effects":paired,
},ensure_ascii=False),flush=True)
