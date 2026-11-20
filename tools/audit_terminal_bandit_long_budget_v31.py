r"""Reproduce 800-epoch REAL terminal-return contextual bandit result (3 seeds).

Crucially this longer run checks whether sparse native terminal rewards
CAN learn the same long-horizon conditional choice with enough updates.
No paired Qstar labels are read by the optimizer. Both arms' deterministic
native terminal records are cached solely for native reproducible action
replay; each REINFORCE update observes only sampled chosen-arm return.

This is not PPO/GRPO or online LLM training.
"""
from pathlib import Path
from hashlib import sha256
import json,random,sys
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"));sys.path.insert(0,str(ROOT/"code"))
from train_v31_cpu_policy_probe import (
    get_rows,compute_native_terminal_returns,pack,NeuralRule,train,evaluate)
BASE=ROOT/"artifacts"/"20261120_v31"
OUT=BASE/"CPU_GATE_B_TERMINAL_BANDIT_800_EPOCHS"
if OUT.exists():raise FileExistsError(OUT)
rows=get_rows();native=compute_native_terminal_returns(rows)
assert native["episodes"]==252 and native["reward_agrees_with_optimal"]==126
group={s:[r for r in rows if r["split"]==s] for s in (
    "train","dev","exploratory_test")}
assert len(group["train"])==72 and len(group["exploratory_test"])==36
xx={s:pack(group[s]) for s in group}
records=[]
for seed in (7,1234,2025):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=NeuralRule()
    log=train(model,xx["train"],group["train"],"terminal_bandit",seed,epochs=800,lr=0.03)
    metrics={s:evaluate(model,xx[s],group[s]) for s in group}
    records.append({"seed":seed,"method":"ONPOLICY_SAMPLED_ACTION_TERMINAL_RETURN_CONTEXTUAL_BANDIT",
                    "epochs":800,"training_examples":72,"metrics":metrics,
                    "loss_log":log})
    print(seed,metrics["exploratory_test"]["accuracy"],
          metrics["exploratory_test"]["paired_flip_accuracy"],
          metrics["exploratory_test"]["mean_native_regret_vs_Qstar"],flush=True)
assert all(x["metrics"]["exploratory_test"]["accuracy"]==1.0 for x in records)
assert all(x["metrics"]["exploratory_test"]["paired_flip_accuracy"]==1.0 for x in records)
OUT.mkdir(parents=True)
(OUT/"RESULTS.json").write_text(json.dumps({
    "status":"REPRODUCED_LONG_BUDGET_TERMINAL_BANDIT_TRAINING",
    "seeds":[7,1234,2025],"epochs":800,"training_worlds":72,
    "real_training_native_arm_rollouts_cached":native,
    "sampling_per_update":"only chosen action return is observed; no Qstar training targets",
    "factor_relative_to_160_epoch_value_pair":5,
    "terminal_rewards_available_same_as_native_Qstar_for_this_fixed_suffix":True,
    "models_are_small_PyTorch_CPU_not_Qwen":True,
    "LLM_weights_updated":False,
    "methods":[x for x in records],
    "dataset_sha256":sha256((BASE/"V31_GATEA_FINAL_CANDIDATE"/"MODEL_OBSERVATIONS.jsonl").read_bytes()).hexdigest(),
},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("AUDIT_REPRODUCED",flush=True)
