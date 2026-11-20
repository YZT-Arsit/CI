r"""Reproduce CPU neural training + verify learned rule models use invariant structure.

Verify relative to UNTRAINED random initialization, save model state_dict
and dataset SHA256, then counterfactually rename every ordinary source fact
S01..S06 and permute order of the public Horn clauses/received history while
preserving their meaning. Held-out test labels remain unchanged by the public
semantics (also checked independently using the visible symbolic planner).

A graph-aware model that truly reasons over the relation structure should be
invariant to systematic relabeling of fact IDs. A bag of shortcuts should not.
"""
from pathlib import Path
from hashlib import sha256
from copy import deepcopy
from collections import defaultdict
from dataclasses import asdict
import json,sys,re,random
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
sys.path.insert(0,str(ROOT/"tools"))
from train_v31_cpu_policy_probe import (
    get_rows,pack,compute_native_terminal_returns,NeuralRule,MLP,
    train,evaluate,
)
from audit_kci_v31_gateA import choose_observable_rules
DATA=ROOT/"artifacts"/"20261120_v31"/"V31_GATEA_FINAL_CANDIDATE"
OUT=ROOT/"artifacts"/"20261120_v31"/"CPU_GATE_B_NEURAL_REPRODUCIBLE_03"
if OUT.exists():raise FileExistsError(OUT)
def rename_observation(obs,permutation,reverse_order=True):
    assert sorted(permutation)==list(range(1,7))
    mapping={f"S{i:02d}":f"S{permutation[i-1]:02d}" for i in range(1,7)}
    def walk(x):
        if isinstance(x,str):
            return re.sub(r"S0[1-6]",lambda m:mapping[m.group(0)],x)
        if isinstance(x,list):return [walk(z) for z in x]
        if isinstance(x,dict):return {k:walk(v) for k,v in x.items()}
        return x
    c=walk(deepcopy(obs))
    if reverse_order:
        c["declared_knowledge_rules"]=c["declared_knowledge_rules"][::-1]
        c["delivered_message_log"]=c["delivered_message_log"][::-1]
    return c
rows=get_rows()
native=compute_native_terminal_returns(rows)
g={s:[r for r in rows if r["split"]==s]
   for s in ("train","dev","exploratory_test")}
assert [len(g[s]) for s in g]==[72,18,36]
OUT.mkdir(parents=True)
(OUT/"MODEL_WEIGHTS").mkdir()
permutations=[]
for shift in range(1,6):
    permutations.append(tuple((i+shift)%6+1 for i in range(6)))
for seed in (11,29,47,59,71,89,101,113,127):
    rng=np.random.default_rng(seed)
    candidate=tuple(int(i) for i in rng.permutation(6)+1)
    if candidate not in permutations:permutations.append(candidate)
    if len(permutations)==9:break
assert len(permutations)==len(set(permutations))==9
records=[]
for variant in ("neural_rule_value_pair","mlp_value_pair",
                "neural_rule_terminal_bandit","neural_rule_value_no_history"):
    for seed in (7,1234,2025):
        torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
        mlp=variant.startswith("mlp")
        nohistory="no_history" in variant
        xx={s:pack(g[s],hide_history=nohistory) for s in g}
        model=MLP(xx["train"]["full"].shape[1]) if mlp else NeuralRule()
        untrained=evaluate(model,xx["exploratory_test"],g["exploratory_test"])
        init={k:v.detach().clone() for k,v in model.state_dict().items()}
        algorithm="terminal_bandit" if "terminal_bandit" in variant else "value_pair"
        trace=train(model,xx["train"],g["train"],algorithm,seed,epochs=160,
                    lr=0.01 if mlp else 0.03)
        updates=sum(float((v-init[k]).abs().sum().item())
                    for k,v in model.state_dict().items())
        assert updates>1e-6
        trained={s:evaluate(model,xx[s],g[s]) for s in g}
        augmented=[]
        for i,perm in enumerate(permutations):
            cloned=[]
            for r in g["exploratory_test"]:
                other=dict(r)
                other["obs"]=rename_observation(r["obs"],perm)
                if not nohistory:
                    truth=choose_observable_rules(other["obs"])
                    assert truth==("OPEN_PRIVATE_CHANNEL" if r["y"] else "SKIP_PRIVATE_CHANNEL")
                cloned.append(other)
            score=evaluate(model,pack(cloned,hide_history=nohistory),cloned)
            augmented.append({"permutation":perm,
                "accuracy":score["accuracy"],
                "paired_flip_accuracy":score["paired_flip_accuracy"],
                "mean_regret":score["mean_native_regret_vs_Qstar"]})
        ckpt=OUT/"MODEL_WEIGHTS"/(variant+"_seed"+str(seed)+".pt")
        torch.save({
            "model_state_dict":model.state_dict(),
            "model_type":"MLP" if mlp else "NeuralRule",
            "variant":variant,"seed":seed,
            "training_epochs":160,
            "origin":"CPU trained on KCI V3.1 train knowledge graph only; NOT a Qwen model",
            "data_hash":sha256((DATA/"MODEL_OBSERVATIONS.jsonl").read_bytes()).hexdigest(),
        },ckpt)
        records.append({
            "variant":variant,"seed":seed,
            "untrained_test":untrained,
            "trained_metrics":trained,
            "parameter_l1_change":round(updates,7),
            "9_label_permutations":augmented,
            "worst_permutation_test_accuracy":min(z["accuracy"] for z in augmented),
            "worst_permutation_test_pair_flip":min(z["paired_flip_accuracy"] for z in augmented),
            "model_checkpoint_sha256":sha256(ckpt.read_bytes()).hexdigest(),
            "checkpoint_file":str(ckpt)
        })
        print("ROBUSTNESS",variant,seed,
              "untrained",untrained["accuracy"],
              "test",trained["exploratory_test"]["accuracy"],
              "min_fact_permuted_test",records[-1]["worst_permutation_test_accuracy"],
              "changed_weights_L1",round(updates,3),flush=True)
report={
    "status":"REAL_CPU_TRAINED_MODELS_WITH_NAME_PERMUTATION_AUDIT",
    "records":records,
    "source_dataset_sha256":sha256((DATA/"MODEL_OBSERVATIONS.jsonl").read_bytes()).hexdigest(),
    "models_saved":len(records),
    "fact_position_renamings":len(permutations),
    "all_permuted_truths_certified_actor_visible_rule_closure":True,
    "trained_own_small_PyTorch_models":True,
    "LLM_Qwen_weights_updated":False,
    "GPU_API_paid":0,
    "not_blind_test_and_no_real_domain_natural_language":True,
}
(OUT/"AUDIT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("FINAL_AUDIT",json.dumps({
    "models":len(records),
    "source_dataset_sha256":report["source_dataset_sha256"],
    "neural_rule_value_pair_min_test_accuracy":min(
       x["worst_permutation_test_accuracy"] for x in records
       if x["variant"]=="neural_rule_value_pair"),
    "neural_rule_value_pair_untrained_accuracy":min(
       x["untrained_test"]["accuracy"] for x in records
       if x["variant"]=="neural_rule_value_pair"),
},ensure_ascii=False),flush=True)
