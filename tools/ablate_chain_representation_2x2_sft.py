r"""2x2 controlled representation bug isolation (ACTUAL CPU native rollouts).

Factors, holding model architecture/training protocol the same except for
the explicit structural factor:
- ORIGINAL soft Horn (false premise firing) vs zero-preserving AND/OR
- no action-risk interaction vs 5 learned action-risk interaction coefficients

All methods use Action SFT only, same seeds, original source train worlds,
same optimizer, same 150 updates, original signed tool observations, and
the same 8 exploratory held-out worlds. This is a POST-HOC repair diagnostic,
not an independent blind test and not a CCE/DPR method result.
"""
from pathlib import Path
from hashlib import sha256
import json,sys,statistics,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
import compare_kci_chain_methods_cpu as old
import compare_kci_chain_methods_cpu_fixed as only_logic
import compare_kci_chain_methods_cpu_soft_action_interaction as only_interaction
import compare_kci_chain_methods_cpu_action_interaction as both
OUT=ROOT/"artifacts"/"20261120_v31"/"CHAIN_CPU_HORN_RULE_BY_ACTION_INTERACTION_2X2_01"
if OUT.exists():raise FileExistsError(OUT)
methods={
    "ORIGINAL_soft_infer_no_interaction":old,
    "FIXED_logic_only":only_logic,
    "INTERACTION_only_SOFT_INFER":only_interaction,
    "BOTH_zero_preserving_and_action_interaction":both,
}
source=old.GROUP
for x in methods.values():
    assert [(s,len(x.GROUP[s])) for s in ("train","dev","exploratory_test")]==[
        ("train",16),("dev",4),("exploratory_test",8)]
routine,more,preferences,counters=old.build_expert_dataset(source["train"])
results=[];started=time.perf_counter()
for variant,trainer in methods.items():
    for seed in (7,1234,2025):
        res=trainer.learning_run(seed,"action_sft",routine,more,preferences,source)
        results.append({
            "variant":variant,"seed":seed,
            "train":res["scores"]["train"],
            "dev":res["scores"]["dev"],
            "exploratory_test":res["scores"]["exploratory_test"],
            "training_wall_seconds":res["training_wall_seconds"],
            "same_native_tool_transition_contract":True,
            "no_shared_model_weights_from_Qwen":True,
        })
        print("TWO_BY_TWO",variant,seed,
            "test_success",res["scores"]["exploratory_test"]["safe_success"],
            "test_privacy_violations",res["scores"]["exploratory_test"]["ci_violations"],
            "train_success",res["scores"]["train"]["safe_success"],flush=True)
aggregate={}
for variant in methods:
    x=[r for r in results if r["variant"]==variant]
    aggregate[variant]={
        "train_safe_success_seeds":[r["train"]["safe_success"] for r in x],
        "dev_safe_success_seeds":[r["dev"]["safe_success"] for r in x],
        "exploratory_safe_success_seeds":[r["exploratory_test"]["safe_success"] for r in x],
        "exploratory_privacy_violations_seeds":[r["exploratory_test"]["ci_violations"] for r in x],
        "mean_exploratory_return":round(statistics.mean(
            r["exploratory_test"]["mean_native_return"] for r in x),6),
    }
record={
    "status":"POSTHOC_SFT_TWO_BY_TWO_CAUSAL_CODE_REPAIR_ABLATION_COMPLETED",
    "factors":{
        "zero_preserving_logical_inference":True,
        "action_conditional_public_risk_features":True,
    },
    "data":"frozen original 7 graph-family synthetic A->R->B->C worlds",
    "trained_model":"small 4.5k parameter shared CPU policy, NOT an LLM",
    "methods":list(methods),
    "train_roots":16,"dev_roots":4,"exploratory_roots":8,
    "seeds":[7,1234,2025],
    "same_optimizer_updates_and_model_initial_seed":True,
    "shared_initial_warmup":30,"continued_updates":120,
    "native_source_data_modified":False,
    "test_set_seen_during_debug":True,
    "aggregate":aggregate,"per_seed":results,
    "real_native_reference_data_sha256":sha256((ROOT/"artifacts"/"20261120_v31"/
        "DYNAMIC_UPSTREAM_DOWNSTREAM_CHAIN_PILOT_01"/
        "CHAIN_CAUSAL_PAIRS_EVAL_ONLY.jsonl").read_bytes()).hexdigest(),
}
OUT.mkdir(parents=True)
(OUT/"RESULTS.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("AGG",json.dumps(aggregate,ensure_ascii=False,indent=2),flush=True)
