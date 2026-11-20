r"""Repair-aware rerun of ONLY CCE+DPR with conservative actual failure backtracking.

This explicitly improves the initial source-only DPR generator:
if immediate source correction unavailable, backtracks to actual B preflight
or same-stage C task-submission decision; never blames the legal upstream
RecordsAgent A. Each proposed training pair is audited from the SAME real
state/actor by native value comparison with a verified successful alternative.

The other methods remain unchanged and their stored results are frozen.
Same 3 seeds, same initialized actor/common SFT warmstart, 120 finetune updates.
"""
from pathlib import Path
from hashlib import sha256
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
import compare_kci_chain_methods_cpu as x
from chain_dpr_backtrack_v2 import collect
OUT=ROOT/"artifacts"/"20261120_v31"/"CHAIN_DPR_BACKTRACKED_POLICY_COMPARISON_02"
if OUT.exists():raise FileExistsError(OUT)
x.compute_DPR_from_failures=collect
common,coverage,high,source_count=x.build_expert_dataset(x.GROUP["train"])
results=[]
for seed in x.SEEDS:
    score=x.learning_run(seed,"cce_reference_dpo_plus_dpr_proxy",
                         common,coverage,high,x.GROUP)
    assert score["DPR_actual_failure_verified_unique_pairs"]>0
    assert all(role!="RecordsAgent" for role in score["DPR_roles"])
    results.append(score)
    print("DPR_BACKTRACK",seed,score["scores"]["exploratory_test"],
          "train_unique_repair_pairs",score["DPR_actual_failure_verified_unique_pairs"],
          "actual_reference_calls",score["training_native_calls"].get("DPR_reference_calls"),
          flush=True)
OUT.mkdir(parents=True)
source=ROOT/"artifacts"/"20261120_v31"/"CHAIN_SHARED_POLICY_METHOD_COMPARISON_01"/"RESULTS.json"
original=json.loads(source.read_text(encoding="utf-8"))
baseline=[p for p in original["per_seed_and_heldout_full_native_model_action_paths"]
          if p["mode"]=="cce_reference_dpo_proxy"]
assert len(baseline)==3
report={
    "status":"ACTUAL_NATIVE_DPR_BACKTRACKED_SHARED_ACTOR_COMPARISON",
    "source_original_comparison_sha256":sha256(source.read_bytes()).hexdigest(),
    "only_changed_factor":"verified DPR actual failed-policy same-state backtracking to downstream roles",
    "CCE_native_Qstar_preference_data_unchanged":True,
    "same_initial_model_seed_and_optimizer_and_updates":True,
    "DPR_actor_roles_exclude_legal_upstream_RecordsAgent":True,
    "seeds":list(x.SEEDS),
    "per_seed_optimized_models":results,
    "matching_CCE_only_baselines":baseline,
    "not_full_LLM_CCE_DPR_or_trained_Qwen":True,
    "not_model_Qpi_estimate":True,
    "all_supervision_norms_synthetic":True,
}
(OUT/"RESULTS.json").write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
print("FINAL_DPR_COMPARE",json.dumps({
    "status":report["status"],
    "data":{str(r["seed"]):{
        "test_success":r["scores"]["exploratory_test"]["safe_success"],
        "test_leaks":r["scores"]["exploratory_test"]["ci_violations"],
        "DPR_unique_pairs":r["DPR_actual_failure_verified_unique_pairs"],
        "DPR_rollouts":r["training_native_calls"].get("native_failed_policy_probe_episodes",0),
        "DPR_searches":r["training_native_calls"].get("DPR_reference_calls",0),
        "roles":r["DPR_roles"]} for r in results}
},ensure_ascii=False),flush=True)
