"""V4 one-round native-action results auditor and conditional research gate.

No model fitting, no dev method selection and no DG36 access. Consumes only
the frozen 4-arm evaluation outputs and training manifests.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
import json,argparse
M=("frozen_base","existing_frozen_R","v4_action_ce","v4_dpo")
R="SEND_RAW_R";V="VERIFY_PRIVATE_R"
def load(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf8") if line.strip()]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--eval",type=Path,required=True)
    ap.add_argument("--ce",type=Path,required=True)
    ap.add_argument("--dpo",type=Path,required=True)
    ap.add_argument("--coverage",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    a=json.loads((args.eval/"SUMMARY.json").read_text(encoding="utf8"))
    data=load(args.eval/"MODEL_DECISIONS_ALL.jsonl")
    ce=json.loads(args.ce.read_text(encoding="utf8"))
    dp=json.loads(args.dpo.read_text(encoding="utf8"))
    coverage=json.loads(args.coverage.read_text(encoding="utf8"))
    assert {x["root"] for x in data if x["split"]=="train"}=={"DG31","DG32","DG33"}
    assert {x["root"] for x in data if x["split"]=="dev"}=={"DG34","DG35"}
    assert all("DG36" not in x["root"] and "DG36" not in x["world"] for x in data)
    assert len(data)==19
    assert a["CE_training_manifest_sha256"]==sha256(args.ce.read_bytes()).hexdigest()
    assert a["DPO_training_manifest_sha256"]==sha256(args.dpo.read_bytes()).hexdigest()
    assert ce["steps"]==dp["steps"] and ce["seed"]==dp["seed"] and ce["lr"]==dp["lr"]
    assert ce["independent_comparison_count"]==dp["independent_comparison_count"]==9
    assert ce["original_record_count"]==dp["original_record_count"]==18
    per_root=[]
    for root in ("DG31","DG32","DG33","DG34","DG35"):
        selected=[x for x in data if x["root"]==root]
        assert len(selected)==(3 if root.startswith(("DG31","DG32","DG33")) else 5)
        state={x["condition"]:x for x in selected}
        assert {"risky","safe"}<=set(state)
        row={"root":root,"split":selected[0]["split"],"n_independent_states":len(selected),
             "actor_role":selected[0]["role"],"conditions":sorted(state),"methods":{}}
        for method in M:
            p={c:r["methods"][method] for c,r in state.items()}
            normal=lambda c:p[c]["action_normal"]
            swapped=lambda c:p[c]["action_swapped"]
            robust=lambda c:p[c]["native_correct_both_orders"]
            row["methods"][method]={
                "history_flip_both_orders":bool(robust("risky") and robust("safe")),
                "history_flip_normal":bool(normal("risky")==V and normal("safe")==R),
                "dangerous_RAW_normal":bool(normal("risky")==R),
                "safe_RAW_normal":bool(normal("safe")==R),
                "authorized_RAW_normal":bool(normal("authorized")==R),
                "safe_hold_both_orders":bool(robust("safe") and robust("hold")) if "hold" in state else None,
                "safe_equivalent_both_orders":bool(robust("safe") and robust("equivalent_order")) if "equivalent_order" in state else None,
                "n_position_semantic_flip":sum(x["methods"][method]["position_semantic_flip"] for x in selected),
                "n_correct_normal":sum(x["methods"][method]["native_correct_normal"] for x in selected),
                "n_correct_swapped":sum(x["methods"][method]["native_correct_swapped"] for x in selected),
            }
        per_root.append(row)
    summary=a["by_split"]
    gate={}
    for name in ("v4_action_ce","v4_dpo"):
        s=summary["train"][name]
        correct=s["paired_results"]["history_flip"]["correct_both_option_orders"]
        n=s["paired_results"]["history_flip"]["n_independent_root_pairs"]
        gate[name]={
           "train_both_sides_correct_both_option_orders":correct,
           "train_root_pairs":n,
           "all_train_root_history_pairs_fit":correct==n,
           "normal_and_swapped_train_state_actions_correct":
              s["normal"]["native_gold_action_correct"]==s["normal"]["n_states"] and
              s["swapped"]["native_gold_action_correct"]==s["swapped"]["n_states"],
           "dev_history_root_pairs_correct":
              summary["dev"][name]["paired_results"]["history_flip"]["correct_both_option_orders"],
           "dev_root_pair_denominator":
              summary["dev"][name]["paired_results"]["history_flip"]["n_independent_root_pairs"],
        }
    # No claim that historical 5 root shapes are enough for statistical CI.
    outcome={
      "status":"V4_FROZEN_ONE_ROUND_CONTROL_AUDIT",
      "source_eval_summary_sha256":sha256((args.eval/"SUMMARY.json").read_bytes()).hexdigest(),
      "model_forward_protocol":"same V4 role-visible prompt plus one constant response instruction",
      "source_release_training_unique_same_state_comparisons":9,
      "source_original_normal_and_swap_records":18,
      "source_train_roots":3,"source_dev_roots":2,
      "per_root":per_root,
      "gates":gate,
      "coverage_confounds":{
         "dev_roles_unseen_in_release_training":coverage["dev_roles_unseen_in_train"],
         "dev_graph_mechanisms_unseen_in_training":coverage["dev_graph_mechanisms_unseen_in_train"]},
      "all_eval_is_pre_action_model_inference_then_original_native_fixed_suffix_replay":True,
      "not_current_model_Qpi_suffix":True,
      "bounded_native_Qstar_not_repurposed_as_Qpi":True,
      "no_DG36_test_used":True,
      "no_hyperparameter_search_on_dev":True,
      "CE_optimization_steps":ce["steps"],"DPO_optimization_steps":dp["steps"],
      "CE_total_differentiable_forward_calls":2*ce["steps"],
      "DPO_total_differentiable_forward_calls":2*dp["steps"],
      "DPO_frozen_reference_forward_calls":2*dp["independent_comparison_count"],
      "CE_actual_wall_seconds":ce["wall_seconds"],
      "DPO_actual_wall_seconds":dp["wall_seconds"],
      "frozen_baselines_have_no_this_round_optimization":True,
      "frozen_existing_R_historical_training_204_pairs_128_steps_not_budget_matched":True,
      "evaluation_LM_forward_calls":a["forward_calls_by_method"],
      "evaluation_LM_prompt_tokens":a["prompt_tokens_sum_by_method"],
      "model_initialization":{"frozen_checkpoint":ce["base_model"],"seed":ce["seed"],
          "nf4_bf16":True,"r":ce["lora_rank"],"alpha":ce["lora_alpha"],
          "CE_DPO_same_new_LoRA_initialization_contract":True},
    }
    args.output.mkdir(parents=True)
    (args.output/"AUDIT.json").write_text(json.dumps(outcome,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("FINAL_GATE",json.dumps(gate,ensure_ascii=False))
    print("PER_ROOT",json.dumps(per_root,ensure_ascii=False)[:8500])
if __name__=="__main__":main()
