"""Re-measure a fixed checkpoint on previously opened DEV after numerical repair.

No checkpoint selection, training, new data, or sealed final-test access.
"""
from pathlib import Path
from hashlib import sha256
import argparse
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"code"), str(ROOT/"tools")]


def main():
    p = argparse.ArgumentParser()
    for name in ("base", "adapter", "data-root", "precision-audit", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    gate = json.loads(a.precision_audit.read_text())
    assert gate["matched_reload_pass"] and gate["strict_correct"]["matched"] == 9
    assert sha256((a.adapter/"adapter_model.safetensors").read_bytes()).hexdigest() == gate["adapter_sha256"]
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("GPU busy; did not load model")
    # Load the repaired runtime before importing historical evaluation tools.
    from kci.agent_rl.hf_policy import HFActionPolicy
    from kci_v4_semantic_action import scoring
    from kci_v4_pairwise_strict_gate import exact_action
    from kci_v4_semantic_eval import source_states
    from kci_v4_paired_repair_full_agent import SingleSharedRepair
    from kci_v4_semantic_full_agent import run_env, spec_table, read
    import torch
    assert HFActionPolicy.__init__.__kwdefaults__["kbit_inference_precision"] == "training"
    train = read(a.data_root/"TRAIN_HISTORY_AWARE.jsonl")
    assert sha256((a.data_root/"TRAIN_HISTORY_AWARE.jsonl").read_bytes()).hexdigest() == gate["training_input_sha256"]
    dev = read(a.data_root/"DEV_DUAL_VIEW_DG34_DG35.jsonl")
    assert {r["root"] for r in dev} == {"DG34", "DG35"}
    states = [r for r in source_states(train, dev) if r["split"] == "dev"]
    start = time.monotonic()
    policy = SingleSharedRepair(str(a.base), str(a.adapter))
    records = []
    with torch.inference_mode():
        for row in states:
            for view, prompt in (("native", row["prompt_native"]), ("entity_alias", row["prompt_alias"])):
                acts, logits = scoring(policy.hf, prompt, row["options"])
                scores = dict(zip(acts, logits.float().cpu().tolist()))
                choice = exact_action(scores)
                records.append(dict(root=row["root"], condition=row["condition"], view=view,
                                    gold=row["gold"], selected=choice, correct=choice==row["gold"], scores=scores))
    specs = spec_table(a.data_root)
    worlds = read(a.data_root/"DEV_WORLD_CONFIGS.jsonl")
    assert len(worlds) == 10 and {w["root_id"] for w in worlds} == {"DG34", "DG35"}
    episodes = [run_env(specs, w, "fixed_grouped_ce_matched_precision", "native",
                        "autonomous_shared", policy) for w in worlds]
    result = dict(scope="previously_opened_DEV_remeasurement_not_new_holdout",
                  no_training_or_checkpoint_selection=True, sealed_final_test_unread=True,
                  adapter_sha256=gate["adapter_sha256"], runtime_policy_id=policy.hf.policy_id,
                  dev_input_sha256=sha256((a.data_root/"DEV_DUAL_VIEW_DG34_DG35.jsonl").read_bytes()).hexdigest(),
                  dev_correct={view:sum(r["correct"] for r in records if r["view"]==view)
                               for view in ("native","entity_alias")},
                  dev_ties=sum(r["selected"]=="UNRESOLVED_TIE" for r in records),
                  autonomous_success=sum(r["native_actual_success"] for r in episodes),
                  autonomous_violations=sum(r["native_actual_violation"] for r in episodes),
                  autonomous_aborts=sum(r["native_abort"] for r in episodes),
                  autonomous_unknown=sum(r["native_incomplete_due_rejected_or_UNKNOWN_action"] for r in episodes),
                  candidate_forwards=60+policy.calls, elapsed_seconds=time.monotonic()-start)
    a.output.mkdir(parents=True)
    for filename, data in (("DECISIONS.jsonl",records),("WORLD_EPISODES.jsonl",episodes)):
        (a.output/filename).write_text("".join(json.dumps(x)+"\n" for x in data),encoding="utf-8")
    (a.output/"SUMMARY.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
