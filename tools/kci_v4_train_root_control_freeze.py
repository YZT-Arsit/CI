"""Freeze V4 TRAIN-root hold/equivalence + entity-alias controls only.

Original V4 remains READ ONLY; all source rows for DG36 are filtered
before JSON parsing. No evaluation data DG34/35 consumed.
Hold/equivalent-order labels are derived from ORIGINAL native action
verdict and full fixed-suffix evidence. This file is EVAL-ONLY, not a
new training data source.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
from hashlib import sha256
import json,sys
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(r"D:\research\kci_data_v4_pilot_20261122\artifacts\20261122_kci_data_v4_pilot_v1")
OUT=ROOT/"artifacts/20261123_v4_paired_repair_v1"
TRAIN=ROOT/"artifacts/20261123_v4_semantic_control_v1/TRAIN_HISTORY_AWARE.jsonl"
RAW="SEND_RAW_R";VERIFY="VERIFY_PRIVATE_R"
def readtrain(file):
    out=[]
    for line in (SOURCE/file).open(encoding="utf8"):
        if '"DG36' in line or '"final_test"' in line:continue
        if not line.strip():continue
        x=json.loads(line)
        if x.get("split")=="train" and x.get("root_id") in ("DG31","DG32","DG33"):
            out.append(x)
    return out
def main():
    rows=readtrain("MODEL_INPUTS_EVAL_ONLY.jsonl")
    native=readtrain("NATIVE_TRAJECTORIES_EVAL_ONLY.jsonl")
    assert len(rows)==len(native)==15
    nmap={x["world_id"]:x for x in native}
    assert set(nmap)=={x["world_id"] for x in rows}
    aware=[json.loads(x) for x in TRAIN.read_text(encoding="utf8").splitlines() if x.strip()]
    seen={(r["root"],r["condition"]):r for r in aware}
    assert len(seen)==9
    output=[]
    for s in rows:
        root=s["root_id"];world=s["world_id"];cond=world.rsplit("__",1)[1]
        assert root in ("DG31","DG32","DG33")
        assert cond in ("risky","safe","authorized","hold","equivalent_order")
        n=nmap[world]
        assert n["replay_state_hash"]==s["state_sha256"]
        assert s["truth_not_exposed_to_model"]
        ev=n["native_action_evaluations"]
        raw=n["fixed_suffix_native_trajectories"][RAW]
        verify=n["fixed_suffix_native_trajectories"][VERIFY]
        assert ev[VERIFY]["verdict"]=="VALID" and verify["terminal_success"]
        if ev[RAW]["verdict"]=="VIOLATION":
            assert cond=="risky" and raw["terminal_violation"]
            gold=VERIFY
        else:
            assert ev[RAW]["verdict"]=="VALID" and raw["terminal_success"]
            assert raw["execution_steps"]<verify["execution_steps"]
            gold=RAW
        if (root,cond) in seen:
            assert seen[root,cond]["prompt"]==s["model_prompt"]
            assert seen[root,cond]["chosen_action"]==gold
            seen_label=True
        else:
            assert cond in ("hold","equivalent_order")
            seen_label=False
        output.append({
           "root":root,"condition":cond,"role":s["actual_role"],"world":world,
           "train_source_state":seen_label,
           "model_prompt":s["model_prompt"],
           "model_prompt_alias":s["entity_aliased_prompt"],
           "action_menu":s["candidate_actions"],
           "action_menu_swapped":s["option_swapped_actions"],
           "model_prompt_swapped":s["options_swapped_prompt"],
           "native_gold_action":gold,"RAW_verdict":ev[RAW]["verdict"],
           "RAW_safe_completion":bool(raw["terminal_success"]),
           "RAW_native_steps":raw["execution_steps"],
           "VERIFY_safe_completion":bool(verify["terminal_success"]),
           "VERIFY_native_steps":verify["execution_steps"],
           "native_state_hash":s["state_sha256"],
           "native_trace_evidence_grade":n["native_evidence_kind"],
           "NO_DG36_data_and_no_dev_labels":True})
    assert len(output)==15 and Counter(x["condition"] for x in output)=={
        "risky":3,"safe":3,"authorized":3,"hold":3,"equivalent_order":3}
    out=OUT/"TRAIN_ROOT_HOLD_ALIAS_EVAL_ONLY.jsonl"
    if out.exists():raise FileExistsError(out)
    out.write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in output),encoding="utf8")
    manifest={
      "status":"FROZEN_TRAIN_ROOT_CONTROLS_ONLY",
      "source_original_V4_model_inputs":str(SOURCE/"MODEL_INPUTS_EVAL_ONLY.jsonl"),
      "training_main_source_sha256":sha256(TRAIN.read_bytes()).hexdigest(),
      "model_inputs_SHA256":sha256((SOURCE/"MODEL_INPUTS_EVAL_ONLY.jsonl").read_bytes()).hexdigest(),
      "native_trajectories_SHA256":sha256((SOURCE/"NATIVE_TRAJECTORIES_EVAL_ONLY.jsonl").read_bytes()).hexdigest(),
      "train_native_control_only_sha256":sha256(out.read_bytes()).hexdigest(),
      "n_worlds":15,"n_unique_roots":3,
      "n_seen_training_states":9,"n_train_root_not_used_in_training_controls":6,
      "views":["original","entity_alias"],
      "DG34_DG35_DG36_never_parsed_in_output":True,
      "NO_NEW_TRAIN_LABELS":True,
    }
    dest=OUT/"TRAIN_ROOT_HOLD_ALIAS_MANIFEST.json"
    if dest.exists():raise FileExistsError(dest)
    dest.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("TRAIN_ROOT_CONTROLS_FROZEN",json.dumps(manifest,ensure_ascii=False))
if __name__=="__main__":main()
