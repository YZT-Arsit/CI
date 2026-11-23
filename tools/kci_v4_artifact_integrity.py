"""Final V4 release comparison integrity gate, all artifacts local to main tree.

Never parses or evaluates DG36; frozen original V4 source remains read-only.
"""
from pathlib import Path
from hashlib import sha256
import json
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/"artifacts/20261123_v4_release_control_v1"
def read(p):return json.loads(p.read_text(encoding="utf8"))
def hash_file(p):
    h=sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(2**20),b""):h.update(chunk)
    return h.hexdigest()
def main():
    source=read(D/"DATA_MANIFEST.json")
    ce=read(D/"train_release_actionCE_s7/TRAINING.json")
    dp=read(D/"train_release_DPO_s7/TRAINING.json")
    scores=read(D/"eval_release_4arms_s7/SUMMARY.json")
    compute=read(D/"COMPUTE_AND_INITIALIZATION.json")
    audit=read(D/"final_audit/AUDIT.json")
    assert source["total_original_train_records"]==36
    assert source["total_independent_train_comparisons"]==18
    assert source["release_original_train_records"]==18
    assert source["release_independent_train_comparisons"]==9
    assert source["dev_roots"]==["DG34","DG35"]
    assert ce["steps"]==dp["steps"]==96
    assert ce["group_source_file_sha256"]==dp["group_source_file_sha256"]
    assert scores["group_data_sha256"]==ce["group_source_file_sha256"]
    assert audit["gates"]["v4_action_ce"]["train_both_sides_correct_both_option_orders"]==0
    assert audit["gates"]["v4_dpo"]["train_both_sides_correct_both_option_orders"]==0
    for tag,folder in (("CE","train_release_actionCE_s7"),("DPO","train_release_DPO_s7")):
        model=D/folder/"adapter/adapter_model.safetensors"
        assert model.is_file() and model.stat().st_size==11815504
        assert hash_file(model)==compute[tag+"_trained_adapter_weights_sha256"]
        assert (D/folder/"adapter/adapter_config.json").exists()
    train_groups=[json.loads(x) for x in (D/"TRAIN_RELEASE_GROUPS.jsonl").read_text(encoding="utf8").splitlines()]
    dev=[json.loads(x) for x in (D/"DEV_NATIVE_STATES_DG34_DG35.jsonl").read_text(encoding="utf8").splitlines()]
    assert len(train_groups)==9 and len(dev)==10
    assert {r["root"] for r in dev}=={"DG34","DG35"}
    assert all("DG36" not in row["world"] for row in dev)
    assert (ROOT/"docs/KCI_V4_4B_RELEASE_CONTROL_20261123_CN.md").exists()
    # Strict file inventory and sha256. Never claim DG36 test secrecy from hashing original source.
    files=sorted((f for f in D.rglob("*") if f.is_file() and f.name!="ARTIFACT_SHA256.json"),
                 key=lambda f:str(f).lower())
    records=[{"relative_path":f.relative_to(D).as_posix(),
              "bytes":f.stat().st_size,"sha256":hash_file(f)} for f in files]
    manifest={
      "status":"COMPLETE_NEGATIVE_CONTROL_STOP_BEFORE_CCE_DPR",
      "artifact_files":len(records),
      "trained_LoRA_checkpoints":2,
      "original_V4_source_read_only":True,
      "source_final_root_not_deserialized_or_scored":True,
      "mixed_18_independent_groups_supported":True,
      "release_9_independent_groups_trained":True,
      "DG34_DG35_evaluated_once_after_fixed_96_step_budget":True,
      "no_new_DG36_metrics":True,
      "models_train_and_dev_pair_gate_passed":False,
      "compute_manifest":"COMPUTE_AND_INITIALIZATION.json",
      "report":"docs/KCI_V4_4B_RELEASE_CONTROL_20261123_CN.md",
      "files":records}
    out=D/"ARTIFACT_SHA256.json"
    assert not out.exists()
    out.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("V4_FINAL_INTEGRITY_OK",len(files),"FILES","2_FULL_LORA_WEIGHTS",
          "DG36_EXCLUDED","TRAIN_PAIR_GATE_FAIL")
if __name__=="__main__":main()
