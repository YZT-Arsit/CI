"""Validate shared-semantic-action development evidence copied back to Windows."""
from pathlib import Path
from hashlib import sha256
import json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts"/"20261123_shared_semantic_action_v1"
def main():
    assert OUT.is_dir()
    for name in ("train_shared_release8_seed7","train_shared_joint3_seed7","train_shared_DPO_CI_future_seed7"):
        folder=OUT/name
        assert (folder/"TRAINING.json").exists(),name
        weights=folder/"adapter/adapter_model.safetensors"
        assert weights.exists() and weights.stat().st_size==11815504,(name,weights.stat().st_size if weights.exists() else 0)
        for filename in ("adapter_config.json","README.md"):
            assert (folder/"adapter"/filename).exists()
    for folder,files in {
        "eval_shared_release8_seed7":["DECISIONS.jsonl","SUMMARY.json","PER_ROOT.json"],
        "eval_shared_joint3_seed7":["DECISIONS.jsonl","SUMMARY.json","PER_ROOT.json"],
        "eval_shared_DPO_CI_future_seed7":["DECISIONS.jsonl","SUMMARY.json","PER_ROOT.json"],
        "native_shared_joint3_seed7":["WORLD_EPISODES.jsonl","SUMMARY.json"],
        "native_shared_DPO_CI_future_seed7":["WORLD_EPISODES.jsonl","SUMMARY.json"],
    }.items():
        for filename in files:assert (OUT/folder/filename).exists(),(folder,filename)
    reports={
        "CE_joint":json.loads((OUT/"native_shared_joint3_seed7/SUMMARY.json").read_text(encoding="utf8")),
        "DPO_joint":json.loads((OUT/"native_shared_DPO_CI_future_seed7/SUMMARY.json").read_text(encoding="utf8"))
    }
    assert sum(x["safe_task_success"] for x in reports["CE_joint"]["splits"].values())==26
    assert sum(x["CI_violations"] for x in reports["CE_joint"]["splits"].values())==2
    assert sum(x["safe_task_success"] for x in reports["DPO_joint"]["splits"].values())==28
    assert sum(x["CI_violations"] for x in reports["DPO_joint"]["splits"].values())==0
    files=sorted(p for p in OUT.rglob("*") if p.is_file() and p.name!="ARTIFACT_SHA256.json")
    manifest=[]
    for p in files:
        h=sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda:f.read(2**20),b""):h.update(chunk)
        manifest.append({"relative_path":p.relative_to(OUT).as_posix(),
             "size_bytes":p.stat().st_size,"sha256":h.hexdigest()})
    summary={
        "status":"LOCAL_EVIDENCE_AND_MODEL_CHECKPOINTS_COMPLETE",
        "n_files":len(files),"n_full_LoRA_checkpoints":3,
        "paper_architecture":"single_shared_base_and_one_active_LoRA_per_run",
        "method_note":"CE calibration and native Qstar teacher DPO are method diagnostics, not complete paper CCE+DPR",
        "source_v31":"artifacts/20261120_v31/V31_GATEA_FINAL_CANDIDATE",
        "source_history":"artifacts/20261122_history_evaluator_dev_v1",
        "report":"docs/KCI_SHARED_SEMANTIC_POLICY_20261123_CN.md",
        "files":manifest}
    result=OUT/"ARTIFACT_SHA256.json"
    assert not result.exists()
    result.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("AUDIT_OK",len(files),"FILES","THREE_LORA_SAFETENSORS_VERIFIED",
          "CE_SUCCESS",26,"DPO_SUCCESS",28,"MANIFEST",result)
if __name__=="__main__":main()
