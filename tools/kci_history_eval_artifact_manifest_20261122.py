"""Immutable audit manifest for a completed history-evaluator development experiment."""
from pathlib import Path
from hashlib import sha256
import json
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/"artifacts"/"20261122_history_evaluator_dev_v1"
MODELS=(
 "train_visible_dpo_s7","train_blind_dpo_s7",
 "train_visible_ce_s7","train_blind_ce_s7",
 "train_visible_strict16_ce_s7","train_blind_strict16_ce_s7")
def main():
    for model in MODELS:
        p=D/model/"adapter"/"adapter_model.safetensors"
        assert p.exists() and p.stat().st_size>10_000_000,(str(p),p.stat().st_size if p.exists() else 0)
        assert (D/model/"TRAINING.json").exists()
    assert (D/"FINAL_DEV_CELL_SUMMARY.json").exists()
    assert (ROOT/"docs"/"KCI_HISTORY_EVALUATOR_DEV_20261122_CN.md").exists()
    names=sorted((p for p in D.rglob("*") if p.is_file() and p.name!="ARTIFACT_SHA256.json"),key=lambda p:str(p))
    files=[]
    for p in names:
        data=p.read_bytes()
        files.append({"relative_path":p.relative_to(D).as_posix(),
                      "size_bytes":len(data),"sha256":sha256(data).hexdigest()})
    x={"n_files":len(files),"n_model_checkpoints":len(MODELS),
       "report":"docs/KCI_HISTORY_EVALUATOR_DEV_20261122_CN.md",
       "source_root":"D:\\research\\longhistory",
       "files":files}
    out=D/"ARTIFACT_SHA256.json"
    assert not out.exists()
    out.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("ARTIFACTS_VERIFIED",len(files),"MODEL_CHECKPOINTS",len(MODELS))
    print("WROTE",out)
if __name__=="__main__":main()
