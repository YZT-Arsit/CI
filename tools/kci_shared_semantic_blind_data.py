"""Paired no-history teacher labels on IDENTICAL first-release shared-policy inputs."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/"artifacts/20261123_shared_semantic_action_v1"
H=ROOT/"artifacts/20261122_history_evaluator_dev_v1"
def load(f):
    return [json.loads(s) for s in f.read_text(encoding="utf8").splitlines() if s.strip()]
def main():
    true=load(D/"RELEASE_TRAIN_STRICT8.jsonl")
    blind=load(H/"TRAIN_BLIND_STRICT8.jsonl")
    assert len(true)==len(blind)==8
    by={(x["world"],x["observation_sha256"]):x for x in true}
    copies=[]
    for r in blind:
        p=by[r["world"],r["observation_hash"]]
        assert p["split"]=="train"
        clone={**p,"chosen_action":r["chosen_action"],
               "provenance":"NATIVE_RECEIPT_BLIND_EVALUATOR_COUNTERFACTUAL",
               "teacher_history_access":"none",
               "true_history_aware_label":p["chosen_action"]}
        copies.append(clone)
    assert len(copies)==8
    assert all(x["chosen_action"]=="SEND_RAW_R" for x in copies)
    f=D/"RELEASE_TRAIN_STRICT8_BLIND.jsonl"
    assert not f.exists()
    f.write_text("".join(json.dumps(x,sort_keys=True,ensure_ascii=False)+"\n" for x in copies),encoding="utf8")
    print("BLIND_MATCHED_SHARED_DATA_OK",len(copies),hashlib.sha256(f.read_bytes()).hexdigest())
if __name__=="__main__":main()
