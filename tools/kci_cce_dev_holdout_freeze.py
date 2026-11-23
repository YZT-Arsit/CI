"""Freeze independent Phase2 development business roots for method comparison.

Parse only split=dev source lines, BEFORE JSON decoding. No final test
P211/P212 or DG36. Only visible business task/native spec, not model labels.
Used once after three TRAIN-only model checkpoints have been frozen.
"""
from __future__ import annotations
from pathlib import Path
from hashlib import sha256
import json
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(r"D:\research\kci_data_v4_phase2_20261122\artifacts\20261122_kci_data_v4_phase2_full_v1")
D=ROOT/"artifacts/20261124_cce_method_comparison_v1"
TARGETS=("P204","P209","P210")
def sha(p):return sha256(Path(p).read_bytes()).hexdigest()
def main():
    x=[]
    for line in (SOURCE/"NEW_ROOTS.jsonl").open(encoding="utf8"):
        if '"root_split":"dev"' not in line:continue
        assert all('"final_test"' not in line for _ in (0,))
        r=json.loads(line)
        assert r["root_id"] in TARGETS and r["root_split"]=="dev"
        assert r["native_spec"]["split"]=="dev"
        x.append(r)
    assert len(x)==3 and set(y["root_id"] for y in x)==set(TARGETS)
    dest=D/"DEV_UNSEEN_BUSINESS_ROOTS_EVAL_ONLY.jsonl"
    if dest.exists():raise FileExistsError(dest)
    dest.write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in x),encoding="utf8")
    meta={"status":"FROZEN_UNSEEN_BUSINESS_DEV_ROOTS_NOT_TRAIN",
          "roots":list(TARGETS),"split":"dev",
          "n_business_roots":3,
          "evaluation_conditions":["risky_noauth","authorized_risky_package","hold_noauth"],
          "paired_seeds":[7,1234,2025,42,314,2718,1618,99],
          "n_episodes_per_policy":3*3*8,
          "policies":["downstream_sft_untrained_A","action_ce","terminal","CCE"],
          "Phase2_root_file_sha256":sha(SOURCE/"NEW_ROOTS.jsonl"),
          "dev_root_file_sha256":sha(dest),
          "test_source_never_parsed":True,
          "not_unseen_knowledge_rule_guarantee":True,
          "all_actions_including_ABORT_retained":True,
          "no_future_oracle_in_model_prompts":True,
          "real_multi_role_4B_no_scripted_outcomes":True,
          "not_tuning_on_DEV":True}
    (D/"DEV_EVALUATION_PRE_REGISTRATION.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("CCE_DEV_HOLDOUT_FROZEN",json.dumps(meta,ensure_ascii=False))
if __name__=="__main__":main()
