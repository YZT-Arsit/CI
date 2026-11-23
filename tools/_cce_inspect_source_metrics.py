"""Metadata-only source inspection for Phase2 CCE, strict train-only."""
from pathlib import Path
import json,collections
P=Path(r"D:\research\kci_data_v4_phase2_20261122\artifacts\20261122_kci_data_v4_phase2_full_v1\TRAIN_AND_DEV_EXACT_NATIVE_FUTURE_EVAL_ONLY.jsonl")
targets={"P201","P202","P203","P205","P206","P207","P208"}
rows=[]
for line in P.open(encoding="utf8"):
    if '"final_test"' in line or '"P211' in line or '"P212' in line:continue
    x=json.loads(line)
    if x.get("root_id") not in targets:continue
    rows.append(x)
    print("ROOT",x["root_id"],"KEYS",sorted(x),flush=True)
    for kind,v in x["A_branches"].items():
        print(" BRANCH",kind,"KEYS",sorted(v),flush=True)
        print("   A ACTION",v.get("actual_A_action"),"B best costs",v.get("best_bounded_safe_success_cost"),
              "B any safe",v.get("any_CI_safe_business_completion"),flush=True)
        print("   B action results",[(act,z["CI_safe_success"],z["CI_violation"],z["native_total_steps"],z["deadline_missed"]) for act,z in v["exact_B_action_paths"].items()],flush=True)
    print(" gap",x.get("A_minus_B_native_success_steps"),"risky-minus-safe",x.get("risky_minus_safe_extra_native_steps"),flush=True)
assert len(rows)==7
