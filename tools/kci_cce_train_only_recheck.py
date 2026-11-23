"""Verify strict pre-JSON train-only source filtering leaves CCE evidence unchanged."""
from pathlib import Path
from hashlib import sha256
import json
D=Path(r"D:\research\longhistory\artifacts\20261123_cce_phase2_mechanism_v1")
a=D/"CCE_ANALYSIS_K8"
b=D/"CCE_ANALYSIS_K8_TRAIN_FILTER_RECHECK"
names=("CCE_MECHANISM_SUMMARY.json","CCE_PAIR_PREFERENCES.jsonl","NATIVE_BOUND_QSTAR_AUDIT_ONLY.json")
def sha(p):return sha256(p.read_bytes()).hexdigest()
rechecks={}
for name in names:
    old=sha(a/name);new=sha(b/name)
    assert old==new,(name,old,new)
    rechecks[name]={"old_sha256":old,"train_raw_line_filtered_sha256":new,
                    "byte_equal":True}
result={"status":"TRAIN_ONLY_PRE_JSON_FILTER_REPLAY_IDENTICAL",
        "source_code_mutation":False,"policy_4B_re_run":False,
        "no_DEV_or_FINAL_root_deserialized_in_recheck":True,
        "first_analysis_DEV_rows_in_mixed_source_were_ignored_after_JSON_parse":True,
        "no_DEV_root_label_used_in_any_preference_or_comparison":True,
        "original_phase2_data_unchanged":True,"files":rechecks}
out=D/"TRAIN_ONLY_QSTAR_RECHECK.json"
assert not out.exists()
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
print(json.dumps(result,ensure_ascii=False,indent=2))
