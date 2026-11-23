"""V4 permitted development evidence schema only; never parse DG36."""
import json,pathlib,collections
P=pathlib.Path(r"D:\research\kci_data_v4_pilot_20261122\artifacts\20261122_kci_data_v4_pilot_v1")
def permitted(name):
    data=[]
    for line in (P/name).open(encoding="utf8"):
        if '"DG36' in line or '"final_test"' in line:continue
        r=json.loads(line)
        if r.get("root_id") in ("DG34","DG35") and r.get("split")=="dev":data.append(r)
    return data
obs=permitted("MODEL_INPUTS_EVAL_ONLY.jsonl")
native=permitted("NATIVE_TRAJECTORIES_EVAL_ONLY.jsonl")
pairs=permitted("CONTROLLED_CONTEXT_PAIRS_EVAL_ONLY.jsonl")
assert len(obs)==len(native)==10,(len(obs),len(native))
print("DEV OBS",[(r["root_id"],r["world_id"],r["actual_role"],r["candidate_actions"]) for r in obs])
print("DEV PAIRS",[(r["pair_id"],r["contrast_type"],r["left_world"],r["right_world"],r["same_candidate_source_message"]) for r in pairs])
for r in native:
    print("\nWORLD",r["world_id"],"ROOT",r["root_id"],"NATIVE_HASH",r["replay_state_hash"])
    print("NATIVE_ACTIONS",[(k, list(v.keys()) if isinstance(v,dict) else type(v).__name__,v.get("verdict") if isinstance(v,dict) else None) for k,v in r["native_action_evaluations"].items()])
    for a,v in r["fixed_suffix_native_trajectories"].items():
        print("SUFFIX",a,{k:v.get(k) for k in ["terminal_done","terminal_success","terminal_violation","execution_steps","outcome_grade"]})
