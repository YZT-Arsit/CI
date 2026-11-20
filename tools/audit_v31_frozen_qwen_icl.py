r"""Independent no-model native replay of local frozen Qwen2.5 named actions.

Reconstruct the actual signed synthetic V3.1 preflight observations and
execute the exact model-selected action under the same documented native
suffix. Only after native replay compare to heldout Qstar label. No model call.
"""
from pathlib import Path
from hashlib import sha256
from collections import Counter,defaultdict
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"));sys.path.insert(0,str(ROOT/"tools"))
from kci.diverse_ci_branch_env import task_catalog,digest
from kci.diverse_ci_long_horizon_v31 import V31Env,enumerate_specs_for_root
from run_v31_shortcut_learning_gate import actual_policy
from solve_v31_longterm import before_preflight
DATA=ROOT/"artifacts"/"20261120_v31"/"V31_GATEA_FINAL_CANDIDATE"
RUN=ROOT/"artifacts"/"20261120_v31"/"FROZEN_QWEN25_V31_PRELFIGHT_01"
OUT=RUN/"INDEPENDENT_NATIVE_REPLAY_AUDIT.json"
if OUT.exists():raise FileExistsError(OUT)
def read(folder,name):
    return [json.loads(s) for s in (folder/name).read_text(encoding="utf-8").splitlines() if s.strip()]
rows=read(RUN,"MODEL_ACTUAL_DECISIONS.jsonl")
assert len(rows)==60
cases={x["case_id"]:x for x in read(DATA,"CASE_SPECS.jsonl")}
prompts={x["world_id"]:x for x in read(DATA,"MODEL_OBSERVATIONS.jsonl")}
truth={x["world_id"]:x for x in read(DATA,"EXACT_NATIVE_ACTION_PREFERENCES_EVAL_ONLY.jsonl")}
tasks={x.id:x for x in task_catalog()}
counts=Counter()
for x in rows:
    key=x["world_id"]
    case_id,hist=key.split("__")
    case=cases[case_id]
    assert x["split"] in ("dev","exploratory_test")
    assert case["split"]==x["split"]
    spec=enumerate_specs_for_root(tasks[case["root_id"]],
            case["business_condition"],int(case_id.split("_M")[-1])-1)
    env=V31Env(spec,hist)
    state,actual=before_preflight(env)
    assert digest(actual)==x["native_state_sha256"]
    assert actual==prompts[key]["model_input"]
    selected=x["selected_action"]
    assert selected in actual["legal_actions"] and x["valid_action"]
    if selected in ("OPEN_PRIVATE_CHANNEL","SKIP_PRIVATE_CHANNEL"):
        native=actual_policy(env,selected)
        assert native==x["actual_native_terminal"],(key,x["variant"])
    gold=truth[key]["oracle_action"]
    assert x["eval_only_native_oracle_action"]==gold
    assert x["correct"]==(selected==gold)
    counts[x["variant"]]+=1
assert counts==Counter({"zero_shot":30,"four_train_examples":30})
assert json.loads((RUN/"SUMMARY.json").read_text(encoding="utf-8"))["total_model_actions"]==60
out={
    "status":"INDEPENDENT_FROZEN_NAMED_ACTION_NATIVE_REPLAY_PASS",
    "qwen_model_name":"qwen2.5:7b-instruct-q4_K_M",
    "Qwen3_4B_original_checkpoint_tested":False,
    "zero_shot_actions":30,"four_train_example_actions":30,
    "all_60_native_preflight_actions_replayed":True,
    "dev_and_exploratory_tests_are_developer_seen_not_blind":True,
    "model_weights_updated":False,
    "no_paid_API":True,
    "source_records_sha256":sha256((RUN/"MODEL_ACTUAL_DECISIONS.jsonl").read_bytes()).hexdigest(),
}
OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)
