r"""Independent evidence replay of real frozen LLM decisions and native outcomes.

Only reads previously saved Ollama outputs; never calls Ollama, GPU, or API.
The model's selected actions must be the exact legal actions supplied by the
actual native observation; replayed states, final business record, reward and
receiver knowledge must match exactly. No writing to the source dataset.
"""
from pathlib import Path
from copy import deepcopy
from collections import Counter,defaultdict
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
sys.path.insert(0,str(ROOT/"tools"))
from kci.diverse_ci_branch_env import task_catalog,FINAL,digest
from kci.diverse_ci_long_horizon_env import LongHorizonCIEnv
from evaluate_frozen_kci_ollama import (
    checkpoint_source,summarize,selected_model_metadata)
CATALOG={t.id:t for t in task_catalog()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run",type=Path,required=True)
    ap.add_argument("--result-name",default="INDEPENDENT_REPLAY_AUDIT.json")
    args=ap.parse_args()
    folder=args.run.resolve()
    report=folder/args.result_name
    if report.exists():raise FileExistsError(report)
    meta=json.loads((folder/"CONFIG.json").read_text(encoding="utf-8"))
    rows=[json.loads(x) for x in (folder/"RESULTS.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    assert rows
    checks=Counter()
    err=[]
    for r in rows:
        env=LongHorizonCIEnv(CATALOG[r["task_id"]],r["history"],r["condition"],
                             r["authorized"])
        if r["experiment"]=="source_probe":
            state,route=checkpoint_source(env,r["history"],r["condition"],r["authorized"])
            assert route==r["route"] and state is not None
            observation=env.observation(state)
            m=r["model"]
            assert m["observation_sha256"]==digest(observation)
            assert m["observation"]==observation
            assert m["action_candidates"]==env.options(state) or (
                m["action_candidates"]==list(env.options(state)))
            assert m["context"]["task"]==r["task_id"]
            raw,raw_a=env.transition(state,"SEND_RAW_R")
            assert raw_a["verdict"]==r["eval_only_raw_candidate_verdict"]
            if m["selected_action"] is not None:
                assert m["selected_action"] in env.options(state)
                nxt,a=env.transition(state,m["selected_action"])
                expected=r["actual_action_native"]
                assert expected and a["verdict"]==expected["verdict"]
                assert nxt.violation==expected["violation"]
                assert nxt.steps==expected["step"]
            checks["source_probes_replayed"]+=1
        elif r["experiment"]=="full_episode":
            state=env.initial()
            for i,step in enumerate(r["trace"]):
                if "model" not in step:raise AssertionError("Missing model decision in trace")
                model=step["model"]
                obs=env.observation(state)
                assert model["observation_sha256"]==digest(obs),(r["task_id"],i)
                assert model["observation"]==obs,(r["task_id"],i)
                assert model["context"]["agent_step"]==i
                assert model["action_candidates"]==list(env.options(state))
                if model["selected_action"] is None:
                    break
                assert model["selected_action"] in env.options(state)
                nxt,a=env.transition(state,model["selected_action"])
                assert a==step["native_audit"],(r["task_id"],i)
                assert state.steps==step["stage_before"]["steps"]
                assert nxt.steps==step["steps_after"]
                if a["verdict"] not in ("VALID","VIOLATION"):break
                state=nxt
                checks["genuine_model_native_actions_replayed"]+=1
            assert state.success==r["native_success"],r["task_id"]
            assert state.violation==r["native_violation"]
            assert state.done==r["native_done"]
            assert state.steps==r["native_steps"]
            assert state.business==r["final_business"]
            assert state.generated_result==r["final_result"]
            assert state.ledger==r["final_actual_deliveries"]
            checks["full_episodes_replayed"]+=1
        else:raise ValueError("Unknown experiment")
    s=summarize(rows)
    existing=folder/"SUMMARY.json"
    if existing.exists():
        saved=json.loads(existing.read_text(encoding="utf-8"))
        assert saved["summary"]==s
    result={
        "status":"INDEPENDENT_FROZEN_POLICY_NATIVE_REPLAY_PASS",
        "model_name":meta["model"]["model_name"],
        "model_digest":meta["model"]["model_digest"],
        "model_is_local_frozen":meta["freeze"],
        "paid_API_calls":0,
        "data_source_inspected_synthetic_not_blind":True,
        "counts":dict(checks),
        "computed_metrics":s,
        "no_training":True,
        "no_oracle_labels_in_actor_calls":True,
    }
    report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="computed_metrics"},
                     ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__":main()
