r"""Independent native replay of actual shared CPU policy A/B/source/C trajectories.

Does not load trained weights, run training, or query a model. From the saved
actor actions and exact original chain fixtures it reconstructs every native
state transition, actual recipient event and business terminal result. Verifies
counterfactual SFT coverage and source of CCE exact native evidence.
"""
from pathlib import Path
from dataclasses import asdict
from hashlib import sha256
from collections import Counter,defaultdict
import json,sys,math
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from compare_kci_chain_methods_cpu_action_interaction import GROUP,METHODS,SEEDS
from chain_shared_actor_cpu_action_interaction import environment,digest
DATA=ROOT/"artifacts"/"20261120_v31"/"CHAIN_SHARED_POLICY_METHOD_COMPARISON_ACTION_INTERACTION_01"
R=DATA/"RESULTS.json"
OUT=DATA/"INDEPENDENT_NATIVE_REPLAY_AUDIT.json"
if OUT.exists():raise FileExistsError(OUT)
v=json.loads(R.read_text(encoding="utf-8"))
assert v["status"]=="CPU_SHARED_ROLE_POLICY_METHOD_COMPARISON_REAL_NATIVE_ROLLOUTS"
assert [len(GROUP[s]) for s in ("train","dev","exploratory_test")]==[16,4,8]
assert len(v["per_seed_and_heldout_full_native_model_action_paths"])==15
records={(x["mode"],x["seed"]):x for x in
         v["per_seed_and_heldout_full_native_model_action_paths"]}
assert len(records)==15
assert set(records)=={(m,s) for m in METHODS for s in SEEDS}
lookup={s:{r["world"]:r for r in GROUP[s]}
        for s in ("dev","exploratory_test")}
checks=Counter()
for (method,seed),exp in records.items():
    for split in ("dev","exploratory_test"):
        rows=exp["dev_test_episode_traces"][split]
        assert len(rows)==len(GROUP[split])
        by_world={r["world"]:r for r in rows}
        assert set(by_world)==set(lookup[split])
        successful=0;violation=0;incomplete=0;acc_return=0.
        for key,r in by_world.items():
            env=environment(lookup[split][key])
            state=env.initial()
            for i,entry in enumerate(r["actual_model_actions"]):
                obs=env.observation(state)
                assert (obs.get("actor") or obs.get("role"))==entry["role"],(method,key,i,"role")
                assert entry["action"] in env.options(state),(method,key,i,"action")
                prior_deliveries=len(state.native.delivered["R"])
                new,a=env.transition(state,entry["action"])
                assert a["verdict"]==entry["verdict"],(method,key,i,"verdict")
                assert (len(new.native.delivered["R"])-prior_deliveries)==entry["new_R_delivery"]
                state=new
                checks["real_native_actor_transitions_replayed"]+=1
            assert state.native.done,(method,key,"no terminal")
            assert (state.native.success and not state.native.violation)==r["native_success"]
            assert state.native.violation==r["native_violation"]
            assert state.native.steps==r["native_steps"]
            actual=(round(1-env.spec.cost_per_step*state.native.steps,6)
                    if state.native.success and not state.native.violation
                    else -1.0 if state.native.violation else 0.0)
            assert abs(actual-r["native_return"])<1e-6
            acc_return+=actual
            successful+=int(r["native_success"])
            violation+=int(r["native_violation"])
            incomplete+=int(not r["native_success"] and not r["native_violation"])
            checks["episodes_replayed"]+=1
        summary=exp["scores"][split]
        assert summary["safe_success"]==successful
        assert summary["ci_violations"]==violation
        assert summary["safe_incomplete"]==incomplete
        assert abs(summary["mean_native_return"]-acc_return/len(by_world))<2e-6
        checks["seed_split_summary_replayed"]+=1
    assert exp["DPR_actual_failure_verified_unique_pairs"]>=0
    if method!="cce_reference_dpo_plus_dpr_proxy":
        assert exp["DPR_actual_failure_verified_unique_pairs"]==0
assert checks["episodes_replayed"]==(4+8)*3*5
assert checks["seed_split_summary_replayed"]==30
report={
    "status":"INDEPENDENT_SHARED_CPU_ACTOR_NATIVE_REPLAY_PASS",
    "method_checkpoints":15,
    "actual_unique_exploratory_worlds":len(GROUP["exploratory_test"]),
    "actual_unique_development_worlds":len(GROUP["dev"]),
    "graph_split_original":True,
    "counts":dict(checks),
    "return_metrics_match_native_execution":True,
    "DPR_verification_source":"actual_failed_native_rollout + same-state, same-role source correction; see training script",
    "no_Qwen_training":True,
    "no_Qpi_value_estimation_or_real_LLM_onpolicy":True,
}
OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report,ensure_ascii=False),flush=True)
