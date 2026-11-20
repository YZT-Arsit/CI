r"""Independent native replay of the improved DPR-backtracking CPU actor results.

Reconstruct each held-out native tool transition from the stored model choices,
verify true R history, business mutations, terminal CI and return. Source runs
are developer-inspected synthetic KCI reference; not original full LLM DPR.
"""
from __future__ import annotations
from pathlib import Path
from hashlib import sha256
from collections import Counter
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from chain_shared_actor_cpu import environment
from compare_kci_chain_methods_cpu import GROUP,SEEDS
BASE=ROOT/"artifacts"/"20261120_v31"/"CHAIN_DPR_BACKTRACKED_POLICY_COMPARISON_02"
R=BASE/"RESULTS.json"
OUT=BASE/"INDEPENDENT_NATIVE_REPLAY_AUDIT.json"
if OUT.exists():raise FileExistsError(OUT)
data=json.loads(R.read_text(encoding="utf-8"))
assert data["status"]=="ACTUAL_NATIVE_DPR_BACKTRACKED_SHARED_ACTOR_COMPARISON"
lookup={s:{x["world"]:x for x in GROUP[s]} for s in ("dev","exploratory_test")}
assert len(data["per_seed_optimized_models"])==3
checks=Counter()
for record in data["per_seed_optimized_models"]:
    assert record["seed"] in SEEDS
    assert record["DPR_actual_failure_verified_unique_pairs"]>0
    assert set(record["DPR_roles"])<=set(("Coordinator","DirectoryAgent","OrderAgent","RecordsAgent","ComplianceAgent","BusinessExecutor"))-{"RecordsAgent"}
    for split in ("dev","exploratory_test"):
        rows=record["dev_test_episode_traces"][split]
        assert len(rows)==len(lookup[split])
        successes=0;violations=0;completions=0;returns=0.0
        for item in rows:
            env=environment(lookup[split][item["world"]])
            state=env.initial()
            for action in item["actual_model_actions"]:
                assert (env.observation(state).get("actor") or env.observation(state).get("role"))==action["role"]
                assert action["action"] in env.options(state)
                nR=len(state.native.delivered["R"])
                nextstate,audit=env.transition(state,action["action"])
                assert audit["verdict"]==action["verdict"],(item["world"],action)
                assert len(nextstate.native.delivered["R"])-nR==action["new_R_delivery"]
                state=nextstate
                checks["native_action_transitions_replayed"]+=1
            assert state.native.done
            assert (state.native.success and not state.native.violation)==item["native_success"]
            assert state.native.violation==item["native_violation"]
            assert state.native.steps==item["native_steps"]
            outcome=(round(1-env.spec.cost_per_step*state.native.steps,6)
                     if state.native.success and not state.native.violation else
                     -1.0 if state.native.violation else 0.0)
            assert abs(outcome-item["native_return"])<1e-6
            successes+=int(item["native_success"])
            violations+=int(item["native_violation"])
            completions+=int(state.native.done)
            returns+=outcome
            checks["actual_native_full_episodes_replayed"]+=1
        prior=record["scores"][split]
        assert successes==prior["safe_success"]
        assert violations==prior["ci_violations"]
        assert abs(returns/len(rows)-prior["mean_native_return"])<1e-6
        checks["seed_split_metrics_rechecked"]+=1
assert checks["actual_native_full_episodes_replayed"]==3*(4+8)
assert checks["seed_split_metrics_rechecked"]==6
assert all(x["scores"]["exploratory_test"]["safe_success"]==0
           for x in data["per_seed_optimized_models"])
report={
    "status":"INDEPENDENT_BACKTRACKED_DPR_NATIVE_REPLAY_PASS",
    "counts":dict(checks),
    "train_unique_verified_repair_pairs_per_seed":[
        x["DPR_actual_failure_verified_unique_pairs"]
        for x in data["per_seed_optimized_models"]],
    "no_punishment_of_legal_upstream_source_actor":True,
    "no_inferred_Qpi_values":True,
    "source_sha256":sha256(R.read_bytes()).hexdigest(),
    "not_training_Qwen_LLM":True
}
OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
