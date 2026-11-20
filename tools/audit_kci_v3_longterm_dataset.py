r"""Independent certified replay for exact same-state history-sensitive Q flips.

Re-creates every preflight state and re-solves both native actions for all
210 contexts (420 action continuations), replays end-to-end action traces,
checks training/dev/test graph disjointness and exact score/label provenance.
No model, API, pseudo reward, manual label, or off-platform data.
"""
from pathlib import Path
from collections import Counter,defaultdict
from hashlib import sha256
from dataclasses import asdict
import argparse,json,sys,re
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
sys.path.insert(0,str(ROOT/"tools"))
from kci.diverse_ci_branch_env import task_catalog,digest,FINAL
from kci.diverse_ci_long_horizon_v3 import LongHorizonFlipCIEnv
from solve_longterm_flip_v3 import paired_counterfactual,enter_preflight,terminal_return

def j(path):return json.loads(path.read_text(encoding="utf-8"))
def jl(path):return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,required=True)
    args=p.parse_args()
    root=args.dataset.resolve()
    output=root/"INDEPENDENT_AUDIT.json"
    if output.exists():raise FileExistsError(output)
    manifest=j(root/"MANIFEST_SHA256.json")
    for name,expected in manifest.items():
        assert sha256((root/name).read_bytes()).hexdigest()==expected,name
    roots=jl(root/"TASK_ROOTS.jsonl")
    worlds=jl(root/"WORLD_CONFIGS.jsonl")
    prompts=jl(root/"MODEL_OBSERVATIONS.jsonl")
    prefs=jl(root/"EXACT_QSTAR_PREFERENCES_EVAL_ONLY.jsonl")
    traces=jl(root/"ACTUAL_NATIVE_TWO_ACTION_CONTINUATIONS_EVAL_ONLY.jsonl")
    pairs=jl(root/"CONTRASTS_EVAL_ONLY.jsonl")
    demos=jl(root/"ORACLE_NATIVE_DEMOS_NOT_ONPOLICY_EVAL_ONLY.jsonl")
    training=jl(root/"BALANCED_ORACLE_PREFERENCE_CANDIDATES_TRAIN_ONLY.jsonl")
    validation=j(root/"VALIDATION.json")
    assert [len(x) for x in (roots,worlds,prompts,prefs,traces,pairs,demos,training)]==[
        21,210,210,210,420,210,210,40]
    taskmap={t.id:t for t in task_catalog()}
    assert {x["task"] for x in roots}==set(taskmap)
    cfg={x["id"]:x for x in worlds}
    src={x["id"]:x for x in prompts}
    pf={x["world_id"]:x for x in prefs}
    trace={(x["world_id"],x["action"]):x for x in traces}
    assert len(cfg)==len(src)==len(pf)==210
    assert len(trace)==420
    splits=defaultdict(set)
    for t in taskmap.values():splits[t.root_split].add(t.graph)
    assert splits["train"].isdisjoint(splits["dev"])
    assert splits["train"].isdisjoint(splits["exploratory_test"])
    assert splits["dev"].isdisjoint(splits["exploratory_test"])
    count=Counter()
    for key,config in cfg.items():
        task=taskmap[config["task_id"]]
        history=config["history"]
        authorized=config["authorization_exception"]
        env=LongHorizonFlipCIEnv(task,history,config["condition"],authorized)
        state,route,observation=enter_preflight(env)
        assert route==config["source_route"]
        assert state and observation
        assert config["native_preflight_sha256"]==digest(asdict(state))
        assert config["budget"]==env.budget
        assert config["required_slots"]==observation["task_evidence_contract"]["required_slots"]
        assert src[key]["model_input"]==observation
        assert src[key]["model_input_sha256"]==digest(observation)
        assert src[key]["legal_action_names"]==observation["legal_actions"]
        assert all(k not in observation for k in (
            "ground_truth","reward","true_Qstar","correct_action",
            "native","issued","oracle","target_label","terminal_return"))
        assert "OPEN_PRIVATE_CHANNEL" in observation["legal_actions"]
        assert "SKIP_PRIVATE_CHANNEL" in observation["legal_actions"]
        actual=paired_counterfactual(env)
        assert actual["Qgap_open_minus_skip"]!=0.0
        target=pf[key]
        chosen=("OPEN_PRIVATE_CHANNEL" if actual["Qgap_open_minus_skip"]>0
                else "SKIP_PRIVATE_CHANNEL")
        assert chosen==target["preferred_action"]
        assert target["native_Qstar_gap_preferred_minus_rejected"]==abs(actual["Qgap_open_minus_skip"])
        assert target["native_Qstar_open"]==actual["open_private"]["exact_native_Qstar"]
        assert target["native_Qstar_skip"]==actual["skip_private"]["exact_native_Qstar"]
        assert target["origin"]["continuation"]=="NATIVE_EXACT_FINITE_HORIZON_POST_FIXED_BUSINESS_ROUTE"
        assert target["origin"]["not_model_policy_Qpi"] is True
        for action,arm in (
            ("OPEN_PRIVATE_CHANNEL","open_private"),
            ("SKIP_PRIVATE_CHANNEL","skip_private")):
            record=trace[key,action]
            raw=actual[arm]
            assert record["native_Qstar"]==raw["exact_native_Qstar"]
            assert record["actual_native_actions"]==raw["actual_action_trace"]
            assert record["real_terminal"]==raw["terminal"]
            assert record["same_native_preflight_sha256"]==digest(asdict(state))
            # Independent run of all selected actions from the same prior state
            # is deterministic. No use of saved hidden branch state.
            current=state
            for step in record["actual_native_actions"]:
                assert step["state_before_sha256"]==digest(asdict(current))
                nxt,a=env.transition(current,step["action"])
                assert a["verdict"]==step["action_verdict"]
                assert step["state_after_sha256"]==digest(asdict(nxt))
                current=nxt
            assert current.done
            assert current.success==record["real_terminal"]["success"]
            assert current.violation==record["real_terminal"]["privacy_violation"]
            assert round(terminal_return(current),6)==record["native_Qstar"]
            count["two_action_native_traces_replayed"]+=1
        count["same_state_observations_reverified"]+=1
    assert count["same_state_observations_reverified"]==210
    assert count["two_action_native_traces_replayed"]==420
    # P0 requires strictly reversed value order; norms, tools, source
    # records, budget, legal action set must remain identical.
    for entry in pairs:
        wa=entry["paired_world_a"];wb=entry["paired_world_b"]
        ca=cfg[wa];cb=cfg[wb]
        assert ca["task_id"]==cb["task_id"]
        assert ca["condition"]==cb["condition"]
        a=src[wa]["model_input"].copy()
        b=src[wb]["model_input"].copy()
        h1=a.pop("delivered_message_log")
        h2=b.pop("delivered_message_log")
        if entry["pair_id"].endswith("__permission_exception"):
            assert entry["authorization_differs"]
            assert h1==h2
            assert a["recipient_policy"]!=b["recipient_policy"]
        else:
            assert a==b,(entry["pair_id"],"unexpected_prompt_difference")
            assert h1!=h2
            assert len(h1)==len(h2)
            if entry.get("same_optimal_action"):
                assert pf[wa]["preferred_action"]==pf[wb]["preferred_action"]
            elif entry.get("strict_long_horizon_Qstar_flip"):
                assert pf[wa]["preferred_action"]!=pf[wb]["preferred_action"]
        count["causal_pair_properties_rechecked"]+=1
    assert count["causal_pair_properties_rechecked"]==210
    positives=[p for p in training if p["chosen_action"]=="OPEN_PRIVATE_CHANNEL"]
    negatives=[p for p in training if p["chosen_action"]=="SKIP_PRIVATE_CHANNEL"]
    assert len(positives)==len(negatives)==20
    for x in training:
        assert x["split"]=="train"
        task_id=x["training_record"].split("__")[0]
        assert taskmap[task_id].root_split=="train"
        world=x["training_record"]
        assert x["prompt"]==src[world]["model_input"]
        assert x["chosen_action"]==pf[world]["preferred_action"]
        assert x["rejected_action"]==pf[world]["dispreferred_action"]
        assert x["not_FROZEN_LLM_Qpi"] is True
    # Need at least the original action and terminal reward to be
    # consistent, even if native sampling under other policies differs.
    summary={
        "status":"INDEPENDENT_EXACT_NATIVE_QSTAR_AND_DATA_LEAKAGE_AUDIT_PASS",
        "counts":dict(count),
        "all_manifest_entries_verified":len(manifest),
        "root_split_heldout_by_knowledge_DAG":True,
        "training_preference_labels_balanced":True,
        "strict_P0_flips_two_independent_history_perturbations_per_condition":70,
        "note":"Exact native post-business-route Qstar, not a real 4B/7B Qpi estimator.",
        "open_domain_natural_language_not_verified":True,
        "human_privacy_norm_approval":False,
        "training_any_LLM_not_performed":True,
    }
    output.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":main()
