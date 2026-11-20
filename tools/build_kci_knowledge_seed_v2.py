r"""Build verified executable multi-agent CI knowledge-accumulation seed dataset.

Only authored synthetic schemas and the native deterministic environment supply
labels. Do not train directly on the public evaluation answer keys.
Command:
  py -3 tools/build_kci_knowledge_seed.py --output artifacts/20261119_composition_signals/knowledge_seed_v1
"""
import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.knowledge_world_seed_v2 import (
    VERSION, DOMAINS, FACTS, RULES, KnowledgeTask, build_seeds,
    certified_contrasts, canonical, topological_orders, find_safe_witness, digest,
)

def rows(path,items):
    if path.exists():raise FileExistsError(path)
    with path.open("w",encoding="utf-8",newline="\n") as f:
        for x in items:f.write(canonical(x)+"\n")

def report_json(path,thing):
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(thing,indent=2,ensure_ascii=False,sort_keys=True)+"\n",encoding="utf-8")

def best_at_first_action(env,mode):
    assert mode in ("raw","private")
    success=[]
    orders=(x for x in topological_orders(env.seed,env.required) if x[0]==env.seed.anchor)
    for order in orders:
        for n in range(len(env.required)+1):
            for subset in combinations(env.required,n):
                if (env.seed.anchor in subset)!=(mode=="private"):continue
                x=env.rollout(order,subset)
                if x["success"] and not x["violation"]:
                    success.append(x)
    return min(success,key=lambda x:(x["steps"],x["order"],x["privately_verified"])) if success else None

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    out=a.output.resolve()
    if out.exists():raise FileExistsError("Refuse to overwrite existing study: "+str(out))
    seeds=build_seeds()
    roots=[];worlds=[];pairs=[];trajectories=[];queries=[];judge=[]
    summary=Counter()
    for seed in seeds:
        r=certified_contrasts(seed)
        linked=KnowledgeTask(seed,"linked",False,"minimal")
        unrelated=KnowledgeTask(seed,"unrelated",False,"minimal")
        link_private=best_at_first_action(linked,"private")
        unlink_raw=best_at_first_action(unrelated,"raw")
        link_raw=best_at_first_action(linked,"raw")
        unlink_private=best_at_first_action(unrelated,"private")
        assert link_raw is None, "linked raw action unexpectedly reaches success"
        assert link_private and link_private["success"]
        assert unlink_raw and unlink_raw["success"]
        assert (unlink_private is None or unlink_raw["steps"]<unlink_private["steps"]), "no optimal utility separation"
        roots.append({
            "seed_id":seed.seed_id,"domain":seed.domain,"topology":seed.topology,
            "fact_permutation":list(seed.fact_permutation),"anchor_fact":seed.anchor,
            "split":seed.split,"workflow_dependency_dag":seed.contract["depends"],
            "task_goal":seed.contract["task"],"task_result":seed.contract["result"],
            "roles":list(seed.contract["roles"]),
            "check_meanings":list(seed.contract["checks"]),
            "inference_rules":seed.rules,
            "protected_relation":seed.contract["secret"],
            "source_mode":"fictional_authored_synthetic_contract",
            "norm_source":"declared_synthetic_policy_not_real_human_label",
            "mechanism_signature":seed.domain+"|"+seed.topology,
        })
        configurations=[
            ("linked_minimal","linked",False,"minimal"),
            ("unrelated_minimal","unrelated",False,"minimal"),
            ("hold_a_minimal","hold_a",False,"minimal"),
            ("hold_b_minimal","hold_b",False,"minimal"),
            ("linked_authorized","linked",True,"minimal"),
            ("linked_full","linked",False,"full"),
            ("order_empty_full","empty",False,"full"),
        ]
        for variant,history,authorized,profile in configurations:
            env=KnowledgeTask(seed,history,authorized,profile)
            initial=env.initial()
            worlds.append({
                "world_id":seed.seed_id+"_"+variant,
                "seed_id":seed.seed_id,"split":seed.split,
                "variant":variant,"history":history,"authorized":authorized,
                "task_profile":profile,"horizon":env.horizon,
                "required_checks":list(env.required),
                "initial_receipts":initial.ledger,
                "initial_state_hash":digest(asdict(initial)),
                "environment_version":VERSION,
            })
            # Send-phase prompt is a genuine reachable Agent state, with the
            # same candidate action vocabulary for linked/unrelated/hold.
            state=initial
            if "PROCESS_"+seed.anchor in env.options(state):
                state,_=env.transition(state,"PROCESS_"+seed.anchor)
                obs=env.observation(state)
                queries.append({
                    "id":seed.seed_id+"_"+variant+"_send_anchor","seed_id":seed.seed_id,
                    "split":seed.split,"world_id":seed.seed_id+"_"+variant,
                    "model_input":obs,
                    "available_actions":list(env.options(state)),
                    "state_hash":digest(asdict(state))
                })
        for kind,left,right in (
            ("flip","unrelated","linked"),
            ("hold","hold_a","hold_b"),
            ("permission","linked","linked_authorized")):
            lp=r["probe"][left];rp=r["probe"][right]
            pairs.append({
                "pair_id":seed.seed_id+"_"+kind,"seed_id":seed.seed_id,"split":seed.split,
                "contrast":kind,"same_requested_task":True,
                "same_available_current_action":"SEND_RAW_"+seed.anchor,
                "left_world":seed.seed_id+"_"+(
                    "linked_authorized" if left=="linked_authorized" else left+"_minimal"),
                "right_world":seed.seed_id+"_"+(
                    "linked_authorized" if right=="linked_authorized" else right+"_minimal"),
                "left_observation_hash":digest(lp["pre_decision_observation"]),
                "right_observation_hash":digest(rp["pre_decision_observation"]),
                "observations_differ":lp["pre_decision_observation"]!=rp["pre_decision_observation"],
                "action_same":lp["action"]==rp["action"],
                "label_source":"native_observed_delivery_and_rule_closure",
                "eval_only_left_immediate_verdict":lp["audit"]["verdict"],
                "eval_only_right_immediate_verdict":rp["audit"]["verdict"],
                "eval_only_left_new_private":lp["audit"]["new_derived_private"],
                "eval_only_right_new_private":rp["audit"]["new_derived_private"],
                "not_a_trajectory_DPO_label":True,
            })
        for kind,trace in (
            ("safe_linked_minimal",link_private),
            ("safe_unrelated_minimal",unlink_raw),
            ("safe_full",r["safe_witness"]),
            ("unsafe_full_raw",r["unsafe_raw"]),
            ("order_earliest_breach",r["order_first"]),
            ("order_latest_breach",r["order_last"]),
        ):
            trajectories.append({"seed_id":seed.seed_id,"split":seed.split,
                                  "trace_kind":kind,**trace})
        trajectories.append({
            "seed_id":seed.seed_id,"split":seed.split,"trace_kind":"all_private_full",
            **KnowledgeTask(seed,"linked",False,"full").rollout(
                topological_orders(seed,FACTS)[0],FACTS)})
        judge.append({
            "seed_id":seed.seed_id,"split":seed.split,
            "flip_true":True,"hold_true":True,
            "linked_raw_anchor_unsafe":True,
            "unrelated_raw_anchor_safe":True,
            "min_safe_private_first_steps":link_private["steps"],
            "min_safe_public_first_steps":unlink_raw["steps"],
            "unrelated_private_first_steps":unlink_private["steps"] if unlink_private else None,
            "raw_first_linked_no_safe_completion":link_raw is None,
            "full_safe_task_completion":r["safe_witness"]["success"],
            "unsafe_full_raw_violates":r["unsafe_raw"]["violation"],
            "all_private_full_fails_utility":True,
            "order_early_violation_step":r["order_first"]["first_derived"]["turn"],
            "order_late_violation_step":r["order_last"]["first_derived"]["turn"],
            "order_trigger_action_changed":r["different_trigger_action"],
            "order_violation_step_changed":r["order_has_distinct_trigger_step"],
            "minimum_safe_full_private_checks":r["safe_witness"]["private_count"],
            "source_certified_external_norm":False,
        })
        summary["seeds"]+=1
        summary["split_"+seed.split]+=1
        summary["order_time_shift"]+=int(r["order_has_distinct_trigger_step"])
        summary["order_action_shift"]+=int(r["different_trigger_action"])
        summary["linked_private_first_required"]+=1
        summary["unrelated_raw_first_optimal"]+=1
        summary["safe_full_exists"]+=1
        summary["all_private_full_incomplete"]+=1
    assert len(roots)==20 and len(worlds)==140 and len(pairs)==60
    assert len(trajectories)==140 and len(queries)==140 and len(judge)==20
    assert len({r["mechanism_signature"] for r in roots})==20
    assert len({r["seed_id"] for r in roots if r["split"]=="test"})==5
    assert summary["order_time_shift"]>=12, "Need genuine temporal order drift in majority of seeds"
    assert summary["linked_private_first_required"]==summary["unrelated_raw_first_optimal"]==20
    out.mkdir(parents=True)
    rows(out/"roots.jsonl",roots)
    rows(out/"worlds.jsonl",worlds)
    rows(out/"action_context_pairs_EVAL.jsonl",pairs)
    rows(out/"verified_native_traces_EVAL.jsonl",trajectories)
    rows(out/"model_inputs.jsonl",queries)
    rows(out/"answer_key_EVAL_ONLY.jsonl",judge)
    report={
        "study":"KCI receiver knowledge seed dataset v2, anti-shortcut fact remapping",
        "schema":VERSION,
        "n_seed_roots":20,
        "n_world_scenarios":len(worlds),
        "n_model_input_states":len(queries),
        "n_matched_pairs":len(pairs),
        "n_executable_witness_traces":len(trajectories),
        "n_heldout_rule_family_test_roots":5,
        "split_roots":{k:summary["split_"+k] for k in ("train","dev","test")},
        "workflow_dags":len(DOMAINS),"inference_rule_graphs":len(RULES),
        "combinations":20,
        "verified_properties":dict(summary),
        "all_label_provenance":"CPU native transition plus deterministic Horn closure",
        "anti_shortcut_seed_specific_fact_permutations":True,
        "requires_strong_history_blind_baseline_audit":True,
        "human_validated_social_norms":False,
        "source_backed_independent_business_tasks":0,
        "model_training_eligible":False,
        "ready_for_cpu_executable_agent_baselines":True,
        "blind_test_note":"TEST root identities deliberately exposed here for development provenance; do not claim blind leaderboard.",
        "api_calls":0,
        "no_model_truth_in_model_inputs":True,
    }
    report_json(out/"VALIDATION.json",report)
    manifest={p.name:sha256(p.read_bytes()).hexdigest()
              for p in sorted(out.iterdir()) if p.is_file()}
    report_json(out/"MANIFEST.sha256.json",manifest)
    print(json.dumps({"status":"BUILT_AND_NATIVE_VERIFIED","path":str(out),
                      "seeds":20,"worlds":len(worlds),"pairs":len(pairs),
                      "order_time_shift":summary["order_time_shift"],
                      "split_roots":report["split_roots"],
                      "backend":"CPU only, zero API calls",
                      "training_ready":False},indent=2),flush=True)

if __name__=="__main__":main()
