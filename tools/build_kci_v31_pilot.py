r"""Gate A development generator: V3.1 knowledge-rule dependency without lexical shortcuts.

This has a carefully bounded --pilot mode (one case per graph) and a separate
--expanded mode (two mechanism cases per original 21 business/graph roots).
Construct scenario specs without reward labels; independently evaluate true
native Qstar for OPEN and SKIP under all five actual delivered histories.

Output is synthetic, reviewer-visible, not Qpi, not a certified business policy.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
from hashlib import sha256
from collections import Counter,defaultdict
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
sys.path.insert(0,str(ROOT/"tools"))
from kci.diverse_ci_branch_env import task_catalog,digest,FINAL
from kci.diverse_ci_long_horizon_v31 import (
    VERSION,V31Env,V31Case,HISTORY_TYPES,enumerate_specs_for_root,
)
from solve_v31_longterm import pair_values,before_preflight,reward

def jsonl(path,records):
    with path.open("x",encoding="utf-8",newline="\n") as f:
        for x in records:
            f.write(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n")
def jfile(path,record):
    with path.open("x",encoding="utf-8") as f:
        f.write(json.dumps(record,ensure_ascii=False,sort_keys=True,indent=2)+"\n")

def specs_for_mode(mode):
    all_tasks=task_catalog()
    if mode=="pilot":
        selected=[]
        seen=set()
        by_graph={}
        for t in all_tasks:
            by_graph.setdefault(t.graph,[]).append(t)
        for i,(graph,options) in enumerate(by_graph.items()):
            selected.append(options[i%len(options)]);seen.add(graph)
        assert len(selected)==7
        return [enumerate_specs_for_root(t,{
           "calendar":"early_open","retail":"pending_yes","document":"external"}[t.business],0)
           for t in selected]
    selected=[]
    for task in all_tasks:
        for variant in range(3):
            condition={
                "calendar":("early_open","early_busy","early_open")[variant],
                "retail":("pending_yes","delivered_yes","pending_yes")[variant],
                "document":("external","external","external")[variant],
            }[task.business]
            selected.append(enumerate_specs_for_root(task,condition,variant))
    assert len(selected)==63
    return selected

def replay_role_samples(env,branch,world_id):
    start,obs=before_preflight(env)
    st=start
    rows=[]
    for i,item in enumerate(branch["action_trace"]):
        assert st.stage==item["stage_before"]
        assert digest(asdict(st))==item["state_before_sha256"]
        before=env.observation(st)
        assert all(k not in before for k in ("oracle","reward","violation","Qstar","chosen_correct_action"))
        role=before["role"]
        next_state,report=env.transition(st,item["action"])
        assert report["verdict"]==item["actual_verdict"]
        assert digest(asdict(next_state))==item["state_after_sha256"]
        rows.append({
            "world_id":world_id,
            "root_id":env.task.id,
            "graph_family":env.task.graph,
            "split":env.task.root_split,
            "role":role,"stage":st.stage,
            "model_visible_observation":before,
            "model_observation_sha256":digest(before),
            "chosen_native_oracle_action":item["action"],
            "human_label_source":"EXACT_NATIVE_OPTIMAL_TRAJECTORY",
            "not_LLM_onpolicy":True,
            "native_before_sha256":item["state_before_sha256"],
            "native_after_sha256":item["state_after_sha256"],
            "current_step":i,
        })
        st=next_state
    assert st.done
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--mode",choices=("pilot","expanded"),default="pilot")
    args=ap.parse_args()
    output=args.output.resolve()
    if output.exists():raise FileExistsError(output)
    specs=specs_for_mode(args.mode)
    cases=[];model=[];paired=[];branches=[];traces=[];train=[];role_trace=[];counts=Counter()
    score_levels=defaultdict(set)
    for spec in specs:
        rec={
            "case_id":spec.case_id,
            "root_id":spec.task.id,
            "graph_family":spec.task.graph,
            "split":spec.root_split,
            "business_kind":spec.task.business,
            "business_condition":spec.condition,
            "fact_permutation":list(spec.fact_permutation),
            "required_facts":list(spec.required),
            "positive_history":list(spec.risky_prior),
            "negative_history":list(spec.safe_prior),
            "hold_history":list(spec.hold_prior),
            "public_setup_fee":spec.prep_fee,
            "public_step_cost":spec.cost_per_step,
            "public_budget":spec.budget,
            "norms_fictional":True,
            "source_graph_is_known_to_actor":True,
        }
        cases.append(rec)
        groups={}
        for variant in HISTORY_TYPES:
            env=V31Env(spec,variant)
            p=pair_values(env)
            world=spec.case_id+"__"+variant
            assert p["model_observation"]["role"]=="Coordinator"
            assert p["Qgap_open_minus_skip"]!=0.0
            expected_positive=(variant=="related")
            assert (p["Qgap_open_minus_skip"]>0)==expected_positive,(
                world,p["Qgap_open_minus_skip"],p["open"]["terminal"],p["skip"]["terminal"])
            assert p["open"]["terminal"]["success"]
            assert p["skip"]["terminal"]["success"]== (not expected_positive)
            assert all(x in "abcdef" for x in spec.risky_prior+spec.safe_prior+spec.hold_prior)
            groups[variant]=p
            prompt=p["model_observation"]
            model.append({
                "world_id":world,"case_id":spec.case_id,
                "root_id":spec.task.id,"split":spec.root_split,
                "model_input":prompt,"model_input_sha256":digest(prompt),
                "legal_actions":prompt["legal_actions"],
                "label_not_in_actor_observation":True,
            })
            branch_record={
                "world_id":world,"case_id":spec.case_id,
                "root_id":spec.task.id,"split":spec.root_split,
                "history_variant":variant,
                "preflight_sha256":p["state_hash"],
                "same_state_for_two_actions":True,
                "native_Qgap_open_minus_skip":p["Qgap_open_minus_skip"],
                "oracle_action":"OPEN_PRIVATE_CHANNEL" if p["Qgap_open_minus_skip"]>0
                       else "SKIP_PRIVATE_CHANNEL",
                "reward_formula":"success?1-public_cost*actual_steps:violation?-1:0",
                "model_Qpi_measured":False,
                "native_Qstar_exact":True,
                "branch_action_values":{
                    "open":p["open"]["native_exact_Qstar"],
                    "skip":p["skip"]["native_exact_Qstar"],
                },
                "terminal_rewards":{
                    "open":p["open"]["terminal"]["reward"],
                    "skip":p["skip"]["terminal"]["reward"],
                },
            }
            paired.append(branch_record)
            for name in ("open","skip"):
                arm=p[name]
                assert arm["first_action"]==("OPEN_PRIVATE_CHANNEL" if name=="open" else "SKIP_PRIVATE_CHANNEL")
                branches.append({
                    "world_id":world,"first_action":arm["first_action"],
                    "same_state_sha256":p["state_hash"],
                    "source_model_observation_sha256":digest(prompt),
                    "full_native_action_trace":arm["action_trace"],
                    "terminal":arm["terminal"],
                    "exact_native_Qstar":arm["native_exact_Qstar"],
                    "native_states_explored":arm["states_explored"],
                    "provenance":arm["provenance"],
                    "NOT_model_Qpi":True,
                })
                counts["real_action_branches"]+=1
            # One actually executable optimal native continuation
            preferred=p["open"] if expected_positive else p["skip"]
            samples=replay_role_samples(env,preferred,world)
            role_trace.extend(samples)
            counts["all_role_action_states"]+=len(samples)
            if spec.root_split=="train" and variant in ("related","same_entity_safe"):
                train.append({
                    "world_id":world,"graph_family":spec.task.graph,
                    "split":"train",
                    "prompt":prompt,
                    "chosen_action":"OPEN_PRIVATE_CHANNEL" if expected_positive
                        else "SKIP_PRIVATE_CHANNEL",
                    "rejected_action":"SKIP_PRIVATE_CHANNEL" if expected_positive
                        else "OPEN_PRIVATE_CHANNEL",
                    "regret_margin":abs(p["Qgap_open_minus_skip"]),
                    "origin":"EXACT_NATIVE_QSTAR_DETERMINISTIC",
                    "NOT_FROZEN_LLM_Qpi":True,
                    "training_family_not_held_out":True,
                })
            score_levels["positive" if expected_positive else "negative"].add(round(p["Qgap_open_minus_skip"],6))
            counts["model_preflight_worlds"]+=1
            counts["native_states_enumerated"]+=p["open"]["states_explored"]+p["skip"]["states_explored"]
        # Opposite-value strict Flip with FULLY normal same-entity facts.
        first=groups["related"]
        for ctrl in ("same_entity_safe","cross_entity_related",
                     "same_entity_hold","authorized_related"):
            other=groups[ctrl]
            x=first["model_observation"].copy()
            y=other["model_observation"].copy()
            lh=x.pop("delivered_message_log")
            rh=y.pop("delivered_message_log")
            if ctrl!="authorized_related":
                assert x==y,("Unexpected different business conditions",spec.case_id,ctrl)
            else:
                # The norm exception only changes this one public policy.
                xx=x.copy();yy=y.copy()
                xx.pop("recipient_policy"); yy.pop("recipient_policy")
                assert xx==yy
            assert len(lh)==len(rh)
            assert len(lh)>0
            assert all("S0" in m["text"] and "S07" not in m["text"]
                       for m in lh+rh)
            assert "OTHER-" not in json.dumps(lh+rh)
            contrasts={
                "case_id":spec.case_id,
                "split":spec.root_split,
                "control_type":ctrl,
                "left_world":spec.case_id+"__related",
                "right_world":spec.case_id+"__"+ctrl,
                "history_message_count_identical":True,
                "both_histories_normal_vocabulary_S01_to_S06":True,
                "same_business_permissions_tools_budget":ctrl!="authorized_related",
                "only_R_prior_deliveries_change":ctrl!="authorized_related",
                "authorization_changed_as_separate_control":ctrl=="authorized_related",
                "related_Qgap":first["Qgap_open_minus_skip"],
                "control_Qgap":other["Qgap_open_minus_skip"],
                "strict_native_Qstar_flip":True,
            }
            traces.append(contrasts)
            counts["contrasts"]+=1
        # One actual hold/no-flip pair, two safe histories with same semantics.
        assert abs(groups["same_entity_safe"]["Qgap_open_minus_skip"]-
                   groups["same_entity_hold"]["Qgap_open_minus_skip"])<1e-7
        counts["hold_pairs_no_flip"]+=1
    assert counts["model_preflight_worlds"]==5*len(specs)
    assert counts["real_action_branches"]==10*len(specs)
    assert counts["contrasts"]==4*len(specs)
    assert len(model)==len(paired)==5*len(specs)
    assert len(branches)==10*len(specs)
    assert len(train)==2*sum(c.root_split=="train" for c in specs)
    assert len({x["chosen_action"] for x in train})==(2 if train else 0)
    assert len(score_levels["positive"])>=4,score_levels
    assert len(score_levels["negative"])>=4,score_levels
    # Test originally simple lexical rule should *fail* on the same-entity normal
    # safe histories. The independent lexical audit is a separate script.
    output.mkdir(parents=True)
    for filename,data in (
        ("CASE_SPECS.jsonl",cases),
        ("MODEL_OBSERVATIONS.jsonl",model),
        ("EXACT_NATIVE_ACTION_PREFERENCES_EVAL_ONLY.jsonl",paired),
        ("TWO_ACTION_TRUE_NATIVE_CONTINUATIONS_EVAL_ONLY.jsonl",branches),
        ("CONTROLLED_HISTORY_PAIRS_EVAL_ONLY.jsonl",traces),
        ("MULTI_ROLE_ORACLE_ACTION_OBSERVATIONS_EVAL_ONLY.jsonl",role_trace),
        ("TRAIN_ONLY_BALANCED_NATIVE_OPTIMAL_ACTION_PREFS.jsonl",train),
    ):
        jsonl(output/filename,data)
    ccount=Counter(x["role"] for x in role_trace)
    report={
        "status":"V31_NATIVE_GATEA_MECHANISMS_GENERATED",
        "mode":args.mode,
        "case_mechanisms":len(specs),
        "knowledge_graph_families":len({x.graph_name for x in specs}),
        "real_task_roots":len({x.task.id for x in specs}),
        "worlds":len(model),
        "same_state_native_action_continuations":len(branches),
        "exact_native_long_horizon_pairs":len(paired),
        "history_causal_pairs":len(traces),
        "total_native_action_states_from_optimal_trajectories":len(role_trace),
        "different_agent_roles_observed":dict(ccount),
        "positive_Qgap_distinct_values":sorted(score_levels["positive"]),
        "negative_Qgap_distinct_values":sorted(score_levels["negative"]),
        "train_balanced_preferences":len(train),
        "split_by_graph_family":True,
        "source_data_uses_only_normal_S01_S06":True,
        "normal_same_entity_safe_history_present":True,
        "variable_required_evidence_counts":sorted(set(len(x.required) for x in specs)),
        "variable_private_setup_steps":sorted(set(x.prep_fee for x in specs)),
        "variable_public_cost_per_step":sorted(set(x.cost_per_step for x in specs)),
        "native_state_evaluations":counts["native_states_enumerated"],
        "Qstar_not_model_Qpi":True,
        "all_business_rules_and_privacy_norms_synthetic":True,
        "paper_grade_blind_test":False,
        "no_LLM_training":True,
        "no_paid_API":True,
    }
    jfile(output/"VALIDATION.json",report)
    jfile(output/"MANIFEST_SHA256.json",{
        p.name:sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir()) if p.is_file()})
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__":main()
