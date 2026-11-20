r"""Actual A->R knowledge accumulation->B action flip->C business completion.

Seven graph mechanisms across three business families; 2 swapped source naming
conditions; 2 upstream package-cost conditions. A chooses from two signed
source results, both immediately legal, actually delivered to receiver R.
B chooses OPEN/SKIP after reading the ACTUALLY UPDATED prior message ledger.
C/other agents execute the true native suffix, including source disclosure,
private correction, and final actual business mutation.

For each B same-state candidate, independently enumerate exact native future
actions and re-execute them from the logged chain. No fictional reward labels,
manual classifier targets or Qwen policy Qpi. This is an executable mechanism
test, NOT trained/shared-LLM multiagent learning.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
from collections import Counter,defaultdict
from hashlib import sha256
import json,sys,argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.diverse_ci_branch_env import task_catalog,digest,SLOT_LABELS,FINAL
from kci.diverse_ci_long_horizon_v31 import enumerate_specs_for_root
from kci.upstream_downstream_chain_pilot import ChainPilotEnv,VERSION
from solve_v31_longterm import native_best

def js(obj):
    return json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(",",":"))
def main():
    out=ROOT/"artifacts"/"20261120_v31"/"DYNAMIC_UPSTREAM_DOWNSTREAM_CHAIN_PILOT_01"
    if out.exists():raise FileExistsError(out)
    tasks=[]
    seen=set()
    for t in task_catalog():
        if t.graph in seen:continue
        candidate=[x for x in task_catalog() if x.graph==t.graph]
        idx=len(tasks)%3
        tasks.append(candidate[idx])
        seen.add(t.graph)
    assert len(tasks)==7 and len({t.business for t in tasks})==3
    rows=[];traces=[];observations=[];source_actions=[];scores=Counter()
    for index,task in enumerate(tasks):
        cond={"calendar":"early_open","retail":"pending_yes","document":"external"}[task.business]
        spec=enumerate_specs_for_root(task,cond,index%3)
        for mode,surcharge in (("matched",0),("tradeoff",4)):
            for swap in (False,True):
                env=ChainPilotEnv(spec,swap,mode,safe_surcharge=surcharge)
                initial=env.initial()
                before_a=env.observation(initial)
                st,a=env.transition(initial,"A_READ_VERIFIED_PACKAGES")
                assert a["verdict"]=="VALID"
                source_obs=env.observation(st)
                assert set(source_obs["legal_actions"])=={
                    "A_SEND_PACKAGE_A","A_SEND_PACKAGE_B","ABORT_CHAIN"}
                # Record one model-visible A choice state per world.
                config=spec.case_id+"__"+mode+"__swap"+str(int(swap))
                source_actions.append({
                    "world":config,"root":task.id,"graph":task.graph,
                    "split":task.root_split,"actor":"RecordsAgent",
                    "current_model_input":source_obs,
                    "current_model_input_sha256":digest(source_obs),
                    "same_true_A_source_state_for_two_actions":True,
                    "both_upstream_options_immediately_legal":True,
                })
                branch={}
                for pkg in ("PACKAGE_A","PACKAGE_B"):
                    parent=st
                    sent,a=env.transition(parent,"A_SEND_"+pkg)
                    assert a["verdict"]=="VALID",(config,pkg,a)
                    assert sent.native.violation is False
                    assert len(sent.source_delivered_ids)==len(
                        sent.source_bundles[pkg]["facts"])
                    assert all(x["sender"]=="RecordsAgent" and x["recipient"]=="R"
                               for x in sent.native.ledger)
                    assert all(x["actual_delivery"] for x in sent.native.ledger)
                    intermediate=sent
                    pre_steps=[
                        {"role":before_a["actor"],"action":"A_READ_VERIFIED_PACKAGES",
                         "after_sha256":digest(asdict(st))},
                        {"role":source_obs["actor"],"action":"A_SEND_"+pkg,
                         "after_sha256":digest(asdict(sent))}
                    ]
                    for act in ("INSPECT_BUSINESS","CHOOSE:"+env.native._route):
                        actor=env.observation(intermediate)["actor"]
                        intermediate,b=env.transition(intermediate,act)
                        assert b["verdict"]=="VALID",(config,pkg,act,b)
                        pre_steps.append({"role":actor,"action":act,
                            "after_sha256":digest(asdict(intermediate))})
                    assert intermediate.native.stage=="preflight"
                    current=env.observation(intermediate)
                    assert current["actor"]=="Coordinator"
                    assert len(current["delivered_message_log"])==len(
                        sent.source_delivered_ids)
                    observed_source_facts=[m["fact"] for m in intermediate.native.ledger]
                    for f in env.source_facts if hasattr(env,"source_facts") else []:
                        assert f in observed_source_facts
                    result={}
                    for preflight_action in ("OPEN_PRIVATE_CHANNEL","SKIP_PRIVATE_CHANNEL"):
                        native=native_best(env.native,intermediate.native,preflight_action)
                        sim=intermediate
                        log=pre_steps[:]
                        first_step_state_hash=digest(asdict(sim.native))
                        for item in native["action_trace"]:
                            assert digest(asdict(sim.native))==item["state_before_sha256"]
                            actor=env.observation(sim)["actor"]
                            sim,a=env.transition(sim,item["action"])
                            assert a["verdict"]==item["actual_verdict"],(
                                config,pkg,item["action"],a)
                            assert digest(asdict(sim.native))==item["state_after_sha256"]
                            log.append({
                                "role":actor,"action":item["action"],
                                "stage":item["stage_before"],
                                "after_sha256":digest(asdict(sim))})
                        assert sim.native.done
                        assert sim.native.success==native["terminal"]["success"]
                        assert sim.native.violation==native["terminal"]["violation"]
                        assert sim.native.generated_result==native["terminal"]["business_result"]
                        assert any(x["role"]=="RecordsAgent" for x in log)
                        assert any(x["role"]=="Coordinator" for x in log)
                        result[preflight_action]=native
                        traces.append({
                            "world":config,"package_selected":pkg,
                            "downstream_B_preflight_action":preflight_action,
                            "native_Qstar_after_A":native["native_exact_Qstar"],
                            "native_full_actions_after_upstream":native["action_trace"],
                            "chain_action_role_log":log,
                            "first_B_state_hash":first_step_state_hash,
                            "actual_final_business":sim.native.business,
                            "actual_final_receiver_messages":sim.native.ledger,
                            "terminal":native["terminal"],
                            "not_model_Qpi":True,
                        })
                        scores["verified_A_B_C_native_paths"]+=1
                    gap=round(result["OPEN_PRIVATE_CHANNEL"]["native_exact_Qstar"]-
                              result["SKIP_PRIVATE_CHANNEL"]["native_exact_Qstar"],7)
                    prior_facts=set(sent.native.delivered["R"])
                    required=set(env.native.required_facts)
                    # Existing connected source records at B depend on A.
                    package_is_risky=tuple(sent.source_bundles[pkg]["facts"])==spec.risky_prior
                    assert (gap>0)==package_is_risky,(
                        task.id,config,pkg,gap,prior_facts,required)
                    branch[pkg]={
                        "package_truly_risky_for_future":package_is_risky,
                        "B_obs":current,
                        "B_gap":gap,
                        "B_Qstar_OPEN":result["OPEN_PRIVATE_CHANNEL"]["native_exact_Qstar"],
                        "B_Qstar_SKIP":result["SKIP_PRIVATE_CHANNEL"]["native_exact_Qstar"],
                        "A_true_source_tool_receipt_sha256":sent.source_certificate,
                        "A_real_delivered_ids":list(sent.source_delivered_ids),
                        "A_steps_charged":sent.source_bundles[pkg]["cost_steps"],
                        "actual_B_preflight_state_sha256":digest(asdict(intermediate.native)),
                    }
                    observations.append({
                        "world":config,"package":pkg,"task":task.id,
                        "split":task.root_split,
                        "actor":"Coordinator","phase":"preflight",
                        "model_input":current,
                        "model_input_sha256":digest(current),
                        "no_oracle_label_in_actor_observation":True,
                    })
                assert len(branch)==2
                risch=[x for x in branch if branch[x]["package_truly_risky_for_future"]]
                safe=[x for x in branch if not branch[x]["package_truly_risky_for_future"]]
                assert len(risch)==len(safe)==1
                left=branch[risch[0]];right=branch[safe[0]]
                assert left["B_gap"]>0>right["B_gap"]
                obsA=left["B_obs"].copy()
                obsB=right["B_obs"].copy()
                historyA=obsA.pop("delivered_message_log")
                historyB=obsB.pop("delivered_message_log")
                if mode=="matched":
                    assert obsA==obsB,("B observation differs beyond R history",config)
                assert len(historyA)==len(historyB)>0
                assert historyA!=historyB
                assert all(x["sender"]=="RecordsAgent" for x in historyA+historyB)
                assert all(x["recipient"]=="R" for x in historyA+historyB)
                assert all("S0" in x["text"] for x in historyA+historyB)
                aval={
                    pkg:max(branch[pkg]["B_Qstar_OPEN"],branch[pkg]["B_Qstar_SKIP"])
                    for pkg in branch}
                record={
                    "world":config,
                    "root":task.id,"graph":task.graph,
                    "business":task.business,"split":task.root_split,
                    "source_mode":mode,"source_cost_surcharge":surcharge,
                    "swapped_package_names":swap,
                    "both_upstream_A_actions_actually_executed":True,
                    "both_A_actions_immediately_privacy_safe":True,
                    "native_verified_source_bundle_hash":left["A_true_source_tool_receipt_sha256"],
                    "B_current_state_only_history_differs":mode=="matched",
                    "related_package":risch[0],"benign_package":safe[0],
                    "upstream_A_values_given_exact_B_continuation":aval,
                    "B_Qgap_risky_package":left["B_gap"],
                    "B_Qgap_benign_package":right["B_gap"],
                    "B_strict_policy_flip_due_to_actual_received_history":True,
                    "B_downstream_current_observation_hashes":[
                        digest(left["B_obs"]),digest(right["B_obs"])],
                    "history_diff":{
                        "risky_A_deliveries":historyA,
                        "safe_A_deliveries":historyB
                    },
                    "source_cost_actual_charged":[left["A_steps_charged"],
                                                   right["A_steps_charged"]],
                    "evidence_grade":"EXACT_FINITE_NATIVE_AFTER_REAL_UPSTREAM_AGENT_ACTION_NOT_QPI",
                }
                rows.append(record)
                scores["strict_A_generated_B_flips"]+=1
                scores["same_observation_B_history_only_flips"]+=int(mode=="matched")
                print("CHAIN",config,"B risky/safe gap",left["B_gap"],right["B_gap"],
                      "A risky/safe best",aval[risch[0]],aval[safe[0]],
                      "B matched",mode=="matched",flush=True)
    assert len(rows)==7*2*2
    assert len(traces)==len(rows)*2*2
    assert len(observations)==len(rows)*2
    assert len(source_actions)==len(rows)
    assert scores["same_observation_B_history_only_flips"]==14
    out.mkdir(parents=True)
    def lines(name,items):
        with (out/name).open("x",encoding="utf-8",newline="\n") as f:
            for x in items:f.write(js(x)+"\n")
    lines("CHAIN_CAUSAL_PAIRS_EVAL_ONLY.jsonl",rows)
    lines("A_SOURCE_DECISION_OBSERVATIONS.jsonl",source_actions)
    lines("B_AFTER_ACTUAL_DELIVERY_OBSERVATIONS.jsonl",observations)
    lines("REAL_AGENT_A_B_C_NATIVE_TRAJECTORIES_EVAL_ONLY.jsonl",traces)
    summary={
        "status":"UPSTREAM_AGENT_ACTUAL_DELIVERY_CAUSES_DOWNSTREAM_Q_FLIP",
        "mechanism_roots":7,"worlds":len(rows),
        "A_pre_delivery_same_state_action_states":len(source_actions),
        "B_post_delivery_observations":len(observations),
        "native_A_B_C_executable_branch_trajectories":len(traces),
        "history_only_strict_B_Q_flips":scores["same_observation_B_history_only_flips"],
        "all_A_actions_immediately_legal":True,
        "all_B_values_derived_by_full_native_future_action_enumeration":True,
        "actual_multiagent_upstream_actor":"RecordsAgent",
        "actual_downstream_actor":"Coordinator, source-specific tool agent, BusinessExecutor",
        "not_preloaded_archive_historical_facts":True,
        "upstream_action_source_read_and_delivery_evidence_verified":True,
        "no_random_GPT_trajectories":True,
        "current_model_Qpi_not_available":True,
        "Qstar_labels_eval_only":True,
        "no_paper_grade_claim":True,
    }
    (out/"VALIDATION.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"MANIFEST_SHA256.json").write_text(json.dumps({
        p.name:sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("SUMMARY",json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
