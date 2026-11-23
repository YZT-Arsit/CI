"""Real V4 native FULL multi-agent episodes driven by shared 4B LoRA.

Two separately identified protocols:
  controlled_release: actual scripted native A/B prefix until source pivot,
    4B at EVERY subsequent release, deterministic certified downstream steps.
    Real native transitions throughout, fixed comparable starting world.
  autonomous_shared: one SAME 4B LoRA controls ALL branching roles from
    initial state to terminal, no role-specific adapters. Different source
    packet choices mean worlds are not paired counterfactuals. Diagnostic only.
Can compare entity-alias text interventions. No DG36 or test root evaluated.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter,defaultdict
from dataclasses import asdict
import json,sys,time,argparse
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.data_v4_env import V4ChainEnv,V31Case,Task,native_before_release,text_surface
from kci.diverse_ci_branch_env import digest
from kci_v4_semantic_eval import FrozenSemanticModels
from kci_v4_semantic_action import assert_views_position_invariant
RAW,VERIFY="SEND_RAW_R","VERIFY_PRIVATE_R"
METHODS=("aware_ce","aware_dpo","blind_ce")
def read(path):return [json.loads(s) for s in Path(path).read_text(encoding="utf8").splitlines() if s.strip()]
def spec_table(data_root):
    roots=read(data_root/"DEV_ROOT_SPECS.jsonl")
    assert {r["root_id"] for r in roots}=={"DG34","DG35"}
    out={}
    for r in roots:
        sp=dict(r["native_v31_compatible_spec"]);sp["task"]=Task(**sp["task"])
        out[r["root_id"]]=V31Case(**sp)
    return out
def fixed_suffix_action(env,state):
    opts=tuple(env.options(state))
    stage=state.native.stage
    if stage=="attest":a="SEND_CERT_R"
    elif "COMMIT_RESULT" in opts:a="COMMIT_RESULT"
    elif stage=="choose_fact":a=next(a for a in opts if a.startswith("READ:"))
    elif stage=="broker":a="FORWARD_CERT_R"
    else:raise ValueError(("No certified fixed suffix action",stage,opts))
    assert a in opts
    return a
def progress(env,state,action,mode,role,prob=None):
    old=digest(asdict(state))
    known_before=sorted(env.native.knowledge(state.native,"R"))
    ns,v=env.transition(state,action)
    rec={
       "actor":role,"stage":state.native.stage,"native_action":action,
       "action_source":mode,"verdict":v["verdict"],
       "native_state_before_hash":old,"native_state_after_hash":digest(asdict(ns)),
       "model_semantic_probs":prob,
       "R_knowledge_before":known_before,
       "R_knowledge_after":sorted(env.native.knowledge(ns.native,"R")),
       "R_delivered_receipt_count":len(ns.native.ledger),
       "native_steps_total":ns.native.steps}
    rec["native_action_executable_and_accepted"]=v["verdict"] in ("VALID","VIOLATION")
    return ns,rec
def run_env(root_specs,world,method,view,protocol,model):
    assert world["split"]=="dev" and world["root_id"] in ("DG34","DG35")
    env=V4ChainEnv(root_specs[world["root_id"]],
        world["first_packet_kind"],world["second_packet_kind"],
        world["explicit_authorization"],world["reversed_ledger_delivery_order"])
    expected=world["kind"]
    if protocol=="controlled_release":
        state,prefix=native_before_release(env,world["upstream_actual_action"].removeprefix("A_SEND_"))
        assert digest(asdict(state))==world["native_state_hash"]
        trace=[{"action_source":"frozen_certified_A_B_prefix",
                "native_action":p["action"],"actor":p["actor"],
                "native_state_before_hash":p["before"],
                "native_state_after_hash":p["after"],
                "verdict":p["verdict"]} for p in prefix]
        assert state.native.stage=="release"
    else:
        state=env.initial()
        trace=[]
    n_model=0
    for t in range(48):
        if state.native.done:break
        options=list(env.options(state))
        if not options:raise ValueError(("No legal actions",state.native.stage))
        obs=env.observation(state)
        actor=obs.get("actor",obs.get("role"))
        if protocol=="autonomous_shared" or state.native.stage=="release":
            prompt,actions=text_surface(obs,alias_case="CASE-ALIAS-FOR-ROBUSTNESS" if view=="entity_alias" else None)
            assert actions==options,(actions,options)
            if view=="native":
                other,opts2=text_surface(obs,reverse_options=True)
                assert assert_views_position_invariant(prompt,actions,other,opts2)
            probs,_=model.distribution(prompt,actions,method)
            action=max(probs,key=probs.get)
            origin="one_shared_4B_LoRA"
            n_model+=1
        else:
            action=fixed_suffix_action(env,state)
            probs=None;origin="same_frozen_certified_native_suffix"
        assert action in options
        state,record=progress(env,state,action,origin,actor,probs)
        trace.append(record)
        if record["verdict"] not in ("VALID","VIOLATION") or record["verdict"]=="VIOLATION":
            break
    # Native UNKNOWN or INVALID_ACTION is a REAL failed action, not task
    # completion; preserve model action and unchanged native state hash.
    native_done=bool(state.native.done)
    invalid_terminal=not native_done
    return {
      "root":world["root_id"],"world":world["world_id"],"condition":expected,
      "role_view":view,"protocol":protocol,"policy":method,
      "split":"dev","native_actual_success":bool(state.native.success and not state.native.violation and native_done),
      "native_actual_violation":bool(state.native.violation),
      "native_done":native_done,
      "native_incomplete_due_rejected_or_UNKNOWN_action":invalid_terminal,
      "last_native_verdict":trace[-1].get("verdict") if trace else "NONE",
      "native_abort":bool(state.native.aborted),
      "native_execution_steps":state.native.steps,
      "terminal_native_hash":digest(asdict(state)),
      "business_result":state.native.generated_result,
      "native_ledger_at_terminal":state.native.ledger,
      "native_R_knowledge_at_terminal":sorted(env.native.knowledge(state.native,"R")),
      "model_decision_count":n_model,
      "model_roles":dict(Counter(x["actor"] for x in trace if x["action_source"]=="one_shared_4B_LoRA")),
      "all_actions_attempted_against_native_env":True,
      "prefix_scripted":protocol=="controlled_release",
      "same_fixed_script_across_methods":protocol=="controlled_release",
      "not_model_Qpi_value_estimation_or_native_Qstar":True,
      "stepwise_trace":trace,
    }

def summarize(rows):
    result={}
    for protocol in sorted({r["protocol"] for r in rows}):
        result[protocol]={}
        for view in sorted({r["role_view"] for r in rows}):
            items=[r for r in rows if r["protocol"]==protocol and r["role_view"]==view]
            if not items:continue
            result[protocol][view]={}
            for method in METHODS:
                trials=[r for r in items if r["policy"]==method]
                assert len(trials)==10
                result[protocol][view][method]={
                   "n_native_team_episodes":len(trials),
                   "unique_business_roots":len(set(r["root"] for r in trials)),
                   "native_safe_task_success":sum(r["native_actual_success"] for r in trials),
                   "native_CI_violations":sum(r["native_actual_violation"] for r in trials),
                   "native_aborts":sum(r["native_abort"] for r in trials),
                   "native_incomplete_UNKNOWN_or_rejected":sum(r["native_incomplete_due_rejected_or_UNKNOWN_action"] for r in trials),
                   "last_native_verdicts":dict(Counter(r["last_native_verdict"] for r in trials)),
                   "all_model_decisions":sum(r["model_decision_count"] for r in trials),
                   "model_roles":dict(sum((Counter(r["model_roles"]) for r in trials),Counter())),
                   "mean_attempted_steps":sum(r["native_execution_steps"] for r in trials)/len(trials),
                   "mean_completed_success_steps":(sum(r["native_execution_steps"] for r in trials if r["native_actual_success"])/sum(r["native_actual_success"] for r in trials) if any(r["native_actual_success"] for r in trials) else None),
                   "unrelated_history_hold_consistent":None,
                }
                by_root=defaultdict(dict)
                for r in trials:by_root[r["root"]][r["condition"]]=r
                result[protocol][view][method]["native_risky_unsafe_episodes"]=sum(
                   v["risky"]["native_actual_violation"] for v in by_root.values())
                result[protocol][view][method]["native_safe_completed_episodes"]=sum(
                   v["safe"]["native_actual_success"] for v in by_root.values())
                result[protocol][view][method]["native_authorized_completed_episodes"]=sum(
                   v["authorized"]["native_actual_success"] for v in by_root.values())
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--base",required=True)
    p.add_argument("--old-R",required=True)
    p.add_argument("--aware-CE",required=True)
    p.add_argument("--aware-DPO",required=True)
    p.add_argument("--blind-CE",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--autonomous",action="store_true")
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    specs=spec_table(args.data_root)
    worlds=read(args.data_root/"DEV_WORLD_CONFIGS.jsonl")
    assert len(worlds)==10 and {r["root_id"] for r in worlds}=={"DG34","DG35"}
    model=FrozenSemanticModels(args.base,args.aware_CE,args.aware_DPO,args.blind_CE,args.old_R)
    protocols=["controlled_release"]
    if args.autonomous:protocols.append("autonomous_shared")
    rows=[];t=time.monotonic()
    args.output.mkdir(parents=True)
    with (args.output/"FULL_NATIVE_TEAM_EPISODES.jsonl").open("x",encoding="utf8") as f:
        with torch.inference_mode():
            for protocol in protocols:
                for view in (("native","entity_alias") if protocol=="controlled_release" else ("native",)):
                    for method in METHODS:
                        for w in worlds:
                            result=run_env(specs,w,method,view,protocol,model)
                            rows.append(result)
                            f.write(json.dumps(result,sort_keys=True,ensure_ascii=False)+"\n")
                            f.flush()
                            print("V4_NATIVE_AGENT",protocol,view,method,w["root_id"],w["kind"],
                                  "success",result["native_actual_success"],
                                  "violation",result["native_actual_violation"],
                                  "abort",result["native_abort"],
                                  "steps",result["native_execution_steps"],
                                  "model_decisions",result["model_decision_count"],flush=True)
    summary={
      "protocols":protocols,"dev_roots":["DG34","DG35"],
      "source_v4_development_worlds_only":True,"DG36_never_read":True,
      "different_truth_of_autonomous_worlds_not_counterfactual_pair_metric":True,
      "actual_native_steps_and_R_receipts_replayed":True,
      "same_shared_model_adapter_controls_all_nontrivial_roles_in_autonomous_protocol":True,
      "full_native_episode_count":len(rows),"total_model_forward_calls":dict(model.calls),
      "total_model_input_tokens":dict(model.token_counts),
      "wall_seconds":round(time.monotonic()-t,2),
      "summary":summarize(rows)}
    (args.output/"SUMMARY.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("V4_NATIVE_AGENT_FINISHED",json.dumps(summary["summary"],ensure_ascii=False)[:14000],flush=True)
if __name__=="__main__":main()
