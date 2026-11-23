"""Full-workflow SFT demonstrations from genuine native safe-success witnesses.

Source = TRAIN Phase2 native roots only. No dev/final Qstar, future preferences,
RL rewards or chosen future A label. All actions including ABORT remain legal.
Enumerate real native branches and retain only complete CI-safe business paths.
A_SEND targets excluded from SFT to prevent baking in CCE's upstream teacher.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from itertools import product
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.v4_phase2_workflows import Phase2Env,Workflow,WorkStep,RECEIVERS
from kci.diverse_ci_branch_env import digest
from kci.v4_phase2_data import render
SOURCE=ROOT/"artifacts/20261123_cce_phase2_mechanism_v1/FROZEN_UPSTREAM_A_STATE_FORKS_TRAIN_ONLY.jsonl"
OUT=ROOT/"artifacts/20261124_cce_downstream_sft_v1"
ROOTS=("P201","P202","P203","P205","P206","P207","P208")
STAGES=("A_READ","A_SEND","pre","preflight","evidence","release","attest","broker","post")
ACTION_CHOICES=("RAW","VERIFY","BROKER")
def jl(p):return [json.loads(l) for l in Path(p).open(encoding="utf8") if l.strip()]
def sha(p):return sha256(Path(p).read_bytes()).hexdigest()
def workflow_from_source(w):
    x=dict(w)
    for stage in ("pre","post"):
        x[stage]=tuple(WorkStep(**{**s,"depends":tuple(s.get("depends",[]))}) for s in x[stage])
    wf=Workflow(**x)
    assert wf.id in ROOTS and wf.split=="train"
    return wf

def run_plan(wf,variant,auth,action_a,private,pattern):
    env=Phase2Env(wf,variant=variant,authorized=auth)
    state=env.initial()
    events=[]
    for _ in range(55):
        if state.done:break
        actions=list(env.options(state))
        if not actions:return None
        role_obs=env.observation(state)
        stage=state.stage
        if stage=="A_READ":action="A_READ_VERIFIED_PACKAGES"
        elif stage=="A_SEND":action=action_a
        elif stage in ("pre","post"):
            enabled=[a for a in actions if a.startswith("DO:")]
            action=enabled[0] if enabled else "COMMIT_RESULT"
        elif stage=="preflight":action="OPEN_PRIVATE_CHANNEL" if private else "SKIP_PRIVATE_CHANNEL"
        elif stage=="evidence":
            action=next(a for a in actions if a.startswith("READ:"))
        elif stage=="release":
            choose=pattern[state.evidence_index] if state.evidence_index<len(pattern) else "RAW"
            action={"RAW":"SEND_RAW_TARGET","VERIFY":"VERIFY_PRIVATE_TARGET",
                    "BROKER":"SEND_RAW_BROKER"}[choose]
        elif stage=="attest":action="SEND_CERT_TARGET"
        elif stage=="broker":action="BROKER_FORWARD_CERT_TARGET"
        else:raise RuntimeError(("Unknown stage",stage,actions))
        if action not in actions:return None
        previous=digest(asdict(state))
        recipient_before={r:sorted(env.knowledge(state,r)) for r in RECEIVERS}
        after,outcome=env.transition(state,action)
        if outcome["verdict"] not in ("VALID","VIOLATION"):return None
        event={
           "role":role_obs["actor"],"stage":stage,"action":action,
           "legal_actions":actions,
           "role_authorized_observation":role_obs,
           "role_observation_sha256":digest(role_obs),
           "native_state_before_sha256":previous,
           "native_state_after_sha256":digest(asdict(after)),
           "native_verdict":outcome["verdict"],
           "recipient_knowledge_before":recipient_before,
           "recipient_knowledge_after":{r:sorted(env.knowledge(after,r)) for r in RECEIVERS},
           "native_steps_after":after.steps}
        events.append(event);state=after
        if state.violation:return None
    if not state.done or not state.success or state.violation or state.deadline_missed or state.aborted:
        return None
    return {"root":wf.id,"variant":variant,"authorized":auth,
       "A_action":action_a,"preflight_private":private,
       "release_pattern":list(pattern),"n_events":len(events),
       "native_final_hash":digest(asdict(state)),
       "terminal_safe_task_success":True,"native_cost":state.steps,
       "actual_business_result":state.business_result,
       "recipient_ledger_final":state.ledger,
       "events":events,"verified_witness_only_not_Qstar_not_4B_Qpi":True}

def select_frozen():
    if OUT.exists():raise FileExistsError(OUT)
    rows=jl(SOURCE)
    seen={}
    for row in rows:
        r=row["root"];assert r in ROOTS and row["split"]=="train"
        if r not in seen:seen[r]=workflow_from_source(row["native_root_spec"])
        assert asdict(seen[r])==row["native_root_spec"] or seen[r].id==r
    assert sorted(seen)==sorted(ROOTS)
    witnesses=[];attempts=Counter()
    for root,wf in sorted(seen.items()):
        for variant in ("risky","safe","hold"):
            for auth in (False,True):
                env=Phase2Env(wf,variant=variant,authorized=auth)
                packet_a=("A_SEND_PACKAGE_A","A_SEND_PACKAGE_B")
                for act_a in packet_a:
                    for open_ in (False,True):
                        # Real source receipt has len(spec.required) signed proofs.
                        for release_mode in product(ACTION_CHOICES,repeat=len(wf.required)):
                            attempts["native_plan_attempts"]+=1
                            trace=run_plan(wf,variant,auth,act_a,open_,release_mode)
                            if trace is None:continue
                            witnesses.append(trace)
                            attempts["native_successful_full_witnesses"]+=1
    assert witnesses and len(witnesses)>100
    # Native witness policy selection for non-A stages: multiple certified
    # successful actions at identical actor observation are allowed. Assign
    # only one deterministic shortest-success witness action per observation
    # for simple SFT, record all alternatives; NO global Qstar statement.
    groups=defaultdict(lambda:defaultdict(list))
    for w in witnesses:
        for step in w["events"]:
            if step["stage"]=="A_SEND":continue
            obs=step["role_authorized_observation"]
            # Include contextual stage and verified signed tool provenance.
            key=digest(obs)
            groups[key][step["action"]].append({
              "root":w["root"],"variant":w["variant"],"authorized":w["authorized"],
              "stage":step["stage"],"role":step["role"],
              "action":step["action"],"witness_cost":w["native_cost"],
              "witness_terminal_hash":w["native_final_hash"],
              "observation":obs,
              "observation_hash":key,
              "preceding_native_state_hash":step["native_state_before_sha256"]})
    demos=[];ambig=Counter()
    for key,candidates in sorted(groups.items()):
        # All messages in the actor observation were actually delivered by
        # previous native transitions; never create a teacher from drafts.
        choices=sorted((min(v,key=lambda x:(x["witness_cost"],x["root"],x["witness_terminal_hash"])),a)
                        for a,v in candidates.items()) if False else None
        best=min(((min(x["witness_cost"] for x in vs),action) for action,vs in candidates.items()),
                 key=lambda x:(x[0],x[1]))
        chosen=best[1]
        record=min(candidates[chosen],key=lambda x:(x["witness_cost"],x["root"],x["witness_terminal_hash"]))
        options=list(record["observation"]["legal_actions"])
        assert chosen in options and "ABORT" in options
        assert "oracle" not in record["observation"]
        assert record["stage"] in STAGES and record["stage"]!="A_SEND"
        assert all(x["root"]==record["root"] for values in candidates.values() for x in values)
        offered={act:min(x["witness_cost"] for x in vs) for act,vs in candidates.items()}
        if len(candidates)>1:ambig[record["stage"]]+=1
        demos.append({
           "root":record["root"],"stage":record["stage"],"role":record["role"],
           "split":"train","native_role_observation":record["observation"],
           "same_observation_sha256":key,
           "chosen_action":chosen,"all_executable_actions":options,
           "verified_native_full_success_witness":True,
           "native_path_witness_hash":record["witness_terminal_hash"],
           "original_path_cost":record["witness_cost"],
           "multiple_successful_native_action_witnesses":offered,
           "note":"min certified witnessed step-cost tie-break; not exact Qstar",
           "ABORT_preserved_in_action_menu":True,
           "not_a_future_A_preference":True,
        })
    assert demos and len(set(x["same_observation_sha256"] for x in demos))==len(demos)
    assert any(x["stage"]=="pre" for x in demos) and any(x["stage"]=="preflight" for x in demos)
    assert any(x["stage"]=="evidence" for x in demos) and any(x["stage"]=="release" for x in demos)
    assert any(x["stage"]=="post" for x in demos) and any(x["stage"]=="attest" for x in demos)
    assert all("ABORT" in x["all_executable_actions"] for x in demos)
    assert "A_SEND" not in {x["stage"] for x in demos}
    OUT.mkdir(parents=True)
    def dump(name,data):
        p=OUT/name
        p.write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in data),encoding="utf8")
        return sha(p)
    whash=dump("NATIVE_SAFE_FULL_TEAM_WITNESSES_TRAIN_ONLY.jsonl",witnesses)
    dhash=dump("SHARED_ROLE_ACTION_SFT_TRAIN_ONLY.jsonl",demos)
    stage_counts=Counter(x["stage"] for x in demos)
    role_counts=Counter(x["role"] for x in demos)
    acts=Counter(x["chosen_action"] for x in demos)
    meta={
       "status":"FROZEN_TRAIN_NATIVE_SUCCESS_WITNESS_ACTION_SFT",
       "source_frozen_CCE_A_state_sha256":sha(SOURCE),
       "native_witness_attempted":attempts["native_plan_attempts"],
       "native_full_success_witnesses":len(witnesses),
       "unique_role_observation_sft_samples":len(demos),
       "independent_train_roots":len(seen),
       "train_roots":sorted(seen),
       "observation_stages":dict(stage_counts),
       "observation_roles":dict(role_counts),
       "demo_action_counts":dict(acts),
       "multisuccess_ambiguous_observations_by_stage":dict(ambig),
       "successful_native_witness_sha256":whash,
       "SFT_rows_sha256":dhash,
       "no_model_Qstar_labels_in_prompts":True,
       "native_original_Phase2_not_modified":True,
       "one_shared_policy_all_roles":True,
       "ABORT_always_left_in_executable_action_options":True,
       "no_A_SEND_optimization_during_basic_execution_SFT":True,
       "A_SEND_left_for_future_CCE_A_policy_comparison":True,
       "not_a_CCE_preference_dataset":True,
       "no_dev_or_final_test_data_loaded":True,
       "dedup_by_unique_actor_observation":True,
    }
    (OUT/"SFT_WITNESS_MANIFEST.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("SFT_WITNESS_FREEZE",json.dumps(meta,ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__":select_frozen()
