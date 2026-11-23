"""Frozen KC8: real shared Qwen3-4B controls every post-history Agent action.

Method-side diagnostic ONLY. Official native CausalEnv + official chance RNG /
independent verifier supplied by the data AI. No native Qstar or future outcome
as model input. TRAIN six roots, 2 signed histories, model samples stochastic
legal actions and real signed tool feedback; no scripted suffix.
True same-A-state forks are new COLLECTED snapshots, model controls all suffix.
K8 exploratory; no post-hoc threshold adjustments, no model updates.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from importlib import util
from dataclasses import asdict
from math import comb
import argparse,json,random,sys,tarfile,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci_pisas_model_side_bridge import FrozenSharedSemantic4B,file_sha,install_isolated_precision_matched_hf
D=ROOT/"artifacts/20261123_kc8_4b_policy_signal_v1"
SRC=D/"KC8_NATIVE_CODE_SNAPSHOT_TRAIN_ONLY.tar.gz"
CONF=D/"FROZEN_PROTOCOL.json"
def load_native_snapshot():
    import kci
    # NO data AI source tree mutation; import unmodified source from checksumed
    # snapshot, with old universal HF policy module still from main project.
    assert SRC.is_file()
    p=json.loads(CONF.read_text(encoding="utf8"))
    assert file_sha(SRC)==p["code_snapshot_sha256"]
    stage=D/"KC8_NATIVE_CODE_SNAPSHOT"
    if not stage.exists():
        stage.mkdir()
        with tarfile.open(SRC,"r:gz") as f:
            assert all(x.name.startswith("code/kci/") for x in f.getmembers())
            f.extractall(stage,filter="data")
    files=stage/"code/kci"
    for k,v in p["all_input_checksums"].items():
        if k.startswith("CODE_"):
            assert file_sha(files/k.removeprefix("CODE_"))==v["SHA256"]
    if str(files) not in kci.__path__:kci.__path__.insert(0,str(files))
    from kci.kci_causal_task_contracts import candidates
    from kci.kci_causal_execution_env import CausalEnv
    from kci.kci_causal_visible_surface import model_only
    from kci.diverse_ci_branch_env import digest
    # Existing official method-side verifier is imported unchanged.
    if "kci_kc8_verified_source" not in sys.modules:
        spec=util.spec_from_file_location("kci_kc8_verified_source",
               str(D/"VERIFIER_ORIGINAL_SNAPSHOT.py"))
        module=util.module_from_spec(spec)
        sys.modules[spec.name]=module
        spec.loader.exec_module(module)
    verifier=sys.modules["kci_kc8_verified_source"]
    return candidates,CausalEnv,model_only,digest,verifier
def stable_rng(seed,root,history,episode,forced):
    seedstr=f"MODEL_CHECKPOINT_ACTION_RNG|{seed}|{root}|{history}|{episode}|{forced or 'UNFORCED'}"
    return random.Random(int(sha256(seedstr.encode()).hexdigest()[:16],16))
def receiver_knowledge(env,st):
    return {env.target:sorted(env.knowledge(st,env.target))}
def clone_prefix(env):
    st=env.initial()
    for a in ("DELIVER_PRIOR_1","DELIVER_PRIOR_2"):
        st,rep=env.transition(st,a)
        assert rep["verdict"]=="VALID" and not st.violation and not st.done
    assert st.stage=="A_SELECT"
    return st
def choices(model,case,history,seed,episode,*,force_first=None,max_events=45,do_native_verify=True):
    candidates,CausalEnv,model_only,digest,verifier=load_native_snapshot()
    env=CausalEnv(case,history)
    state=clone_prefix(env)
    # One native chance RNG per root/history/episode is used by all branches.
    chance_rng=verifier.deterministic_rng(case.id,history,seed,episode)
    model_rng=stable_rng(seed,case.id,history,episode,force_first)
    events=[];rich=[];n_model=0;forced_count=0;last_verdict=None
    for ix in range(max_events):
        if state.done:break
        before=digest(state.__dict__)
        knowledge_before=receiver_knowledge(env,state)
        actual_delivery_before=len(state.delivered)
        if state.stage in ("PRIMARY_CHANCE","SECOND_CHANCE","AUTH_CHANCE"):
            support=env.chance_outcomes(state)
            selected,nextstate=verifier.choose_chance(support,chance_rng)
            assert nextstate.signed_feedback and nextstate.signed_feedback["signature"]
            e={"type":"chance","outcome":selected,
               "before_native_state_sha256":before,
               "after_native_state_sha256":digest(nextstate.__dict__),
               "signed_feedback_signature":nextstate.signed_feedback["signature"]}
            selection_provenance="ACTUAL_SIGNED_NATIVE_CHANCE_NOT_MODEL"
            scores=None
        else:
            actor_obs=env.actor_observation(state)
            view=model_only(actor_obs)
            if ix==0 and force_first is not None:
                assert state.stage=="A_SELECT" and force_first in ("SEND_A","SEND_B")
                assert force_first in view["available_actions"]
                selected=force_first
                forced_count+=1
                scores=None
                selection_provenance="EXPERIMENT_FORCED_CURRENTLY_VALID_A"
            else:
                selected,scores=model.sample(view,model_rng)
                n_model+=1
                selection_provenance="LIVE_SHARED_QWEN3_4B_ACTION"
            nextstate,rep=env.transition(state,selected)
            last_verdict=rep["verdict"]
            if last_verdict not in ("VALID","VIOLATION"):
                raise ValueError(("invalid model sampled legal action",state.stage,selected,rep))
            e={"type":"model_action","actor":actor_obs["actor"],"action":selected,
               "observed_prompt_sha256":view["prompt_sha256"],
               "before_native_state_sha256":before,
               "after_native_state_sha256":digest(nextstate.__dict__)}
        after=digest(nextstate.__dict__)
        rec={
          "event_index":ix,"event_type":e["type"],
          "native_stage":state.stage,"model_role":(
             actor_obs["actor"] if e["type"]=="model_action" else "SIGNED_TOOL_SOURCE"),
          "native_action_or_chance":selected,
          "native_state_before_SHA256":before,
          "native_state_after_SHA256":after,
          "actual_knowledge_before":knowledge_before,
          "actual_knowledge_after":receiver_knowledge(env,nextstate),
          "actual_delivered_count_before":actual_delivery_before,
          "actual_delivered_count_after":len(nextstate.delivered),
          "new_actual_deliveries":nextstate.delivered[actual_delivery_before:],
          "business_result_after":nextstate.result,
          "native_CI_violation_after":bool(nextstate.violation),
          "native_stage_after":nextstate.stage,"native_elapsed_after":nextstate.steps,
          "selection_provenance":selection_provenance,
          "native_verdict":last_verdict if e["type"]=="model_action" else "SIGNED_CHANCE"
        }
        if scores is not None:
            rec["model_candidate_probabilities"]=scores
            rec["model_prompt_SHA256"]=view["prompt_sha256"]
            rec["model_observation_SHA256"]=view["role_observation_sha256"]
            rec["actor_legal_action_ids"]=view["available_actions"]
        events.append(e);rich.append(rec);state=nextstate
    incomplete=not state.done
    # Independent original native verifier ONLY verifies actual actions,
    # signed chance outcomes and business transition. It does NOT prove
    # checkpoint action authenticity; model runner provenance is separate.
    verified=None
    if not incomplete and do_native_verify:
        probe={
          "root_id":case.id,"history":history,"method":"OTHER_EXPLICIT",
          "checkpoint_sha256":model.adapter_sha256,"seed":seed,
          "episode_id":episode,"decision_source":"LIVE_CHECKPOINT_MODEL",
          "sampling_source":"EVALUATOR_DETERMINISTIC_NATIVE_RNG",
          "events":events,
          "claimed_safe_success":bool(state.success and not state.violation),
          "claimed_CI_violation":bool(state.violation)
        }
        verified=verifier.verify_one(probe,{case.id:case})
        assert verified["native_execution_and_actor_observations_independently_verified"]
    out={
       "root_id":case.id,"history":history,"seed":seed,
       "episode_id":episode,
       "checkpoint_sha256":model.adapter_sha256,
       "policy_id":"one_shared_Qwen3_4B_action_SFT_from_Phase2",
       "not_native_Qstar":True,
       "A_forced_by_experiment":force_first,
       "post_A_model_controls_every_agent":True,
       "n_model_selected_actions":n_model,"n_forced_A_actions":forced_count,
       "n_actual_signed_feedbacks":sum(x["type"]=="chance" for x in events),
       "task_safe_success":bool(state.done and state.success and not state.violation),
       "CI_violation":bool(state.violation),
       "safe_abort":bool(state.aborted and not state.violation),
       "deadline_missed":bool(state.deadline_missed),
       "unknown_or_incomplete":bool(incomplete),
       "native_elapsed":state.steps,
       "native_initial_A_state_sha256":rich[0]["native_state_before_SHA256"],
       "native_final_SHA256":digest(state.__dict__),
       "native_business_result":state.result,
       "recipient_knowledge_final":receiver_knowledge(env,state),
       "recipient_delivered_log_final":state.delivered,
       "real_model_action_trace":rich,
       "official_native_verifier":verified,
       "official_verifier_checks_native_not_cryptographic_model_authenticity":True
    }
    if force_first is None:
        out["official_full_episode_record"]={
           "root_id":case.id,"history":history,"method":"SFT",
           "checkpoint_sha256":model.adapter_sha256,"seed":seed,"episode_id":episode,
           "decision_source":"LIVE_CHECKPOINT_MODEL",
           "sampling_source":"EVALUATOR_DETERMINISTIC_NATIVE_RNG",
           "events":events,"claimed_safe_success":out["task_safe_success"],
           "claimed_CI_violation":out["CI_violation"]}
    else:
        out["forced_A_has_not_been_claimed_model_generated"]=True
    return out
def two_sided_sign(aa,bb):
    assert len(aa)==len(bb)
    plus=sum(x>y for x,y in zip(aa,bb))
    minus=sum(x<y for x,y in zip(aa,bb))
    n=plus+minus
    exact=1. if not n else min(1.,2*sum(comb(n,j) for j in range(max(plus,minus),n+1))/2**n)
    return {"n_discordant":n,"A_better":plus,"B_better":minus,"ties":len(aa)-n,
            "exact_two_sided_p":exact}
def analyze_cells(records,plan):
    by=defaultdict(dict)
    for x in records:
        key=x["root_id"]+"__"+x["history"]
        assert (x["seed"],x["A_forced_by_experiment"]) not in by[key]
        by[key][(x["seed"],x["A_forced_by_experiment"])]=x
    assert len(by)==plan["fork_n_distinct_train_roots"]
    final=[];accepted=[]
    for key,cases in sorted(by.items()):
        roots=key.split("__")
        seeds=plan["fork_seeds"]
        a=[cases[(s,"SEND_A")] for s in seeds]
        b=[cases[(s,"SEND_B")] for s in seeds]
        assert all(x["native_initial_A_state_sha256"]==a[0]["native_initial_A_state_sha256"]
                   for x in a+b)
        assert all(not x["unknown_or_incomplete"] for x in a+b)
        risk=two_sided_sign([int(not x["CI_violation"]) for x in a],
                            [int(not x["CI_violation"]) for x in b])
        task=two_sided_sign([int(x["task_safe_success"]) for x in a],
                            [int(x["task_safe_success"]) for x in b])
        reward=lambda x:(-1. if x["CI_violation"] else 1. if x["task_safe_success"] else 0.)
        ra=sum(map(reward,a))/len(a);rb=sum(map(reward,b))/len(b)
        terminal=("SEND_A" if ra>rb else "SEND_B" if rb>ra else None)
        # Pre-registered K8 exploratory ONLY; min exact p = 0.0078125
        # cannot pass familywise <0.00625 even if all pairs disagree.
        confirmed=None
        reason="K8_EXPLORATORY_NO_CONFIRMATORY_CCE_LABEL"
        if risk["n_discordant"] and risk["exact_two_sided_p"]<=plan["bonferroni_CI_first_threshold"]:
            confirmed="SEND_A" if risk["A_better"]>risk["B_better"] else "SEND_B"
            reason="FAMILYWISE_RISK_CONFIRMED"
        elif risk["n_discordant"]:
            reason="ABSTAIN_UNRESOLVED_FUTURE_CI_RISK"
        elif task["n_discordant"]:
            reason="ABSTAIN_NO_STRUCTURAL_RISK_EQUIVALENCE_AND_K8_LOW_POWER"
        else:
            reason="ABSTAIN_EQUAL_OR_UNINFORMATIVE_TERMINAL_OUTCOME"
        if confirmed:accepted.append((key,confirmed))
        final.append({"cell_id":key,"root_id":roots[0],"history":roots[1],
           "n_paired_seeds":len(seeds),
           "A_immediately_valid":True,"B_immediately_valid":True,
           "risk_pair_sign":risk,"CI_safe_completion_pair_sign":task,
           "actual_Qpi_A": {"CI":sum(x["CI_violation"] for x in a),"safe_success":sum(x["task_safe_success"] for x in a),
                           "abort":sum(x["safe_abort"] for x in a),"deadline":sum(x["deadline_missed"] for x in a)},
           "actual_Qpi_B":{"CI":sum(x["CI_violation"] for x in b),"safe_success":sum(x["task_safe_success"] for x in b),
                           "abort":sum(x["safe_abort"] for x in b),"deadline":sum(x["deadline_missed"] for x in b)},
           "ordinary_terminal_scalar_branch_mean": {"A":ra,"B":rb,"unfiltered_direction":terminal},
           "CCE_confirmed_direction":confirmed,"CCE_evidence_status":reason,
           "DPO_teacher_export_allowed":bool(confirmed and not plan["K8_is_exploratory_NOT_confirmatory"]),
           "same_policy_same_trajectories_for_CCE_vs_terminal":True,
           "not_native_Qstar":True})
    assert not accepted
    return final
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model-base",required=True)
    p.add_argument("--model-adapter",required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    assert args.output.parent.resolve()==D.resolve()
    if args.output.exists():raise FileExistsError(args.output)
    prereg=json.loads(CONF.read_text(encoding="utf8"))
    assert file_sha(SRC)==prereg["code_snapshot_sha256"]
    assert file_sha(Path(args.model_adapter)/"adapter_model.safetensors")==prereg["checkpoint_SHA256"]
    candidates,CausalEnv,model_only,digest,verifier=load_native_snapshot()
    cases={c.id:c for c in candidates() if c.split=="train" and c.id in prereg["train_root_ids"]}
    assert len(cases)==6
    patch=json.loads((D/"PRECISION_RUNTIME_ISOLATED_PATCH.json").read_text(encoding="utf8"))
    assert patch["source_frozen_protocol_sha256"]==file_sha(CONF)
    install_isolated_precision_matched_hf(
        study_dir=D,expected_sha256=patch["audited_HF_policy_sha256"])
    model=FrozenSharedSemantic4B(args.model_base,args.model_adapter,
                                max_prompt_tokens=prereg["max_prompt_tokens"])
    start=time.monotonic()
    args.output.mkdir()
    full=[]
    with (args.output/"REAL_4B_TRAIN_FULL_MODEL_NATIVE_EPISODES.jsonl").open("x",encoding="utf8") as f:
        for root,c in sorted(cases.items()):
            for h in prereg["two_histories"]:
                for seed in prereg["full_episode_seeds"]:
                    x=choices(model,c,h,seed,prereg["full_episode_id"])
                    assert not x["A_forced_by_experiment"]
                    f.write(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n");f.flush()
                    full.append(x)
                    print("KC8_REAL_4B_FULL",root,h,seed,
                        x["task_safe_success"],x["CI_violation"],
                        x["safe_abort"],x["deadline_missed"],x["n_model_selected_actions"],flush=True)
    assert len(full)==prereg["n_full_model_episodes"]==24
    # Freeze same-current-legal A pairs with four independent roots selected
    # by hash BEFORE reading future fork outcomes.
    eligible=[]
    for root,c in sorted(cases.items()):
        for h in prereg["two_histories"]:
            env=CausalEnv(c,h);st=clone_prefix(env)
            assert all(a in env.options(st) for a in ("SEND_A","SEND_B"))
            current=[env.transition(st,a)[1]["verdict"] for a in ("SEND_A","SEND_B")]
            if current==["VALID","VALID"]:
                eligible.append((root,h,digest(st.__dict__)))
    chosen=[]
    for root in sorted(cases):
        variants=[x for x in eligible if x[0]==root]
        if variants:
            chosen.append(min(variants,key=lambda x:sha256(f"{x[0]}|{x[1]}".encode()).hexdigest()))
    chosen=sorted(chosen,key=lambda x:sha256(f"KC8_FORK_SELECTED|{x[0]}|{x[1]}".encode()).hexdigest())[:prereg["fork_n_distinct_train_roots"]]
    assert len(chosen)==4 and len({x[0] for x in chosen})==4
    selections=[{"root_id":r,"history":h,"A_initial_state_SHA256":sh,
                 "both_A_current_native_valid":True} for r,h,sh in chosen]
    (args.output/"FROZEN_FORK_SELECTION_BEFORE_BRANCH_OUTCOMES.json").write_text(
       json.dumps({"predeclared_selection_sha256":file_sha(CONF),
          "n_eligible":len(eligible),"selected":selections,"not_selected_by_Qpi_or_native_Qstar":True},
          indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    forks=[]
    with (args.output/"REAL_4B_TRAIN_FORCED_A_FORKS.jsonl").open("x",encoding="utf8") as f:
        for root,h,_ in chosen:
            for seed in prereg["fork_seeds"]:
                for action in ("SEND_A","SEND_B"):
                    x=choices(model,cases[root],h,seed,0,force_first=action)
                    assert x["A_forced_by_experiment"]==action
                    f.write(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n");f.flush()
                    forks.append(x)
            print("KC8_REAL_4B_FORKS",root,h,2*len(prereg["fork_seeds"]),flush=True)
    assert len(forks)==4*2*8
    stats=analyze_cells(forks,prereg)
    (args.output/"EXPLORATORY_CCE_VS_TERMINAL_SIGNALS.json").write_text(
       json.dumps({"protocol":file_sha(CONF),"K8_exploratory":True,
          "labels_confirmed_for_training":0,"states":stats},ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    # Only exact TRAIN teacher *prompt hashes* count as existing model-safe
    # observation supervision coverage. No Qstar input to model.
    saved=ROOT/"artifacts/20261123_kc8_4b_policy_signal_v1/FROZEN_KC8_TRAIN_INPUTS"
    teacherfiles=[
      saved/"TRAIN_EARLY_A_ACTION_CE_IDENTICAL_STATES.jsonl",
      saved/"TRAIN_STRICT_SIGNED_FEEDBACK_ACTION_CE.jsonl"]
    known=set()
    for path in teacherfiles:
       for line in path.open(encoding="utf8"):
           row=json.loads(line)
           known.add(row["prompt_sha256"])
    reached=set(
       ev["model_prompt_SHA256"] for row in full for ev in row["real_model_action_trace"]
       if ev.get("model_prompt_SHA256"))
    visited_root=Counter(x["root_id"] for x in full)
    report={
        "status":"KC8_TRAIN_REAL_4B_POLICY_DIAGNOSTIC_NO_CCE_TRAINING",
        "real_full_native_4B_episodes":len(full),
        "distinct_business_train_roots":len(visited_root),
        "full_model_actions":sum(x["n_model_selected_actions"] for x in full),
        "actual_signed_chance_feedback":sum(x["n_actual_signed_feedbacks"] for x in full),
        "CI_safe_task_success":sum(x["task_safe_success"] for x in full),
        "CI_violations":sum(x["CI_violation"] for x in full),
        "safe_abort":sum(x["safe_abort"] for x in full),
        "deadline_missed":sum(x["deadline_missed"] for x in full),
        "unknown_incomplete":sum(x["unknown_or_incomplete"] for x in full),
        "actual_native_elapsed_sum":sum(x["native_elapsed"] for x in full),
        "official_native_replay_pass":sum(x["official_native_verifier"] is not None for x in full),
        "n_independent_A_legal_fork_states":len(chosen),
        "real_full_frozen_4B_suffix_fork_trajectories":len(forks),
        "fork_CCE_confirmed_training_labels":0,
        "n_visited_distinct_exact_actor_prompt_SHA256":len(reached),
        "n_visited_exact_teacher_covered":len(reached&known),
        "n_visited_exact_teacher_uncovered":len(reached-known),
        "checkpoint_SHA256":model.adapter_sha256,
        "model_score_costs":model.costs(),
        "wall_seconds_excluding_model_loading":time.monotonic()-start,
        "role_summary":dict(Counter(ev["model_role"] for x in full for ev in x["real_model_action_trace"] if ev["selection_provenance"]=="LIVE_SHARED_QWEN3_4B_ACTION")),
        "native_action_summary":dict(Counter(ev["native_action_or_chance"] for x in full for ev in x["real_model_action_trace"] if ev["selection_provenance"]=="LIVE_SHARED_QWEN3_4B_ACTION")),
        "Qpi_not_native_Qstar":True,
        "no_4B_gradient_updates":True,
        "no_DEV_TEST_native_labels_accessed":True,
        "checkpoint_actual_source":"Phase2 shared workflow Action SFT; not trained on KC8",
        "notes":"K8 low-power exploratory fork signals are not reliable preference labels; same K8 samples for scalar terminal and CI-first CCE."}
    (args.output/"SUMMARY.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("KC8_REAL_4B_DIAGNOSTIC_FINISH",json.dumps(report,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
