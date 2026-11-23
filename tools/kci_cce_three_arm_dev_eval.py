"""Genuine frozen 4B shared-team held-out Phase2 business-root evaluation.

Evaluate ONE prior downstream SFT adapter + three separately fine-tuned A
signal arms, each with same model architecture, same visible information,
same 7 TRAIN A states and same 48 optimizer updates.

These are THREE new business roots (P204/P209/P210), but knowledge Boolean
mechanisms have known cross-cohort overlap. DG36, P211,P212 final test sealed.
No CCE training labels or Qstar ever enter model DEV prompts.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
from random import Random
import argparse,json,time,torch,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci.agent_rl.hf_policy import HFActionPolicy
from kci.v4_phase2_workflows import Phase2Env,RECEIVERS
from kci_cce_phase2_frozen4b_rollout import (
    model_candidate_prompt,create_workflow,native_after_action,sample_action)
from kci_cce_adaptive_protocol import FIRST_SEEDS,sha
METHODS=("downstream_sft_untrained_A","action_ce","terminal","CCE")
ROOTS=("P204","P209","P210")
CONDITIONS=("risky_noauth","authorized_risky_package","hold_noauth")
def jl(p):return [json.loads(s) for s in Path(p).open(encoding="utf8") if s.strip()]
def deep_hash(p):
    h=sha256()
    with Path(p).open("rb") as f:
        for ch in iter(lambda:f.read(1048576),b""):h.update(ch)
    return h.hexdigest()
class Frozen4BAdapters:
    def __init__(self,base,sft,trained):
        self.hf=HFActionPolicy(model_path=base,adapter=str(sft),
            trainable=False,quantization="4bit",dtype="bf16",
            max_prompt_tokens=4096,allow_download=False)
        for method,path in trained.items():
            self.hf.model.load_adapter(str(path),adapter_name=method,is_trainable=False)
        self.hf.model.requires_grad_(False)
        self.hf.model.eval()
        self.cache={}
        self.forward_counts=Counter()
        self.token_counts=Counter()
        self.cache_hits=Counter()
    @torch.inference_mode()
    def get_distribution(self,obs,method):
        acts=tuple(sorted(obs["legal_actions"]))
        prompt_hash=sha256(json.dumps(obs,ensure_ascii=False,sort_keys=True).encode("utf8")).hexdigest()
        key=(method,prompt_hash,acts)
        if key in self.cache:
            self.cache_hits[method]+=1
            return self.cache[key]
        adapter="default" if method=="downstream_sft_untrained_A" else method
        self.hf.model.set_adapter(adapter)
        scores=[];token=0
        for a in acts:
            prompt=model_candidate_prompt(obs,a)
            p=self.hf.log_distribution(prompt,("ACCEPT","REJECT"))
            assert torch.isfinite(p).all()
            scores.append(float((p[0]-p[1]).detach()))
            token+=int(self.hf._input_ids(prompt).numel())
        probs=torch.softmax(torch.tensor(scores,dtype=torch.float32),dim=0).tolist()
        result={"P":dict(zip(acts,[float(x) for x in probs])),
            "scores":dict(zip(acts,scores)),"method":method,
            "native_actions":list(acts)}
        self.forward_counts[method]+=len(acts)
        self.token_counts[method]+=token
        self.cache[key]=result
        return result

def episode(wf,condition,method,seed,policies):
    assert wf.split=="dev" and wf.id in ROOTS
    if condition=="risky_noauth":
        env=Phase2Env(wf,variant="risky",authorized=False)
    elif condition=="authorized_risky_package":
        env=Phase2Env(wf,variant="risky",authorized=True)
    elif condition=="hold_noauth":
        env=Phase2Env(wf,variant="hold",authorized=False)
    else:raise ValueError(condition)
    st=env.initial()
    rng=Random(seed)
    actions=[];chosen_A=None;stop_cause=None
    for _ in range(50):
        if st.done:break
        obs=env.observation(st)
        pred=policies.get_distribution(obs,method)
        a,draw=sample_action(pred["P"],rng)
        assert a in env.options(st)
        stage=st.stage
        next_,event=native_after_action(env,st,a,"SINGLE_SHARED_FROZEN_4B",
               prob=pred["P"],draw=draw)
        event["candidate_scores"]=pred["scores"]
        event["actor_visible_observation_sha256"]=sha256(json.dumps(obs,ensure_ascii=False,sort_keys=True).encode("utf8")).hexdigest()
        event["upstream_A_choice"]=stage=="A_SEND"
        actions.append(event);st=next_
        if stage=="A_SEND":chosen_A=a
        if event["native_verdict"] not in ("VALID","VIOLATION"):
            stop_cause="UNKNOWN_ACTION_OR_TRANSITION"
            break
        if event["native_verdict"]=="VIOLATION":break
    if not st.done and not stop_cause:stop_cause="MAX_50_ACTIONS_NO_TERMINAL"
    safe=chosen_A=="A_SEND_"+env.packet_by_kind["safe"]
    risky=chosen_A=="A_SEND_"+env.packet_by_kind["risky"]
    return {
       "root":wf.id,"condition":condition,"split":"dev",
       "model":method,"seed":seed,
       "A_choice":chosen_A,"A_chose_safe_package":bool(safe),
       "A_chose_risky_package":bool(risky),
       "A_action_was_executed":chosen_A in ("A_SEND_PACKAGE_A","A_SEND_PACKAGE_B"),
       "task_CI_safe_success":bool(st.done and st.success and not st.violation),
       "CI_violation":bool(st.violation),
       "aborted":bool(st.aborted),
       "deadline_missed":bool(st.deadline_missed),
       "UNKNOWN_or_incomplete":bool(stop_cause),
       "completion_failure_reason":stop_cause,
       "native_execution_steps":st.steps,
       "native_business_result":st.business_result,
       "native_ledger":st.ledger,
       "native_recipient_knowledge":{r:sorted(env.knowledge(st,r)) for r in RECEIVERS},
       "n_model_actions":len(actions),
       "model_decision_roles":dict(Counter(e["actor"] for e in actions)),
       "abort_stage":next((ev["stage"] for ev in actions if ev["native_action"]=="ABORT"),None),
       "all_steps_actual_native_transitions":True,
       "real_4B_no_Qstar_in_prompt":True,
       "native_stepwise_trace":actions}

def summarize(rows):
    bymethod={}
    for method in METHODS:
        data=[x for x in rows if x["model"]==method]
        assert len(data)==len(ROOTS)*len(CONDITIONS)*len(FIRST_SEEDS)==72
        d={"n_native_4B_end_to_end_episodes":len(data),
          "n_independent_DEV_business_roots":3,
          "safe_task_success":sum(x["task_CI_safe_success"] for x in data),
          "CI_violations":sum(x["CI_violation"] for x in data),
          "safe_aborts":sum(x["aborted"] and not x["CI_violation"] for x in data),
          "deadline_misses":sum(x["deadline_missed"] for x in data),
          "UNKNOWN_or_incomplete":sum(x["UNKNOWN_or_incomplete"] for x in data),
          "n_model_actions":sum(x["n_model_actions"] for x in data),
          "A_executed_count":sum(x["A_action_was_executed"] for x in data),
          "A_safe_on_unlicensed_risky":sum(x["A_chose_safe_package"] for x in data if x["condition"]=="risky_noauth"),
          "A_safe_on_unlicensed_risky_den":sum(x["condition"]=="risky_noauth" for x in data),
          "A_risky_on_unlicensed_risky":sum(x["A_chose_risky_package"] for x in data if x["condition"]=="risky_noauth"),
          "A_abort_or_unreached_on_unlicensed_risky":sum(not x["A_action_was_executed"] for x in data if x["condition"]=="risky_noauth"),
          "mean_successful_native_cost":(sum(x["native_execution_steps"] for x in data if x["task_CI_safe_success"])/sum(x["task_CI_safe_success"] for x in data) if any(x["task_CI_safe_success"] for x in data) else None),
          "stage_abort_counts":dict(Counter(x["abort_stage"] for x in data if x["aborted"])),
          "by_condition":{},"by_root":{}}
        for condition in CONDITIONS:
            xs=[x for x in data if x["condition"]==condition]
            d["by_condition"][condition]={
             "n":len(xs),"safe_success":sum(x["task_CI_safe_success"] for x in xs),
             "CI_violations":sum(x["CI_violation"] for x in xs),
             "abort":sum(x["aborted"] for x in xs),
             "deadline":sum(x["deadline_missed"] for x in xs),
             "A_correct_safe_where_required":sum(x["A_chose_safe_package"] for x in xs) if condition=="risky_noauth" else None}
        for root in ROOTS:
            xs=[x for x in data if x["root"]==root]
            d["by_root"][root]={"n":len(xs),"safe_success":sum(x["task_CI_safe_success"] for x in xs),
                 "CI_violations":sum(x["CI_violation"] for x in xs),
                 "abort":sum(x["aborted"] for x in xs)}
        bymethod[method]=d
    return bymethod

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",required=True)
    p.add_argument("--sft",type=Path,required=True)
    p.add_argument("--action-ce",type=Path,required=True)
    p.add_argument("--terminal",type=Path,required=True)
    p.add_argument("--cce",type=Path,required=True)
    p.add_argument("--dev",type=Path,required=True)
    p.add_argument("--dev-pre-reg",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    reg=json.loads(args.dev_pre_reg.read_text(encoding="utf8"))
    assert reg["dev_root_file_sha256"]==sha(args.dev)
    assert reg["roots"]==list(ROOTS)
    assert reg["paired_seeds"]==list(FIRST_SEEDS)
    raw=jl(args.dev)
    assert len(raw)==3 and {r["root_id"] for r in raw}==set(ROOTS)
    wf=[create_workflow(r["native_spec"]) for r in raw]
    assert all(w.split=="dev" and w.id in ROOTS for w in wf)
    paths={"downstream_sft_untrained_A":args.sft,
           "action_ce":args.action_ce,"terminal":args.terminal,"CCE":args.cce}
    adapter_sha={m:deep_hash(p/"adapter_model.safetensors") for m,p in paths.items()}
    model=Frozen4BAdapters(args.base,args.sft,{
        "action_ce":args.action_ce,"terminal":args.terminal,"CCE":args.cce})
    start=time.monotonic()
    args.output.mkdir(parents=True)
    data=[]
    with (args.output/"REAL_UNSEEN_DEV_NATIVE_EPISODES.jsonl").open("x",encoding="utf8") as f:
        with torch.inference_mode():
            for method in METHODS:
                for flow in sorted(wf,key=lambda x:x.id):
                    for condition in CONDITIONS:
                        for seed in FIRST_SEEDS:
                            result=episode(flow,condition,method,seed,model)
                            data.append(result)
                            f.write(json.dumps(result,ensure_ascii=False,sort_keys=True)+"\n")
                            f.flush()
                            print("CCE_3ARM_DEV",method,flow.id,condition,seed,
                                "safe",int(result["task_CI_safe_success"]),
                                "CI",int(result["CI_violation"]),
                                "abort",int(result["aborted"]),
                                "deadline",int(result["deadline_missed"]),
                                "A",result["A_choice"],flush=True)
    summary=summarize(data)
    doc={
       "protocol":"Phase2 3 dev business roots one shared 4B model full native team",
       "n_methods":len(METHODS),"n_distinct_dev_roots":3,
       "n_complete_native_4B_dev_episodes":len(data),
       "seed_set":list(FIRST_SEEDS),
       "model_checkpoint_sha256":adapter_sha,
       "all_from_same_SFT_shared_policy_start":True,
       "same_model_candidate_scoring_for_all":True,
       "native_verifier_and_reward_unchanged":True,
       "all_role_actions_selected_by_4B_and_executed_native":True,
       "no_A_optimal_Qstar_labels_in_4B_PROMPT":True,
       "DEV_roots_not_seen_in_model_training":True,
       "final_test_roots_P211_P212_DG36_UNREAD":True,
       "DEV_are_business_unseen_not_all_rule_families_unseen":True,
       "forward_counts_by_method":dict(model.forward_counts),
       "cache_hits_by_method":dict(model.cache_hits),
       "tokenized_model_input_tokens_by_method":dict(model.token_counts),
       "unique_method_role_obs_distribution_count":len(model.cache),
       "all_models_real_native_steps":sum(x["native_execution_steps"] for x in data),
       "all_models_model_decisions":sum(x["n_model_actions"] for x in data),
       "runtime_seconds":round(time.monotonic()-start,2),
       "by_method":summary,
    }
    (args.output/"SUMMARY.json").write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("CCE_3ARM_DEV_SUMMARY",json.dumps({
      k:{t:v[t] for t in ("safe_task_success","CI_violations","safe_aborts","deadline_misses","A_safe_on_unlicensed_risky")}
      for k,v in summary.items()},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
