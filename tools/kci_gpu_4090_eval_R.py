"""Native KCI before/after evaluation of an actual pretrained Qwen3-4B LoRA.

Loads ONE 4-bit HF base with the trained SourceAgent adapter and compares
greedy decisions with adapter disabled (same frozen base) vs enabled ONLY at
Coordinator/preflight. All other roles use the same frozen base.
No Qstar or hidden native state in prompts; no test labels in training.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
from hashlib import sha256
from collections import Counter
import sys,json,time,argparse
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.agent_rl.hf_policy import HFActionPolicy
from kci_llm_text_surface import render_role_text
from kci_exact_policy_continuation_cpu import environment_public
from chain_shared_actor_cpu_action_interaction import baseline_automatic_action,valid_decision_candidates
from compare_kci_chain_methods_cpu_action_interaction import GROUP
from kci.diverse_ci_branch_env import digest

def preflight(env,package):
    st=env.initial()
    for action in ("A_READ_VERIFIED_PACKAGES",package):
        st,au=env.transition(st,action)
        assert au["verdict"]=="VALID"
    for _ in range(10):
        ob=env.observation(st)
        if ob.get("current_stage")=="preflight":return st
        auto=baseline_automatic_action(env,st)
        assert auto
        st,au=env.transition(st,auto)
        assert au["verdict"]=="VALID"
    raise AssertionError("No reachable B preflight")

def first_release(env,package):
    st=preflight(env,package)
    st,au=env.transition(st,"OPEN_PRIVATE_CHANNEL")
    assert au["verdict"]=="VALID"
    for _ in range(25):
        obs=env.observation(st)
        if obs.get("current_stage")=="release":
            return st
        action=baseline_automatic_action(env,st)
        if action is None:raise AssertionError(("no release",obs))
        st,au=env.transition(st,action)
        assert au["verdict"]=="VALID"
    raise AssertionError("No release state reachable")

class Policy:
    def __init__(self,base_path,adapter_path):
        self.p=HFActionPolicy(model_path=base_path,adapter=adapter_path,
            trainable=False,quantization="4bit",dtype="bf16",
            allow_download=False,max_prompt_tokens=4096)
        self.p.model.eval()
        self.cache={}
        self.cache_hits=0
        self.model_forward_calls=0
        self.tokens=0
    @torch.inference_mode()
    def decision(self,obs,mode):
        prompt=render_role_text(obs)
        choices=tuple(obs["legal_actions"])
        stage=obs.get("stage",obs.get("current_stage"))
        adapter_on=(mode=="trained_R" and stage=="release")
        key=(sha256(prompt.encode()).hexdigest(),choices,adapter_on)
        if key in self.cache:
            self.cache_hits+=1
            probs=self.cache[key]
        else:
            if adapter_on:
                self.p.model.set_adapter("default")
                logp=self.p.log_distribution(prompt,choices)
            else:
                with self.p.model.disable_adapter():
                    logp=self.p.log_distribution(prompt,choices)
            probs=logp.exp().detach().cpu().tolist()
            self.cache[key]=probs
            self.model_forward_calls+=1
            self.tokens+=len(self.p._input_ids(prompt)[0])
        idx=max(range(len(probs)),key=lambda i:probs[i])
        return choices[idx],{action:round(prob,7) for action,prob in zip(choices,probs)}

def episode(row,policy,mode):
    e=environment_public(row)
    state=e.initial();trace=[]
    for step in range(60):
        if state.native.done:break
        obs=e.observation(state)
        avail=list(e.options(state))
        selected=valid_decision_candidates(obs,avail)
        if len(selected)>1:
            action,probs=policy.decision(obs,mode)
            assert action in avail
            # All native legal decision options include CI-correctable ABORT;
            # no oracle replacement of the generated action.
            assert action in selected
            source="pretrained_LM"
        else:
            action=baseline_automatic_action(e,state)
            probs=None;source="native_shared_scripted_tool"
        if action is None:raise AssertionError("Missing native action")
        nxt,audit=e.transition(state,action)
        trace.append({"role":obs.get("actor",obs.get("role")),
            "phase":obs.get("stage",obs.get("current_stage")),"action":action,
            "source":source,"native_verdict":audit["verdict"],
            "receipts_delta":len(nxt.native.delivered["R"])-len(state.native.delivered["R"]),
            "before_sha256":digest(asdict(state)), "after_sha256":digest(asdict(nxt)),
            "probs":probs})
        if audit["verdict"] not in ("VALID","VIOLATION"):
            raise AssertionError((row["world"],audit))
        state=nxt
    success=state.native.done and state.native.success and not state.native.violation
    violated=bool(state.native.violation)
    return {"world":row["world"],"root":row["root"],"source_mode":row["source_mode"],
            "mode":mode,"safe_task_success":bool(success),"CI_violation":violated,
            "native_steps":state.native.steps,
            "reward":round(1-e.spec.cost_per_step*state.native.steps,6) if success else(-1 if violated else 0.),
            "n_R_receipts":len(state.native.delivered["R"]),"trace":trace}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",required=True)
    p.add_argument("--adapter",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    target=Path(args.output)
    if target.exists():raise FileExistsError(target)
    target.mkdir(parents=True)
    model=Policy(args.base,args.adapter)
    modes=("base","trained_R")
    data=[]
    # Original simulated CI workflow, one fixed A package (not an oracle A).
    for split,rows in GROUP.items():
        for row in rows:
            e=environment_public(row)
            st=first_release(e,"A_SEND_PACKAGE_A")
            obs=e.observation(st)
            decisions={}
            for mode in modes:
                act,prob=model.decision(obs,mode)
                decisions[mode]={"action":act,"probs":prob}
            data.append({"split":split,"world":row["world"],
                         "root":row["root"],"source_mode":row["source_mode"],
                         "R_choice":decisions,
                         "R_native_state_hash":digest(asdict(st))})
        print("R_PROBE",split,
              Counter(x["R_choice"]["base"]["action"] for x in data if x["split"]==split),
              Counter(x["R_choice"]["trained_R"]["action"] for x in data if x["split"]==split),
              flush=True)
    with (target/"B_DECISIONS.jsonl").open("w",encoding="utf8") as f:
        for r in data:f.write(json.dumps(r)+"\n")
    full=[]
    for split in ("dev","exploratory_test"):
        for row in GROUP[split]:
            for mode in modes:
                r=episode(row,model,mode)
                full.append({"split":split,**r})
                with (target/"FULL_EPISODES.jsonl").open("a",encoding="utf8") as f:
                    f.write(json.dumps({"split":split,**r})+"\n")
                print("NATIVE_4B",split,row["world"],mode,
                      "SUCCESS",r["safe_task_success"],"CI",r["CI_violation"],
                      "REWARD",r["reward"],"POLICY_STEPS",
                      [(x["phase"],x["action"]) for x in r["trace"] if x["source"]=="pretrained_LM"],
                      flush=True)
    summary={
        "status":"REAL_PRETRAINED_QWEN3_4B_Qlora_NATIVE_BEFORE_AFTER_EVAL",
        "trained_role":"SourceAgent/release only",
        "base_checkpoint":args.base,
        "trained_adapter":args.adapter,
        "train_test_task_roots_split_not_mixed":True,
        "exploratory_test_seen_by_developers":True,
        "R_probe":{split:{mode:dict(Counter(
            r["R_choice"][mode]["action"] for r in data if r["split"]==split))
            for mode in modes} for split in GROUP},
        "R_probe_mean_verify_prob":{split:{mode:
            sum(r["R_choice"][mode]["probs"]["VERIFY_PRIVATE_R"]
                for r in data if r["split"]==split) /
              len([r for r in data if r["split"]==split])
            for mode in modes} for split in GROUP},
        "full":{split:{mode:{
            "n":sum(r["split"]==split and r["mode"]==mode for r in full),
            "safe_task_success":sum(r["safe_task_success"] for r in full if r["split"]==split and r["mode"]==mode),
            "CI_violations":sum(r["CI_violation"] for r in full if r["split"]==split and r["mode"]==mode),
            "mean_native_reward":sum(r["reward"] for r in full if r["split"]==split and r["mode"]==mode)/len(GROUP[split])
            } for mode in modes} for split in ("dev","exploratory_test")},
        "hf_forward_calls":model.model_forward_calls,
        "cache_hits":model.cache_hits,
        "measured_prompt_tokens":model.tokens,
    }
    (target/"SUMMARY.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("EVAL_SUMMARY",json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
