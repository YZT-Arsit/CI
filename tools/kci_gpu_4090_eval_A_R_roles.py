"""Independent pretrained Qwen3-4B A-LoRA + frozen R-LoRA native evaluation.

One frozen HF 4-bit pretrained base, with two role-specific LoRA adapters.
Baseline: A/B/C = pretrained base, SourceAgent/release = learned R adapter.
Treatment: RecordsAgent/A_SEND = learned A adapter, R remains identical,
           B/C remain identical pretrained base.

No native Qstar in LM prompt or DPO training. Qstar reference is used only
for independent preference-regret diagnostics in this EVAL executable.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
from collections import Counter
from hashlib import sha256
import sys,json,argparse,time,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.agent_rl.hf_policy import HFActionPolicy
from kci_llm_text_surface import render_role_text
from kci_exact_policy_continuation_cpu import environment_public
from compare_kci_chain_methods_cpu_action_interaction import GROUP
from chain_shared_actor_cpu_action_interaction import baseline_automatic_action,valid_decision_candidates
from kci.diverse_ci_branch_env import digest

class RoutedActual4B:
    def __init__(self,base_model,A_adapter,R_adapter):
        self.hf=HFActionPolicy(
            model_path=base_model,adapter=A_adapter,trainable=False,
            quantization="4bit",dtype="bf16",allow_download=False,max_prompt_tokens=4096)
        self.hf.model.load_adapter(R_adapter,adapter_name="SourceAgent",is_trainable=False)
        self.hf.model.requires_grad_(False)
        self.hf.model.eval()
        self.cache={}
        self.model_calls=0
        self.cache_hits=0
        self.prompt_tokens=0
    def policy(self,observation,mode):
        prompt=render_role_text(observation)
        options=tuple(observation["legal_actions"])
        stage=observation.get("stage",observation.get("current_stage"))
        routed_role=("A" if stage=="A_SEND" and mode=="A_plus_R"
                     else "R" if stage=="release" else "base")
        key=(sha256(prompt.encode()).hexdigest(),options,routed_role)
        if key in self.cache:
            self.cache_hits+=1
            probabilities=self.cache[key]
        else:
            with torch.inference_mode():
                if routed_role=="A":
                    self.hf.model.set_adapter("default")
                    p=self.hf.log_distribution(prompt,options)
                elif routed_role=="R":
                    self.hf.model.set_adapter("SourceAgent")
                    p=self.hf.log_distribution(prompt,options)
                else:
                    with self.hf.model.disable_adapter():
                        p=self.hf.log_distribution(prompt,options)
            probabilities={a:float(v) for a,v in zip(options,p.exp().cpu())}
            self.cache[key]=probabilities
            self.model_calls+=1
            self.prompt_tokens+=int(len(self.hf._input_ids(prompt)[0]))
        act=max(options,key=lambda a:probabilities[a])
        return act,probabilities

def policy_run(env,model,mode,world):
    st=env.initial();trace=[];A_choice=None
    for i in range(65):
        if st.native.done:break
        obs=env.observation(st)
        valid=list(env.options(st))
        candidates=valid_decision_candidates(obs,valid)
        if len(candidates)>=2:
            choice,probs=model.policy(obs,mode)
            assert choice in candidates,(choice,candidates)
            source="Qwen3-4B"
        else:
            choice=baseline_automatic_action(env,st)
            probs=None;source="common_native_tool"
        assert choice in valid,(choice,valid)
        if choice.startswith("A_SEND_PACKAGE"):
            A_choice=choice.removeprefix("A_SEND_")
        before=digest(asdict(st))
        n_before=len(st.native.delivered["R"])
        nxt,verdict=env.transition(st,choice)
        trace.append({"stage":obs.get("stage",obs.get("current_stage")),
            "role":obs.get("actor",obs.get("role")),
            "action":choice,"source":source,
            "native_verdict":verdict["verdict"],
            "before_sha256":before,"after_sha256":digest(asdict(nxt)),
            "new_R_receipts":len(nxt.native.delivered["R"])-n_before,
            "action_probs":probs})
        if verdict["verdict"] not in ("VALID","VIOLATION"):
            raise AssertionError(verdict)
        st=nxt
    if not st.native.done:raise AssertionError("NO_TERMINAL")
    safe=bool(st.native.success and not st.native.violation)
    CI=bool(st.native.violation)
    reward=(round(1-env.spec.cost_per_step*st.native.steps,6)
            if safe else -1. if CI else 0.)
    q=world["upstream_A_values_given_exact_B_continuation"]
    gap=abs(q["PACKAGE_A"]-q["PACKAGE_B"])
    return {"safe_success":safe,"CI_violation":CI,"reward":reward,
            "native_steps":st.native.steps,"A_choice":A_choice,
            "A_strict_Qstar":gap>1e-8,
            "A_reference_regret":(float(max(q.values())-q[A_choice]) if A_choice else None),
            "trace":trace}

def initial_A(env,model,mode):
    st,verdict=env.transition(env.initial(),"A_READ_VERIFIED_PACKAGES")
    assert verdict["verdict"]=="VALID"
    act,p=model.policy(env.observation(st),mode)
    return {"action":act,"probs":p,"state_sha256":digest(asdict(st)),
            "observation_sha256":digest(env.observation(st))}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",required=True)
    p.add_argument("--A-adapter",required=True)
    p.add_argument("--R-adapter",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    path=Path(args.output)
    if path.exists():raise FileExistsError(path)
    path.mkdir(parents=True)
    model=RoutedActual4B(args.base,args.A_adapter,args.R_adapter)
    all_=[],[]
    modes=("R_only","A_plus_R")
    results=[];start=time.monotonic()
    for split,worlds in GROUP.items():
        for world in worlds:
            env=environment_public(world)
            states={mode:initial_A(env,model,mode) for mode in modes}
            executions={mode:policy_run(env,model,mode,world) for mode in modes}
            item={"split":split,"world":world["world"],
                  "root":world["root"],"graph":world["graph"],
                  "source_mode":world["source_mode"],
                  "A_distribution":states,"runs":executions}
            results.append(item)
            with (path/"WORLD_PAIRS.jsonl").open("a",encoding="utf8") as f:
                f.write(json.dumps(item,ensure_ascii=False)+"\n")
            print("QWEN4B_A_AFTER_DPO",split,world["world"],
                  "A",[(mode,states[mode]["action"],
                        round(states[mode]["probs"].get("A_SEND_PACKAGE_A",0),4),
                        executions[mode]["safe_success"],executions[mode]["CI_violation"],
                        executions[mode]["reward"]) for mode in modes],
                  "t",round(time.monotonic()-start,1),flush=True)
    summary={}
    for split in GROUP:
        by=[row for row in results if row["split"]==split]
        summary[split]={}
        for mode in modes:
            x=[r["runs"][mode] for r in by]
            p=[r["A_distribution"][mode] for r in by]
            strict=[r for r in x if r["A_strict_Qstar"]]
            summary[split][mode]={
                "worlds":len(x),"independent_roots":len(set(row["root"] for row in by)),
                "A_choose_A":sum(q["action"]=="A_SEND_PACKAGE_A" for q in p),
                "A_choose_B":sum(q["action"]=="A_SEND_PACKAGE_B" for q in p),
                "mean_P_A":sum(q["probs"].get("A_SEND_PACKAGE_A",0) for q in p)/len(p),
                "A_Qstar_optimal":sum(q["A_reference_regret"]<1e-8 for q in strict),
                "A_Qstar_strict_n":len(strict),
                "mean_A_Qstar_regret":sum(q["A_reference_regret"] for q in x)/len(x),
                "safe_success":sum(q["safe_success"] for q in x),
                "CI_violations":sum(q["CI_violation"] for q in x),
                "mean_reward":sum(q["reward"] for q in x)/len(x),
            }
    report={"status":"ACTUAL_QWEN3_4B_TRAINED_A_PLUS_FROZEN_R_NATIVE_EVAL",
            "roles":{"RecordsAgent":"A-only LoRA in treatment",
                     "SourceAgent":"same frozen R LoRA in both arms",
                     "Coordinator_and_business":"identical frozen pretrained base"},
            "source_train_only":True,
            "no_model_oracle_inputs":True,
            "development_seen_test_not_blind":True,
            "summary":summary,"model_forward_calls":model.model_calls,
            "cache_hits":model.cache_hits,"model_prompt_tokens":model.prompt_tokens,
            "runtime_seconds":round(time.monotonic()-start,2)}
    (path/"SUMMARY.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("FINAL_A_ROLE_ADAPTER",json.dumps(report,ensure_ascii=False),flush=True)
if __name__=="__main__":
    main()
