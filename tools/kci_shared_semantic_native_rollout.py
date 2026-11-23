"""One shared 4B LoRA controls ALL learned decision roles in native KCI chains.

No A/B/R per-role adapters. Common certified source reads and deterministic
tool transitions are identical between policy conditions. No native Qstar or
verdict labels are passed to the model.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from dataclasses import asdict
from pathlib import Path
import argparse,json,sys,time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci.agent_rl.hf_policy import HFActionPolicy
from kci_shared_semantic_action import SharedSemanticPolicy,canonical_actions
from kci_exact_policy_continuation_cpu import environment_public
from compare_kci_chain_methods_cpu_action_interaction import GROUP
from chain_shared_actor_cpu_action_interaction import (baseline_automatic_action,
                                                       valid_decision_candidates)
from kci.diverse_ci_branch_env import digest

def episode(world,policy,mode):
    env=environment_public(world)
    st=env.initial()
    trace=[]
    for t in range(65):
        if st.native.done:break
        native_options=list(env.options(st))
        obs=env.observation(st)
        decisions=valid_decision_candidates(obs,native_options)
        if len(decisions)>1:
            # Native options remain identical, but only audited decision types
            # are scored; e.g. SEND_RAW_M is not an R-recipient disclosure.
            view={**obs,"legal_actions":list(decisions)}
            a,prob=policy.decide(view)
            assert a in decisions
            origin="shared_one_LoRA"
        else:
            a=baseline_automatic_action(env,st)
            prob=None
            origin="shared_certified_common_tool"
        assert a is not None and a in native_options,(st.phase,obs,a)
        old=digest(asdict(st));before=len(st.native.delivered["R"])
        nxt,v=env.transition(st,a)
        assert v["verdict"] in ("VALID","VIOLATION"),v
        trace.append({
             "role":obs.get("actor",obs.get("role")),
             "phase":obs.get("stage",obs.get("current_stage")),
             "action":a,"policy_source":origin,"native_verdict":v["verdict"],
             "R_new_receipts":len(nxt.native.delivered["R"])-before,
             "state_before_sha256":old,"state_after_sha256":digest(asdict(nxt)),
             "action_probabilities":prob})
        st=nxt
    assert st.native.done,("Nonterminal",world["world"])
    return {"split":world["split"] if "split" in world else None,
            "world":world["world"],"root":world["root"],
            "safe_success":bool(st.native.success and not st.native.violation),
            "CI_violation":bool(st.native.violation),
            "aborted":bool(st.native.aborted),
            "native_steps":st.native.steps,
            "trace":trace,
            "n_model_choices":sum(e["policy_source"]=="shared_one_LoRA" for e in trace)}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    assert not args.output.exists()
    hf=HFActionPolicy(model_path=args.base,adapter=args.adapter,trainable=False,
        quantization="4bit",dtype="bf16",max_prompt_tokens=4096,allow_download=False)
    hf.model.eval()
    scorer=SharedSemanticPolicy(hf)
    start=time.monotonic()
    raw=[];counts=Counter()
    for split,worlds in GROUP.items():
        for world in worlds:
            row=episode({**world,"split":split},scorer,"shared")
            raw.append(row)
            counts["native_steps"]+=len(row["trace"])
            counts["model_choices"]+=row["n_model_choices"]
            print("SHARED_NATIVE_EPISODE",split,world["world"],
                 row["safe_success"],row["CI_violation"],
                 row["native_steps"],row["n_model_choices"],
                 round(time.monotonic()-start,1),flush=True)
    args.output.mkdir(parents=True)
    (args.output/"WORLD_EPISODES.jsonl").write_text("".join(
        json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in raw),encoding="utf8")
    summ={}
    for split in GROUP:
        items=[x for x in raw if x["split"]==split]
        summ[split]={
            "n_worlds":len(items),"n_roots":len({x["root"] for x in items}),
            "safe_task_success":sum(x["safe_success"] for x in items),
            "CI_violations":sum(x["CI_violation"] for x in items),
            "safe_aborts":sum(x["aborted"] and not x["CI_violation"] for x in items),
            "mean_steps":sum(x["native_steps"] for x in items)/len(items),
            "model_choices":sum(x["n_model_choices"] for x in items)}
    report={"architecture":"one Qwen3-4B base + exactly one shared LoRA adapter",
        "per_role_adapters":False,
        "no_native_hidden_labels_in_actor_prompt":True,
        "non_decision_tools_certified_and_scripted":True,
        "exploratory_test_previously_seen_by_developers":True,
        "elapsed_sec":round(time.monotonic()-start,1),
        "total_model_choices":counts["model_choices"],
        "model_decision_roles":dict(Counter(e["role"] for r in raw for e in r["trace"]
                    if e["policy_source"]=="shared_one_LoRA")),
        "one_adapter_used_at_every_model_decision":True,
        "total_native_transitions":counts["native_steps"],
        "splits":summ}
    (args.output/"SUMMARY.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("SHARED_NATIVE_FINAL",json.dumps(report,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
