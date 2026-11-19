"""GPU inference-only Qpi vs native Qstar at frozen KCI agent history forks.

Reads a frozen TRAIN-prefix panel. Every candidate action is immediately valid.
The frozen LLM samples the REAL remaining multi-agent trajectory. No training,
no gradient, no oracle witness in model prompts or continuation decisions.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from dataclasses import asdict
from hashlib import sha256
import argparse,json,math,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.agent_rl.core import Scenario
from kci.agent_rl.policy_continuation import (
    FrozenFork,frozen_policy_counterfactual,compare_oracle_policy
)

def sha(p):
    h=sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()

def json_file(p):return json.loads(p.read_text(encoding="utf-8"))
def entries(p):return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]
def dump(p,r):
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(r,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
def add(p,r):
    with p.open("a",encoding="utf-8") as f:
        f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n")
        f.flush()

def qpi_confidence(result,replicates:int=1000):
    """Paired bootstrap of K common-seed rollout outcomes, audit-only."""
    import random
    values=result["actions"]
    if len(values)!=2:return {"pairwise":False}
    a,b=sorted(values)
    ra=[x["reward"] for x in values[a]["rollouts"]]
    rb=[x["reward"] for x in values[b]["rollouts"]]
    assert len(ra)==len(rb) and len(ra)>1
    rng=random.Random(432198)
    sample=[]
    for _ in range(replicates):
        sample.append(sum(ra[j]-rb[j] for j in
                          (rng.randrange(len(ra)) for _ in ra))/len(ra))
    sample.sort()
    low=sample[int(.025*len(sample))]
    high=sample[int(.975*len(sample))]
    return {"paired_bootstrap_Qpi_gap_95CI":[low,high],
            "sign_is_supported_at_95pct":low>0 or high<0,
            "n_branch_pairs":len(ra),
            "paired_action_names":[a,b],
            "descriptive_only_no_multiple_testing_correction":True}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--adapter",default=None)
    p.add_argument("--selection",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--source-root",default=None)
    p.add_argument("--bits4",action="store_true")
    p.add_argument("--dtype",choices=("bf16","fp16","fp32"),default="bf16")
    p.add_argument("--group-continuations",type=int,default=8)
    p.add_argument("--seed",type=int,default=19921119)
    p.add_argument("--temperature",type=float,default=1.)
    p.add_argument("--explore-epsilon",type=float,default=.35)
    p.add_argument("--max-prompt-tokens",type=int,default=4096)
    a=p.parse_args()
    if not 2<=a.group_continuations<=16:raise ValueError("Invalid continuation budget")
    root=Path(a.selection).resolve()
    forks=entries(root/"FORKS.jsonl")
    selection=json_file(root/"SELECTION.json")
    if len(forks)!=selection["selected_forks"]:raise ValueError("Selection changed")
    if not 2<=len(forks)<=32:raise ValueError("Pilot supports 2..32 forks")
    if a.source_root is None:
        from experiments.receipt_dataset import SOURCE
        source_root=SOURCE
    else:source_root=Path(a.source_root).resolve()
    # HFActionPolicy supports ONLY inference mode here.
    from kci.agent_rl.hf_policy import HFActionPolicy
    policy=HFActionPolicy(
        model_path=a.model,adapter=a.adapter,trainable=False,
        quantization="4bit" if a.bits4 else "none",dtype=a.dtype,
        temperature=a.temperature,explore_epsilon=a.explore_epsilon,
        max_prompt_tokens=a.max_prompt_tokens)
    assert not policy.trainable
    assert not any(param.requires_grad for param in policy.model.parameters())
    assert not policy.model.training, "Frozen policy must be in deterministic eval mode"
    output=Path(a.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Frozen Qpi output already exists")
    output.mkdir(parents=True,exist_ok=True)
    from kci.agent_rl.core import digest
    args=dict(vars(a))
    if a.adapter is not None:
        args["adapter_is_frozen_reference"]=True
    dump(output/"RUN_CONFIG.json",{
        **args,"selection_sha256":{n:sha(root/n) for n in ("FORKS.jsonl","SELECTION.json")},
        "source_hashes":{**Scenario("meeting",False,False,11).build(source_root).source_hashes,
                         **Scenario("query",False,False,11).build(source_root).source_hashes},
        "policy_id":policy.policy_id,
        "training_allowed":False,
        "no_oracle_in_policy_input":True,
        "model_frozen":True
    })
    results=[]
    for index,item in enumerate(forks):
        s=Scenario(**item["scenario"])
        f=FrozenFork(s,tuple(item["prefix"]),item["phase"],item["role"],item["state_hash"])
        r=frozen_policy_counterfactual(
            f,policy,n_continuations=a.group_continuations,
            seed=a.seed+101399*(index//2),source_root=source_root,
            compute_oracle=True,require_early=True)
        r["selection_index"]=index
        r["history_pair_id"]=item["pair_id"]
        r["related_history"]=s.related
        comparison=compare_oracle_policy(r,qpi_margin=.25)
        comparison.update(qpi_confidence(r))
        r["comparison"]=comparison
        add(output/"fork_outputs.jsonl",r)
        results.append(r)
        print(json.dumps({
            "completed_forks":len(results),"total_forks":len(forks),
            "scenario":s.key,"pair_id":item["pair_id"],
            "teacher_vs_policy":comparison["ranking"],
            "Qpi_gap":comparison["policy_gap"],
            "Qstar_gap":comparison["teacher_gap"]
        }),flush=True)
    byrank=Counter(x["comparison"]["ranking"] for x in results)
    counts=Counter()
    paired=defaultdict(list)
    for x in results:
        paired[x["history_pair_id"]].append(x)
        counts["model_forks"]+=1
        counts["branch_rollouts"]+=sum(len(v["rollouts"]) for v in x["actions"].values())
        counts["high_confidence_qpi_action_preference"]+=x["comparison"]["sign_is_supported_at_95pct"]
        counts["policy_qpi_rank_different_from_Qstar"]+=x["comparison"]["ranking"]=="conflict"
        counts["policy_ties_despite_teacher_preference"]+=x["comparison"]["ranking"]=="policy_tie"
    flips=Counter()
    for idx,group in paired.items():
        if len(group)!=2 or {g["related_history"] for g in group}!={False,True}:
            raise AssertionError("Broken related/unrelated history pairing")
        before,after=sorted(group,key=lambda x:x["related_history"])
        for a in before["actions"]:
            if a not in after["actions"]:raise AssertionError("Paired action contract changed")
        flips["history_pairs"]+=1
        flips["Qstar_action_value_changed"]+=any(
            before["actions"][a]["qstar"]["q"]!=after["actions"][a]["qstar"]["q"]
            for a in before["actions"])
        flips["Qpi_action_value_changed"]+=any(
            abs(before["actions"][a]["mean_policy_reward"]-
                after["actions"][a]["mean_policy_reward"])>.25
            for a in before["actions"])
        flips["Qpi_preference_changed"]+=(
            before["comparison"]["policy_preference"] !=after["comparison"]["policy_preference"])
    summary={
        "status":"COMPLETE_INFERENCE_ONLY",
        "selected_prefixes":len(results),"source_train_conditions_not_blind_TEST":True,
        "rankings":dict(byrank),"counts":dict(counts),"paired_history":dict(flips),
        "checkpoint":policy.policy_id,
        "cannot_claim_joint_CI_policy_gain":True,
        "no_optimizer_or_parameter_updates":True,
        "Qstar_not_policy_expected_value":True,
        "not_a_significance_test":True,
        "recommended_next_step":"Review high-confidence Qpi/Qstar conflicts before using policy-conditioned CCE to train."
    }
    dump(output/"SUMMARY.json",summary)
    manifest={q.relative_to(output).as_posix():sha(q)
              for q in output.iterdir() if q.is_file()}
    dump(output/"MANIFEST.json",manifest)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
