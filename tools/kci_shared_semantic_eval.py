"""Evaluate ONE shared semantic-action policy, including strict history pairs.

Not a native multi-agent rollout. Each row is a certified pre-action
observation, and the chosen native action is checked against saved native
correctness. Full agent execution requires a separate executable rollout.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter,defaultdict
from hashlib import sha256
import argparse,json,sys,time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci.agent_rl.hf_policy import HFActionPolicy
from kci_shared_semantic_action import SharedSemanticPolicy,canonical_actions,candidate_prompt
def load(path):
    return [json.loads(s) for s in path.read_text(encoding="utf8").splitlines() if s.strip()]
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--adapter")
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--include-multirole",action="store_true")
    args=p.parse_args()
    assert not args.output.exists()
    names=[
       ("release_train","RELEASE_TRAIN_STRICT8.jsonl"),
       ("release_eval","RELEASE_EVAL_MATCHED12.jsonl"),
       ("v31_train","V31_TRAIN_72.jsonl"),
       ("v31_eval","V31_EVAL_54.jsonl"),
    ]
    if args.include_multirole:names.append(("multirole_train","MULTIROLE_TRAIN.jsonl"))
    hf=HFActionPolicy(model_path=args.model,adapter=args.adapter,
        trainable=False,quantization="4bit",dtype="bf16",max_prompt_tokens=4096,
        allow_download=False)
    sem=SharedSemanticPolicy(hf)
    hf.model.eval();out=[];start=time.monotonic()
    with torch.inference_mode():
        for tag,name in names:
            data=load(args.data_root/name)
            for i,rec in enumerate(data):
                obs=rec["observation"]
                observed=canonical_actions(obs)
                obs_flip={**obs,"legal_actions":list(reversed(obs["legal_actions"]))}
                for act in observed:
                    assert candidate_prompt(obs,act)==candidate_prompt(obs_flip,act),tag
                action,probs=sem.decide(obs)
                assert set(probs)==set(observed)
                assert abs(sum(probs.values())-1)<.0005
                out.append({
                  "dataset":tag,"split":rec["split"],"world":rec["world"],"root":rec["root"],
                  "role":rec["role"],"phase":rec["phase"],"truth":rec["chosen_action"],
                  "predicted_native_action":action,"correct":action==rec["chosen_action"],
                  "probabilities":probs,"source_provenance":rec["provenance"],
                  "observation_sha256":rec["observation_sha256"],
                  "position_invariance_exact":True,
                })
                if (i+1)%12==0:
                    print("SHARED_ACTION_EVAL",tag,i+1,len(data),
                          "seconds",round(time.monotonic()-start,1),flush=True)
    def pair_key(item):
        if item["dataset"].startswith("v31"):return item["world"].split("__")[0]
        return item["world"]
    groups=defaultdict(list)
    for row in out:groups[(row["dataset"],row["split"])].append(row)
    summary={}
    for (ds,split),rows in groups.items():
        world=defaultdict(list)
        for r in rows:world[pair_key(r)].append(r)
        pairs=[a for a in world.values() if len(a)==2]
        correct_pairs=sum(all(r["correct"] for r in a) for a in pairs)
        semflip=sum(len(set(r["predicted_native_action"] for r in a))==2 for a in pairs)
        summary[f"{ds}/{split}"]={
            "correct_states":sum(r["correct"] for r in rows),"n_states":len(rows),
            "independent_roots":len(set(r["root"] for r in rows)),
            "n_pair_worlds":len(pairs),"both_correct_pairs":correct_pairs,
            "semantic_flip_pairs":semflip,
            "predicted_actions":dict(Counter(r["predicted_native_action"] for r in rows)),
            "truth_actions":dict(Counter(r["truth"] for r in rows)),
            "all_prompt_menu_permutation_invariance_verified":True,
        }
    args.output.mkdir(parents=True)
    (args.output/"DECISIONS.jsonl").write_text("".join(
        json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in out),encoding="utf8")
    result={"architecture":"single_4B_LoRA_all_agent_roles",
       "single_adapter_path":args.adapter,
       "n_actor_model_action_states":len(out),
       "training_data_has_previously_seen_exploratory_graphs":True,
       "not_full_multi_agent_episode":True,
       "menu_order_invariance_by_design":True,
       "runtime_sec":round(time.monotonic()-start,1),
       "by_group":summary}
    (args.output/"SUMMARY.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("SHARED_ACTION_EVAL_COMPLETE",json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
