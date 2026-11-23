"""CPU-only readout of saved LM action log-probabilities at matched native states."""
import json
import math
import statistics
import tarfile
from pathlib import Path
from collections import Counter, defaultdict
from diagnose_state_matched_cpu import BASE, ARCHIVE, OUT, armfile, jlines, identity

def quantile(values,q):
    if not values:return None
    s=sorted(values)
    return s[min(len(s)-1,int((len(s)-1)*q))]

def cmp_episodes(left,right):
    result={"same_actions":[],"same_states":[],"first_changed":[],"sequence_changed":[]}
    for key in sorted(left):
        a,b=left[key],right[key]
        assert a["initial_state_hash"]==b["initial_state_hash"]
        assert a["rollout_seed"]==b["rollout_seed"]
        sig_a=[d["chosen_action"] for d in a["decisions"]]
        sig_b=[d["chosen_action"] for d in b["decisions"]]
        if sig_a!=sig_b:
            result["sequence_changed"].append(key)
        divergence=None
        for idx,(da,db) in enumerate(zip(a["decisions"],b["decisions"])):
            if da["state_hash"]!=db["state_hash"]:
                break
            if da["actions"] != db["actions"]:
                raise RuntimeError("Same state but action set differs")
            result["same_states"].append((key,idx))
            if da["chosen_action"]!=db["chosen_action"]:
                divergence={
                    "scenario":key[0],"sample":key[1],"decision_step":idx,
                    "role":da["actor"],"state_hash":da["state_hash"],
                    "options":da["actions"],
                    "A_chosen":da["chosen_action"],"B_chosen":db["chosen_action"],
                    "A_selected_prob":math.exp(da["old_logprob"]),
                    "B_selected_prob":math.exp(db["old_logprob"]),
                    "A_choice_entropy":da["entropy"],"B_choice_entropy":db["entropy"],
                    "A_outcome":("violation" if a["violation"] else "success" if a["success"] else "safe_failure"),
                    "B_outcome":("violation" if b["violation"] else "success" if b["success"] else "safe_failure"),
                }
                if len(da["actions"])==2 and da["chosen_action"]!=db["chosen_action"]:
                    divergence["A_prob_of_A_choice"]=math.exp(da["old_logprob"])
                    divergence["B_prob_of_A_choice"]=1.-math.exp(db["old_logprob"])
                    divergence["A_minus_B_prob_of_A_choice"]=(
                        divergence["A_prob_of_A_choice"]-divergence["B_prob_of_A_choice"])
                result["first_changed"].append(divergence)
                break
            if da["chosen_action"]==db["chosen_action"]:
                result["same_actions"].append({
                    "key":key,"idx":idx,"role":da["actor"],"action":da["chosen_action"],
                    "A_logprob":da["old_logprob"],"B_logprob":db["old_logprob"],
                    "delta_logprob_B_minus_A":db["old_logprob"]-da["old_logprob"],
                    "delta_entropy_B_minus_A":db["entropy"]-da["entropy"],
                })
    vals=[abs(r["delta_logprob_B_minus_A"]) for r in result["same_actions"]]
    em=[abs(r["delta_entropy_B_minus_A"]) for r in result["same_actions"]]
    per_role=defaultdict(list)
    for r in result["same_actions"]:
        per_role[r["role"]].append(abs(r["delta_logprob_B_minus_A"]))
    return {
        "matched_common_choice_records":len(vals),
        "matching_state_comparisons":len(result["same_states"]),
        "different_paths":len(result["sequence_changed"]),
        "different_choices_at_same_state":result["first_changed"],
        "abs_delta_selected_logprob":{
            "mean":statistics.mean(vals) if vals else None,
            "median":quantile(vals,.5),
            "p90":quantile(vals,.9),
            "p99":quantile(vals,.99),
            "max":max(vals) if vals else None,
        },
        "abs_delta_selected_entropy":{
            "mean":statistics.mean(em) if em else None,
            "p90":quantile(em,.9),
            "max":max(em) if em else None,
        },
        "abs_delta_selected_logprob_by_role":{
            k:{"count":len(v),"median":quantile(v,.5),"max":max(v)}
            for k,v in per_role.items()
        },
    }

def main():
    with tarfile.open(ARCHIVE,"r:gz") as tf:
        raw={}
        for arm in ("terminal","policy_cce"):
            raw[arm]={}
            for split in ("dev_before","dev_after"):
                eps=jlines(tf,armfile(arm,split+"/episodes.jsonl"))
                raw[arm][split]={identity(e):e for e in eps}
    A,B=raw["terminal"],raw["policy_cce"]
    result={
        "A_vs_B_after":cmp_episodes(A["dev_after"],B["dev_after"]),
        "A_before_vs_after":cmp_episodes(A["dev_before"],A["dev_after"]),
        "B_before_vs_after":cmp_episodes(B["dev_before"],B["dev_after"]),
        "A_vs_B_before":cmp_episodes(A["dev_before"],B["dev_before"]),
        "interpretation":"Differences in logged chosen-action probability are observations, not direct parameter-gradient measurements; epsilon=0.35 is part of action sampling."
    }
    file=OUT/"CPU_LOGPROB_DIAGNOSTIC.json"
    if file.exists():raise FileExistsError(file)
    file.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    text=["# KCI CPU policy-output differences", ""]
    for name in ("A_vs_B_before","A_before_vs_after","B_before_vs_after","A_vs_B_after"):
        r=result[name]
        text.append(f"## {name}")
        text.append(f"- Identical action at same state: {r['matched_common_choice_records']} recorded decisions.")
        text.append(f"- Changes in sampled action path: {r['different_paths']} episodes.")
        text.append(f"- |Delta logged chosen-action log-probability|: {r['abs_delta_selected_logprob']}.")
        text.append(f"- First diverging decisions: {json.dumps(r['different_choices_at_same_state'],ensure_ascii=False)}.")
        text.append("")
    text.append("CAUTION: chosen-action probability differences are not policy gradient norms or a statistical conclusion.")
    (OUT/"CPU_LOGPROB_DIAGNOSTIC.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":main()
