"""CPU analysis of *trainable* (multi-option) DEV decision probabilities only."""
import json,tarfile,math,statistics
from collections import defaultdict
from diagnose_state_matched_cpu import OUT,ARCHIVE,armfile,jlines,identity
with tarfile.open(ARCHIVE,"r:gz") as tf:
    a={identity(e):e for e in jlines(tf,armfile("terminal","dev_after/episodes.jsonl"))}
    b={identity(e):e for e in jlines(tf,armfile("policy_cce","dev_after/episodes.jsonl"))}
rows=[]
for key in sorted(a):
    for da,db in zip(a[key]["decisions"],b[key]["decisions"]):
        if da["state_hash"]!=db["state_hash"] or da["chosen_action"]!=db["chosen_action"]:
            break
        if len(da["actions"])>1:
            assert len(db["actions"])==len(da["actions"])
            rows.append({"role":da["actor"],"state_hash":da["state_hash"],
                "logp_diff":db["old_logprob"]-da["old_logprob"],
                "prob_diff":math.exp(db["old_logprob"])-math.exp(da["old_logprob"])})
assert rows
def qu(x,q):
    z=sorted(x);return z[min(len(z)-1,int(q*(len(z)-1)))]
vals=[abs(x["logp_diff"]) for x in rows]
probs=[abs(x["prob_diff"]) for x in rows]
out={"matched_branch_decisions":len(rows),
     "unique_role_state_pairs":len({(r["role"],r["state_hash"]) for r in rows}),
     "abs_selected_logprob_diff":{"median":qu(vals,.5),"p90":qu(vals,.9),"max":max(vals)},
     "abs_selected_probability_diff":{"median":qu(probs,.5),"p90":qu(probs,.9),"max":max(probs)},
     "per_role":{}}
for role in sorted(set(r["role"] for r in rows)):
    v=[abs(x["prob_diff"]) for x in rows if x["role"]==role]
    out["per_role"][role]={"count":len(v),"median_abs_probability_diff":qu(v,.5),
                           "p90_abs_probability_diff":qu(v,.9),"max":max(v)}
target=OUT/"CPU_BRANCH_PROB_DIAGNOSTIC.json"
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(out,ensure_ascii=False,indent=2))
