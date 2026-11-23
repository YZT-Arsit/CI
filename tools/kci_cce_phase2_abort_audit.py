from pathlib import Path
import json,collections
D=Path(r"D:\research\longhistory\artifacts\20261123_cce_phase2_mechanism_v1")
x=[json.loads(l) for l in (D/"POLICY_SAMPLED_4B_K8/SAMPLED_ACTUAL_FROZEN_4B_TRAJECTORIES.jsonl").open(encoding="utf-8")]
c=collections.Counter();byroot=collections.Counter();probs=collections.defaultdict(list);stage_p=collections.defaultdict(list)
for row in x:
  reason="violation" if row['ci_violation'] else "success" if row['ci_safe_success'] else "abort" if row['aborted'] else "deadline" if row['deadline_missed'] else "unknown"
  c[("outcome",reason)]+=1
  byroot[(row["root"],row["condition"],row["A_first_action"],reason)]+=1
  for ev in row["native_transition_trace"]:
    if ev["native_action"]=="ABORT":
      c[("abort_stage",ev["stage"])]+=1;c[("abort_actor",ev["actor"])]+=1
      c[("abort_after_a",row["A_first_action"])]+=1
      if ev.get("model_action_probs"):probs[ev["stage"]].append(ev["model_action_probs"].get("ABORT",0))
    if ev["decision_source"].startswith("FROZEN_"):
      c[("model_stage_decision",ev["stage"])]+=1
      if ev.get("model_action_probs"):stage_p[ev["stage"]].append(ev["model_action_probs"].get("ABORT",0))
print("OUTCOMES AND ABORTS",dict(c))
print("STAGE ABORT P AT ACTUAL ABORT", {k:(len(v),round(sum(v)/len(v),4),min(v),max(v)) for k,v in probs.items()})
print("STAGE ALL MODEL P ABORT",{k:(len(v),round(sum(v)/len(v),4)) for k,v in stage_p.items()})
print("ROOT HEAD",dict(list(byroot.items())[:25]))
print("TRACES",len(x))
