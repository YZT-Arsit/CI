"""Deterministically select 8 SAME-PREFIX recipient-history pairs from
Qwen 4B's 64-group TRAIN trajectories for a frozen-policy Qpi vs Qstar pilot.
Selection uses native exact teacher only to stratify; these are NOT TEST cases.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from dataclasses import asdict
from hashlib import sha256
import json,re,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/"code"))
from kci.agent_rl.core import Scenario,digest
from kci.agent_rl.cce import EARLY,native_future_value
from kci.agent_rl.policy_continuation import make_fork,replay_prefix,immediately_legal
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
BACKUP=PACKAGE/"grpo_64_completed_backup.tar.gz"
PREFIX="gpu_4b_grpo_64_01/"
RX=re.compile(r"^(meeting|query)_related([01])_allowed([01])_h(\d+)$")
def decode(s):
    x=RX.fullmatch(s)
    if x is None:raise ValueError(s)
    return Scenario(x[1],x[2]=="1",x[3]=="1",int(x[4]))
def load_train():
    with tarfile.open(BACKUP,"r:gz") as z:
        data=z.extractfile(PREFIX+"episodes.jsonl").read()
        manifest=json.load(z.extractfile(PREFIX+"MANIFEST.json"))
    assert sha256(data).hexdigest()==manifest["episodes.jsonl"]
    return [json.loads(s) for s in data.decode().splitlines() if s.strip()]
def evaluate_pair(kind,allowed,horizon,prefix):
    worlds={}
    options={}
    for related in (False,True):
        spec=Scenario(kind,related,allowed,horizon)
        env,s=replay_prefix(spec,tuple(prefix))
        opts=immediately_legal(env,s)
        if len(opts)<2:return None
        worlds[related]=(env,s)
        options[related]=set(opts)
    if options[False]!=options[True]:return None
    if len(options[False])!=2:return None
    q={}
    for related,(env,s) in worlds.items():
        q[related]={action:native_future_value(env,s,action) for action in options[related]}
    # Difference in the value ranking when only recipient history changes.
    actions=sorted(options[False])
    sig={(r,a):q[r][a]["q"] for r in (False,True) for a in actions}
    flip=any(q[False][a]["q"]!=q[True][a]["q"] for a in actions)
    reach=any(q[False][a]["reachable"]!=q[True][a]["reachable"] for a in actions)
    margins={r:abs(q[r][actions[0]]["q"]-q[r][actions[1]]["q"]) for r in (False,True)}
    return {
       "kind":kind,"allow":allowed,"horizon":horizon,
       "prefix":list(prefix),"role":env.actor(s),"phase":env.observation(s)["phase"],
       "actions":actions,"native_qstar_unrelated":q[False],
       "native_qstar_related":q[True],
       "history_sensitive_qstar":flip,"history_sensitive_reachability":reach,
       "largest_within_state_qstar_gap":max(margins.values()),
    }
def main():
    out=HERE/"selection_v2"
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(exist_ok=True)
    ep=load_train()
    assert len(ep)==512
    candidate_prefixes={}
    for e in ep:
        spec=decode(e["scenario_key"])
        env,s=spec.build(),spec.build().decision_start()
        # Actual sampled action path, no fabricated prefix.
        prefix=[]
        for d in e["decisions"]:
            if s.done:break
            assert digest(asdict(s))==d["state_hash"]
            obs=env.observation(s)
            if (env.actor(s),obs["phase"]) in EARLY and len(env.options(s))>=2:
                key=(spec.kind,spec.allow,spec.horizon,tuple(prefix))
                candidate_prefixes.setdefault(key,None)
            nxt,audit=env.transition(s,d["chosen_action"])
            assert audit["verdict"]==d["verdict"]
            s=nxt
            prefix.append(d["chosen_action"])
    verified=[]
    for kind,allow,horizon,prefix in sorted(candidate_prefixes):
        result=evaluate_pair(kind,allow,horizon,prefix)
        if result is not None:verified.append(result)
    # Predeclare half genuinely history-sensitive Qstar pairs and half
    # paired controls, with family/permission/role-phase diversity.
    verified.sort(key=lambda x:(x["kind"],x["allow"],x["phase"],
                               x["horizon"],x["prefix"]))
    sensitive=[x for x in verified if x["history_sensitive_qstar"]]
    controls=[x for x in verified if not x["history_sensitive_qstar"]]
    chosen=[]
    def take(pool,n):
        seen={(x["kind"],x["allow"],x["phase"]) for x in chosen}
        for item in pool:
            if len(chosen)>=n:return
            k=(item["kind"],item["allow"],item["phase"])
            if k not in seen and item not in chosen:
                chosen.append(item);seen.add(k)
        for item in pool:
            if len(chosen)>=n:return
            if item not in chosen:chosen.append(item)
    take(sensitive,4)
    take(controls,8)
    take(sensitive,8)
    if len(chosen)<6:
        raise RuntimeError(f"Insufficient supported history pairs: {len(chosen)}")
    forks=[]
    for pair_idx,entry in enumerate(chosen):
        for related in (False,True):
            spec=Scenario(entry["kind"],related,entry["allow"],entry["horizon"])
            fork=make_fork(spec,entry["prefix"])
            assert (fork.role,fork.phase)==(entry["role"],entry["phase"])
            forks.append({"pair_id":pair_idx,"scenario":asdict(spec),
                          "prefix":list(fork.prefix),"state_hash":fork.state_hash,
                          "role":fork.role,"phase":fork.phase,
                          "train_source_history_pair":True})
    report={"version":"policy-conditioned-CI-Qpi-diagnostic-v1",
      "source_archive_sha256":sha256(BACKUP.read_bytes()).hexdigest(),
      "train_episodes":len(ep),"candidate_history_prefixes":len(candidate_prefixes),
      "eligible_history_pairs":len(verified),
      "eligible_history_sensitive":len(sensitive),
      "eligible_history_insensitive":len(controls),
      "selected_history_pairs":len(chosen),
      "selected_forks":len(forks),"n_rollouts_per_action":8,
      "max_policy_action_branch_rollouts":len(forks)*2*8,
      "selected_history_sensitive":sum(x["history_sensitive_qstar"] for x in chosen),
      "selected_reachability_sensitive":sum(x["history_sensitive_reachability"] for x in chosen),
      "source_business_generalization":False,
      "uses_known_train_prefixes":True,
      "fixed_checkpoint":"Qwen3-4B-Instruct-2507-ms base, local, frozen, temp=1 epsilon=.35",
      "policy_gradient_allowed":False,
      "qstar_teacher_only_selection_and_diagnostic":True,
      "selected_pair_metadata":chosen}
    (out/"ELIGIBLE.jsonl").write_text("".join(
       json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in verified),encoding="utf-8")
    (out/"FORKS.jsonl").write_text("".join(
       json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in forks),encoding="utf-8")
    (out/"SELECTION.json").write_text(
       json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="selected_pair_metadata"},ensure_ascii=False))
if __name__=="__main__":main()
