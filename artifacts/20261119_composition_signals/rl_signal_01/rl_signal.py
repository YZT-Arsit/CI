"""KCI-Align RL signal viability pilot (NO MODEL FIT, NO API, NO GPU).

Authoritative env: code/kci/dynamic_tasks.py (full multi-step tool contract).
Goal: verify actual state->action->outcome reward contrast BEFORE scaling RL.
The model must never see hidden flags, target values or oracle continuations.

This is a deterministic evaluator smoke test, NOT evidence of learned ability.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
import json,random,sys,statistics
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"code"))
from kci.dynamic_tasks import DynamicTask,canonical
from experiments.receipt_dataset import SOURCE

HERE=Path(__file__).resolve().parent
SEED=9111926
G=8
HORIZONS=(7,9,11,13)
KINDS=("meeting","query")
def sha(p):return sha256(p.read_bytes()).hexdigest()
def dump(path,records):
    if path.exists():raise FileExistsError(path)
    path.write_text("".join(canonical(r)+"\n" for r in records),encoding="utf-8")

def make_env(kind,related,allowed,horizon):
    return DynamicTask(kind,SOURCE,related=related,allow=allowed,horizon=horizon)

def score(state):
    # Reward is an EXPERIMENTAL trajectory reward, not a social-norm oracle.
    # Safety > progress > lower cost, without allowing task refusal to tie success.
    if state.violation:return -3.0
    if state.success and state.done:return 2.0-0.01*state.turn
    return -0.25

def episode(env,seed):
    rng=random.Random(seed)
    state=env.decision_start()
    trajectory=[]
    while not state.done:
        actor=env.actor(state);options=env.options(state)
        if not options:break
        action=rng.choice(options)
        observation=env.observation(state)
        assert not any(k in observation for k in ("related","success","violation","expected"))
        nxt,verdict=env.transition(state,action)
        if verdict["verdict"]=="UNKNOWN":raise AssertionError("Declared action had UNKNOWN transition")
        trajectory.append({"turn":state.turn,"actor":actor,
            "observation":observation,"selected_action":action,
            "valid_actions":list(options),"verdict":verdict["verdict"],
            "new_relations":verdict.get("new_protected_relations",[]),
            "transitioned":nxt.turn>state.turn,
            "done_after":nxt.done})
        state=nxt
        if len(trajectory)>env.horizon+3:raise AssertionError("Rollout never terminates")
    return {"outcome":{"success":bool(state.success),"ci_violation":bool(state.violation),
                       "done":bool(state.done),"steps":int(state.turn),"reward":score(state)},
            "trajectory":trajectory}

def first_action_value(env):
    s=env.decision_start()
    choices=env.options(s)
    assert len(choices)==2
    outcomes={}
    for a in choices:
        after,audit=env.transition(s,a)
        assert audit["verdict"]!="UNKNOWN"
        proof=env.exact(after)
        outcomes[a]={"immediate_ci":audit["verdict"],
            "safe_success_reachable":proof["status"]=="EXACT",
            "optimal_total_steps":proof["min_total_cost"],
            "witness":proof["witness"],
            "enumerated_nodes":proof["nodes"]}
    return outcomes

def main():
    out=HERE/"pilot"
    if out.exists() and any(out.iterdir()):raise FileExistsError("No overwrite: choose a fresh output")
    out.mkdir(parents=True,exist_ok=True)
    starts,episodes,groups=[],[],[]
    for kind in KINDS:
      for related in (False,True):
       for allow in (False,True):
        for horizon in HORIZONS:
          env=make_env(kind,related,allow,horizon)
          name=f"{kind}_related{int(related)}_allow{int(allow)}_h{horizon}"
          starts.append({"state_group":name,"task_family":kind,"related":related,
                         "allowed":allow,"horizon":horizon,
                         "initial_state_options":first_action_value(env),
                         "source_hashes":env.source_hashes})
          group=[]
          for i in range(G):
              result=episode(env,SEED+len(starts)*1000+i)
              full={"state_group":name,"sample":i,**result}
              episodes.append(full);group.append(full)
          rewards=[r["outcome"]["reward"] for r in group]
          terminal=[r["outcome"]["success"] for r in group]
          violations=[r["outcome"]["ci_violation"] for r in group]
          # DAPO-style group filter: keep only samples with nonzero outcome contrast.
          # Group advantages use centered rewards; no learning occurs in this script.
          mean=statistics.mean(rewards)
          advantage=[round(x-mean,6) for x in rewards]
          informative=(max(rewards)>min(rewards))
          groups.append({"state_group":name,"G":G,"rewards":rewards,
              "advantages":advantage,"informative":informative,
              "terminal_safe_successes":sum(terminal),"violations":sum(violations),
              "distinct_rewards":len(set(rewards)),
              "n_actions":sum(len(r["trajectory"]) for r in group)})
    dump(out/"starts.jsonl",starts)
    dump(out/"rollouts.jsonl",episodes)
    dump(out/"groups.jsonl",groups)
    meaningful=[]
    for s in starts:
        vals=list(s["initial_state_options"].values())
        both_legal=all(v["immediate_ci"]=="VALID" for v in vals)
        distinct=[(v["safe_success_reachable"],v["optimal_total_steps"]) for v in vals]
        if both_legal and distinct[0]!=distinct[1]:meaningful.append(s["state_group"])
    info=sum(g["informative"] for g in groups)
    stats={"env":"Native DynamicTask v2 source-adapted","examples":len(starts),"task_families":len(KINDS),
      "rollout_groups":len(groups),"rollouts":len(episodes),
      "G":G,"informative_groups":info,
      "informative_group_fraction":round(info/len(groups),4),
      "mean_safe_success_per_group":round(statistics.mean(g["terminal_safe_successes"] for g in groups),3),
      "total_safe_successes":sum(g["terminal_safe_successes"] for g in groups),
      "total_violations":sum(g["violations"] for g in groups),
      "both_legal_actions_with_future_value_gap":len(meaningful),
      "future_value_gap_groups":meaningful,
      "no_model_training":True,"real_norms_verified":False,
      "business_task_independence":False,"GPU_go":False,
      "recommendation":"First prove reward contrast and policy sampling variability; then enable actual model rollouts and on-policy updates. Synthetic task family not an ACL generalization claim."}
    (out/"SUMMARY.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding="utf-8")
    manifest={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    (out/"MANIFEST.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(stats,ensure_ascii=False))
if __name__=="__main__":main()
