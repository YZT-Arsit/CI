"""Independent replay of frozen-model Qpi branch trajectories and native Qstar.

No model/GPU required. Rejects forged actions, receipts, terminal outcomes,
selection tampering, prefix mismatches and Qstar/Qpi result manipulation.
"""
from __future__ import annotations
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json,math,sys,statistics

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.agent_rl.core import Scenario,digest,terminal_reward
from kci.agent_rl.policy_continuation import (
    replay_prefix,immediately_legal,compare_oracle_policy
)
from kci.agent_rl.cce import native_future_value

def sha(path):
    h=sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def rows(p):return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]

def audit_record(record,source_root,*,require_oracle=True):
    s=Scenario(**record["scenario"])
    env,start=replay_prefix(s,tuple(record["prefix"]),source_root)
    assert record["scenario_key"]==s.key
    assert record["prefix_state_hash"]==digest(asdict(start))
    assert record["role"]==env.actor(start)
    assert record["phase"]==env.observation(start)["phase"]
    options=immediately_legal(env,start)
    assert set(options)==set(record["actions"])
    assert len(options)==2
    n=int(record["n_continuations_per_action"])
    assert 2<=n<=16
    for action,branch in record["actions"].items():
        all_rollouts=branch["rollouts"]
        assert len(all_rollouts)==n
        for i,rr in enumerate(all_rollouts):
            assert rr["forced_action"]==action
            assert rr["prefix_hash"]==record["prefix_state_hash"]
            assert rr["policy_id"]==record["policy_id"]
            assert rr["seed"]==record["seed"]+104729*i
            assert rr["continuation_actions"][0]==action
            current=start
            choices=rr["continuation_actions"]
            for act in choices:
                if current.done or act not in env.options(current):
                    raise AssertionError("Recorded model action was impossible")
                nxt,audit=env.transition(current,act)
                assert audit["verdict"]!="UNKNOWN"
                assert nxt.turn>current.turn
                current=nxt
            assert current.done
            assert rr["safe_success"]==bool(current.success and not current.violation)
            assert rr["ci_violation"]==bool(current.violation)
            assert rr["safe_failure"]==bool(not current.success and not current.violation)
            assert rr["final_steps"]==current.turn
            assert rr["final_receipts_hash"]==digest(current.ledger)
            assert rr["continuation_logprob_count"]==len(choices)-1
            target=terminal_reward(success=current.success,violation=current.violation,
                                   done=current.done,steps=current.turn)
            assert abs(rr["reward"]-target)<1e-9
        rewards=[rr["reward"] for rr in all_rollouts]
        assert abs(branch["mean_policy_reward"]-statistics.mean(rewards))<1e-10
        assert abs(branch["std_policy_reward"]-statistics.pstdev(rewards))<1e-10
        assert branch["safe_successes"]==sum(rr["safe_success"] for rr in all_rollouts)
        assert branch["violations"]==sum(rr["ci_violation"] for rr in all_rollouts)
        assert branch["safe_failures"]==sum(rr["safe_failure"] for rr in all_rollouts)
        assert abs(branch["mean_terminal_steps"]-
                   statistics.mean(rr["final_steps"] for rr in all_rollouts))<1e-10
        if branch["qstar"] is None:
            if require_oracle:
                raise AssertionError("Expected native Qstar but no teacher was recorded")
        else:
            if not require_oracle:
                raise AssertionError("Policy-conditioned CCE must not use exact Qstar")
            ideal=native_future_value(env,start,action)
            assert ideal["legal"]==branch["qstar"]["legal"]
            assert ideal["reachable"]==branch["qstar"]["reachable"]
            assert ideal["optimal_terminal_turn"]==branch["qstar"]["optimal_terminal_turn"]
            assert abs(ideal["q"]-branch["qstar"]["q"])<1e-8
    values=[v["mean_policy_reward"] for v in record["actions"].values()]
    assert abs(record["policy_return_range"]-(max(values)-min(values)))<1e-10
    if "comparison" in record:
        if not require_oracle:raise AssertionError("Unexpected teacher ranking in pure Qpi")
        rerank=compare_oracle_policy(record,qpi_margin=.25)
        for name,value in rerank.items():
            if name in record["comparison"]:
                assert record["comparison"][name]==value,name
    return True

def main(out:Path):
    if not (out/"MANIFEST.json").exists():
        raise RuntimeError("Incomplete inference run; MANIFEST missing")
    manifest=read(out/"MANIFEST.json")
    assert set(manifest)=={q.name for q in out.iterdir() if q.is_file() and q.name!="MANIFEST.json"}
    for name,h in manifest.items():
        assert sha(out/name)==h,name
    config=read(out/"RUN_CONFIG.json")
    assert config["training_allowed"] is False
    assert config["model_frozen"] is True
    selection=Path(config["selection"]).resolve()
    for n,h in config["selection_sha256"].items():
        assert sha(selection/n)==h
    original=rows(selection/"FORKS.jsonl")
    results=rows(out/"fork_outputs.jsonl")
    assert len(original)==len(results)
    root=Path(config["source_root"]).resolve() if config["source_root"] else None
    src={}
    for kind in ("meeting","query"):
        src.update(Scenario(kind,False,False,11).build(root).source_hashes if root
                   else Scenario(kind,False,False,11).build().source_hashes)
    assert src==config["source_hashes"]
    for ix,(selected,r) in enumerate(zip(original,results)):
        assert ix==r["selection_index"]
        assert selected["scenario"]==r["scenario"]
        assert selected["state_hash"]==r["prefix_state_hash"]
        assert selected["prefix"]==r["prefix"]
        assert selected["pair_id"]==r["history_pair_id"]
        assert r["policy_id"]==config["policy_id"]
        assert r["related_history"]==r["scenario"]["related"]
        assert audit_record(r,root)
    print(json.dumps({
        "INDEPENDENT_QPI_TRAJECTORY_REPLAY":"PASS",
        "forks":len(results),
        "branches":sum(len(r["actions"]) for r in results),
        "policy_continuations":sum(len(a["rollouts"]) for r in results for a in r["actions"].values()),
        "source_sha256":"PASS","all_output_checksums":"PASS",
        "policy_gradients_used":False,"model_superiority_established":False}))
if __name__=="__main__":
    if len(sys.argv)!=2:raise SystemExit("Usage: python tools/audit_kci_policy_qpi.py RUN_DIR")
    main(Path(sys.argv[1]).resolve())
