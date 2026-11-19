"""Read-only independent native replay of policy-conditioned local CCE proofs.

Every proof contains K candidate branch continuations sampled by one frozen
policy version before the model update. Verify actual state prefix, role,
choices, terminal outcome, reward, Qpi aggregation and unnormalized credit.
Unlike legacy CCE, native Qstar must be absent from these training proofs.
"""
from __future__ import annotations
from collections import Counter
import json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.agent_rl.core import Scenario
from audit_kci_policy_qpi import audit_record,read,rows,sha

def audit_proof(proof,episode,source_root,cfg):
    r=proof["action_returns"]
    assert proof["teacher_kind"]=="frozen_same_step_policy_Qpi"
    assert proof["no_native_Qstar_used"] is True
    assert r["scenario_key"]==proof["scenario"]==episode["scenario_key"]
    assert r["policy_id"]==proof["policy_id"]==episode["policy_id"]
    i=proof["decision_index"]
    assert type(i) is int and 0<=i<len(episode["decisions"])
    d=episode["decisions"][i]
    assert d["state_hash"]==proof["state_hash"]==r["prefix_state_hash"]
    assert d["chosen_action"]==proof["chosen_action"]
    assert d["actor"]==proof["role"]==r["role"]
    assert proof["phase"]==r["phase"]
    assert proof["prefix"]==r["prefix"]==[x["chosen_action"] for x in episode["decisions"][:i]]
    assert r["n_continuations_per_action"]==cfg["n_frozen_continuations_per_action"]
    assert audit_record(r,source_root,require_oracle=False)
    actions=r["actions"]
    assert len(actions)==2 and d["chosen_action"] in actions
    a,b=sorted(actions)
    rewards_a=[x["reward"] for x in actions[a]["rollouts"]]
    rewards_b=[x["reward"] for x in actions[b]["rollouts"]]
    n=len(rewards_a)
    assert len(rewards_b)==n
    positive=sum(x-y>.1 for x,y in zip(rewards_a,rewards_b))
    negative=sum(x-y<-.1 for x,y in zip(rewards_a,rewards_b))
    consensus=max(positive,negative)/n
    gap=actions[a]["mean_policy_reward"]-actions[b]["mean_policy_reward"]
    g=r["signal_gate"]
    assert g["supported"] is True
    assert abs(g["mean_return_gap"]-gap)<1e-10
    assert abs(g["pairwise_consensus"]-consensus)<1e-10
    assert g["positive_paired_samples"]==positive
    assert g["negative_paired_samples"]==negative
    assert abs(gap)>=cfg["minimum_raw_return_gap"]
    assert consensus>=cfg["minimum_paired_direction_consensus"]
    assert g["terminal_reward_not_replaced"] is True
    avg=sum(x["mean_policy_reward"] for x in actions.values())/2.
    recomputed=max(-1.,min(1.,
                    actions[d["chosen_action"]]["mean_policy_reward"]-avg))
    assert abs(recomputed-proof["local_credit"])<1e-10
    assert abs(proof["local_credit"])>1e-8
    return {"raw_margin":gap,"local_credit":recomputed,"role":d["actor"]}

def audit(out):
    man=read(out/"MANIFEST.json")
    for n,expected in man.items():
        assert sha(out/n)==expected,n
    config=read(out/"RUN_CONFIG.json")
    assert config["credit_mode"]=="policy_cce"
    policy=read(out/"POLICY_CCE_CONFIG.json")
    assert policy["uses_native_Qstar"] is False
    assert policy["terminal_grpo_gradient_preserved"] is True
    assert policy["local_credit_style"]=="additive"
    source_root=Path(config["source_root"]).resolve()
    groups=rows(out/"groups.jsonl")
    episodes=rows(out/"episodes.jsonl")
    if not groups:raise AssertionError("Empty training groups")
    proofs=rows(out/"policy_cce_proofs.jsonl") if (out/"policy_cce_proofs.jsonl").exists() else []
    grouped=Counter(p["group_index"] for p in proofs)
    seen=set();counts=Counter()
    first=0
    for group_idx,g in enumerate(groups):
        n=g["sample_size"]
        batch=episodes[first:first+n];first+=n
        assert len(batch)==n
        for item in batch:
            assert item["scenario_key"]==g["group_id"]
        b={x["sample_index"]:x for x in batch}
        assert len(b)==len(batch)
        own=[p for p in proofs if p["group_index"]==group_idx]
        assert len(own)==g["policy_cce_proof_count"]==grouped[group_idx]
        assert g["credit_style"]=="additive"
        for p in own:
            key=(group_idx,p["sample_index"],p["decision_index"])
            assert key not in seen
            seen.add(key)
            episode=b[p["sample_index"]]
            v=audit_proof(p,episode,source_root,policy)
            counts["verified_local_credits"]+=1
            counts[v["role"]]+=1
            counts["positive"]+=v["local_credit"]>0
            counts["negative"]+=v["local_credit"]<0
        counts["optimizer_updates"]+=g["updated"]
        counts["training_groups"]+=1
        counts["training_episodes"]+=n
    assert first==len(episodes)
    assert counts["verified_local_credits"]==len(proofs)
    result={
        "INDEPENDENT_POLICY_CCE_REPLAY":"PASS",
        "groups":counts["training_groups"],
        "episodes":counts["training_episodes"],
        "local_credits":counts["verified_local_credits"],
        "optimizer_updates":counts["optimizer_updates"],
        "local_credit_signs":{"positive":counts["positive"],"negative":counts["negative"]},
        "roles":{k:v for k,v in counts.items() if k in ("Planner","Broker","HR")},
        "Qstar_used_in_model_or_policy_training":False,
        "policy_conditioned_continuation_verified":True,
        "full_method_novelty_or_superiority_established":False
    }
    print(json.dumps(result,ensure_ascii=False))
    return result
if __name__=="__main__":
    if len(sys.argv)!=2:raise SystemExit("Usage: python tools/audit_kci_policy_cce.py RUN_DIR")
    audit(Path(sys.argv[1]).resolve())
