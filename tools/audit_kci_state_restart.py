"""Independent native KCI state-restart/policy-credit replay auditor.

Reconstructs EACH native intermediate state solely from authorized action
prefixes. It cannot trust serialized hidden state or self-reported rewards.
Replays suffix decisions through the original simulator; recomputes Qpi branch
outcomes, confidence gates and local credit against raw return evidence.

Checks source corpus hashes, group identity, no unknown/unsafe state injection,
recorded receipts and immutable artifact manifests. For developed DEV,
delegates to the separately implemented root-start native replay auditor.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json,math,sys,statistics

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.agent_rl.core import Scenario,digest,prompt_for,terminal_reward,LETTERS
from kci.agent_rl.state_restarts import RestartSpec,restore
from audit_kci_agent_rl import audit_folder as audit_root
from audit_kci_policy_qpi import audit_record
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def rows(p):return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]
def sha(p):
    h=sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()

def verify_native_restart(ep,spec,source):
    assert ep["scenario_key"]==spec.scenario.key
    assert ep["initial_state_hash"]==spec.state_hash
    env,state=restore(spec,source)
    assert not state.done and not state.violation
    decisions=ep["decisions"]
    assert decisions and decisions[0]["state_hash"]==spec.state_hash
    assert all(math.isfinite(x["old_logprob"]) for x in decisions)
    for index,row in enumerate(decisions):
        assert not state.done,("Extra action after terminal",index)
        valid=tuple(env.options(state))
        obs=env.observation(state)
        assert row["state_hash"]==digest(asdict(state))
        assert row["observation_hash"]==digest(obs)
        assert row["prompt"]==prompt_for(obs,valid)
        assert row["actor"]==env.actor(state)
        assert row["actions"]==list(valid)
        chosen=row["chosen_index"]
        assert type(chosen) is int and 0<=chosen<len(valid)
        assert row["chosen_action"]==valid[chosen]
        assert row["chosen_letter"]==LETTERS[chosen]
        assert row["receipt_count_before"]==len(state.ledger)
        nxt,audit=env.transition(state,valid[chosen])
        assert audit["verdict"]!="UNKNOWN"
        assert audit["verdict"]==row["verdict"]
        assert json.dumps(audit.get("new_protected_relations",[]))==json.dumps(row["newly_derived"])
        assert row["receipt_count_after"]==len(nxt.ledger)
        assert nxt.turn>state.turn
        state=nxt
    assert state.done and ep["done"] is True
    assert ep["total_steps"]==state.turn
    assert ep["success"]==bool(state.success)
    assert ep["violation"]==bool(state.violation)
    assert ep["final_receipt_digest"]==digest(state.ledger)
    assert ep["reward"]==terminal_reward(
        success=state.success,violation=state.violation,done=state.done,
        steps=state.turn)
    return True

def audit_credit_proof(item,episode,spec,source,config):
    assert item["group_index"]>=0
    assert item["teacher_kind"]=="frozen_policy_Qpi_from_native_restart"
    assert item["native_Qstar_used"] is False
    assert item["restart_state_hash"]==spec.state_hash
    assert tuple(item["restart_prefix"])==spec.prefix
    assert item["scenario"]==spec.scenario.key
    assert item["role"]==spec.role and item["phase"]==spec.phase
    assert item["policy_id"]==episode["policy_id"]
    assert item["sample_index"]==episode["sample_index"]
    assert item["decision_index"]==0
    assert item["chosen_action"]==episode["decisions"][0]["chosen_action"]
    result=item["Qpi_evidence"]
    assert result["prefix_state_hash"]==spec.state_hash
    assert result["policy_id"]==item["policy_id"]
    assert result["prefix"]==list(spec.prefix)
    assert result["n_continuations_per_action"]==config["K"]
    assert all(x["qstar"] is None for x in result["actions"].values())
    assert audit_record(result,source,require_oracle=False)
    a,b=sorted(result["actions"])
    v=result["actions"]
    ra=[r["reward"] for r in v[a]["rollouts"]]
    rb=[r["reward"] for r in v[b]["rollouts"]]
    diffs=[x-y for x,y in zip(ra,rb)]
    n=len(diffs)
    positive=sum(x>.1 for x in diffs)
    negative=sum(x<-.1 for x in diffs)
    consensus=max(positive,negative)/n
    gap=v[a]["mean_policy_reward"]-v[b]["mean_policy_reward"]
    gate=result["signal_gate"]
    assert gate["supported"] is True
    assert abs(gate["mean_reward_gap"]-gap)<1e-10
    assert abs(gate["paired_sign_consensus"]-consensus)<1e-10
    assert gate["pos_paired"]==positive and gate["neg_paired"]==negative
    assert abs(gap)>=config["min_return_gap"]
    assert consensus>=config["min_paired_direction_consensus"]
    mean=sum(y["mean_policy_reward"] for y in v.values())/len(v)
    intended=max(-1.,min(1.,v[item["chosen_action"]]["mean_policy_reward"]-mean))
    assert abs(item["local_credit"]-intended)<1e-10
    assert abs(intended)>1e-8
    return {"role":spec.role,"sign":"positive" if intended>0 else "negative"}

def main(root):
    if not (root/"MANIFEST.json").exists():
        raise RuntimeError("Incomplete run; MANIFEST.json missing")
    manifest=load(root/"MANIFEST.json")
    actual={p.relative_to(root).as_posix() for p in root.rglob("*")
            if p.is_file() and p.name!="MANIFEST.json"}
    assert actual==set(manifest),("Extra or missing file",actual^set(manifest))
    for n,h in manifest.items():assert sha(root/n)==h,n
    args=load(root/"RUN_CONFIG.json")
    if "runner_sha256" in args:
        assert sha(ROOT/"tools"/"run_kci_state_directed.py")==args["runner_sha256"]
        assert sha(ROOT/"tools"/"audit_kci_state_restart.py")==args["native_replay_auditor_sha256"]
        for name,h in args["code_sha256"].items():
            assert sha(ROOT/"code"/"kci"/"agent_rl"/name)==h,name
    source=Path(args["source_root"]).resolve()
    selection=Path(args["selection"]).resolve()
    for name,h in args["selection_sha256"].items():
        assert sha(selection/name)==h
    from kci.agent_rl.core import Scenario
    current={}
    for kind in ("meeting","query"):
        current.update(Scenario(kind,False,False,11).build(source).source_hashes)
    assert current==args["source_hashes"]
    train=rows(root/"episodes.jsonl")
    groups=rows(root/"groups.jsonl")
    assert groups and train
    proofs=rows(root/"restart_cce_proofs.jsonl") if (root/"restart_cce_proofs.jsonl").exists() else []
    proof_group=defaultdict(list)
    for proof in proofs:proof_group[proof["group_index"]].append(proof)
    cfg=load(root/"STATE_CCE_CONFIG.json") if (root/"STATE_CCE_CONFIG.json").is_file() else None
    source_selection=rows(selection/"FORKS.jsonl")
    specs=[RestartSpec.from_record(r) for r in source_selection]
    count=Counter()
    examined=0
    unique=set()
    for idx,g in enumerate(groups):
        assert g["group_index"]==idx
        spec=specs[idx]
        actual_spec=RestartSpec.from_record(g["restart"])
        assert actual_spec==spec
        assert g["source_state_hash"]==spec.state_hash
        assert g["group_id"]==spec.scenario.key
        n=g["sample_size"]
        batch=train[examined:examined+n]
        assert len(batch)==n
        examined+=n
        assert [x["sample_index"] for x in batch]==list(range(n))
        assert [x["reward"] for x in batch]==g["rewards"]
        assert len({x["initial_state_hash"] for x in batch})==1
        assert len({x["policy_id"] for x in batch})==1
        assert {x["policy_id"] for x in batch}=={g["policy_id"]}
        assert spec.key not in unique
        unique.add(spec.key)
        for ep in batch:
            verify_native_restart(ep,spec,source)
            count["native_episodes"]+=1
            count["action_decisions"]+=len(ep["decisions"])
        mine=proof_group[idx]
        assert len(mine)==g.get("restart_cce_proof_count",0)
        if cfg is not None:
            assert g["state_cce_evidence"].get("credited_decisions",0)==len(mine)
            assert g["state_cce_evidence"].get("supported_prefixes",0) in (0,1)
            for pr in mine:
                ep=batch[pr["sample_index"]]
                audit_credit_proof(pr,ep,spec,source,cfg)
                count["validated_counterfactual_proofs"]+=1
        elif mine:
            raise AssertionError("Found local credit without CCE configuration")
        count["groups"]+=1
        count["supported_start_states"]+=g["state_cce_evidence"].get("supported_prefixes",0)
        count["extra_continuations"]+=g["state_cce_evidence"].get("policy_branch_rollouts",0)
        count["optimizer_updates"]+=int(g.get("updated",False))
    assert examined==len(train)
    assert sum(len(v) for v in proof_group.values())==len(proofs)
    dev={}
    if args["mode"]=="train":
        for phase in ("dev_before","dev_after"):
            dev[phase]=audit_root(root/phase,source)
    assert len(groups)==args["max_groups"]
    result={
      "STATE_DIRECTED_INDEPENDENT_REPLAY":"PASS",
      "groups":count["groups"],"native_episodes":count["native_episodes"],
      "native_action_decisions":count["action_decisions"],
      "distinct_restart_states":len(unique),
      "supported_policy_value_prefixes":count["supported_start_states"],
      "accepted_policy_cce_proofs":count["validated_counterfactual_proofs"],
      "extra_frozen_policy_rollouts":count["extra_continuations"],
      "optimizer_updates":count["optimizer_updates"],
      "dev_replay":dev,
      "no_Qstar_teacher_in_policy_cce_training":True,
      "separate_business_norm_generalization_proven":False,
      "source_integrity":"PASS","artifact_sha256":"PASS"}
    print(json.dumps(result,ensure_ascii=False))
if __name__=="__main__":
    if len(sys.argv)!=2:
        raise SystemExit("Usage: python tools/audit_kci_state_restart.py RUN_DIR")
    main(Path(sys.argv[1]).resolve())
