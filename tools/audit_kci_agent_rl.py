"""Independent native execution and source-integrity audit for KCI Agent RL.

No model inference, no optimizer. Replays each recorded selected action in the
original finite executor and rejects any mismatch in observed prompts, delivered
receipts, action choices, rewards, or group initial context.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json, math, re, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.agent_rl.core import (
    LETTERS, Scenario, digest, group_signal, prompt_for, terminal_reward
)
from kci.dynamic_tasks import DynamicTask

PATTERN=re.compile(r"(meeting|query)_related([01])_allowed([01])_h(\d+)$")


def read_json(p):
    return json.loads(p.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def hash_file(p):
    h=sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def scenario_for(key):
    m=PATTERN.fullmatch(key)
    if not m:raise ValueError("Unrecognized scenario key: "+key)
    return Scenario(m[1],m[2]=="1",m[3]=="1",int(m[4]))


def audit_episode(raw,source_root):
    spec=scenario_for(raw["scenario_key"])
    env=spec.build(source_root)
    state=env.decision_start()
    if digest(asdict(state))!=raw["initial_state_hash"]:
        raise AssertionError("Starting state changed")
    observed=raw["decisions"]
    for row in observed:
        if state.done:raise AssertionError("Recorded decision after terminal")
        options=tuple(env.options(state))
        obs=env.observation(state)
        assert row["state_hash"]==digest(asdict(state))
        assert row["observation_hash"]==digest(obs)
        assert row["prompt"]==prompt_for(obs,options)
        assert row["actor"]==env.actor(state)
        assert row["actions"]==list(options)
        idx=row["chosen_index"]
        assert type(idx) is int and 0<=idx<len(options)
        assert row["chosen_action"]==options[idx]
        assert row["chosen_letter"]==LETTERS[idx]
        assert math.isfinite(row["old_logprob"]) and math.isfinite(row["entropy"])
        if row["chosen_token_id"] is not None:
            assert type(row["chosen_token_id"]) is int and row["chosen_token_id"]>=0
        assert row["receipt_count_before"]==len(state.ledger)
        after,audit=env.transition(state,options[idx])
        assert audit["verdict"]!="UNKNOWN"
        assert row["verdict"]==audit["verdict"]
        # JSON turns native (subject,attribute) tuples into lists.
        assert json.dumps(row["newly_derived"])==json.dumps(audit.get("new_protected_relations",[]))
        assert row["receipt_count_after"]==len(after.ledger)
        state=after
    assert state.done and raw["done"]==state.done
    assert raw["success"]==state.success and raw["violation"]==state.violation
    assert raw["total_steps"]==state.turn
    assert raw["final_receipt_digest"]==digest(state.ledger)
    assert raw["reward"]==terminal_reward(
        success=state.success,violation=state.violation,
        done=state.done,steps=state.turn)
    return raw


def audit_folder(folder,source_root):
    ps=rows(folder/"episodes.jsonl")
    gs=rows(folder/"groups.jsonl")
    if not ps or not gs:raise ValueError("Empty run cannot be called complete")
    total_groups=len(gs)
    expected=sum(len(g["rewards"]) for g in gs)
    assert len(ps)==expected
    index=0;stats=Counter()
    for g in gs:
        n=len(g["rewards"])
        episode_block=ps[index:index+n]
        index+=n
        assert len({e["initial_state_hash"] for e in episode_block})==1
        assert len({e["scenario_key"] for e in episode_block})==1
        assert len({e["policy_id"] for e in episode_block})==1
        assert len({e["split"] for e in episode_block})==1
        assert [e["reward"] for e in episode_block]==g["rewards"]
        for ep in episode_block:
            audit_episode(ep,source_root)
            stats["rollouts"]+=1
            stats["safe_success"]+=ep["success"] and not ep["violation"]
            stats["violations"]+=ep["violation"]
        stats["groups"]+=1
        # Cost-only signals and zero-variance groups must not be mislabeled.
        distinct_outcomes={(e["violation"],e["success"]) for e in episode_block}
        if len(distinct_outcomes)>1: kind="outcome_contrast"
        elif len(set(g["rewards"]))>1:kind="cost_only"
        else:kind="zero_variance"
        assert g.get("signal_kind",kind)==kind
        stats[kind]+=1
        if "updated" in g and kind=="zero_variance":
            assert not g["updated"]
        if "updated" in g and kind=="cost_only" and not g["updated"]:
            assert g.get("skip_reason") in ("cost_only_signal_not_enabled","zero_reward_variance")
    assert index==len(ps) and stats["groups"]==total_groups
    return dict(stats)


def main():
    if len(sys.argv)!=2:raise SystemExit("Usage: python tools/audit_kci_agent_rl.py FRESH_RUN_DIRECTORY")
    out=Path(sys.argv[1]).resolve()
    if not (out/"MANIFEST.json").is_file():raise RuntimeError("Incomplete run: MANIFEST.json absent")
    manifest=read_json(out/"MANIFEST.json")
    actual={p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and p.name!="MANIFEST.json"}
    assert actual==set(manifest),("Missing or unexpected output artifacts",actual^set(manifest))
    for name,expected in manifest.items():
        assert hash_file(out/name)==expected,("Tampered file",name)
    config=read_json(out/"RUN_CONFIG.json")
    source_root=Path(config["source_root"]).resolve()
    current={}
    for kind in ("meeting","query"):
        env=DynamicTask(kind,source_root,related=False,allow=False,horizon=11)
        current.update(env.source_hashes)
    assert config["source_hashes"]==current
    results={}
    folders=[f for f in (out,out/"dev_before",out/"dev_after") if (f/"episodes.jsonl").exists()]
    for folder in folders:
        results[folder.relative_to(out).as_posix()]=audit_folder(folder,source_root)
    assert results
    total=sum(x["rollouts"] for x in results.values())
    print(json.dumps({"independent_replay":"PASS",
        "run":str(out),"folders":results,"total_verified_rollouts":total,
        "all_file_hashes":"PASS","real_model_training_not_inferred_from_probe":True},
        ensure_ascii=False))
if __name__=="__main__":
    main()
