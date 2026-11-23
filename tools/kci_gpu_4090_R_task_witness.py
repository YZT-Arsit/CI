"""Verified native task-completion witnesses vs ABORT for KCI source role.

This only adds *task-completion* repair signal absent from simple RAW/VERIFY
privacy labels. A witness forces the chosen legal release action and then
replays a fully legal native continuation to business success. Not Qstar,
not current LM Qpi. Training roots only; all evidence comes from native env.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
from dataclasses import asdict
from hashlib import sha256
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from compare_kci_chain_methods_cpu_action_interaction import GROUP
from chain_shared_actor_cpu_action_interaction import baseline_automatic_action
from kci_exact_policy_continuation_cpu import environment_public,enumerate_decisions
from kci_llm_text_surface import render_role_text
from kci.diverse_ci_branch_env import digest

OUTPUT=ROOT/"artifacts"/"20261122_kci_R_task_witness_train_v1"
SOURCE=ROOT/"artifacts"/"20261122_kci_pretrained_llm_native"/"source_R_verified_positive_v1"/"DPO_SOURCE_TRAIN.jsonl"

def legal_route(env,state):
    """Act only after a forced legal release. Verify the entire native suffix."""
    cur=state
    moves=[]
    for _ in range(60):
        if cur.native.done:break
        ob=env.observation(cur)
        avail=list(env.options(cur))
        phase=ob.get("current_stage",ob.get("stage"))
        if "COMMIT_RESULT" in avail:
            act="COMMIT_RESULT"
        elif phase=="release":
            candidates=[]
            for option in ("SEND_RAW_R","VERIFY_PRIVATE_R"):
                if option not in avail:continue
                _,v=env.transition(cur,option)
                if v["verdict"]=="VALID":candidates.append(option)
            # Prefer verified private completion if available; otherwise use
            # valid RAW. No hidden task labels drive this local preference.
            act=("VERIFY_PRIVATE_R" if "VERIFY_PRIVATE_R" in candidates
                 else candidates[0] if candidates else None)
        else:
            act=baseline_automatic_action(env,cur)
        if act is None:return None
        nxt,v=env.transition(cur,act)
        if v["verdict"]!="VALID":return None
        moves.append((phase,act))
        cur=nxt
    if cur.native.done and cur.native.success and not cur.native.violation:
        return {"reward":round(1-env.spec.cost_per_step*cur.native.steps,6),
                "native_steps":cur.native.steps,"trace":moves,
                "end_state_sha256":digest(asdict(cur))}
    return None

def native_witness_pairs():
    samples=[]
    counts=Counter()
    for w in GROUP["train"]:
        env=environment_public(w)
        states,_=enumerate_decisions(env)
        for p in states:
            if p["phase"]!="release":continue
            st=p["state"]
            obs=p["obs"]
            actions=obs["legal_actions"]
            if "ABORT" not in actions:
                counts["no_abort_control"]+=1
                continue
            end,v=env.transition(st,"ABORT")
            if v["verdict"]!="VALID" or not end.native.done or end.native.success or end.native.violation:
                counts["abort_not_safe_incomplete"]+=1
                continue
            successes=[]
            for candidate in ("SEND_RAW_R","VERIFY_PRIVATE_R"):
                if candidate not in actions:continue
                next_state,verdict=env.transition(st,candidate)
                if verdict["verdict"]!="VALID":continue
                cont=legal_route(env,next_state)
                if cont is not None:
                    successes.append((candidate,cont))
            if not successes:
                counts["no_verified_completion_witness"]+=1
                continue
            winner,proof=max(successes,key=lambda v:v[1]["reward"])
            assert proof["reward"]>0
            label=lambda a:chr(ord("A")+actions.index(a))
            samples.append({
                "source":"verified_native_completion_vs_safe_abort",
                "evidence_type":"verified_witness_only_not_oracle_Qstar",
                "split":"train","role":"SourceAgent",
                "native_role":obs.get("actor",obs.get("role")),
                "stage":"release","world":w["world"],"root":w["root"],
                "state_hash":p["hash"],"observation_hash":digest(obs),
                "chosen_action":winner,"rejected_action":"ABORT",
                "chosen_letter":label(winner),"rejected_letter":label("ABORT"),
                "all_executable_actions":actions,
                "prompt":render_role_text(obs),
                "native_evidence":{winner:{"immediate_verdict":"VALID",
                                   "witness":proof},
                                   "ABORT":{"immediate_verdict":"VALID",
                                            "safe_task_success":False}},
                "not_Qstar_oracle_teacher":True,
                "no_test_labels":True,
            })
            counts["native_witness_vs_abort_pairs"]+=1
    return samples,dict(counts)

def main():
    if OUTPUT.exists():raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    native,counts=native_witness_pairs()
    prior=[json.loads(l) for l in SOURCE.read_text(encoding="utf8").splitlines() if l.strip()]
    pools={
        "ci_violation":[r for r in prior if r["chosen_action"]=="VERIFY_PRIVATE_R"],
        "legal_raw":[r for r in prior if r["chosen_action"]=="SEND_RAW_R"],
        "task_witness":native,
    }
    assert len(pools["ci_violation"])==28 and len(pools["legal_raw"])==72
    assert len(native)>=20,counts
    # Cycle all three evidence groups to prevent a long early run of only
    # one kind of preference when training has a small update budget.
    interleaved=[]
    n=max(len(p) for p in pools.values())
    for i in range(n):
        for group in ("ci_violation","legal_raw","task_witness"):
            if i<len(pools[group]):
                r=dict(pools[group][i]);r["training_evidence_group"]=group
                interleaved.append(r)
    file=OUTPUT/"DPO_R_TASK_AND_CI_TRAIN.jsonl"
    file.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in interleaved),encoding="utf8")
    note={"status":"SOURCE_ROLE_CI_PRIVACY_AND_VERIFIED_TASK_COMPLETION",
          "training_roots":sorted(set(r["root"] for r in interleaved)),
          "unique_witness_pairs":len(native),"total_pairs":len(interleaved),
          "group_sizes":{k:len(v) for k,v in pools.items()},
          "native_replay_counts":counts,
          "file_sha256":sha256(file.read_bytes()).hexdigest(),
          "witness_type":"verified successful native continuation, not Qstar/Qpi"}
    (OUTPUT/"SUMMARY.json").write_text(json.dumps(note,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("SOURCE_TASK_TRAIN",json.dumps(note,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
