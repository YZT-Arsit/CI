"""Turn verified frozen-Qwen3-4B A continuations into usable CI-first training pairs.

Do NOT confuse a public *potential* inference with an actual CI violation.
Each rollout is independently re-executed through the unchanged native verifier.
An observed native violation dominates any cost. When all suffixes finish
safely, task success comes next and completed-task cost comes last.
Small-K uncertainty and possible-inference markers remain auditable fields,
not hard exclusions of every otherwise legal preference.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
from collections import Counter
from hashlib import sha256
import json,argparse,sys,statistics,math
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from compare_kci_chain_methods_cpu_action_interaction import GROUP
from kci_exact_policy_continuation_cpu import environment_public
from kci_llm_text_surface import render_role_text
from kci.diverse_ci_branch_env import digest

def verify_rollout(env,origin,record):
    curr=origin
    for i,event in enumerate(record["trace"]):
        assert digest(asdict(curr))==event["parent_sha256"],("before",i)
        assert event["action"] in env.options(curr),("candidate",i)
        nxt,audit=env.transition(curr,event["action"])
        assert audit["verdict"]==event["verdict"],("verdict",i)
        assert digest(asdict(nxt))==event["state_sha256"],("after",i)
        assert len(nxt.native.delivered["R"])-len(curr.native.delivered["R"])==event["new_receipts"]
        curr=nxt
    assert curr.native.done
    success=bool(curr.native.success and not curr.native.violation)
    violated=bool(curr.native.violation)
    reward=round(1-env.spec.cost_per_step*curr.native.steps,6) if success else(-1. if violated else 0.)
    assert abs(reward-record["reward"])<1e-7
    assert success==record["safe_success"] and violated==record["CI_violation"]
    assert digest(asdict(curr))==record["terminal_sha256"]
    return len(record["trace"])

def adjudicate(branches,K):
    a,b=branches
    aa,bb=[x["CI_violation"] for x in a],[x["CI_violation"] for x in b]
    v1,v2=sum(aa),sum(bb)
    sa,sb=sum(x["safe_success"] for x in a),sum(x["safe_success"] for x in b)
    signed=[x["reward"]-y["reward"] for x,y in zip(a,b)]
    mean=statistics.mean(signed)
    pos=sum(x>1e-8 for x in signed)
    neg=sum(x< -1e-8 for x in signed)
    if v1!=v2:
        # An actual violation has been witnessed; never prefer more
        # witnessed violations solely because that branch is cheaper.
        if abs(v1-v2)>=1:
            return (1 if v1<v2 else -1),"observed_native_CI_violation"
    if v1 or v2:
        return 0,"both_branches_have_witnessed_CI_violations"
    if sa!=sb:
        # A task-success difference needs two paired observations for K=4,
        # to prevent a single sample fluctuation becoming a whole-model label.
        if abs(sa-sb)>=2 and (max(pos,neg)>=math.ceil(.75*K)):
            return (1 if sa>sb else -1),"observed_task_success"
        return 0,"task_success_uncertain"
    if sa<K or sb<K:
        return 0,"task_completion_not_robust"
    if abs(mean)<.01 or max(pos,neg)<math.ceil(.75*K):
        return 0,"cost_not_decisive"
    return (1 if mean>0 else -1),"executed_CI_safe_success_cost"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    src=Path(args.source)
    dest=Path(args.output)
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir(parents=True)
    worlds={w["world"]:w for rows in GROUP.values() for w in rows}
    records=[json.loads(x) for x in (src/"NATIVE_A_FORKS.jsonl").read_text(encoding="utf8").splitlines()]
    counts=Counter()
    points=[]
    detailed=[]
    for row in records:
        w=worlds[row["world"]]
        e=environment_public(w)
        state,a=e.transition(e.initial(),"A_READ_VERIFIED_PACKAGES")
        assert a["verdict"]=="VALID"
        obs=e.observation(state)
        assert digest(asdict(state))==row["A_state_hash"]
        assert digest(obs)==row["A_observation_hash"]
        two=tuple(row["legal_A_options"])
        branch=[row["branches"][k] for k in two]
        K=row["K"]
        assert len(branch[0])==len(branch[1])==K
        for items in branch:
            for trace in items:
                counts["native_replayed_suffixes"]+=1
                counts["native_replayed_actions"]+=verify_rollout(e,state,trace)
        label,reason=adjudicate(branch,K)
        counts["audited_worlds"]+=1
        counts["source_split_"+row["split"]]+=1
        counts["label_reason_"+reason]+=1
        p={
            "world":w["world"],"root":w["root"],"split":row["split"],
            "action_gap":statistics.mean(x["reward"]-y["reward"] for x,y in zip(*branch)),
            "sampled_violation_counts":[sum(x["CI_violation"] for x in branch[0]),
                                        sum(x["CI_violation"] for x in branch[1])],
            "sampled_safe_success_counts":[sum(x["safe_success"] for x in branch[0]),
                                           sum(x["safe_success"] for x in branch[1])],
            "potential_future_hazard":row["public_inference_hazards"],
            "CI_first_old_filter_accepted":row["CI_first_accepted"],
            "new_empirical_direction":label,"adjudication_reason":reason,
        }
        detailed.append(p)
        if label and row["split"]=="train":
            good=two[0] if label==1 else two[1]
            bad=two[1] if label==1 else two[0]
            choices=obs["legal_actions"]
            record={
                "source":"Qwen3_4B_same_actor_frozen_policy_executed_CI_first",
                "evidence_type":"native_replayed_K4_actual_frozen_pretrained_Qpi_suffixes",
                "split":"train","role":"RecordsAgent","stage":"A_SEND",
                "world":w["world"],"root":w["root"],
                "state_hash":digest(asdict(state)),
                "observation_hash":digest(obs),
                "chosen_action":good,"rejected_action":bad,
                "chosen_letter":chr(65+choices.index(good)),
                "rejected_letter":chr(65+choices.index(bad)),
                "all_executable_actions":list(choices),
                "prompt":render_role_text(obs),
                "native_evidence":{act:{
                    "K":K,"CI_violations":sum(x["CI_violation"] for x in items),
                    "safe_successes":sum(x["safe_success"] for x in items),
                    "mean_reward":statistics.mean(x["reward"] for x in items)}
                    for act,items in zip(two,branch)},
                "preference_reason":reason,
                "risk_flag":"public_future_inference_is_potential_not_verified_violation",
                "limitation":"K4_sampling_not_zero-risk_proof",
                "not_Qstar_oracle_teacher":True,
                "sample_count_per_action":K,
            }
            points.append(record)
            with (dest/"DPO_ROLE_A_TRAIN.jsonl").open("a",encoding="utf8") as f:
                f.write(json.dumps(record,ensure_ascii=False)+"\n")
        print("A_CREDIT_ADJUDICATED",w["world"],label,reason,
              "reward_gap",round(p["action_gap"],4),
              "n_violation",p["sampled_violation_counts"],flush=True)
    assert counts["audited_worlds"]==len(records)
    summary={
        "status":"AUDITED_REAL_QWEN3_4B_A_CI_FIRST_TRAINING_CREDIT",
        "input_sha256":sha256((src/"NATIVE_A_FORKS.jsonl").read_bytes()).hexdigest(),
        "train_credit_pairs":len(points),
        "train_roots":sorted(set(x["root"] for x in points)),
        "train_preferred_actions":dict(Counter(x["chosen_action"] for x in points)),
        "train_reasons":dict(Counter(x["preference_reason"] for x in points)),
        "audit_counts":dict(counts),
        "no_future_Qstar_in_teacher":True,
        "native_CI_verifier_unchanged":True,
        "public_risk_is_not_a_hard_violation":True,
        "zero_observed_violation_not_zero_true_risk":True,
        "detailed":detailed,
    }
    (dest/"SUMMARY.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("FINAL_A_ADJUDICATION",json.dumps(summary,ensure_ascii=False)[:6500],flush=True)

if __name__=="__main__":main()
