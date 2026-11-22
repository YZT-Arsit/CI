"""Export TRAIN-only native CI source-role preferences as real LM-action DPO pairs.

Negative: same-state SEND_RAW_R causes immediate CI violation, VERIFY safe.
Positive: both actions legal, and executing RAW under frozen baseline CPU actor
has noninferior safety/completion and better actual task reward.

Source of negative is the native verifier; positive is explicitly a bounded
CPU policy-value diagnostic, NOT a true Qwen-7B continuation or hidden Qstar.
The language prompt contains only the acting role's authorized observation.
"""
from __future__ import annotations
from pathlib import Path
import argparse,hashlib,json,sys,collections
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from compare_kci_chain_methods_cpu_action_interaction import GROUP
from run_kci_role_local_credit_cpu import load_base
from kci_exact_policy_continuation_cpu import environment_public,enumerate_decisions,FrozenPolicyValue
from kci_llm_text_surface import render_role_text
from kci.diverse_ci_branch_env import digest

def dataset():
    base,_=load_base(7)
    result=[]
    count=collections.Counter()
    for w in GROUP["train"]:
        env=environment_public(w)
        pts,_=enumerate_decisions(env)
        value=FrozenPolicyValue(env,base)
        for p in pts:
            if p["phase"]!="release":continue
            options=tuple(p["actions"])
            if "SEND_RAW_R" not in options or "VERIFY_PRIVATE_R" not in options:continue
            raw=p["verdicts"]["SEND_RAW_R"]
            verify=p["verdicts"]["VERIFY_PRIVATE_R"]
            if raw=="VIOLATION" and verify=="VALID":
                chosen="VERIFY_PRIVATE_R"
                rejected="SEND_RAW_R"
                kind="actual_native_immediate_CI_verdict"
                count["verified_violation_repair"]+=1
                details={"SEND_RAW_R":{"immediate_verdict":raw},
                         "VERIFY_PRIVATE_R":{"immediate_verdict":verify}}
            elif raw==verify=="VALID":
                qa=value.q(p["state"],"SEND_RAW_R")
                qb=value.q(p["state"],"VERIFY_PRIVATE_R")
                count["both_immediate_legal"]+=1
                if not (qa[0]>qb[0]+.005 and
                        qa[2]<=qb[2]+1e-4 and
                        qa[1]>=qb[1]-1e-4):
                    count["safe_raw_not_dominant_abstained"]+=1
                    continue
                chosen="SEND_RAW_R"
                rejected="VERIFY_PRIVATE_R"
                kind="safe_legitimate_raw_CPU_frozen_policy_value"
                count["verified_legal_raw_positive"]+=1
                details={"SEND_RAW_R":{"immediate_verdict":raw,
                                        "CPU_policy_expected_return":qa[0],
                                        "CPU_policy_success":qa[1],
                                        "CPU_policy_violation":qa[2]},
                         "VERIFY_PRIVATE_R":{"immediate_verdict":verify,
                                             "CPU_policy_expected_return":qb[0],
                                             "CPU_policy_success":qb[1],
                                             "CPU_policy_violation":qb[2]}}
            else:raise AssertionError(("Unexpected source verdicts",raw,verify))
            obs=p["obs"]
            all_actions=obs["legal_actions"]
            label=lambda action:chr(65+all_actions.index(action))
            assert chosen in all_actions and rejected in all_actions
            result.append({
              "source":kind,
              "evidence_type":kind,"split":"train",
              "role":"SourceAgent",
              "native_role":obs.get("actor",obs.get("role")),
              "world":w["world"],"root":w["root"],
              "stage":"release","state_hash":p["hash"],
              "observation_hash":digest(obs),
              "chosen_action":chosen,"rejected_action":rejected,
              "chosen_letter":label(chosen),"rejected_letter":label(rejected),
              "all_executable_actions":all_actions,
              "prompt":render_role_text(obs),
              "native_evidence":details,
              "not_Qstar_oracle_teacher":True,
              "no_test_labels":True,
            })
    return result,dict(count)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--name",required=True)
    args=parser.parse_args()
    out=ROOT/"artifacts"/"20261122_kci_pretrained_llm_native"/args.name
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    records,metrics=dataset()
    file=out/"DPO_SOURCE_TRAIN.jsonl"
    with file.open("w",encoding="utf8") as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False)+"\n")
    report={"status":"SOURCE_ROLE_NATIVE_CI_AND_LEGAL_POSITIVE_DPO_TRAIN_ONLY",
            "pair_count":len(records),"source_verdict_balance":metrics,
            "independent_graph_roots":4,
            "all_training_records_have_authorized_role_text":True,
            "legitimate_positive_uses_CPU_baseline_not_LLM":True,
            "no_hidden_Qstar_labels":True,
            "sha256":hashlib.sha256(file.read_bytes()).hexdigest()}
    (out/"SUMMARY.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print("SOURCE_ROLE_DPO_EXPORTED",json.dumps(report),flush=True)
if __name__=="__main__":main()
