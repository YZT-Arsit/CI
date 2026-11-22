"""Collect TRAIN-only true native counterfactual preferences from frozen Qwen7B.

A action comes from the model's own earlier TRAIN-only independent decision
record. All deterministic tool actions are native, and two B actions are forced
from the exact same reconstructed preflight state. The remainder is decided by
the same frozen, locally running pretrained model. No test references used.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
import argparse,json,time,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from compare_kci_chain_methods_cpu_action_interaction import GROUP,baseline_automatic_action
from kci_exact_policy_continuation_cpu import environment_public
from kci_pretrained_llm_native_runner import OllamaActor
from kci_pretrained_llm_counterfactual import force_and_follow
from kci_llm_text_surface import render_role_text
from kci.diverse_ci_branch_env import digest

def to_preflight(row,A):
    env=environment_public(row)
    state=env.initial()
    action_order=[]
    for action in ("A_READ_VERIFIED_PACKAGES",A):
        assert action in env.options(state)
        state,audit=env.transition(state,action)
        assert audit["verdict"]=="VALID"
        action_order.append(action)
    while not state.native.done:
        obs=env.observation(state)
        if obs.get("current_stage")=="preflight":break
        chosen=baseline_automatic_action(env,state)
        if chosen is None:raise RuntimeError(("Unexpected non-scripted action before B",obs))
        state,audit=env.transition(state,chosen)
        assert audit["verdict"]=="VALID"
        action_order.append(chosen)
    assert obs.get("current_stage")=="preflight"
    return env,state,action_order

def compare(left,right):
    if left.get("terminal")!="DONE" or right.get("terminal")!="DONE":
        return None,"incomplete_branch_evidence"
    # Strict CI safety first, successful task next, cost last.
    if left["CI_violation"] != right["CI_violation"]:
        return (1 if not left["CI_violation"] else -1),"verified_CI_verdict"
    if left["safe_success"] != right["safe_success"]:
        return (1 if left["safe_success"] else -1),"verified_task_success"
    if left["CI_violation"] or not left["safe_success"]:
        return None,"same_CI_task_outcome"
    if abs(left["reward"]-right["reward"]) <.010:
        return None,"cost_gap_too_small"
    return (1 if left["reward"]>right["reward"] else -1),"verified_completed_cost"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--name",required=True)
    p.add_argument("--limit",type=int,default=8)
    a=p.parse_args()
    src=ROOT/"artifacts"/"20261122_kci_pretrained_llm_native"/"qwen7b_A_all28_v1"/"EPISODES.jsonl"
    rows={r["world"]:r for r in (json.loads(s) for s in src.read_text(encoding="utf8").splitlines())
          if r["split"]=="train"}
    chosen=[row for row in GROUP["train"] if (not row["swapped_package_names"])]
    chosen=chosen[:a.limit]
    out=ROOT/"artifacts"/"20261122_kci_pretrained_llm_native"/a.name
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    model=OllamaActor("qwen2.5:7b-instruct-q4_K_M")
    prefs=[];raws=[]
    start=time.monotonic()
    for i,row in enumerate(chosen):
        A=rows[row["world"]]["action"]
        assert A in ("A_SEND_PACKAGE_A","A_SEND_PACKAGE_B")
        env,st,prefix=to_preflight(row,A)
        obs=env.observation(st)
        actions=("OPEN_PRIVATE_CHANNEL","SKIP_PRIVATE_CHANNEL")
        assert all(env.transition(st,x)[1]["verdict"]=="VALID" for x in actions)
        outcomes=[force_and_follow(env,st,x,model) for x in actions]
        left,right=outcomes
        direction,reason=compare(left,right)
        reference={
          "world":row["world"],"root":row["root"],"split":"train",
          "A_chosen_by_pretrained_LLM":A,
          "native_exact_B_preflight_hash":digest(asdict(st)),
          "preflight_role_observation_hash":digest(obs),
          "real_native_prefix":prefix,
          "forced_B_actions":list(actions),
          "continuation_RESULTS":{actions[0]:left,actions[1]:right},
          "preference_direction":direction,"reason":reason,
        }
        raws.append(reference)
        with (out/"FORKS_NATIVE_TRAIN_ONLY.jsonl").open("a",encoding="utf8") as f:
            f.write(json.dumps(reference,ensure_ascii=False)+"\n")
        if direction is not None:
            selected=actions[0] if direction==1 else actions[1]
            rejected=actions[1] if direction==1 else actions[0]
            letter=lambda action:chr(ord("A")+obs["legal_actions"].index(action))
            pitem={
              "source":"actual_frozen_Qwen7B_native_continuation_both_actions",
              "evidence_type":"one_complete_actual_model_suffix_per_forced_action",
              "split":"train","role":"Coordinator","stage":"preflight",
              "world":row["world"],"root":row["root"],
              "state_hash":digest(asdict(st)),"observation_hash":digest(obs),
              "chosen_action":selected,"rejected_action":rejected,
              "chosen_letter":letter(selected),"rejected_letter":letter(rejected),
              "all_executable_actions":list(obs["legal_actions"]),
              "prompt":render_role_text(obs),
              "native_evidence":{x:{"safe_success":out.get("safe_success"),
                        "CI_violation":out.get("CI_violation"),"reward":out.get("reward"),
                        "model_calls":out.get("model_calls")}
                        for x,out in zip(actions,outcomes)},
              "preference_reason":reason,
              "language_model":"qwen2.5:7b-instruct-q4_K_M",
              "not_Qstar_oracle_teacher":True,
              "sample_count_per_action":1,
            }
            prefs.append(pitem)
            with (out/"DPO_ROLE_B_TRAIN.jsonl").open("a",encoding="utf8") as f:
                f.write(json.dumps(pitem,ensure_ascii=False)+"\n")
        print("REAL_MODEL_TRAIN_FORK",i+1,len(chosen),row["world"],
            A,[(x,y.get("reward"),y.get("CI_violation"),y.get("terminal")) for x,y in zip(actions,outcomes)],
            "PREFERENCE",direction,reason,
            "ELAPSED",round(time.monotonic()-start,1),flush=True)
    summary={"status":"TRAIN_ONLY_B_CCE_DATA_FROM_ACTUAL_FROZEN_PRETRAINED_LM",
             "worlds":len(raws),"distinct_roots":len(set(x["root"] for x in raws)),
             "both_branches_rollouts":len(raws)*2,"accepted_pref_pairs":len(prefs),
             "accepted_roots":len(set(x["root"] for x in prefs)),
             "teacher":"one full native model-generated suffix per legal B action",
             "all_CI_verdicts_from_native_not_model":True,
             "reference_Qstar_only_for_independent_future_eval":True,
             "model_unmodified":True,"cost_seconds":round(time.monotonic()-start,1)}
    (out/"SUMMARY.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("FINAL",json.dumps(summary),flush=True)
if __name__=="__main__":main()
