"""CCE Phase2: freeze same upstream actual native state and both legal A forks.

Only TRAIN Phase2 root lines are parsed. No final test or dev examples
deserialized or used for selection. This is not a preference-training dataset.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json,sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(r"D:\research\kci_data_v4_phase2_20261122\artifacts\20261122_kci_data_v4_phase2_full_v1")
OUT=ROOT/"artifacts/20261123_cce_phase2_mechanism_v1"
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.v4_phase2_workflows import Workflow,WorkStep,Phase2Env,RECEIVERS
from kci.v4_phase2_data import render
from kci.diverse_ci_branch_env import digest
ROOTS=("P201","P202","P203","P205","P206","P207","P208")
SEEDS=(7,1234,2025,42,314,2718,1618,99)
VARIANTS=("risky","authorized")
ACTIONS=("A_SEND_PACKAGE_A","A_SEND_PACKAGE_B")
def sha(p):return sha256(Path(p).read_bytes()).hexdigest()
def source_train_rows():
    rows=[]
    for line in (SOURCE/"NEW_ROOTS.jsonl").open(encoding="utf8"):
        if '"final_test"' in line or '"P211' in line or '"P212' in line:continue
        x=json.loads(line)
        if x.get("root_split")=="train" and x.get("root_id") in ROOTS:
            rows.append(x)
    assert {x["root_id"] for x in rows}==set(ROOTS)
    return rows
def reconstruct(raw):
    spec=dict(raw["native_spec"])
    assert spec["split"]=="train" and spec["id"] in ROOTS
    for phase in ("pre","post"):
        spec[phase]=tuple(WorkStep(**{**s,"depends":tuple(s.get("depends",[]))}) for s in spec[phase])
    return Workflow(**spec)
def preflight():
    if OUT.exists():raise FileExistsError(OUT)
    roots=source_train_rows()
    states=[]
    for raw in sorted(roots,key=lambda r:r["root_id"]):
        spec=reconstruct(raw)
        assert spec.deadline_type==raw["deadline_class"]
        assert raw["deadline_units"]==spec.budget
        for condition in VARIANTS:
            env=Phase2Env(spec,variant="risky",authorized=condition=="authorized")
            initial=env.initial()
            pre=env.observation(initial)
            assert "A_READ_VERIFIED_PACKAGES" in pre["legal_actions"]
            s,receipt=env.transition(initial,"A_READ_VERIFIED_PACKAGES")
            assert receipt["verdict"]=="VALID" and s.stage=="A_SEND"
            obs=env.observation(s)
            assert obs["actor"]=="RecordsAgent"
            assert all(a in obs["legal_actions"] for a in ACTIONS)
            prefix,_=render(obs,style=0)
            assert "A_SEND_PACKAGE_A" in prefix and "A_SEND_PACKAGE_B" in prefix
            choices={}
            for action in ACTIONS:
                assert action in env.options(s)
                nxt,report=env.transition(s,action)
                assert report["verdict"]=="VALID",(spec.id,action,report)
                assert not nxt.violation and not nxt.done
                assert len(nxt.ledger)>len(s.ledger)
                receiver=spec.target
                before=set(env.knowledge(s,receiver));after=set(env.knowledge(nxt,receiver))
                delta=sorted(after-before)
                assert delta,(spec.id,action)
                choices[action]={
                  "post_A_state":asdict(nxt),
                  "post_A_native_hash":digest(asdict(nxt)),
                  "actual_delivery_count_delta":len(nxt.ledger)-len(s.ledger),
                  "recipient":receiver,
                  "knowledge_before":sorted(before),"knowledge_after":sorted(after),
                  "knowledge_delta":delta,
                  "A_verdict":report["verdict"],
                  "A_immediate_task_success":nxt.success,
                  "A_immediate_CI_violation":nxt.violation,
                  "A_workflow_stage_after":nxt.stage}
            assert choices[ACTIONS[0]]["knowledge_after"]!=choices[ACTIONS[1]]["knowledge_after"]
            ident=spec.id+"__"+condition
            states.append({
              "cell_id":ident,"root":spec.id,"condition":condition,"split":"train",
              "business_kind":spec.kind,"business_family":spec.family,
              "deadline_class":spec.deadline_type,"deadline_budget":spec.budget,
              "recipient":spec.target,
              "workflow_DAG_signature":raw["abstract_task_DAG_shape_sha256"],
              "native_root_spec":raw["native_spec"],
              "A_pre_state":asdict(s),
              "A_pre_state_sha256":digest(asdict(s)),
              "A_model_observation":obs,
              "A_model_prompt":prefix,
              "A_model_prompt_sha256":sha256(prefix.encode("utf8")).hexdigest(),
              "A_legal_actions":list(env.options(s)),
              "A_valid_candidates":choices,
              "current_A_both_actions_native_CI_legal":True,
              "fixed_prefix_A_read_verdict":receipt["verdict"],
              "no_model_Qstar_or_future_in_legal_observation":True,
            })
    assert len(states)==len(ROOTS)*len(VARIANTS)
    OUT.mkdir(parents=True)
    p=OUT/"FROZEN_UPSTREAM_A_STATE_FORKS_TRAIN_ONLY.jsonl"
    p.write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in states),encoding="utf8")
    manifest={
      "cell":"CCE_Phase2_true_upstream_same_state_4B_frozen_policy",
      "not_CCE_training_yet":True,
      "root_ids":list(ROOTS),
      "deadline_tight":[x for x in ROOTS if any(y["root"]==x and y["deadline_class"]=="tight" for y in states)],
      "deadline_relaxed":[x for x in ROOTS if any(y["root"]==x and y["deadline_class"]=="relaxed" for y in states)],
      "context_conditions":list(VARIANTS),
      "n_independent_business_roots":len(ROOTS),
      "n_distinct_current_A_states":len(states),
      "n_legal_A_actions_per_state":len(ACTIONS),
      "K_seeds_per_action":list(SEEDS),
      "same_state_and_same_Frozen_downstream_policy":True,
      "output_sha256":sha(p),
      "source_NEW_ROOTS_SHA256":sha(SOURCE/"NEW_ROOTS.jsonl"),
      "source_Phase2_DATA_MANIFEST_SHA256":sha(SOURCE/"DATA_MANIFEST.json"),
      "full_original_source_read_only":True,
      "final_test_root_P211_P212_and_DG36_never_parsed_or_evaluated":True,
      "no_dev_root_used":True,
    }
    (OUT/"PRE_REGISTERED_CELL.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("CCE_PHASE2_A_FORKS_FROZEN",json.dumps(manifest,ensure_ascii=False))
if __name__=="__main__":preflight()
