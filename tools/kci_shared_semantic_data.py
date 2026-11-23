"""Rebuild one-shared-4B dataset directly from verified V3.1 and first-release records.

No test labels in train. V3.1 and first-release are different decision
contracts and stay separately tagged. All actor observations are role-visible.
"""
from __future__ import annotations
from pathlib import Path
from hashlib import sha256
from collections import Counter,defaultdict
from dataclasses import asdict
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci_shared_semantic_action import canonical_actions, candidate_prompt
from kci_exact_policy_continuation_cpu import environment_public
from compare_kci_chain_methods_cpu_action_interaction import GROUP
from kci_history_eval_dev_probe_20261122 import at_first_release
from kci.diverse_ci_branch_env import digest
V31=ROOT/"artifacts/20261120_v31/V31_GATEA_FINAL_CANDIDATE"
HIST=ROOT/"artifacts/20261122_history_evaluator_dev_v1"
OUT=ROOT/"artifacts/20261123_shared_semantic_action_v1"
def read(p):return [json.loads(s) for s in p.read_text(encoding="utf8").splitlines() if s.strip()]
def dump(p,rows):
    assert not p.exists()
    p.write_text("".join(json.dumps(r,sort_keys=True,ensure_ascii=False)+"\n" for r in rows),encoding="utf8")
    return sha256(p.read_bytes()).hexdigest()
def make(source,obs,chosen,world,root,split,provenance,tag):
    assert chosen in canonical_actions(obs)
    assert obs["legal_actions"] and len(obs["legal_actions"])>=2
    assert "oracle_Qstar" not in obs and "violation" not in obs
    for act in canonical_actions(obs):
        s=candidate_prompt(obs,act)
        assert "Candidate action under consideration: "+act in s
    return {"source":source,"dataset_tag":tag,"observation":obs,
        "chosen_action":chosen,"world":world,"root":root,"split":split,
        "role":obs.get("actor",obs.get("role")),"phase":obs.get("stage",obs.get("current_stage")),
        "provenance":provenance,"observation_sha256":digest(obs)}
def main():
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    hashes={}
    balanced=read(V31/"TRAIN_ONLY_BALANCED_NATIVE_OPTIMAL_ACTION_PREFS.jsonl")
    assert len(balanced)==72
    vtrain=[make("v31",x["prompt"],x["chosen_action"],x["world_id"],x["world_id"].split("_M")[0],"train",
                 x["origin"],"v31_early_Qstar") for x in balanced]
    assert Counter(x["chosen_action"] for x in vtrain)=={"OPEN_PRIVATE_CHANNEL":36,"SKIP_PRIVATE_CHANNEL":36}
    hashes["V31_TRAIN_72.jsonl"]=dump(OUT/"V31_TRAIN_72.jsonl",vtrain)
    observed=read(V31/"MODEL_OBSERVATIONS.jsonl")
    labels=read(V31/"EXACT_NATIVE_ACTION_PREFERENCES_EVAL_ONLY.jsonl")
    labelmap={x["world_id"]:x for x in labels}
    assert len(observed)==len(labels)==315
    veval=[]
    for r in observed:
        world=r["world_id"]
        if not world.endswith(("__related","__same_entity_safe")):continue
        if r["split"]=="train":continue
        label=labelmap[world]
        assert label["split"]==r["split"]
        veval.append(make("v31",r["model_input"],label["oracle_action"],world,r["root_id"],r["split"],
                          "EVAL_ONLY_exact_native_Qstar", "v31_early_Qstar"))
    assert Counter(x["split"] for x in veval)=={"dev":18,"exploratory_test":36}
    hashes["V31_EVAL_54.jsonl"]=dump(OUT/"V31_EVAL_54.jsonl",veval)
    prior=read(HIST/"TRAIN_VISIBLE_STRICT8.jsonl")
    all_pairs=read(HIST/"PAIR_STATES.jsonl")
    refs={(r["world"],r["branch"]):r for r in all_pairs}
    chainworlds={r["world"]:r for subset in GROUP.values() for r in subset}
    def release(r,split):
        assert r["world"] in chainworlds
        env=environment_public(chainworlds[r["world"]])
        st=at_first_release(env,r["upstream_package"])
        obs=env.observation(st)
        assert digest(asdict(st))==r["native_state_sha256"]
        assert digest(obs)==r["visible_observation_sha256"]
        return make("first_release",obs,r["aware_label"],r["world"],r["root"],split,
                    "NATIVE_first_release_actual_receipt_teacher","raw_or_private")
    rtrain=[release(refs[x["world"],x["branch"]],"train") for x in prior]
    assert len(rtrain)==8 and len(set(x["root"] for x in rtrain))==2
    assert Counter(x["chosen_action"] for x in rtrain)=={"SEND_RAW_R":4,"VERIFY_PRIVATE_R":4}
    hashes["RELEASE_TRAIN_STRICT8.jsonl"]=dump(OUT/"RELEASE_TRAIN_STRICT8.jsonl",rtrain)
    reval=[]
    for r in all_pairs:
        if r["split"]=="train" or "__matched__" not in r["world"]:continue
        reval.append(release(r,r["split"]))
    assert Counter(x["split"] for x in reval)=={"dev":4,"exploratory_test":8}
    hashes["RELEASE_EVAL_MATCHED12.jsonl"]=dump(OUT/"RELEASE_EVAL_MATCHED12.jsonl",reval)
    multi=read(V31/"TRAIN_ONLY_NATIVE_MULTIROLE_ACTION_DEMOS.jsonl")
    assert len(multi)==485
    kept=[]
    for r in multi:
        if len(r["prompt"]["legal_actions"])<2:continue
        kept.append(make("v31_multirole",r["prompt"],r["oracle_native_action"],r["world_id"],
                         r["world_id"].split("_M")[0],"train",
                         r["provenance"],"oracle_role_demo"))
    # Same observation/action can recur in the original trajectories.
    unique={}
    for rec in kept:
        key=(rec["observation_sha256"],rec["chosen_action"],rec["role"])
        unique.setdefault(key,rec)
    kept=list(unique.values())
    hashes["MULTIROLE_TRAIN.jsonl"]=dump(OUT/"MULTIROLE_TRAIN.jsonl",kept)
    def roots(rows):return sorted(set(x["root"] for x in rows))
    counts={name:{
        "states":len(xs),
        "roots":len(roots(xs)),
        "roots_ids":roots(xs),
        "roles":dict(Counter(x["role"] for x in xs)),
        "labels":dict(Counter(x["chosen_action"] for x in xs)),
    } for name,xs in (
        ("v31_train",vtrain),("v31_eval",veval),
        ("release_train",rtrain),("release_eval",reval),("multirole_train",kept))}
    assert set(roots(vtrain)).isdisjoint(roots(veval))
    assert set(roots(rtrain)).isdisjoint(roots(reval))
    summary={"dataset_id":"SHARED_SEMANTIC_20261123","status":"FROZEN_TRAIN_EVAL_ROLES",
        "hashes":hashes,"counts":counts,
        "source_v31_root":"artifacts/20261120_v31/V31_GATEA_FINAL_CANDIDATE",
        "source_release_root":"artifacts/20261122_history_evaluator_dev_v1",
        "v31_exact_Qstar_not_Qpi":True,
        "release_expert_is_native_immediate_CI_plus_fixed_safe_suffix":True,
        "multi_role_native_oracle_not_Qpi":True,
        "all_splits_root_disjoint_per_dataset":True,
        "exploratory_seen_by_developers":True,
        "root_counts_not_independent_graph_structure_counts":True,
        "training_not_leaking_evaluation_labels":True}
    (OUT/"MANIFEST.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(summary,ensure_ascii=False,indent=2)[:13000])
if __name__=="__main__":main()
