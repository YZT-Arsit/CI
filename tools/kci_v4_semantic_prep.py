"""Freeze V4 semantic 4B cell: aware CE/DPO and verified blind-teacher CE.

Exact V4 source is READ ONLY. Only train roots DG31-33 and dev DG34-35
are deserialized; DG36 is filtered before JSON decoding. The actual actor
prompt in every teacher arm still contains TRUE receiver deliveries.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from hashlib import sha256
from collections import Counter,defaultdict
import importlib.util,json,sys
ROOT=Path(__file__).resolve().parents[1]
V4=Path(r"D:\research\kci_data_v4_pilot_20261122")
SRC=V4/"artifacts/20261122_kci_data_v4_pilot_v1"
PREVIOUS=ROOT/"artifacts/20261123_v4_release_control_v1"
OUT=ROOT/"artifacts/20261123_v4_semantic_control_v1"
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci_v4_semantic_action import semantic_v4_candidate,assert_views_position_invariant
from kci_v4_loader import checked_read_dev,checked_read_train
from kci.diverse_ci_branch_env import Task,digest
from kci.diverse_ci_long_horizon_v31 import V31Case
RAW,VERIFY="SEND_RAW_R","VERIFY_PRIVATE_R"
def read(path):return [json.loads(s) for s in Path(path).read_text(encoding="utf8").splitlines() if s.strip()]
def sha(path):return sha256(Path(path).read_bytes()).hexdigest()
def native_blind_evidence():
    envpath=V4/"code/kci/data_v4_env.py"
    spec=importlib.util.spec_from_file_location("kci.data_v4_env",envpath)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    roots={}
    # Strict raw-line filtering before JSON parsing, including all DEV metadata.
    def only_train_rows(path):
        for line in Path(path).open(encoding="utf8"):
            if '"DG36' in line or '"final_test"' in line:continue
            r=json.loads(line)
            if r.get("split")=="train":yield r
    # Only TRAIN rows deserialized; no DG36 spec/task instance generated.
    for r in only_train_rows(SRC/"ROOTS.jsonl"):
        sp=dict(r["native_v31_compatible_spec"]);sp["task"]=Task(**sp["task"])
        roots[r["root_id"]]=V31Case(**sp)
    assert set(roots)=={"DG31","DG32","DG33"}
    worlds=list(only_train_rows(SRC/"WORLD_CONFIGS.jsonl"))
    assert len(worlds)==15
    evidence={}
    for w in worlds:
        if w["kind"] not in ("risky","safe","authorized"):continue
        env=module.V4ChainEnv(roots[w["root_id"]],w["first_packet_kind"],
                              w["second_packet_kind"],w["explicit_authorization"],
                              w["reversed_ledger_delivery_order"])
        state,_=module.native_before_release(env,w["upstream_actual_action"].removeprefix("A_SEND_"))
        assert digest(asdict(state))==w["native_state_hash"]
        assert digest(env.observation(state))==w["observation_hash"]
        clone=deepcopy(state)
        clone.native.delivered["R"]=[]
        clone.native.ledger=[x for x in clone.native.ledger if x["recipient"]!="R"]
        assert not clone.native.delivered["R"]
        raw=module.native_continue(env,clone,RAW)
        verified=module.native_continue(env,clone,VERIFY)
        nxt,r=env.transition(clone,RAW)
        assert r["verdict"]=="VALID"
        assert raw["terminal_success"] and not raw["terminal_violation"]
        assert verified["terminal_success"] and not verified["terminal_violation"]
        assert raw["execution_steps"]<verified["execution_steps"]
        evidence[(w["root_id"],w["kind"])]={
            "source_world":w["world_id"],"root":w["root_id"],"condition":w["kind"],
            "original_state_hash":w["native_state_hash"],
            "teacher_history_intervention":"clear_only_R_prior_ledger_and_delivered_in_clone",
            "teacher_action":RAW,"native_RAW_verdict":r["verdict"],
            "raw_fixed_suffix_success":raw["terminal_success"],
            "raw_fixed_suffix_steps":raw["execution_steps"],
            "verify_fixed_suffix_steps":verified["execution_steps"],
            "not_model_Qpi_or_global_Qstar":True}
    assert len(evidence)==9
    return evidence

def prepare():
    if OUT.exists():raise FileExistsError(OUT)
    groups=read(PREVIOUS/"TRAIN_RELEASE_GROUPS.jsonl")
    dev=read(PREVIOUS/"DEV_NATIVE_STATES_DG34_DG35.jsonl")
    assert len(groups)==9 and len(dev)==10
    evidence=native_blind_evidence()
    aware=[];blind=[]
    for g in groups:
        a,b=g["paired_option_records"]
        assert assert_views_position_invariant(a["prompt"],a["option_order"],b["prompt"],b["option_order"])
        root=g["root"];condition=g["category"]
        kind={"D_CI":"risky","D_positive_RAW":"safe",
              "D_authorized_positive_RAW":"authorized"}[condition]
        native_blind=evidence[root,kind]
        assert native_blind["original_state_hash"]==g["state_hash"]
        actions=tuple(sorted(a["option_order"]))
        assert actions==( "ABORT","SEND_RAW_R","VERIFY_PRIVATE_R")
        obs_signatures={act:sha256(semantic_v4_candidate(a["prompt"],a["option_order"],act).encode()).hexdigest() for act in actions}
        common={
          "comparison_id":g["group_id"],"root":root,"condition":kind,
          "role":g["role"],"stage":g["stage"],"source_split":"train",
          "observation_hash":g["observation_hash"],"native_state_hash":g["state_hash"],
          "canonical_actions":list(actions),"prompt":a["prompt"],"original_action_order":a["option_order"],
          "swapped_prompt":b["prompt"],"swapped_action_order":b["option_order"],
          "semantic_candidate_sha256":obs_signatures,
          "source_original_row_count":2,"native_evidence_grade":g["evidence_grade"],
          "true_history_aware_action":g["chosen_action"],
          "true_history_blind_action":native_blind["teacher_action"],
          "blind_native_justification":native_blind,
          "no_oracle_label_visible_in_actor_prompt":True,
        }
        aware.append({**common,"teacher_type":"V4_real_history_native_teacher",
                      "chosen_action":g["chosen_action"],"rejected_action":g["rejected_action"]})
        blind.append({**common,"teacher_type":"V4_native_teacher_R_receipts_cleared_only",
                      "chosen_action":native_blind["teacher_action"],
                      "rejected_action":VERIFY})
    assert Counter(x["chosen_action"] for x in aware)=={RAW:6,VERIFY:3}
    assert Counter(x["chosen_action"] for x in blind)=={RAW:9}
    assert all(x["prompt"]==y["prompt"] for x,y in zip(aware,blind))
    assert all(x["semantic_candidate_sha256"]==y["semantic_candidate_sha256"] for x,y in zip(aware,blind))
    # Double view on dev: native V4 actor prompt and exact source-supplied
    # entity/workflow-alias text; both again permute V4 option menu unchanged.
    source_dev=checked_read_dev(SRC/"MODEL_INPUTS_EVAL_ONLY.jsonl")
    original={r["world_id"]:r for r in source_dev}
    assert set(original)=={r["world"] for r in dev}
    double=[]
    for r in dev:
        source=original[r["world"]]
        assert source["model_prompt"]==r["prompt"]
        assert source["candidate_actions"]==r["options"]
        assert source["options_swapped_prompt"]==r["swapped_prompt"]
        assert source["option_swapped_actions"]==r["swapped_options"]
        alias=source["entity_aliased_prompt"]
        assert alias!=r["prompt"]
        assert assert_views_position_invariant(r["prompt"],r["options"],r["swapped_prompt"],r["swapped_options"])
        # For alias, swapped order is a compatibility check, not an extra
        # example: candidate scores must be exactly the same per action.
        for act in r["options"]:
            semantic_v4_candidate(alias,r["options"],act)
        double.append({
          "world":r["world"],"root":r["root"],"role":r["role"],"stage":r["stage"],
          "condition":r["world"].rsplit("__",1)[1],"source_split":"dev",
          "native_prompt":r["prompt"],"native_swapped_prompt":r["swapped_prompt"],
          "aliased_prompt":alias,"canonical_actions":sorted(r["options"]),
          "original_action_order":r["options"],"swapped_action_order":r["swapped_options"],
          "native_state_hash":r["state_hash"],"observation_hash":r["observation_hash"],
          "true_native_action":r["gold_action"],
          "raw_native_verdict":r["raw_verdict"],"verify_native_verdict":r["verify_verdict"],
          "raw_fixed_suffix":r["raw_fixed_suffix"],"verify_fixed_suffix":r["verify_fixed_suffix"],
          "source_native_trace_grade":r["native_evidence_grade"],
          "alias_is_text_only_exact_V4_source":True,
        })
    assert len(double)==10
    OUT.mkdir(parents=True)
    def dump(name,records):
        path=OUT/name
        path.write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in records),encoding="utf8")
        return sha(path)
    hashes={
      "TRAIN_HISTORY_AWARE.jsonl":dump("TRAIN_HISTORY_AWARE.jsonl",aware),
      "TRAIN_HISTORY_BLIND.jsonl":dump("TRAIN_HISTORY_BLIND.jsonl",blind),
      "DEV_DUAL_VIEW_DG34_DG35.jsonl":dump("DEV_DUAL_VIEW_DG34_DG35.jsonl",double),
      "BLIND_NATIVE_REPLAY_EVIDENCE.jsonl":dump("BLIND_NATIVE_REPLAY_EVIDENCE.jsonl",list(evidence.values())),
    }
    manifest={
       "cell":"V4_semantic_three_arm_native_pair_comparison",
       "source_V4_is_read_only":str(SRC),
       "train_groups_source":str(PREVIOUS/"TRAIN_RELEASE_GROUPS.jsonl"),
       "train_source_sha256":sha(PREVIOUS/"TRAIN_RELEASE_GROUPS.jsonl"),
       "source_input_metadata_sha256":sha(SRC/"MODEL_INPUTS_EVAL_ONLY.jsonl"),
       "records_by_training_arm":9,"independent_train_release_comparisons":9,
       "train_roots":["DG31","DG32","DG33"],"dev_roots":["DG34","DG35"],
       "train_aware":{RAW:6,VERIFY:3},"train_blind":{RAW:9},
       "train_prompts_identical_between_arms":True,
       "train_normal_swapped_original_records_collapsed_to_one_semantic_input":True,
       "dev_worlds":10,"dev_views":["native","entity_alias"],
       "dev_native_action_permutation_invariance_exact":True,
       "blind_labels_native_replayed_at_true_training_states":True,
       "no_Qpi_claim_from_native_Qstar_or_fixed_suffix":True,
       "no_DG36_deserialized_or_scored":True,
       "derived_hashes":hashes,
    }
    (OUT/"DATA_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=="__main__":prepare()
