"""V4 strict schema loader, compatible with 2026-11-22 history-evaluator format.

Only the original training pairs are loaded; each independent same-state
comparison has two ALREADY-SUPPLIED option orders. No synthetic swap is made.
The V4 data source is READ ONLY. DG36 is forbidden in every operation.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json,re

FROZEN_FINAL_ROOT="DG36"
ALLOWED_TRAIN_ROOTS=frozenset(("DG31","DG32","DG33"))
ALLOWED_DEV_ROOTS=frozenset(("DG34","DG35"))
RELEASE_CATEGORIES=frozenset(("D_CI","D_positive_RAW","D_authorized_positive_RAW"))
ROOT=Path(__file__).resolve().parents[1]
V4_DEFAULT=Path(r"D:\research\kci_data_v4_pilot_20261122\artifacts\20261122_kci_data_v4_pilot_v1")
def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()
def checked_read_train(path):
    path=Path(path)
    x=[json.loads(s) for s in path.open(encoding="utf8") if s.strip()]
    assert x and all(r.get("split")=="train" for r in x),path
    # Never allow DG36 or evaluation samples through training loader.
    assert all(r.get("root_id",r.get("root")) in ALLOWED_TRAIN_ROOTS for r in x),path
    return x
def checked_read_dev(path):
    """Filter raw lines BEFORE JSON parsing. Never inspect final-test payload."""
    data=[]
    for line in Path(path).open(encoding="utf8"):
        if '"DG36' in line or '"final_test"' in line:
            continue
        if not line.strip():continue
        row=json.loads(line)
        if row.get("root_id",row.get("root")) in ALLOWED_DEV_ROOTS and row.get("split")=="dev":
            data.append(row)
    assert data
    return data

def option_lines(prompt,actions):
    # Exact user-facing V4 prompt, after the native 'Options:' marker.
    head,sep,tail=prompt.rpartition("\nOptions:\n")
    assert sep and head and tail
    lines=tail.splitlines()
    assert len(lines)==len(actions)
    for j,(line,action) in enumerate(zip(lines,actions)):
        assert line==(chr(65+j)+". "+action),(line,action)
    return head

def strict_train_groups(source=V4_DEFAULT):
    source=Path(source)
    canonical=checked_read_train(source/"TRAIN_SAME_STATE_PREFERENCES.jsonl")
    compat=checked_read_train(source/"TRAIN_V4_HISTORY_EVALUATOR_COMPAT.jsonl")
    assert len(canonical)==len(compat)
    # Unlike old trainer, roles, stages and order are data-driven.
    match={(x["root"],x["state_hash"],tuple(x["all_executable_actions"]),
            x["chosen_action"],x["rejected_action"]):x for x in compat}
    assert len(match)==len(compat),"Repeated compat rows"
    pairs=defaultdict(list)
    for r in canonical:
        root=r["root_id"]
        role=r["acting_role"]
        stage=(r["prompt"].split("\nStage: ",1)[1].split("\n",1)[0])
        opts=list(r["option_order"])
        assert 2<=len(opts)<=26 and len(set(opts))==len(opts)
        assert r["chosen_action"] in opts and r["rejected_action"] in opts
        assert r["chosen_action"]!=r["rejected_action"]
        assert opts[ord(r["chosen"])-65]==r["chosen_action"]
        assert opts[ord(r["rejected"])-65]==r["rejected_action"]
        assert r["not_LLM_policy_Qpi"] and r["same_role_state_and_model_input"]
        assert not r["cross_history_pair"]
        assert sha256(r["prompt"].encode("utf8")).hexdigest()==r["prompt_sha256"]
        base=option_lines(r["prompt"],opts)
        key=(root,r["state_sha256"],tuple(opts),r["chosen_action"],r["rejected_action"])
        assert key in match,key
        other=match[key]
        assert (other["role"],other["stage"])==(role,stage)
        assert (other["branch"],other["observation_hash"])==(r["category"],r["same_state_observation_sha256"])
        assert other["prompt"]==r["prompt"]
        assert other["not_model_Qpi"] and other["split"]=="train"
        assert other["native_evidence"]==r["evidence"]
        assert other["chosen_letter"]==r["chosen"] and other["rejected_letter"]==r["rejected"]
        assert other["state_hash"]==r["state_sha256"]
        assert opts==other["all_executable_actions"]
        assert r["evidence_grade"]==r["evidence"]["grade"]
        if stage=="release":
            assert r["category"] in RELEASE_CATEGORIES
            assert {r["chosen_action"],r["rejected_action"]}=={"SEND_RAW_R","VERIFY_PRIVATE_R"}
        item={
           "root":root,"role":role,"stage":stage,"category":r["category"],
           "prompt":r["prompt"],"option_order":opts,
           "chosen_action":r["chosen_action"],"rejected_action":r["rejected_action"],
           "chosen_letter":r["chosen"],"rejected_letter":r["rejected"],
           "state_hash":r["state_sha256"],"observation_hash":r["same_state_observation_sha256"],
           "evidence":r["evidence"],"evidence_grade":r["evidence_grade"],
           "preference_id":r["preference_id"],"original_option_body":base,
           "not_model_Qpi":True,"source_split":"train",
        }
        k=(root,role,stage,r["category"],r["state_sha256"],
           r["same_state_observation_sha256"],r["chosen_action"],r["rejected_action"])
        pairs[k].append(item)
    groups=[]
    for k,items in sorted(pairs.items()):
        assert len(items)==2,("Missing original paired option ordering",k,len(items))
        assert len({r["preference_id"] for r in items})==2
        assert len({r["prompt"] for r in items})==2
        assert len({r["original_option_body"] for r in items})==1
        a,b=sorted(items,key=lambda x:"__swapped" in x["preference_id"])
        assert a["preference_id"].endswith("__normal") and b["preference_id"].endswith("__swapped")
        assert a["option_order"][:2]==list(reversed(b["option_order"][:2]))
        assert a["option_order"][2:]==b["option_order"][2:]
        assert a["chosen_action"]==b["chosen_action"] and a["rejected_action"]==b["rejected_action"]
        gid=sha256(("|".join(str(v) for v in k)).encode("utf8")).hexdigest()
        groups.append({"group_id":gid,"root":a["root"],"role":a["role"],"stage":a["stage"],
                       "category":a["category"],"state_hash":a["state_hash"],
                       "observation_hash":a["observation_hash"],
                       "chosen_action":a["chosen_action"],"rejected_action":a["rejected_action"],
                       "evidence_grade":a["evidence_grade"],"paired_option_records":[a,b],
                       "n_original_records":2})
    assert len(groups)*2==len(canonical)
    assert sum(x["stage"]=="release" for x in groups)>0
    return groups

def native_dev_rows(source=V4_DEFAULT):
    """Read-only original V4 development labels; no Q* or Qpi inference."""
    source=Path(source)
    obs=checked_read_dev(source/"MODEL_INPUTS_EVAL_ONLY.jsonl")
    native=checked_read_dev(source/"NATIVE_TRAJECTORIES_EVAL_ONLY.jsonl")
    contrasts=checked_read_dev(source/"CONTROLLED_CONTEXT_PAIRS_EVAL_ONLY.jsonl")
    om={x["world_id"]:x for x in obs}
    nm={x["world_id"]:x for x in native}
    assert len(om)==len(obs)==len(native)==len(nm)
    assert set(om)==set(nm)
    rows=[]
    for world in sorted(om):
        x=om[world];n=nm[world]
        assert x["root_id"]==n["root_id"] and x["state_sha256"]==n["replay_state_hash"]
        assert x["truth_not_exposed_to_model"]
        choices=x["candidate_actions"]
        swapped=x["option_swapped_actions"]
        assert choices[:2]==list(reversed(swapped[:2])) and choices[2:]==swapped[2:]
        assert option_lines(x["model_prompt"],choices)==option_lines(x["options_swapped_prompt"],swapped)
        assert sha256(x["model_prompt"].encode("utf8")).hexdigest()==x["model_prompt_sha256"]
        raw=n["fixed_suffix_native_trajectories"]["SEND_RAW_R"]
        verify=n["fixed_suffix_native_trajectories"]["VERIFY_PRIVATE_R"]
        rv=n["native_action_evaluations"]["SEND_RAW_R"]["verdict"]
        vv=n["native_action_evaluations"]["VERIFY_PRIVATE_R"]["verdict"]
        assert rv in ("VALID","VIOLATION") and vv=="VALID"
        assert verify["terminal_success"] and not verify["terminal_violation"]
        assert verify["outcome_grade"]=="verified_native_fixed_suffix_not_Qpi_not_optimal"
        if rv=="VIOLATION":
            assert raw["terminal_violation"] and not raw["terminal_success"]
            label="VERIFY_PRIVATE_R"
        else:
            assert raw["terminal_success"] and not raw["terminal_violation"]
            assert raw["execution_steps"]<verify["execution_steps"],world
            label="SEND_RAW_R"
        rows.append({
          "world":world,"root":x["root_id"],"role":x["actual_role"],"stage":"release",
          "source_split":"dev","options":choices,"swapped_options":swapped,
          "prompt":x["model_prompt"],"swapped_prompt":x["options_swapped_prompt"],
          "model_prompt_sha256":x["model_prompt_sha256"],
          "observation_hash":x["observation_sha256"],"state_hash":x["state_sha256"],
          "gold_action":label,"raw_verdict":rv,"verify_verdict":vv,
          "raw_fixed_suffix":raw,"verify_fixed_suffix":verify,
          "native_evidence_grade":n["native_evidence_kind"],
          "trace_origin":"V4_NATIVE_TRAJECTORIES_EVAL_ONLY",
        })
    parsed_contrasts=[]
    for r in contrasts:
        assert r["root_id"] in ALLOWED_DEV_ROOTS
        assert r["left_world"] in om and r["right_world"] in om
        assert r["same_candidate_source_message"] and r["same_native_candidate"]
        parsed_contrasts.append({
            "pair_id":r["pair_id"],"root":r["root_id"],"contrast_type":r["contrast_type"],
            "left_world":r["left_world"],"right_world":r["right_world"],
            "same_length_history":r["same_length_history"],
            "only_R_history_differs":r["only_R_history_differs"],
            "only_authorization_differs":r["only_authorization_differs"],
            "native_verifier_provenance":r["native_verifier_provenance"],
        })
    assert {r["root"] for r in rows}==ALLOWED_DEV_ROOTS
    assert len(rows)==5*len(ALLOWED_DEV_ROOTS)
    assert len(parsed_contrasts)==4*len(ALLOWED_DEV_ROOTS)
    return rows,parsed_contrasts

def prepared_manifest(src,groups,dev):
    src=Path(src)
    return {
      "status":"V4_STRICT_COMPATIBILITY_AND_BLIND_TEST_EXCLUSION",
      "input_files_sha256":{name:sha(src/name) for name in [
          "TRAIN_SAME_STATE_PREFERENCES.jsonl",
          "TRAIN_V4_HISTORY_EVALUATOR_COMPAT.jsonl",
          # hashes for dev inputs recorded without sending their content to trainers
          "MODEL_INPUTS_EVAL_ONLY.jsonl","NATIVE_TRAJECTORIES_EVAL_ONLY.jsonl",
          "CONTROLLED_CONTEXT_PAIRS_EVAL_ONLY.jsonl"]},
      "not_model_Qpi":True,"native_future_Qstar_file_NOT_USED_as_training_Qpi":True,
      "total_original_train_records":2*len(groups),
      "total_independent_train_comparisons":len(groups),
      "release_original_train_records":2*sum(g["stage"]=="release" for g in groups),
      "release_independent_train_comparisons":sum(g["stage"]=="release" for g in groups),
      "train_roots":sorted({g["root"] for g in groups}),
      "release_roots":sorted({g["root"] for g in groups if g["stage"]=="release"}),
      "train_by_category":dict(Counter(g["category"] for g in groups)),
      "train_by_role":dict(Counter(g["role"] for g in groups)),
      "train_by_stage":dict(Counter(g["stage"] for g in groups)),
      "train_only_normal_and_supplied_swap_once_each":True,
      "dev_worlds":len(dev),
      "dev_roots":sorted({r["root"] for r in dev}),
      "final_test_DG36_not_deserialized_scored_or_used":True,
    }

def prepare_v4(source=V4_DEFAULT,out=None):
    source=Path(source)
    groups=strict_train_groups(source)
    dev,contrasts=native_dev_rows(source)
    if out is None:
        return groups,dev,contrasts,prepared_manifest(source,groups,dev)
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    def dump(name,records):
        path=out/name
        path.write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in records),encoding="utf8")
        return sha(path)
    manifest=prepared_manifest(source,groups,dev)
    hashes={
        "TRAIN_ALL_GROUPS.jsonl":dump("TRAIN_ALL_GROUPS.jsonl",groups),
        "TRAIN_RELEASE_GROUPS.jsonl":dump("TRAIN_RELEASE_GROUPS.jsonl",[g for g in groups if g["stage"]=="release"]),
        "DEV_NATIVE_STATES_DG34_DG35.jsonl":dump("DEV_NATIVE_STATES_DG34_DG35.jsonl",dev),
        "DEV_CONTROLLED_CONTRASTS_DG34_DG35.jsonl":dump("DEV_CONTROLLED_CONTRASTS_DG34_DG35.jsonl",contrasts),
    }
    manifest["derived_files_sha256"]=hashes
    (out/"DATA_MANIFEST.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    return groups,dev,contrasts,manifest
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,default=V4_DEFAULT)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    data,dev,contrasts,manifest=prepare_v4(a.source,a.output)
    print(json.dumps(manifest,indent=2,ensure_ascii=False))
