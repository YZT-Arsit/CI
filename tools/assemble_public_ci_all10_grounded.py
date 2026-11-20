r"""Combine 2 existing + 8 newly validated grounded synthetic task families.

This is NOT extra independent roots, an official benchmark, or a certified DPO
training corpus. Only model-visible observations enter MODEL_INPUTS.
All oracle verdicts remain in explicitly separate EVAL_ONLY files.
"""
from __future__ import annotations
from pathlib import Path
from hashlib import sha256
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"artifacts"/"20261119_composition_signals"
OLD=BASE/"public_workflow_grounded_W01_R03_v1"
NEW=BASE/"public_workflow_remaining_8_grounded_v1"

def load(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def jd(path):
    return json.loads(path.read_text(encoding="utf-8"))

def save(path,items):
    with path.open("x",encoding="utf-8",newline="\n") as f:
        for row in items:
            f.write(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n")

def jfile(path,v):
    with path.open("x",encoding="utf-8") as f:
        json.dump(v,f,ensure_ascii=False,sort_keys=True,indent=2)
        f.write("\n")

def validated(parent,expected):
    m=jd(parent/"MANIFEST_SHA256.json")
    for name,h in m.items():
        assert sha256((parent/name).read_bytes()).hexdigest()==h,name
    a=jd(parent/"INDEPENDENT_AUDIT.json")
    assert a.get("status",a.get("result"))==expected
    return m,a

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    out=args.output.resolve()
    if out.exists():raise FileExistsError(out)
    man_old,aold=validated(OLD,"INDEPENDENT_CPU_AUDIT_PASS")
    man_new,anew=validated(NEW,"INDEPENDENT_REPLAY_PASS")
    roots=[];worlds=[];inputs=[];pairs=[];traces=[];negative=[]
    splits={"W01":"train","W02":"train","W03":"dev","W04":"train",
            "W05":"exploratory_test","R01":"train","R02":"train",
            "R03":"dev","R04":"exploratory_test","R05":"train"}
    for base,engine in ((OLD,"w01_r03_v1"),(NEW,"remaining_eight_v1")):
        for item in load(base/"ROOTS.jsonl"):
            root=item["root_id"]
            assert root in splits
            roots.append({
                "root_id":root,"split":splits[root],
                "underlying_engine":engine,
                "source_adapted_task":item.get("business_task",item.get("task")),
                "upstream_reference":item.get("source_reference"),
                "native_knowledge_topology":item.get(
                    "native_knowledge_structure",item.get("knowledge_topology")),
                "synthetic_privacy_policy":True,
                "human_norm_certified":False,
                "official_upstream_harness":False,
            })
        for x in load(base/"WORLDS.jsonl"):
            name=x.get("id",x.get("world_id"))
            root=x["root_id"]
            history=x.get("history_variant",x.get("history"))
            assert name and root in splits and history
            worlds.append({
                "world_id":name,"root_id":root,"split":splits[root],
                "engine":engine,
                "history":history,
                "authorized":x["authorized"],
                "task_profile":x["task_profile"] if "task_profile" in x else x["profile"],
                "is_action_probe":x.get("probe_only_not_full_task_success",
                    x.get("minimal_is_action_probe",False)),
                "initial_native_sha256":x["initial_native_hash"],
                "initial_business_sha256":x.get("business_fixture_hash",
                                               x.get("initial_business_store_hash")),
            })
        for x in load(base/"MODEL_DECISIONS.jsonl"):
            prompt=x.get("model_input",x.get("observation"))
            assert isinstance(prompt,dict) and prompt
            world=x.get("case_id",x.get("world_id"))
            refhash=x.get("input_sha256",x.get("model_prompt_sha256"))
            assert refhash
            import hashlib
            calculated=sha256(json.dumps(prompt,ensure_ascii=False,sort_keys=True,
                separators=(",",":"),allow_nan=False).encode("utf-8")).hexdigest()
            assert calculated==refhash,(world,calculated,refhash)
            assert "ground_truth" not in prompt and "violation" not in prompt
            assert "issued" not in prompt and "native" not in prompt
            assert "grounded_fact_internal" not in json.dumps(prompt)
            inputs.append({
                "world_id":world,"root_id":x["root_id"],
                "split":splits[x["root_id"]],"engine":engine,
                "model_input":prompt,
                "available_actions":prompt["available_actions"],
                "prompt_sha256":refhash,
                "eval_only_truth_withheld":True})
        for x in load(base/"CONTEXT_PAIRS_EVAL_ONLY.jsonl"):
            root=x["root_id"]
            typ=x.get("kind",x.get("type"))
            left=x.get("left",x.get("left_world"))
            right=x.get("right",x.get("right_world"))
            pairs.append({
                "pair_id":x.get("id",x.get("pair_id")),
                "root_id":root,"split":splits[root],
                "engine":engine,"type":typ,
                "left_world_id":left,"right_world_id":right,
                "same_candidate_action":x["same_current_action"] if (
                    "same_current_action" in x) else x["same_candidate_action"],
                "eval_only_left_verdict":x["eval_only_left_verdict"],
                "eval_only_right_verdict":x["eval_only_right_verdict"],
                "not_a_terminal_DPO_preference_label":True})
        for x in load(base/"VERIFIED_EXECUTIONS_EVAL_ONLY.jsonl"):
            traces.append({**x,"root_id":x["root_id"],
                           "split":splits[x["root_id"]],
                           "engine":engine})
        if engine=="remaining_eight_v1":
            for x in load(base/"NEGATIVE_CONTROLS_EVAL_ONLY.jsonl"):
                negative.append({"root_id":x["root_id"],"engine":engine,**x})
        else:
            for x in aold["negative_controls"]:
                negative.append({"root_id":x["name"].split("_")[0],
                    "engine":engine,"control":x["name"],"passed":x["passed"]})
    assert(len(roots),len(worlds),len(inputs),len(pairs),len(traces),len(negative))==(10,80,80,30,60,24)
    assert len({x["root_id"] for x in roots})==10
    assert len({x["world_id"] for x in worlds})==80
    assert len({x["world_id"] for x in inputs})==80
    assert len({x["pair_id"] for x in pairs})==30
    assert all(x["left_world_id"]!=x["right_world_id"] for x in pairs)
    assert all(x["split"]==splits[x["root_id"]] for x in traces)
    assert all(x["split"]==splits[x["root_id"]] for x in inputs)
    assert all(x["split"]==splits[x["root_id"]] for x in pairs)
    out.mkdir(parents=True)
    for name,rows in (
        ("ROOTS.jsonl",roots),("WORLDS.jsonl",worlds),
        ("MODEL_INPUTS.jsonl",inputs),("CONTEXT_PAIRS_EVAL_ONLY.jsonl",pairs),
        ("FULL_TRACES_EVAL_ONLY.jsonl",traces),
        ("NEGATIVE_CONTROLS_EVAL_ONLY.jsonl",negative)):
        save(out/name,rows)
    v={
        "status":"ALL10_GROUNDED_ASSEMBLY_PASS",
        "distinct_business_workflow_roots":10,
        "independent_business_workflow_sources":2,
        "total_executable_scenarios":80,
        "on_policy_action_state_observations":80,
        "controlled_history_flip_hold_authorization_pairs":30,
        "real_simulated_business_and_knowledge_traces":60,
        "negative_controls_total":24,
        "split_roots":{"train":6,"dev":2,"exploratory_test":2},
        "generation_method":"Two pinned native simulators with business-bound source language certificates, no API.",
        "all_source_facts_are_synthetic":True,
        "official_upstream_harness_not_used":True,
        "uncertified_natural_language_beyond_exact_templates":True,
        "strong_template_lexical_baseline_must_be_kept":True,
        "norm_validity_human_verified":False,
        "blind_test_ready":False,
        "training_approved":False,
        "no_GPU_or_model_training":True,
        "original_datasets_left_untouched":True,
        "provenance":{
            "w01_r03_dataset":str(OLD),
            "remaining_eight_dataset":str(NEW),
            "original_2_manifest_sha256":sha256((OLD/"MANIFEST_SHA256.json").read_bytes()).hexdigest(),
            "new_8_manifest_sha256":sha256((NEW/"MANIFEST_SHA256.json").read_bytes()).hexdigest(),
            "w01_r03_independent_audit":aold["status"],
            "remaining_eight_independent_audit":anew["result"],
        },
    }
    jfile(out/"VALIDATION.json",v)
    jfile(out/"MANIFEST_SHA256.json",
        {f.name:sha256(f.read_bytes()).hexdigest() for f in sorted(out.iterdir()) if f.is_file()})
    print(json.dumps(v,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":main()
