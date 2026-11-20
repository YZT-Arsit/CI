r"""Cross-family consistency validation of the complete synthetic KCI 10 workflow pilot."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from hashlib import sha256
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[1]
def load(p):
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
def digest(v):
    return sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",type=Path,required=True)
    a=ap.parse_args()
    p=a.dataset.resolve()
    output=p/"INDEPENDENT_CROSS_FAMILY_AUDIT.json"
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((p/"MANIFEST_SHA256.json").read_text(encoding="utf-8"))
    for name,expected in manifest.items():
        assert sha256((p/name).read_bytes()).hexdigest()==expected,name
    roots=load(p/"ROOTS.jsonl")
    worlds=load(p/"WORLDS.jsonl")
    prompts=load(p/"MODEL_INPUTS.jsonl")
    pairs=load(p/"CONTEXT_PAIRS_EVAL_ONLY.jsonl")
    traces=load(p/"FULL_TRACES_EVAL_ONLY.jsonl")
    negatives=load(p/"NEGATIVE_CONTROLS_EVAL_ONLY.jsonl")
    assert [len(x) for x in (roots,worlds,prompts,pairs,traces,negatives)]==[10,80,80,30,60,24]
    root_lookup={x["root_id"]:x for x in roots}
    assert len(root_lookup)==10
    assert len({x["world_id"] for x in worlds})==80
    world_lookup={x["world_id"]:x for x in worlds}
    prompt_lookup={x["world_id"]:x for x in prompts}
    assert set(world_lookup)==set(prompt_lookup)
    for x in prompts:
        assert x["root_id"] in root_lookup
        assert x["split"]==root_lookup[x["root_id"]]["split"]
        assert x["engine"]==world_lookup[x["world_id"]]["engine"]
        assert digest(x["model_input"])==x["prompt_sha256"]
        assert x["model_input"]["available_actions"]==x["available_actions"]
        assert all(k not in x["model_input"] for k in (
            "ground_truth","oracle","native","issued","reward","violation","seed_id"))
        serialized=json.dumps(x["model_input"],ensure_ascii=False,sort_keys=True)
        assert "grounded_fact_internal" not in serialized
        assert '"assertion"' not in serialized
    for x in pairs:
        a=x["left_world_id"];b=x["right_world_id"]
        assert a!=b and a in prompt_lookup and b in prompt_lookup
        assert world_lookup[a]["root_id"]==world_lookup[b]["root_id"]==x["root_id"]
        assert x["split"]==root_lookup[x["root_id"]]["split"]
        assert x["type"] in ("flip","hold","permission")
        if x["type"]=="flip":
            assert x["eval_only_left_verdict"]=="VALID"
            assert x["eval_only_right_verdict"]=="VIOLATION"
        elif x["type"]=="hold":
            assert x["eval_only_left_verdict"]==x["eval_only_right_verdict"]=="VALID"
        else:
            assert x["eval_only_left_verdict"]=="VIOLATION"
            assert x["eval_only_right_verdict"]=="VALID"
    byroot=Counter(x["root_id"] for x in roots)
    assert len(byroot)==10
    for rid in byroot:
        assert sum(w["root_id"]==rid for w in worlds)==8
        assert sum(x["root_id"]==rid for x in pairs)==3
        assert sum(x["root_id"]==rid for x in traces)==6
    for x in traces:
        assert x["split"]==root_lookup[x["root_id"]]["split"]
    counters=Counter(x["split"] for x in roots)
    assert counters==Counter({"train":6,"dev":2,"exploratory_test":2})
    for x in negatives:
        assert x["root_id"] in root_lookup
        if "passed" in x:assert x["passed"] is True
        elif x["kind"]=="tampered_natural_language":
            assert x["verdict"]=="UNKNOWN"
        else:
            assert x["observed_safe_success"] is False
    out={
        "result":"CROSS_FAMILY_AUDIT_PASS",
        "source_manifest_files_validated":len(manifest),
        "model_prompts_checked":len(prompts),
        "root_group_leakage":False,
        "roots":10,"worlds":80,"history_contrasts":30,
        "business_native_traces_referenced":60,
        "negative_controls_referenced":24,
        "source_task_roots_10_but_only_two_public_benchmark_suites":True,
        "all_10_roots_have_immediate_flip_hold_authorization":True,
        "manual_norm_certification":False,
        "language_templates_not_unseen":True,
        "blind_test_not_claimed":True,
        "model_training_not_run":True,
    }
    output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__":main()
