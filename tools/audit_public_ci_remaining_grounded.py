r"""Independent CPU replay for eight evidence-bound synthetic public workflows.

Re-loads all saved files, verifies the manifest, reconstructs true simulator
observations and actions, checks paired knowledge verdicts, and replays
each full native/tool trajectory exactly. No API/LLM/GPU.
"""
from __future__ import annotations
import argparse,json,sys
from copy import deepcopy
from pathlib import Path
from hashlib import sha256
from dataclasses import asdict
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
sys.path.insert(0,str(ROOT/"tools"))
from kci.public_workflow_ci_remaining_grounded import (
    RemainingGroundedCIEnv,REMAINING,source_business_result,logical_assertions,
    parse_claim,proof_assertion,
)
from kci.public_workflow_ci import LOOKUP
from kci.knowledge_world_seed_v2 import FACTS,digest,canonical
from build_public_ci_grounded_two import action_probe
from build_public_ci_remaining_grounded import negative_case,tamper_case

def read(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",type=Path,required=True)
    args=ap.parse_args()
    root=args.dataset.resolve()
    audit=root/"INDEPENDENT_AUDIT.json"
    if audit.exists():raise FileExistsError(audit)
    manifest=json.loads((root/"MANIFEST_SHA256.json").read_text(encoding="utf-8"))
    for filename,h in manifest.items():
        assert sha256((root/filename).read_bytes()).hexdigest()==h,filename
    roots=read(root/"ROOTS.jsonl")
    worlds=read(root/"WORLDS.jsonl")
    model=read(root/"MODEL_DECISIONS.jsonl")
    pairs=read(root/"CONTEXT_PAIRS_EVAL_ONLY.jsonl")
    traces=read(root/"VERIFIED_EXECUTIONS_EVAL_ONLY.jsonl")
    answers=read(root/"ANSWER_KEYS_EVAL_ONLY.jsonl")
    neg=read(root/"NEGATIVE_CONTROLS_EVAL_ONLY.jsonl")
    assert (len(roots),len(worlds),len(model),len(pairs),
            len(traces),len(answers),len(neg))==(8,64,64,24,48,8,16)
    assert set(x["root_id"] for x in roots)==set(REMAINING)
    group={r["root_id"]:r["group"] for r in roots}
    info={w["world_id"]:w for w in worlds}
    prompts={m["world_id"]:m for m in model}
    count=Counter()
    for world_id,w in info.items():
        env=RemainingGroundedCIEnv(LOOKUP[w["root_id"]],w["history"],
                                   w["authorized"],w["profile"])
        st=env.initial()
        assert w["initial_native_hash"]==digest(asdict(st.native))
        assert w["initial_business_store_hash"]==digest(st.store)
        assert group[w["root_id"]]==w["split"]
        assert w["step_horizon"]==env.knowledge.horizon
        expected=[{"sender":r["sender"],"recipient":r["recipient"],
                   "event_id":r["event_id"],
                   "content":st.delivered_texts.get(r["event_id"],"")}
                  for r in st.native.ledger if r["recipient"]=="R"]
        assert expected==w["initial_received_messages"]
        claims=logical_assertions(env.spec,st.store)
        for f in FACTS:
            text=proof_assertion(claims[f])
            assert parse_claim(text)=={k:claims[f][k]
                         for k in ("relation","subject","object")}
        p=action_probe(env)
        saved=prompts[world_id]
        assert p["model_input"]==saved["observation"],world_id
        assert p["model_input_sha256"]==saved["model_prompt_sha256"]
        assert saved["available_actions"]==p["model_input"]["available_actions"]
        assert not any(k in p["model_input"] for k in (
            "ground_truth","reward","violation","native","issued","seed_id"))
        assert "grounded_fact_internal" not in canonical(p["model_input"])
        assert '"assertion":' not in canonical(p["model_input"])
        count["world_states_and_prompts"]+=1
        count["certified_language_relations"]+=4
    for p in pairs:
        left=info[p["left_world"]]
        right=info[p["right_world"]]
        l=RemainingGroundedCIEnv(LOOKUP[p["root_id"]],left["history"],
                                 left["authorized"],left["profile"])
        r=RemainingGroundedCIEnv(LOOKUP[p["root_id"]],right["history"],
                                 right["authorized"],right["profile"])
        one=action_probe(l);two=action_probe(r)
        assert one["action"]==two["action"]==p["same_candidate_action"]
        assert one["audit"]["verdict"]==p["eval_only_left_verdict"]
        assert two["audit"]["verdict"]==p["eval_only_right_verdict"]
        assert one["model_input_sha256"]==p["left_prompt_hash"]
        assert two["model_input_sha256"]==p["right_prompt_hash"]
        count["history_pairs_checked"]+=1
    for t in traces:
        e=RemainingGroundedCIEnv(LOOKUP[t["root_id"]],
                                 t["history"],t["authorized"],t["profile"])
        current=e.run_script(t["order"],t["private_checks"])
        for key,value in current.items():
            assert t[key]==value,(t["root_id"],t["trace_type"],key)
        count["full_traces_checked"]+=1
    for n in neg:
        if n["kind"]=="no_business_source":
            got=negative_case(LOOKUP[n["root_id"]],n["kind"])
        else:
            got=tamper_case(LOOKUP[n["root_id"]])
        assert got==n,(n["root_id"],n["kind"])
        count["negative_controls_checked"]+=1
    assert count["world_states_and_prompts"]==64
    assert count["history_pairs_checked"]==24
    assert count["full_traces_checked"]==48
    assert count["negative_controls_checked"]==16
    report={
        "result":"INDEPENDENT_REPLAY_PASS",
        "manifest_files_verified":len(manifest),
        "counts":dict(count),
        "all_8_workflows_replayed":True,
        "business_events_and_knowledge_verdicts_replayed":True,
        "original_2_workflows_unmodified":True,
        "official_upstream_harness_tested":False,
        "human_norm_certified":False,
        "formal_training_ready":False,
        "API_calls":0}
    audit.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":main()
