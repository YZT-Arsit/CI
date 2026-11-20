"""Independent CPU dataset replay/manifest and shortcut-policy checks."""
from __future__ import annotations
from collections import Counter,defaultdict
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.knowledge_world_seed import (
    VERSION, Seed,KnowledgeTask,build_seeds,certified_contrasts,topological_orders,
    find_safe_witness,FACTS,canonical,digest,
)

def records(p):
    return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",type=Path,required=True)
    args=ap.parse_args()
    root=args.dataset
    manifest=json.loads((root/"MANIFEST.sha256.json").read_text(encoding="utf-8"))
    for name,expected in manifest.items():
        assert sha256((root/name).read_bytes()).hexdigest()==expected, name
    data={name:records(root/(name+".jsonl")) for name in
          ("roots","worlds","action_context_pairs_EVAL",
           "verified_native_traces_EVAL","model_inputs","answer_key_EVAL_ONLY")}
    assert len(data["roots"])==20 and len(data["worlds"])==140
    assert len(data["action_context_pairs_EVAL"])==60
    assert len(data["verified_native_traces_EVAL"])==140
    assert len(data["model_inputs"])==140
    assert len(data["answer_key_EVAL_ONLY"])==20
    expected={s.seed_id:s for s in build_seeds()}
    assert set(expected)=={s["seed_id"] for s in data["roots"]}
    split=defaultdict(set)
    for r in data["worlds"]:
        s=expected[r["seed_id"]]
        assert r["split"]==s.split
        split[s.seed_id].add(r["split"])
        env=KnowledgeTask(s,r["history"],r["authorized"],r["task_profile"])
        initial=env.initial()
        assert r["initial_state_hash"]==digest(asdict(initial))
        assert r["initial_receipts"]==initial.ledger
        assert r["required_checks"]==list(env.required)
    assert all(len(s)==1 for s in split.values())
    all_pairs={x["pair_id"]:x for x in data["action_context_pairs_EVAL"]}
    assert len(all_pairs)==60
    for s in expected.values():
        r=certified_contrasts(s)
        for kind,left,right in (
            ("flip","unrelated","linked"),
            ("hold","hold_a","hold_b"),
            ("permission","linked","linked_authorized")):
            pair=all_pairs[s.seed_id+"_"+kind]
            l=r["probe"][left];rr=r["probe"][right]
            assert pair["eval_only_left_immediate_verdict"]==l["audit"]["verdict"]
            assert pair["eval_only_right_immediate_verdict"]==rr["audit"]["verdict"]
            assert pair["left_observation_hash"]==digest(l["pre_decision_observation"])
            assert pair["right_observation_hash"]==digest(rr["pre_decision_observation"])
    # The non-leaky model-input file must not contain private evaluator annotations.
    inputs=data["model_inputs"]
    assert all(set(e)=={"id","seed_id","split","world_id",
                        "model_input","available_actions","state_hash"} for e in inputs)
    assert all("eval_only_left_immediate_verdict" not in canonical(e) for e in inputs)
    assert all(e["model_input"]["available_actions"]==e["available_actions"] for e in inputs)
    baselines=defaultdict(lambda:Counter())
    for s in expected.values():
        for history,profile in (("linked","full"),("unrelated","full"),
                                ("linked","minimal"),("unrelated","minimal")):
            env=KnowledgeTask(s,history,False,profile)
            order=topological_orders(s,env.required)[0]
            policies={
                "always_raw":env.rollout(order,()),
                "always_private":env.rollout(order,env.required),
                "fixed_redact_f0":env.rollout(order,("f0",)),
                "fixed_redact_f0_f2":env.rollout(order,tuple(f for f in ("f0","f2") if f in env.required)),
                "best_safe_environment_oracle":find_safe_witness(env),
            }
            assert policies["best_safe_environment_oracle"] is not None
            for name,v in policies.items():
                k=(profile,name)
                baselines[k]["n"]+=1
                baselines[k]["safe_success"]+=int(v["success"] and not v["violation"])
                baselines[k]["violations"]+=int(v["violation"])
                baselines[k]["safe_incomplete"]+=int(not v["success"] and not v["violation"])
                baselines[k]["steps_on_success"]+=v["steps"] if v["success"] else 0
    summary={
        "status":"AUDIT_PASS",
        "verified_manifest_files":len(manifest),
        "verified_scenario_records":140,
        "verified_pair_records":60,
        "verified_reconstructed_native_moments":140+60*2,
        "splits":{"train":12,"dev":3,"test":5},
        "baselines":{profile:{
            policy:{
                "n":a["n"],"safe_success":a["safe_success"],
                "violations":a["violations"],
                "safe_incomplete":a["safe_incomplete"],
                "mean_steps_on_success":round(a["steps_on_success"]/a["safe_success"],3)
                if a["safe_success"] else None,
            } for (p,policy),a in baselines.items() if p==profile
        } for profile in ("minimal","full")},
        "not_a_model_evaluation":True,
        "blind_test_not_certified":True,
        "real_business_norm_certification":False,
    }
    output=root/"AUDIT_AND_SHORTCUTS.json"
    if output.exists():raise FileExistsError("Refuse overwrite audit")
    output.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":main()
