"""Extract evaluator-only continuation tables from existing ProofWriter archive.

Read-only with respect to source archives. Never use evaluator targets as actor input.
Run into a NEW output directory. No LLM calls, parameter updates or locked split reads.
"""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

def records(path):
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def key(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def require(cond, message):
    if not cond:
        raise ValueError(message)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path,
                   help="d_future_core_20261124 root")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    require(not args.out.exists(), "Output must not already exist")
    core = args.source / "D_future_core_multisource_v2"
    paths = [core / "D_future_core.jsonl",
             core / "EVALUATOR_ONLY_WITNESS_INDEX.jsonl",
             args.source / "data" / "PROOFWRITER_CORE_EXECUTED_V1" / "EVAL_ONLY_ALL_NATIVE_BRANCHES.jsonl.gz"]
    pw = {r["row_id"]: r for r in records(paths[0]) if r["source_dataset"] == "PROOFWRITER"}
    require(all(r["split"] == "source_train" for r in pw.values()), "Unexpected source split")
    idx = {w["witness"]["evaluator_witness"]["spec_id"]: w["row_id"]
           for w in records(paths[1]) if w["row_id"] in pw}
    require(len(idx) == len(pw), "Spec join collision")
    cases, cells, seen = [], [], set()
    counts, labels = collections.Counter(), collections.Counter()
    for raw in records(paths[2]):
        rid = idx.get(raw["spec"]["spec_id"])
        if rid is None:
            continue
        require(rid not in seen, "Duplicate witness " + rid)
        seen.add(rid)
        r = pw[rid]
        e = raw["evaluated_contracts"][r["communication_contract"]]
        require(e["pass"], "Unaccepted contract")
        actions = e["actions"]
        aks = [key(a) for a in actions]
        require(len(aks) == 2 and set(aks) == {key(r["supervision"]["chosen_action"]), key(r["supervision"]["rejected_action"])},
                "Pair identity mismatch")
        full = set(aks) == {key(a) for a in r["executable_actions"]}
        counts["full_upstream_menu_rows"] += full
        catalog = {key(a) for a in r["actor_context"]["authorized_downstream_candidate_catalog"]}
        fs, fa = [], []
        for ai, (orig, auth) in enumerate(zip(e["original"], e["authorized"])):
            require(orig["first_audit"]["verdict"] == auth["first_audit"]["verdict"] == "VALID",
                    "Immediate validity failure")
            om = {key(d["action"]): d for d in orig["downstream"]}
            am = {key(d["action"]): d for d in auth["downstream"]}
            require(om.keys() == am.keys() and set(om) == catalog, "Catalog or authorization mismatch")
            ss, sa = [], []
            for bkey, d in sorted(om.items()):
                da = am[bkey]
                t = int(d["all_task_claims_verifiable"])
                ta = int(da["all_task_claims_verifiable"])
                require(t == ta, "Task truth changed under authorization")
                v, va = d["audit"]["verdict"], da["audit"]["verdict"]
                require(v in ("VALID", "VIOLATION") and va == "VALID", "Uncertain or unmatched authorization")
                z, za = int(v == "VIOLATION"), int(va == "VIOLATION")
                for dd, tt in [(d,t),(da,ta)]:
                    final = int(bool(dd.get("final_if_executed") and dd["final_if_executed"].get("task_success")))
                    require(tt == final, "Native terminal task mismatch")
                y, ya = t*(1-z), ta*(1-za)
                ss.append(y); sa.append(ya)
                labels[f"T{t}_V{z}"] += 1
                counts["task_complete_blocked_by_target_norm"] += int(t == 1 and z == 1 and za == 0)
                cells.append({"row_id":rid, "split_group":r["split_group"], "first_action_index":ai,
                              "first_action":actions[ai], "next_action":d["action"],
                              "evaluator_target":{"task_supported":t, "ci_violation":z,
                                                  "safe_success":y, "authorized_task_supported":ta,
                                                  "authorized_ci_violation":za, "authorized_safe_success":ya},
                              "origin":"matched_native_branch_archive",
                              "reference_only_not_target_Qpi":True})
            require(max(ss,default=0) == orig["safe_success"] and max(sa,default=0) == auth["safe_success"],
                    "Branch backup mismatch")
            fs.append(max(ss,default=0));fa.append(max(sa,default=0))
            counts["branch_tables"] += 1
        ci = aks.index(key(r["supervision"]["chosen_action"]))
        ri = aks.index(key(r["supervision"]["rejected_action"]))
        require(fs[ci] > fs[ri] and fa[0] == fa[1], "Stored preference/authorization direction mismatch")
        cases.append({"row_id":rid,"split_group":r["split_group"],"stratum":r["stratum"],
                      "actor_input":{"actor_context":r["actor_context"],"executable_actions":r["executable_actions"]},
                      "annotated_first_actions":actions,"full_upstream_menu":full,
                      "evaluator_only":{"safe_reference":fs,"authorized_reference":fa}})
    require(seen == set(pw), "Missing witness rows")
    summary = {"status":"PASS_ARCHIVE_CONTINUATION_TABLE_EXTRACTION",
               "rows":len(cases),"source_groups":len({r["split_group"] for r in cases}),
               "cells":len(cells),"joint_class_counts":dict(labels),**dict(counts),
               "input_sha256":{str(x):sha(x) for x in paths},
               "limitations":["Rows/cells are correlated within split groups.",
                              "All are source_train conditional reference evidence, not target model rollouts.",
                              "Unannotated upstream actions are UNKNOWN; never label them negative.",
                              "Authorization labels are evaluator-only until the native authorized observation is reconstructed.",
                              "No new model training, no independent holdout claim."]}
    args.out.mkdir(parents=True, exist_ok=False)
    for name, seq in [("CASES.jsonl",cases),("EVAL_ONLY_CELLS.jsonl",cells)]:
        with (args.out/name).open("w",encoding="utf-8",newline="\n") as f:
            for x in seq:
                f.write(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n")
    (args.out/"SUMMARY.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
