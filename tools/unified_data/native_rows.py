"""Rebind teacher targets to the observations of the agent who actually acts."""
import ast
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
sys.path[:0] = [str(ART / "20261125_multisource_policy"), str(ART / "20261125_continuation_4b")]
import prepare as proof
import prepare_multi as legacy
import bind_native as native
import kci
kci.__path__.append(str(legacy.BASE.parent/"kci_pisas_dynamic_20261123/code/kci"))
import dynamic_env as dynamic

CORE = legacy.CORE
cn = legacy.cn
data = CORE / "data"


def lines(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def observation(ctx, menu):
    return dict(ctx=ctx, actions=sorted(set(map(cn, menu))))


def example(row, ob, positives, tier, detail, *, negatives=(), route="sft", grade="closed_action_set"):
    return dict(source=row["source_group_key"], root=row["group"], domain=row["source_dataset"],
                row=row["row_id"], tier=tier, evidence=grade, detail=detail, route=route,
                ob=ob, positive=sorted(set(positives)), negative=sorted(set(negatives)))


def negotiation(rows):
    engine = native.Negotiation.__new__(native.Negotiation)
    engine.ng = native.ng
    native.visible = {}
    for game in lines(data / "NEGOTIATION_CORE_V2/EVAL_ONLY_ALL_NATIVE_BRANCHES.jsonl.gz"):
        r = rows.get(game["id"])
        if r is None:
            continue
        assert "DEAL_OR_NO_DEAL:deal-"+game["source_root"] == r["source_group_key"]
        engine.neg = {game["id"]: game}
        for first in r["executable_actions"]:
            for branch in engine.negotiation(r, first):
                outcomes = branch["outcomes"]
                ok = [a for a,v in outcomes.items() if v["stsr"]]
                if not ok:
                    ok = [a for a,v in outcomes.items() if v["abort"] and not (v["direct"] or v["inferred"])]
                assert ok
                ob = native.visible.pop(branch["policy_input"])
                yield example(r, ob, ok, "native_mediator", dict(event=branch["condition"], probability="1/7",
                              first=first), negatives=set(outcomes)-set(ok))


def proofwriter(rows):
    cases = list(lines(proof.TABLE/"CASES.jsonl"))
    specs, sources = proof.bind_native(cases)
    cells = defaultdict(dict)
    for c in lines(proof.TABLE/"EVAL_ONLY_CELLS.jsonl"):
        cells[c["row_id"]][cn(c["first_action"]), cn(c["next_action"])] = c["evaluator_target"]
    for case in cases:
        r = rows[case["row_id"]]
        for auth in (False, True):
            env = proof.env_for(case, specs, sources, auth)
            ledger, _ = env.prefix()
            aa, bb = env.messages("source_A"), env.messages("source_B")
            ctx = proof.context(env, ledger, "A")
            assert ctx["authorized_downstream_candidate_catalog"] == bb
            assert auth or ctx == r["actor_context"]
            reachable = []
            for a in aa:
                next_ledger, _, verdict = env.deliver(ledger, a)
                if verdict["verdict"] != "VALID":
                    continue
                cb = proof.context(env, next_ledger, "B")
                assert cb["recipient_actual_deliveries"][-1] == a and "authorized_downstream_candidate_catalog" not in cb
                stop = cn(dict(kind="SAFE_ABORT", actor="source_B"))
                menu = observation(cb, bb+[json.loads(stop)])
                ok = []
                for b in bb:
                    cell = cells[r["row_id"]].get((cn(a),cn(b)))
                    if cell is None:
                        fin, _, v = env.deliver(next_ledger, b)
                        allowed = v["verdict"]=="VALID" and env.final(fin) is not None
                    else:
                        p = "authorized_" if auth else ""
                        allowed = cell[p+"task_supported"] and not cell[p+"ci_violation"]
                    if allowed:
                        ok.append(cn(b))
                if ok:
                    reachable.append(cn(a))
                else:
                    ok = [stop]
                yield example(r, menu, ok, "native_proof_B", dict(authorized=auth,first=a),
                              negatives=set(menu["actions"])-set(ok))
            assert reachable
            menu = observation(ctx, aa)
            yield example(r, menu, reachable, "native_proof_A", dict(authorized=auth),
                          negatives=set(menu["actions"])-set(reachable))


def aggregate(rows):
    for s in lines(data/"AGGREGATE_K_CORE_V1/PREREG_ALL_CANDIDATES.jsonl"):
        r = rows.get(s["spec_id"])
        if not r:
            continue
        assert legacy.sha(native.ag.SRC/s["source_file"]) == s["source_sha256"]
        for auth in (False, True):
            native.visible = {}
            g = native.aggregate(r,s,auth)
            branches = [bs[0] for bs in g["paths"].values()]
            assert len({b["policy_input"] for b in branches}) == 1
            ok = set.intersection(*[{a for a,v in b["outcomes"].items() if v["stsr"]} for b in branches])
            if ok:
                ob = native.visible[branches[0]["policy_input"]]
                yield example(r,ob,ok,"native_aggregate_B",dict(authorized=auth),
                              negatives=set(ob["actions"])-ok, grade="safe_success_across_indistinguishable_histories")


def teacher_trajectories(rows):
    for record in lines(data/"EXISTING_602_CORE_AUDIT_V1/EVAL_ONLY_REPLAYED_CORE_AND_ABLATION.jsonl.gz"):
        r = rows.get(record["row_id"])
        if r is None:
            continue
        assert record["source_root"] == r["source_group_key"]
        if "full_traces" in record:
            for auth, branches in record["full_traces"].items():
                for branch in branches:
                    tr = branch["complete_native_continuation"]
                    if not tr["terminal_outcome"]["safe_task_success"]:
                        continue
                    for st in tr["true_agent_trace"]:
                        ctx = dict(st["observation"])
                        menu = ctx.pop("legal_actions")
                        for k in ("env_version","root_id"):
                            ctx.pop(k,None)
                        a = st["model_action"]
                        assert ctx["acting_agent"] == st["actor"] and a in menu
                        yield example(r,observation(ctx,menu),[cn(a)],"pisas_reference_witness",
                                      dict(authorized=auth=="True", state=st["state_before_hash"]),
                                      grade="executed_safe_success_witness")
        else:
            spec,source = dynamic.instance("relational",r["row_id"])
            for key, branches in record["exact_reference_chance_traces"].items():
                auth, first = key.split("|")
                for br in branches:
                    result = br["result"]
                    if not result["success"] or result["risk"]:
                        continue
                    env = dynamic.aud.ExactRel(spec,source,ast.literal_eval(br["bits"]),auth=="True")
                    mid = result["trace"][0]["action"].get("source_stage") == "bridge"
                    st = dynamic.aud.rel_start(env,mid)
                    for step in result["trace"]:
                        assert env.state_id(st)==step["before"]
                        ctx = dict(env.observation(st))
                        menu = env.legal_actions(st)
                        a = step["action"]
                        assert a in menu and ctx["acting_agent"]==step["actor"]
                        ctx.pop("legal_actions",None)
                        ctx.pop("root_id",None)
                        yield example(r,observation(ctx,menu),[cn(a)],"relational_reference_witness",
                                      dict(authorized=auth=="True",state=step["before"],chance=br["bits"]),
                                      grade="executed_safe_success_witness")
                        st, _ = env.transition(st,a)
                        assert env.state_id(st)==step["after"]
                    assert st.success and not st.any_violation


def historical_repairs(rows):
    obs = legacy.rgz(legacy.OLD/"observations.gz")
    gs = legacy.rgz(legacy.OLD/"graphs.gz")
    graphs = {g["root"]:g for g in gs if g["partition"]=="train"}
    selection = legacy.rd(ART/"20261125_process_policy/S3_SELECTION.json")
    for root, records in selection["roots"].items():
        g = graphs[root]
        r = rows[g["row_id"]]
        for repair in records:
            if repair["kind"]!="repair":
                continue
            o = obs[repair["tag"]]
            ctx = json.loads(o["prompt"].split("\n",1)[1].split("\nExecutable actions:\n")[0])
            a = repair["chosen"]
            if a not in o["actions"] or a in repair["accepted"]:
                continue
            for p in repair["accepted"]:
                yield example(r,dict(ctx=ctx,actions=sorted(o["actions"])),[p],"student_failure_repair",
                              dict(snapshot=selection["snapshot"],visit=repair["visit"],norm=repair["norm"]),
                              negatives=[a],route="dpo",grade="historical_student_failure")

