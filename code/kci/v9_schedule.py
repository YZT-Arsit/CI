"""Build source-balanced schedules only from fit roots and native outcomes."""
from hashlib import sha256
import json
import random


def rng(seed, *parts):
    return random.Random(sha256(json.dumps([seed, *parts]).encode()).digest())


def native_examples(g, initial):
    common, success, failure, ci, ordinary, repair = [], [], [], [], [], []
    reached_failures = []
    for first, branches in g["paths"].items():
        if not g["valid_first"][first]:
            continue
        for b in branches:
            t = b["policy_input"]
            outcomes = b["outcomes"]
            good = [a for a, v in outcomes.items() if v["stsr"]]
            bad = [a for a, v in outcomes.items() if not v["stsr"]]
            unsafe = [a for a, v in outcomes.items() if v["direct"] or v["inferred"]]
            lawful = [a for a in outcomes if a not in unsafe]
            for a in outcomes:
                path = [[g["first_input"], first], [t, a]]
                (success if a in good else failure).append(path)
            common.extend([[t, a]] for a in good)
            positives = good or [a for a in lawful if not outcomes[a]["abort"]] or lawful
            ci.extend(([[t, a]], [[t, v]]) for a in positives for v in unsafe)
            ordinary.extend(([[t, a]], [[t, v]]) for a in good for v in bad)
            if first == initial["first"]:
                observed = next(x for x in initial["events"][first] if x["condition"] == b["condition"])
                assert observed["input"] == t and observed["stsr"] == outcomes[observed["action"]]["stsr"]
                if not observed["stsr"]:
                    reached_failures.append(dict(event=b["condition"], input=t, action=observed["action"],
                                                 direct=observed["direct"], inferred=observed["inferred"],
                                                 abort=observed["abort"], alternatives=good))
                    repair.extend(([[t, a]], [[t, observed["action"]]]) for a in good)
    assert common and success
    return dict(common=common, success=success, failure=failure, ci=ci, ordinary=ordinary,
                repair=repair, reached_failures=reached_failures)


def schedule(graphs, initials, seed):
    arms = {k: [] for k in ("proof_ci", "proof_sft", "proof_terminal", "proof_cce",
                            "neg_ordinary", "neg_dpr", "neg_repair_ce")}
    evidence = {}
    for domain in ("proof", "neg"):
        baseline = {s["id"]: s for s in initials[domain]["root_results"] if s["split"] == "fit"}
        roots = []
        for g in graphs:
            if g["domain"] != domain or g["split"] != "fit":
                continue
            state = baseline[g["id"]]
            pool = native_examples(g, state)
            if domain == "neg":
                evidence[g["id"]] = pool["reached_failures"]
                if not (pool["repair"] and pool["ordinary"]):
                    continue
            roots.append((g, pool))
        for epoch in range(2):
            ordering = list(roots)
            rng(seed, domain, epoch).shuffle(ordering)
            for g, pool in ordering:
                gid = g["id"]
                def pick(k):
                    return rng(seed, epoch, gid, k).choice(pool[k])
                common = pick("common")
                def add(arm, pos, neg=None):
                    arms[arm].append(dict(root=g["source_root"], id=gid, group=g["split_group"],
                                          common=common, pos=pos, neg=neg or []))
                if domain == "proof":
                    first = g["first_input"]
                    a, b = g["chosen"], g["rejected"]
                    assert g["valid_first"][a] and g["valid_first"][b] and g["reference_stsr"][a] > g["reference_stsr"][b]
                    add("proof_ci", *pick("ci"))
                    add("proof_sft", pick("success"))
                    add("proof_terminal", pick("success"), pick("failure"))
                    add("proof_cce", [[first, a]], [[first, b]])
                else:
                    add("neg_ordinary", *pick("ordinary"))
                    p, n = pick("repair")
                    add("neg_dpr", p, n)
                    add("neg_repair_ce", p)
    return dict(seed=seed, arms=arms, downstream_failures=evidence)

