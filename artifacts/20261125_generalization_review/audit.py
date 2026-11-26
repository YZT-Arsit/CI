"""Post-hoc CPU audit of two already opened KCI research panels."""
import gzip
import json
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path
from statistics import median

OUT = Path(__file__).resolve().parent
PILOT = OUT.parent / "20261125_constraint_regression_pilot"
PROOF = OUT.parent / "20261125_continuation_4b"


def load(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def route_rule(obs, path):
    """Only applied to the pilot's conditional, successfully delivered path library."""
    return (obs["known_authorization_status"]
            or (obs["role_stage"] != "public" and not obs["actual_delivery_receipt_metadata"])
            or any(action["kind"] == "SEND_BROKER_PRIVATE" for action in path))


def relation():
    train = load(PILOT / "training.json.gz")
    matrix = load(PILOT / "matrix.json.gz")
    obs = load(PILOT / "observations.json.gz")
    mid = load(PILOT / "C_mid_eval.json")["compat_margins"]
    end = load(PILOT / "C_eval.json")["compat_margins"]
    routes = defaultdict(Counter)
    distinct = defaultdict(set)
    strata = defaultdict(Counter)
    def score(split, prompt, path, truth):
        routes[split]["cells"] += 1
        routes[split]["correct"] += int(route_rule(json.loads(prompt), path) == truth)
        distinct[split].add((prompt, json.dumps(path, sort_keys=True)))

    for r in train:
        for c in r["cells"]:
            score("train_library", r["p"], c["path"], c["y"])
    for c, m, e in zip(matrix, mid, end, strict=True):
        if c["role"] != "development":
            continue
        score("development", obs[c["tag"]]["p"], c["path"], c["y"])
        if c["auth"]:
            continue
        s = strata[f"{c['family']}_{len(c['path'])}"]
        s["cells"] += 1
        s["conflicts"] += int(not c["y"])
        s["mid_conflicts_detected"] += int(not c["y"] and m < 0)
        s["end_conflicts_detected"] += int(not c["y"] and e < 0)
    for split, keys in distinct.items():
        routes[split]["unique_observation_path"] = len(keys)
    return {"route_only_rule": routes, "unauthorized_conflict_recall": strata}


def proofwriter():
    """Fix the second literal and the non-content fields of the first delivery."""
    train_ids = {r["row_id"] for r in load(PROOF / "SPLIT.json")["roots"]
                 if r["partition"] == "train"}
    panels = [g for g in load(PROOF / "graphs.gz") if g["row_id"] in train_ids]
    assert len(panels) == len(train_ids) == 128
    matched = []
    for g in panels:
        original = g["worlds"]["original"]["paths"]
        authorized = g["worlds"]["authorized"]["paths"]
        counts = Counter(pairs=0)
        for a, b in combinations(original, 2):
            left, right = json.loads(a), json.loads(b)
            if left["literal"] == right["literal"]:
                continue
            for field in ("message", "literal"):
                left.pop(field)
                right.pop(field)
            if left != right:
                continue
            first, second = original[a]["classes"], original[b]["classes"]
            abort, = (key for key in first if json.loads(key)["kind"] == "SAFE_ABORT")
            if first[abort] != 0 or second[abort] != 0:
                continue
            for tail in first.keys() & second.keys():
                if {first[tail], second[tail]} != {2, 3}:
                    continue
                if authorized[a]["classes"][tail] == authorized[b]["classes"][tail] == 2:
                    counts["pairs"] += 1
                    if 2 in first.values() and 2 in second.values():
                        counts["both_histories_allow_success"] += 1
                    if set(original[a]["accepted"]).isdisjoint(original[b]["accepted"]):
                        counts["disjoint_accepted_actions"] += 1
        matched.append({"root": g["root"], "stratum": g["stratum"], **counts})
    return {"train_roots": len(matched), "roots_with_pairs": sum(r["pairs"] > 0 for r in matched),
            "pairs_within_roots": sum(r["pairs"] for r in matched),
            "pairs_both_histories_allow_success": sum(r.get("both_histories_allow_success", 0) for r in matched),
            "pairs_disjoint_accepted_actions": sum(r.get("disjoint_accepted_actions", 0) for r in matched),
            "roots_by_contract": Counter(r["stratum"] for r in matched if r["pairs"]),
            "by_root": matched}


def margin_audit():
    cells = load(PILOT / "matrix.json.gz")
    public = "SEND_PUBLIC_PROJECTION"
    summary = {}
    for arm in ("initial", "A", "B", "C_mid", "C", "any_private_in_plan"):
        scores = ([int(any(a["kind"] == "SEND_BROKER_PRIVATE" for a in c["path"])) for c in cells]
                  if arm == "any_private_in_plan" else load(PILOT / f"{arm}_eval.json")["compat_margins"])
        same_first = defaultdict(list)
        cross = defaultdict(dict)
        for c, m in zip(cells, scores, strict=True):
            if (c["role"], c["family"], c["auth"]) != ("development", "TWO_HOP_JOIN", False):
                continue
            first = c["path"][0]["kind"]
            same_first[c["tag"], first].append((c["y"], m))
            if c["history"] != "initial" and first == public:
                key = (c["history"] == public, c["path"][1]["kind"] == public)
                assert key not in cross[c["group"]]
                cross[c["group"]][key] = (c["y"], m)
        rankings, interactions = [], []
        for group in same_first.values():
            safe = [m for y, m in group if y]
            unsafe = [m for y, m in group if not y]
            if safe and unsafe:
                rankings.append(min(safe) - max(unsafe))
        for v in cross.values():
            assert all(v[h,s][0] == (not (h and s)) for h,s in product((False, True), repeat=2))
            # Added public receipt should reduce the all-public route's margin more.
            interactions.append((v[False,True][1] - v[True,True][1])
                                - (v[False,False][1] - v[True,False][1]))
        summary[arm] = {name: {"n": len(seq), "positive": sum(x > 0 for x in seq),
                               "negative": sum(x < 0 for x in seq), "ties": seq.count(0),
                               "median": median(seq)}
                        for name, seq in (("suffix_order", rankings), ("receipt_suffix_interaction", interactions))}
    return summary


if __name__ == "__main__":
    result = {"relation": relation(), "proofwriter": proofwriter(), "margin_audit": margin_audit()}
    (OUT / "EVIDENCE.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**result["relation"], **{k: v for k, v in result["proofwriter"].items()
                                              if k != "by_root"}}, indent=2))
