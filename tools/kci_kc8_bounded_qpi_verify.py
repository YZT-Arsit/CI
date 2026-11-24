"""CPU replay of a completed bounded Qpi run; no model inference."""
from __future__ import annotations

import argparse
import json

from kci_kc8_bounded_qpi_run import read, write, sweep, setup
from kci_bounded_policy_value import normalized, evaluate
import kci_kc8_real4b_train_diagnostic as native
from kci_pisas_model_side_bridge import file_sha


def verify(study, source):
    p, after = read(study / "PROTOCOL.json"), read(study / "AFTER.json")
    start = read(study / "START.json")
    if start["protocol_sha256"] != file_sha(study / "PROTOCOL.json") or start["checkpoint"] != p["checkpoint_sha256"]:
        raise ValueError("run identity mismatch")
    cache = read(study / "KNOWN_DISTRIBUTIONS.json")
    if file_sha(study / "KNOWN_DISTRIBUTIONS.json") != p["known_distributions_sha256"]:
        raise ValueError("imported cache changed")
    queries = [json.loads(line) for line in (study / "GPU_DISTRIBUTIONS.jsonl").read_text(encoding="utf8").splitlines()]
    query_views, forwards = {}, 0
    sampled_priority_checks = sorted({0, len(queries) // 2, len(queries) - 1}) if queries else []
    for i, row in enumerate(queries):
        key = row["prompt_sha256"]
        if key in cache or key != row["actor_view"]["prompt_sha256"]:
            raise ValueError("duplicate or corrupt query")
        if i in sampled_priority_checks:
            _, frontier, views = sweep(source, p["selected"], cache)
            chosen = min(frontier, key=lambda k: (-frontier[k], k))
            if chosen != key or row["actor_view"] != views[key] or abs(frontier[key] - row["frontier_mass_before_scoring"]) > 1e-12:
                raise ValueError("priority query replay mismatch")
        normalized(row["distribution"], row["actor_view"]["available_actions"])
        cache[key] = row["distribution"]
        query_views[key] = row["actor_view"]
        n = len(row["actor_view"]["available_actions"])
        if row["cost_before"]["candidate_model_forwards"] != forwards or row["cost_after"]["candidate_model_forwards"] != forwards + n:
            raise ValueError("forward accounting mismatch")
        forwards += n
    cells, frontier, _ = sweep(source, p["selected"], cache)
    if cells != after["cells"] or len(frontier) != after["frontier_size"]:
        raise ValueError("final value replay mismatch")
    cases, Env, view, digest = setup(source)
    observed = {}
    def record_view(obs):
        result = view(obs)
        observed[result["prompt_sha256"]] = result
        return result
    for cell in p["selected"]:
        env = Env(cases[cell["root_id"]], cell["history"])
        for action in ("SEND_A", "SEND_B"):
            state, verdict = env.transition(native.clone_prefix(env), action)
            if verdict["verdict"] != "VALID":
                raise ValueError("illegal first action")
            evaluate(env, state, cache, record_view, digest, structural_bounds=True)
    if any(observed.get(k) != v for k, v in query_views.items()):
        raise ValueError("queried model observation not reproduced from native state")
    if forwards > p["max_new_candidate_forwards"] or forwards != after["incremental_model_costs"]["candidate_model_forwards"]:
        raise ValueError("frozen query budget exceeded")
    result = {"status": "PASS_NATIVE_QPI_BOUNDS_REPLAY", "new_queries_checked": len(queries),
              "candidate_forwards_checked": forwards, "query_priority_checks": sampled_priority_checks,
              "all_model_views_reproduced_from_native_states": True,
              "all_final_intervals_exactly_recomputed": True,
              "source_known_distributions": p["existing_policy_distributions"],
              "new_model_training": False, "white_box_finite_environment_only": True,
              "result_sha256": file_sha(study / "AFTER.json"),
              "query_trace_sha256": file_sha(study / "GPU_DISTRIBUTIONS.jsonl")}
    write(study / "INDEPENDENT_REPLAY.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    verify(args.study, args.source)
