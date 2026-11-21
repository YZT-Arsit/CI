"""Regression/causal tests for the separate 2026-11-21 KCI CPU research gate."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"code"), str(ROOT/"tools")]

from compare_kci_chain_methods_cpu_action_interaction import GROUP
from chain_shared_actor_cpu_action_interaction import (
    SharedHornActor, score_actions, valid_decision_candidates,
)
from kci_exact_policy_continuation_cpu import (
    environment_public, FrozenPolicyValue, enumerate_decisions,
    policy_preference_points,
)

torch.set_num_threads(2)


def _frozen_sft():
    checkpoint = (ROOT/"artifacts"/"20261120_v31"/
                  "CHAIN_REPAIRED_POSTHOC_NEW_VARIANTS_01"/
                  "SMALL_CPU_CHECKPOINTS"/"action_sft_seed7.pt")
    net = SharedHornActor()
    net.load_state_dict(torch.load(checkpoint, map_location="cpu",
                                   weights_only=True)["model_state_dict"])
    return net.eval()


def test_actual_native_state_tree_and_legality_gate():
    e = environment_public(GROUP["train"][0])
    points, audit = enumerate_decisions(e)
    assert audit["points_A_SEND"] == 1
    assert audit["points_preflight"] == 2
    assert audit["points_release"] >= 2
    assert "privacy_resource_contract" in points[0]["obs"]
    for p in points:
        assert all(a not in ("ABORT", "ABORT_CHAIN") for a in p["actions"])
        for a in p["actions"]:
            assert p["verdicts"][a] in ("VALID", "VIOLATION")


def test_dpr_does_not_pollute_legal_CCE():
    e = environment_public(GROUP["train"][0])
    points, _ = enumerate_decisions(e)
    buckets, audit = policy_preference_points(e, _frozen_sft(), points,
                                              include_repair=True)
    assert audit["dpr_verified_pairs"] >= 1
    source_violations = [
        p for p in points if p["phase"] == "release"
        and "VIOLATION" in p["verdicts"].values()]
    assert len(buckets["repair"]) == len(source_violations)
    for obs, good, bad, gap, key in buckets["repair"]:
        assert good == "VERIFY_PRIVATE_R" and bad == "SEND_RAW_R"
        assert gap > 0
    assert all("VIOLATION" not in p["verdicts"].values()
               for p in points if p["hash"] in
               {k for group in ("A_SEND","preflight","release")
                for _,_,_,_,k in buckets[group]})


def test_exact_Qpi_law_and_no_weight_mutation():
    e = environment_public(GROUP["train"][0])
    net = _frozen_sft()
    weights = [t.detach().clone() for t in net.parameters()]
    st, audit = e.transition(e.initial(), "A_READ_VERIFIED_PACKAGES")
    assert audit["verdict"] == "VALID"
    eval = FrozenPolicyValue(e, net)
    obs = e.observation(st)
    actions = valid_decision_candidates(obs, e.options(st))
    with torch.no_grad():
        probs = torch.softmax(score_actions(net, obs, actions), dim=0).tolist()
    q = [eval.q(st, act) for act in actions]
    lhs = eval.value(st)
    for j in range(3):
        assert abs(lhs[j] - sum(float(p)*v[j] for p,v in zip(probs,q))) < 1e-7
    assert all(torch.equal(x, y) for x, y in zip(weights, net.parameters()))
    assert eval.queries > 0 and eval.transitions > 0


def test_Qstar_vs_Qpi_reversal_in_existing_dev_seen_world():
    world = next(w for w in GROUP["exploratory_test"]
                 if w["world"] == "DG19_M01__tradeoff__swap0")
    e = environment_public(world)
    st, _ = e.transition(e.initial(), "A_READ_VERIFIED_PACKAGES")
    eval = FrozenPolicyValue(e, _frozen_sft())
    vals = {a:eval.q(st, "A_SEND_"+a)[0]
            for a in ("PACKAGE_A","PACKAGE_B")}
    qstar = world["upstream_A_values_given_exact_B_continuation"]
    assert qstar["PACKAGE_A"] > qstar["PACKAGE_B"]
    assert vals["PACKAGE_A"] < vals["PACKAGE_B"]
    assert e.observation(st)["privacy_resource_contract"]["step_cost"] > 0
