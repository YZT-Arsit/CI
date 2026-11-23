"""Reproducible finite counterexamples; these are not LLM efficacy experiments."""
from itertools import product
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    # A falling DPO loss can still leave the preferred action below 0.5.
    logodds = lambda p: math.log(p/(1-p))
    prior, learned, beta = 0.001, 0.01, 2.0
    dpo_before = math.log(2)
    dpo_after = math.log1p(math.exp(-beta*(logodds(learned)-logodds(prior))))
    assert dpo_after < dpo_before and learned < 0.5

    # Enumerate a bounded grid independently of any training implementation.
    # If sup |Qhat-Q| <= eps and Qhat(a) >= max Qhat-eta, regret <= 2eps+eta.
    checks, violations = 0, 0
    epsilon = 0.25
    for q in product([0., .25, .5, .75, 1.], repeat=2):
        for error in product([-.25, 0., .25], repeat=2):
            estimated = [v+e for v,e in zip(q,error)]
            for eta in (0., .25):
                for action in (0,1):
                    if estimated[action] >= max(estimated)-eta:
                        checks += 1
                        violations += max(q)-q[action] > 2*epsilon+eta+1e-12
    assert checks > 0 and violations == 0
    # A fully informed optimal SFT policy already achieves the optimum.
    q = [0., 1.]
    sft_value = q[1]
    cce_value = max(q)
    assert sft_value == cce_value
    result = dict(
        scope="analytic_counterexamples_and_finite_grid_NOT_model_results",
        no_unconditional_superiority=dict(optimal_sft=sft_value, exact_cce=cce_value),
        dpo_low_loss_wrong_greedy=dict(prior_good_probability=prior,
            learned_good_probability=learned, beta=beta, initial_loss=dpo_before,
            final_loss=dpo_after, greedy_preferred_action=False),
        conditional_value_bound=dict(epsilon=epsilon, checked=checks, violations=violations,
            bound="2*epsilon+eta; same admissible action set and uniform value error required"),
        hoeffding_paired_return_radius={str(n):math.sqrt(8*math.log(40)/n) for n in (4,16,128)},
        radius_assumptions="two-sided 95%, iid seed pairs, each return in [-1,1]; not lexicographic safety proof",
        illustrative_all_steps_correct={str(t):.95**t for t in (8,32,64)},
        step_assumptions="independent 95% correct meaningful actions; illustration, not measured success")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
