# Pre-registered KCI CCE upper-bound study — 2026-11-19

Research direction unchanged: executable multi-agent action alignment under evolving recipient knowledge. The baseline is the already independently verified Qwen3-4B-Instruct-2507 terminal-only GRPO run.

## Frozen baseline and CPU feasibility

- Baseline 64 groups: 512 genuine on-policy trajectories; 62 optimizer steps, 2 cost-only skips; dev 96 episodes: safe success 41 -> 47, CI violations 19 -> 13, safe failures 36 -> 36.
- Native exact continuation search on 512 real 4B trajectories found 352 distinct branching states: 235 future-reachability gaps and 94 cost/value gaps among currently legal actions. Some reachability gaps are trivial ABORT/terminal.
- Controlled related/unrelated history pairs at the same source+permission+action prefix: 384 comparable, 40 changed action values, 24 changed between alternatives still legal, 24 changed immediate CI applicability.
- The CCE teacher is native exact Q* (safe success reachable and shortest terminal cost), NOT a frozen-policy continuation value. This is a mechanism upper-bound diagnostic, **not** the final full method.

## Frozen algorithm and scope

For a real sampled episode e at role/action t:
- Require role/phase among Planner/choose, Broker/choose or collect, HR/hr.
- At the exact same execution state, score each alternative action that is currently executable and immediately VALID. Exclude ABORT/WITHHOLD; do not score late single-option, final notification or simulator-evident task abandonment.
- Each alternative is evaluated by native exact legal continuation: Q*=1-0.01*shortest terminal step for safe task success, else 0. Immediate violation is never eligible.
- If at least two alternatives are legal and Q spread >= 0.005, compute bounded local contrast 2*(Qchosen - mean Qvalid)/spread, clipped to [-1,1]. Zero credits are excluded.
- Where real episode terminal reward has outcome variance, mix 0.5 terminal group advantage and 0.5 local action credit at eligible acting-agent decisions; at other steps retain original terminal credit.
- When a group has zero reward variance or cost-only variance, update only actions with nonzero local CCE credit, never amplify a mere 0.01 cost difference via group normalization.
- The exact teacher is called ONLY after policy sampling, as a training-time verifier. All policy prompts, actions and terminal rewards are unchanged; record independent source hashes and per-state proof. No hidden labels go to the LM.

## Identical model + rollout conditions

Model: local Qwen3-4B-Instruct-2507, new LoRA rank 8 alpha 16, 4bit nf4 bf16, CUDA RTX5090. Common seed 9111926; train worlds fixed 32 cells repeated in same order; group size G=8; same random seeds and action sampling temperature 1.0, epsilon-mixture exploration 0.35; same AdamW 1e-5, clip ratio 0.2, frozen-base reference KL 0.02, 64 attempted groups. Fixed DEV 12 scenario groups x 8 independent action samples = 96, evaluations before/after and at groups 16,32,48.

Difference from archived terminal-only baseline: training-time CCE Q* teacher and local action advantage ONLY. This uses additional trusted CPU oracle calls; budgets match on-policy model trajectory samples and attempted updates, not total CPU instructions. Record actual optimizer steps and teacher calls, and note if train trajectories diverge after updating the policy.

## Execution stages and gate

Stage A (software): 8-group CCE smoke with same model/seed, ensure from-baseline DEV counts align and at least one independently replayable nonzero local credit, no CUDA training crashes, and complete end-to-end artifact. This is NOT an efficacy comparison and will not select hyperparameters.
Stage B: repeat the same fixed CCE algorithm from the original base with 64 groups if Stage A passes *software and signal checks*. Do not tune after seeing DEV.
Outcome gate for 64 groups: safe-success >= 50/96 and CI violations <=13/96 OR safe-success>=47/96 and violations<=10/96, while safe failures <=36/96 and no collapsed action entropy. Other outcomes are failure or mixed; no paper advantage claim. Also report the full 0/16/32/48/64 curve rather than choosing best milestone.

TRAIN and DEV share source Meeting/Query instances and synthetic norms, differ in horizons only. The policy selects among provided legal options (constrained action interface). No human-reviewed external CI normative source, business generalization or autonomous unconstrained tool-call claim. DPR is explicitly excluded from this study. No paid external API.

## Evidence

- CPU source tests for CCE oracle, native replay and gradient, and full original tests.
- GPU run manifest and exact standard native rollout auditor.
- Separate independent CCE audit recomputes every counterfactual proof from the frozen native executor, checking no oracle contamination of policy prompts.
- Save original and CCE outputs under distinct immutable directories; preserve original baseline and all negative results.
