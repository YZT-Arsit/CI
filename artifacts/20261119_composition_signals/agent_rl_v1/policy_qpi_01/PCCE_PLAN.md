# KCI-Align PCCE — frozen-policy counterfactual credit repair pilot (2026-11-19)

**Objective unchanged:** dynamic recipient knowledge + multi-agent long-horizon CI action learning.
This is a single preregistered mechanism test, not a new benchmark or established ACL contribution.

## Why the old exact-Qstar CCE failed

Qwen3-4B terminal-only GRPO (64 groups): 41/96 -> 47/96 safe success, 19/96 -> 13/96 violations.
Native exact-Qstar CCE (same attempted groups, 64 vs 62 actual updates): 41/96 -> 41/96, violations 19/96 -> 19/96. It changed only 3/96 DEV action trajectories vs terminal GRPO's 20/96, with mean preclip norm 0.658 vs 1.597. Its 975 training credits contained 881 small one/few-step synthetic ideal-cost differences. Former objective mixed terminal reward 50/50, diminishing the original action signal.

Frozen Qwen3-4B policy inference on 8 TRAIN-related/unrelated history pairs (16 forks, 2 valid actions, 8 continuations each = 256 true native branch rollouts) was independently audited. Qstar/Qpi pairwise ranking: 10 concordant, 6 Qpi too close to rank, zero demonstrably reversed. Four of eight history pairs showed distinct Qpi outcomes. Therefore teacher ranking reversal has not been established; direct reward magnitude and policy feasibility differences are demonstrated, and the approach must preserve terminal gradient.

## One fixed candidate: Policy-Calibrated CCE (PCCE)

At a real on-policy sampled, acting-agent prefix, consider only two simultaneously immediately valid non-ABORT/non-WITHHOLD alternatives in Planner/choose, Broker/choose/collect or HR/hr.
Freeze the just-sampled LM policy *before* the optimizer step.
Perform K=4 complete native rollouts under that policy following each forced action, using coupled random seeds for alternatives.
Calculate mean return Q_pi(s,a), and compare K paired rollout return differences. Require abs(mean gap) >= 0.5 and at least 75% consistent reward-difference signs (differences below 0.1 considered ties).
Apply only the chosen action's raw centered return:
  local = clip(Qpi(s,chosen) - mean_a Qpi(s,a), -1, +1).
Do NOT divide a tiny Qstar cost gap by its own standard deviation.
Train with A_total = A_terminal + 0.5*local for locally supported actual action tokens; all other actions use unchanged original terminal GRPO advantage. On zero-variance terminal groups only supported local decisions can update. Never put Qpi/Qstar or outcome labels into the model prompt.
Cache Qpi only within the same frozen on-policy group and cap 4 distinct early prefixes per group; no cache across weight updates. Record complete branch actions/rewards/receipts and source hashes for independent audit.
No Qstar teacher in this mechanism. No frozen-LM continuation changes or additional fabricated task data.

## Identical pilot model and budget

Start from Qwen3-4B-Instruct-2507-ms BASE, same LoRA rank8 alpha16, quantization NF4 BF16, seed 9111926, group size8, first 8 identical TRAIN groups, same dev horizon groups 12x8=96, temperature1 epsilon.35, AdamW 1e-5, ratio clipping .2, reference KL .02, same executable source and reward.

Stage A: 8 attempted groups of PCCE training, plus DEV before/after. Upper bound on Qpi branch calls: 8 groups * 4 distinct prefixes * 2 actions * 4 continuations = 256 branch rollouts, in addition to ordinary GRPO trajectories. This is NOT equal-compute to terminal GRPO, despite matched base/sample/update settings.
Require full native rollout audit, separate independent Qpi credit proof audit, positive model optimizer steps, at least 2 supported distinct early prefixes across 8 groups, nonzero local credit counts, and no degradation of DEV safe completion below 41/96 or violations above 19/96. If any fails, do not scale.

Stage B only if A passes: one full 64-group PCCE experiment, evaluated at groups 0/16/32/48/64 with the same tested hyperparameters, no post hoc tuning. Compare to archived terminal-only 64-group and exact-Qstar CCE with ALL differences and any extra LM inference budget acknowledged.
Performance gate for expanded CCE: safe success >= 50/96 and violations <=13/96, OR safe success>=47/96 and violations<=10/96, with safe incomplete <=36/96, no significant entropy collapse, and meaningful early action-history Flip/Hold changes. Otherwise record NO-GO.

## Limitations

These are synthetic two-source Meeting/Query workflows and horizon-defined DEV only; no independent task family and no externally verified normative authority. The policy receives finite valid action options, not unconstrained tool call generation. Short pilot is engineering/learning-signal evidence, not high-power statistical significance. DPR is excluded. No external API calls.
