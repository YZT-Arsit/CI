# KCI-Align: matched state-restart GRPO vs state-directed Qpi credit (frozen pilot)

Pre-registered before model updates on the RTX 5090.

## Research question

Does adding **frozen same-policy, same native state Qpi credit to the acting
Agent's first action** improve joint CI-safe task execution compared with the
same state-restart terminal GRPO policy, given the SAME TRAIN restart corpus,
the SAME rollout generator, and the SAME extra Qpi continuation workload?

This is NOT a test of an independent business task family or a paper-ready method.

## Fixed source and cohorts

Use the exact 16 frozen TRAIN restart specs in policy_qpi_01/selection_v2:
8 paired related/unrelated recipient-history roots, Planner8, Broker4, HR4.
The cohort was selected using past synthetic Qstar and old TRAIN trajectories:
therefore it is research DEV/TUNE, not blind TEST.
Each prefix is independently replayed via the native DynamicTask and hashed.
No hidden state labels, selected curriculum tags, final success or Qpi data
are passed to the language model prompt.

DEV remains 12 developed synthetic horizon condition groups x8 full episodes,
matched before and after (96). Root-case DEV is not independent business transfer.

## Matched actual model costs and starts

Qwen3-4B-Instruct-2507-ms same local base, fresh LoRA r8 alpha16,
NF4 BF16, CUDA RTX5090, seed9111926, G=8, 16 group updates,
AdamW 1e-5, clip0.2, KL frozen reference0.02,
temperature1, epsilon-mixture0.35.

Each arm computes exactly one frozen Qpi fork per one of the 16 TRAIN
state groups: 2 actions x K=4 continuation rollouts =8 extra per group.
Each arm also collects G=8 actual on-policy restart suffix trajectories per
group, and the same fixed DEV before+after 96 episodes. The extra frozen Qpi
probe happens before each optimizer step. Some realized action sequence lengths
may differ once weights diverge; record token usage, not just rollout totals.

Arm A, **Matched-compute terminal-only restart GRPO**:
compute all policy Qpi branches and record proofs, but DO NOT use them as
gradient. Perform ordinary terminal group-advantage action-token GRPO only.

Arm B, **State-directed PCCE**:
compute identical Qpi branches, require abs mean reward gap>=0.5 and at least
75% paired sign agreement, then apply bounded raw centered Qpi to the current
state's model-selected FIRST action token only, with additive weight0.5. The
remaining action tokens use unchanged terminal GRPO; zero-variance groups are
eligible only for supported local-credit actions.

The only intended algorithmic difference is applying accepted local
counterfactual credit to the model. Both arms have equal intended counts of
Qpi branch calls and optimizer opportunities, not guaranteed identical sampled
token/forward FLOPs.

## Before launching the 4B pilot

1. Windows CPU native state restart tests and full original regression pass.
2. Frozen 4B inference-only 16-group probe, no gradients, shows:
   8/16 supported states (Planner3/8, Broker3/4, HR2/4),
   64 model action local proof records, 128 extra policy continuations,
   two independent replay audits PASS.
3. New CLI matched-compute flag must pass source compile and no-training
   random/replay test.

## Planned outcome and stop

Execute both arms to 16 groups, no post-hoc hyperparameter changes.
For both record: exact native train and DEV outputs; strict trajectory and
policy-Qpi evidence re-audit; model hashes, actual optimizer count, mean
gradient, true cost/rollout counts, per-role local credit distribution.

Pilot positive candidate only if:
- replay and exact source hashes pass for both arms;
- both start from same 96 DEV behavior and same base LoRA initialization;
- local-credit treatment has >=3 supported distinct restart prefixes with at
  least 2 early roles and nonzero gradients;
- PCCE DEV safe-success improves by >=2 episodes vs terminal-only arm without
  increasing CI violations, OR violations improve by >=2 with no reduction in
  safe success;
- safe incomplete does not rise, and no invalid actions/entropy collapse.
Otherwise stop training and report a negative or inconclusive signal.

No 64-group expansion or claims of ACL novelty until these metrics pass and
an independent source/norm test family exists. DPR excluded.
