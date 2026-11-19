# KCI State-Directed Rollout: matched 4B curriculum and credit experiment

Date: 2026-11-19. KCI-Align direction unchanged: receiver-history-conditioned long-horizon CI action learning.

## Evidence for state restart

Archived original root-start on-policy 4B 8-group PCCE: only 1 / 18 early distinct prefixes passed gap >=0.5 and 75% direction-consistency. No DEV gain.
Frozen Qwen3-4B 16-group STATE RESTART probe, selected from 512 source-adapted TRAIN trajectories with four history-dependent and four hold control pairs:
- 16 native states, related/unrelated pairs, deterministic state hashes, already defined before this run.
- 128 real LM episodes, 472 model-sampled action decisions; 16/16 groups have outcome contrast.
- 128 extra model policy continuation rollouts.
- 8/16 restart states have supported mean reward gaps; 64 accepted action-specific local credit records.
- role coverage: Planner 3/8, Broker 3/4, HR 2/4.
- original native action/receipt/terminal reward independent replay PASS and SHA256 manifest PASS.

Caution: this is deliberately selected TRAIN-derived curriculum, not unbiased prevalence, not model improvement and not independent business generalization.

## Frozen matched experiment

Arms:
  A: state-directed root native policy GRPO with terminal rewards only.
  B: same state-directed GRPO + policy-calibrated CCE on model-generated actions.

Same source version: pinned Meeting + Query source-adapted DynamicTask and synthetic CI norm, original seed 9111926.
Same model: local Qwen3-4B-Instruct-2507-ms, fresh LoRA rank8 alpha16, NF4 BF16, Qwen base. Both start from the same initialization via torch/manual seed.
Same training restart schedule: all 16 states in exact fixed selection_v2/FORKS.jsonl order; each state is a real native TRAIN prefix with verified state hash, related/unrelated paired context. Each model samples 8 complete suffix episodes starting at the same state; no action labels are forced.
Same target optimizer: AdamW 1e-5, group GRPO clipping .2, base reference KL .02, gradient norm clip 1., temperature 1., epsilon explore .35, fixed model action contract.
Both get 16 attempted groups, identical model-action sample count per group, fixed developed ROOT DEV conditions (12 scenario groups x 8 rollouts each = 96 full trajectories) before and after.
Same action-token normalization and episode reward. Only sampled post-restart LM actions receive gradient. Original prefix is never teacher-forced into the gradient.
Model meta restart selection Qstar/result flags never added to prompt.

PCCE only:
- At each exact restarted prefix, run four independent complete continuations under the same pre-update LM per immediately VALID nonabort action, paired random seeds.
- Require |mean Qpi difference| >= .5 and >= .75 pairwise sign consensus (> .1 return diff). Reject insufficient signals.
- At chosen first actual model action, local centered RAW Qpi return (bounded [-1,1]) weighted .5, ADDITIVELY on top of terminal GRPO. Other tokens get unchanged terminal credit.
- No native Qstar teacher in PCCE.
- Extra frozen policy continuations make run B more expensive in LM inference, NOT equal FLOPs or total rollout cost. Report counts and actual trainable action decisions.

## Decision and reporting

First verify both run manifests and independent native restart, real DEV and Qpi credit proof replay.
Initial DEV must be identical / matched before training; if not, invalidate pair.
Main exploratory comparisons: safe completed tasks, CI violations, safe incomplete/overwithholding, correct history-conditioned action changes, role action distribution, optimizer steps, entropy/KL, gradients. All 96 DEV episodes retained, not cherry picked.
Pilot learning gate: PCCE improves safe completion >=2 with no more violations than state-terminal, OR preserves safe completion with >=2 fewer violations; no increase in safe incomplete beyond state-terminal. Do not retroactively alter.
State-directed terminal vs archived root terminal 64-group comparison is a separate unbalanced-budget observation, NEVER a controlled superiority claim.
No ACL novelty claim from a single 16-group seed or two source task instances.
If both fail, STOP additional on-policy scaling and diagnose state-task mismatch and optimization credit; no posthoc relaxation, no data templating.

No external API. DPR not included.
