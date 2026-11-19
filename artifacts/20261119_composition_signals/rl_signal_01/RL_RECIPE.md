# KCI-Align: move from classification rehearsal to true agent RL

This is an executable-system specification, not a claim that RL model training has completed.

## Technical objective

Learn πθ(a_t | observation of agent i) from actual delivered-recipient history and a fixed task norm. The target orders terminal outcomes: (1) no direct/cumulative CI violations, (2) verified task completion, (3) cost/steps among safe successes.

The current ReceiptTask has just three main phases, no deadline, and zero continuation cost. It is useful for static composition diagnosis, not the main long-horizon RL objective.

DynamicTask has actual delivered receipts, multiple decision phases, horizon and safe-completion exact continuations. It remains source-adapted and synthetic with only two source business instances. Its use must not be presented as independent business task generalization.

## Environment RL-signal preflight (real simulator, no model)

Under rl_signal_01/pilot, seed 9111926:
- 32 conditions (2 tasks x 2 history relation settings x 2 permissions x 4 horizons);
- G=8 uniform legal action trajectories per condition, 256 complete rollouts;
- 29/32 groups with nonzero reward variance;
- 44/256 safe completed trajectories; 19/256 trajectories with CI violation;
- 32/32 had two currently legitimate initial actions with different finite future values;
- five tests pass, no model training, no API, no GPU.
A random policy with useful group variance does not prove the intended 4B or 8B model samples useful outcomes. The next required evidence is on-policy.

## Grounded methods to reuse

- CI-RL (NeurIPS 2025): verified privacy/task RL reward and synthetic task/norm diversity. Around 700 examples, with external PrivacyLens transfer; not proof that our two source instances are diverse.
- PrivAct (ICML 2026): multi-agent tree sampling, final leakage+helpfulness scoring, value iteration and per-agent DPO preference extraction. Compare equal rollout budgets.
- DAPO (NeurIPS 2025): dynamic sampling filters zero-variance groups, stable policy update/clipping, per-token loss; avoid wasting updates.
- RAGEN (2025): real multi-turn rollouts, informative diverse initial states, monitor Echo Trap reward/entropy collapse.
- Agent Lightning (2025): explicitly separate agent execution and RL training with transition-level credit assignment.
References:
https://github.com/EricGLan/CI-RL
https://github.com/chengyh23/PrivAct
https://dapo-sia.github.io/
https://arxiv.org/abs/2504.20073
https://arxiv.org/abs/2508.03680

## V1 true RL system to implement

1. Agent transition adapter: observe the acting agent with legal receipts; model samples action in valid action set; executable transition changes task and receipt state; independent verifier checks CI and task result. Each record binds model checkpoint, policy input, generated action tokens/log-probability, role, actual delivered messages, terminal result, oracle checksum. Only model-generated action tokens are optimized, never tool outputs or hidden judge labels.
2. On-policy sampling: G=4–8 COMPLETE trajectories from each identical world state under the current old policy. Reward vectors: compliance, verified task success, cost. For an initial scalar surrogate, safe success gets +1 minus a small bounded cost penalty, safe incomplete gets 0, any CI violation gets -1. Privacy-violating success is never a positive reward. Report all components.
3. Plain terminal GRPO baseline: action-token log-probs only, grouped advantage centered by rewards, clip/KL. DAPO-inspired skip zero-variance groups but include their frequency in evidence. Keep naive SFT and equal-budget PrivAct-style tree DPO for comparison.
4. Proposed KCI increment (only if baseline is feasible): evaluate two currently legal actions from the SAME real world prefix using a frozen reference continuation; compute counterfactual safe-completion advantage. Credit the acting role at the decision point. A downstream failure should generate a DPR repair pair at the actual failed decision, not automatically punish the upstream legally sent message. Compare to terminal GRPO and tree-DPO with matched rollout budgets.
5. Data: at least one additional independent source-backed multi-agent task family with its own tool contract and explicit normative authority. Keep source/rule grouped TEST blind. UNKNOWN cannot silently count as safe/unsafe. Vary actual knowledge, recipient, permission, purpose, relevance and task necessity, not just names.
6. Metrics: safe task success, explicit/derived violations, task completion, overwithholding, cost and steps, flip/hold pair outcomes, invalid actions, actionable reward variance, KL/entropy, and retained language/tool competence. Include external planning baseline.

## Target-model pilot rather than 135M perfection

- Pilot 4B actual policy, 24–48 paired roots, G=4–8 on-policy trajectories each, no gradients yet. First confirm meaningful safe success and mixed outcome rewards under the REAL policy. If all success or all failure, change difficulty/curriculum, not the RL loss.
- Then short 4B LoRA GRPO with matched terminal-only control and KCI counterfactual credit. Sample small but enough to show repeatable reward slopes and no privacy/utility regression on fixed DEV. Record actual model rollouts, steps and compute.
- If controlled slope exists, diversify tasks/norms and run two further seeds and larger 7B/8B. Use measured 4B throughput to extrapolate compute and trajectories; small-data results cannot guarantee large-run success.
- Do not use 135M inability to answer symbolic 4-hop relation questions as an absolute NO-GO for 4B RL. Use it only for debugging action schema, reward, gradient and environment contracts.

## Stop criteria

Hidden truth in policy prompt, incorrect state transition labels, nonexistent reward variance, majority class shortcut, new leakage or overwithholding, success based only on static judgments, missing task-source holdout. No expensive long run until model-based signal and matching baseline are known.

Direction unchanged: dynamic receiver knowledge x multi-agent long horizon x CI policy parameter training. CCE/DPR are potential algorithmic contributions, not currently proven.
