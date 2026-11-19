# KCI-Align: policy-conditioned CCE repair and 4B pilot (2026-11-19)

**Decision: code and reference mechanism repaired, but the preregistered 8-group signal gate FAILED. Do not scale to 64 groups.**

This is a mechanism feasibility and state-exposure diagnosis, not a proven ACL method improvement.

## Research background

Original Qwen3-4B terminal-only GRPO (64 groups): safe task completion 41/96 -> 47/96, CI violations 19/96 -> 13/96.
Original exact-native-Qstar CCE (64 groups): safe task completion 41/96 -> 41/96, violations 19/96 -> 19/96, only 3/96 DEV action paths changed. Mixing 50% normalized ideal Qstar advantage diluted terminal reward and made tiny ideal cost differences unit-sized signals. Exact Qstar is not the same as imperfect model policy expected return.

## Frozen Qpi versus Qstar experiment

From the original 512 TRAIN on-policy Qwen3-4B sampled trajectories, selected 8 fixed history pairs (16 identical-source action prefixes) with 2 immediately legal actions each. For every action, completed K=8 frozen 4B model continuations through the native environment: 256 recorded branches, independently replayed with full manifest hashes. No model parameters updated.

Comparison of action ranking: 10/16 Qstar and Qpi agreed, 6/16 had Qstar preference but Qpi difference below the preregistered margin, 0/16 decisive reversals. Related versus unrelated historical knowledge changed Qstar action values and Qpi return differences in 4/8 history pairs. No evidence that Qstar routinely reverses action rankings on this small selected panel; there is a clear discrepancy in reward magnitudes and model policy capabilities.

## Repaired candidate: Policy-Calibrated CCE (PCCE)

- Freeze the actual on-policy LM weights for the entire rollout+counterfactual+single update group.
- At real early Planner/Broker/HR decision states, evaluate two currently legal, non-ABORT alternatives.
- Under that SAME LM, sample K=4 full native multi-agent continuations per forced alternative with shared per-rollout random seed.
- Require absolute mean terminal-reward gap >= 0.5 and >=75% directional agreement among paired differences. Tiny cost-only differences or uncertain pairs yield NO local credit.
- Compute local credit from raw centered achievable Qpi returns, bounded to [-1,1], never dividing by the tiny Qstar reward gap.
- Preserve terminal GRPO: acting-agent token advantage becomes terminal advantage + 0.5 * local Qpi credit, where confident local credit exists. Other tokens retain the exact original terminal GRPO advantage.
- No exact Qstar teacher or future outcome injected into model inference, and Qpi cache never reused across weight updates.
- Every accepted local credit retains underlying branch actions, reward, source state and final receipt hashes for separate independent replay.
- DPR is explicitly excluded.

Full original repository regression tests: **241/241 PASS**. Separate CPU gradient tests prove additive no-local equals baseline terminal GRPO, and supported local signal can update under zero terminal group variance. Independent model-free branch proof smoke: PASS.

## Real RTX 5090 / Qwen3-4B 8-group pilot

Same original base, NF4 BF16, LoRA r8 alpha16, temperature 1.0 and epsilon 0.35, seed 9111926, G=8, LR 1e-5, reference KL0.02, 8 attempted groups, matched developed DEV (96 full Agent episodes).

| Fixed DEV | Before | After PCCE 8 |
|---|---:|---:|
| Safe task completion /96 | 41 | 41 |
| CI violations /96 | 19 | 19 |
| Safe incomplete /96 | 36 | 36 |
| Mean terminal reward | 0.202604 | 0.202604 |

Real optimizer updates 8/8; independent native execution replay PASS; independent Qpi proof replay PASS.
Extra frozen-policy continuation rollouts: **144**.
Early unique prefixes queried across eight training groups: **18**.
Prefixes with a sufficiently robust and large Qpi return gap: **1**.
Action-specific local credit proofs: **8** (Planner: 8; positive 1, negative 7).
Preregistered pilot gate required at least TWO distinct supported early prefixes as well as no safety/utility regression. Thus pilot NO-GO; no 64-group PCCE run was started. Additional LM inference cost makes this unequal-compute to terminal GRPO despite matched base and updates.

## Mechanism diagnosis and next task

The policy continuation teacher was not commonly misranking actions relative to Qstar in the 16 selected states. The actionable issue is low exposure to reliable counterfactual decision prefixes during ordinary on-policy group training: only 1 of 18 observed early prefixes cleared the signal gate.

The next candidate experiment must address state visitation/controlled prefix replay rather than posthoc lowering the confidence gate or raising local loss weight:

1. Quantify historical Flip/Hold and relevant early branch coverage among archived 512 real on-policy episodes; separate true safety/completion gaps from synthetic fixed-route cost differences.
2. Develop a native state-restart curriculum for actual historically sensitive prefixes with complete authorized observations and no hidden future truth; use frozen-model Qpi to verify branch outcome contrast before any training.
3. Only then preregister a matched-resource comparison of state-directed policy credit versus terminal-only GRPO, reporting extra LM rollout tokens and verifying external norms and independent task families.

This study does NOT establish KCI-Align algorithmic novelty, strong external generalization or an improved 4B deployed policy.

## Reproducible code and verified backups

Source: code/kci/agent_rl/policy_continuation.py, policy_cce.py, grpo.py; tools/run_kci_agent_rl.py, run_kci_policy_qpi.py, audit_kci_policy_qpi.py, audit_kci_policy_cce.py.

Frozen input and source: policy_qpi_01/selection_v2 and pinned native Meeting/Query anchors.

Frozen inference backup: policy_qpi_01/qpi_inference_backup.tar.gz (SHA-256 7bae406f44dd093781500eb7e5ba671470daa60d3d149c5adaf85becfba035fe).
Frozen 8-group 4B PCCE backup: policy_qpi_01/policy_cce_8_backup.tar.gz (SHA-256 c97fa752c712dac362c69535658c9b8380939262f70b67ff0aed243a92ba20e0).

The pretrained Qwen3-4B base checkpoint is NOT included in these smaller backups. No further GPU training has been scheduled.
