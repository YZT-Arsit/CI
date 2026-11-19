# KCI-Align Agent RL v0.1 — implementation audit (local Windows)

## Scope

This is a **terminal-reward GRPO-style baseline** on the original multi-step DynamicTask. It is *not* an experimental result for 4B/8B and does not yet implement CCE or DPR. The two source-adapted task instances have synthetic CI rules; no claim about independent real business norms.

## Completed implementation

- Native per-agent authorized observations; current executable options only.
- Hugging Face policy returning a single sampled legal action-letter token and the corresponding logprob.
- Real native state transition, successful delivery receipts, environment-only privacy and task result scoring.
- G multiple policy-sampled full trajectories from identical initial state, terminal rewards and group advantages.
- LoRA/4-bit QLoRA option, adapter-disabled frozen base KL on legal-action distribution, GRPO clipped ratio update on **only chosen agent option tokens**.
- Zero-reward-variance groups skipped; cost-only groups distinguished and skipped by default to prevent a small cost difference being normalized into a large training gradient.
- Immutable JSONL traces, source hashes for Meeting+Query, optional local model weight fingerprints, unique run directories and independent exact action replay verifier.
- Training command requires explicit --confirm-gpu-training. No 4B CPU runs allowed by default.

## Actual verification on DESKTOP-M8QBBO1

- Full existing project regression run from D:\research\longhistory\code: **211 tests, OK**.
- Native action contract tests: **12**, PyTorch GRPO gradient tests: **5**, replay/forgery tests: **7**. All pass.
- New CLI preflight: PASS, 32 TRAIN and 32 developed DEV controlled conditions, all relevant source SHA-256 verified, CUDA not available on this machine.
- Random-policy CLI: **32 complete trajectories**, 8 groups, 6 safe success, 7 CI violations, 19 safe failures; independent output replay and manifest verification: PASS.
- Real local 135M HF inference-only model: 2 completed safe trajectories, correct one-token action selection, no training; this is a model integration smoke only.
- Real local 135M CPU train command: 1 sampled group with identical outcomes, optimizer_steps=0 (correctly skipped). The old smoke output's boolean claimed an update because of a reporting defect; the CLI was subsequently fixed to use actual optimizer_steps. Do not use that old summary boolean as evidence.
- Separate REAL causal LM + LoRA gradient *software* test using explicitly artificial rewards: one optimizer update, selected higher-reward action logprob increased from about -0.227 to -0.198; finite gradient, frozen-reference adapter context functioning. **These were NOT native real trajectory rewards and do not prove RL effectiveness.**
- No paid API calls and no GPU training while implementing.

## What remains before any research claim

1. Move to an actual CUDA server with model weights and pinned Python environment. Run HF 4B inference-only model rollouts; measure success, violations, group outcome variance and cost-only signals. Inspect action-role distribution.
2. Then a short 4B LoRA GRPO run; confirm real optimizer_steps positive and compare fixed developed DEV before/after, including over-withholding, violations, tasks and costs.
3. Independent business/rule families and human-reviewed CI norms are still missing. TRAIN/DEV currently differ only in horizons, not source domains.
4. CCE (same-prefix counterfactual long-horizon action value) and DPR (targeting downstream failed action) remain **future method increments** to compare against terminal-only GRPO. There is no evidence yet of novelty or performance advantage.
5. Existing v1 static micro-classification failures cannot be extrapolated to 4B; equally, a successful 135M software smoke is not evidence of 4B learning.

## Location and commands

Code: code/kci/agent_rl and tools/run_kci_agent_rl.py
Auditor: tools/audit_kci_agent_rl.py
Tests: code/tests/test_agent_rl*.py
Run instructions and dependency hints: artifacts/20261119_composition_signals/agent_rl_v1/README.md and requirements_gpu.txt.

Native source and model flags fail closed on missing input/invalid action; no silent coercion of UNKNOWN to safe or successful.
