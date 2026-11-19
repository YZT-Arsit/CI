# KCI-Align 4B CCE 64-group completed evaluation

**Decision: NO-GO for exact-native-Qstar local CCE mixing** at the evaluated budget. This is not a rejection of policy-conditioned CCE generally.

## Matched experiment

Both arms: Qwen3-4B-Instruct-2507, fresh rank-8 alpha-16 LoRA, 4-bit NF4 BF16, seed 9111926, group size 8, 64 attempted groups, same 32 TRAIN scenario cells and 12 DEV cells (96 sampled episodes), LR 1e-5, clipped-ratio 0.2, frozen reference KL 0.02, temperature 1.0, exploratory epsilon 0.35, same initial checkpoint and action sampling seeds. All 96 DEV action paths and 420 matched log probabilities were **identical before training**.

The additional CCE objective uses native exact safe-completion Q* values for currently legal non-ABORT/non-WITHHOLD alternatives in Planner/choose, Broker/choose and collect, HR/hr. Q* is used at training only, never given to the model at inference. This is an optimistic ideal-policy reachability teacher, NOT the current model's expected continuation value. The CCE trial gets 64 actual updates vs 62 baseline updates, so realized optimizer steps and CPU-oracle workloads differ.

## Final developed DEV result

| Metric | Initial | Terminal GRPO | CCE Q* + GRPO |
|---|---:|---:|---:|
| Safe task success /96 | 41 | 47 | 41 |
| CI violations /96 | 19 | 13 | 19 |
| Safe incomplete /96 | 36 | 36 | 36 |
| Mean terminal reward | 0.202604 | 0.321771 | 0.202604 |
| Actual optimizer updates | - | 62 | 64 |
| Different DEV action paths vs initial /96 | - | 20 | 3 |
| Different DEV outcome labels vs initial /96 | - | 10 | 0 |

CCE checkpoints at 16/32/48/64 attempted groups never changed the above joint DEV outcome counts.

Both independent verifiers PASS: native state/action replay for 992 trajectories across all checkpoints and independent native Q* recomputation for all 975 local credit proofs. These are source-adapted synthetic Meeting/Query tasks and a horizon-split development panel, NOT externally validated business/norm generalization.

## Offline diagnostics (read-only)

- Mean preclip gradient norm: terminal GRPO 1.59668 vs CCE 0.65787.
- Mean reference KL: 0.006125 vs 0.001399.
- Mean absolute same-action logprob change along comparable DEV prefixes: approximately 0.0529 for terminal GRPO vs 0.00810 for CCE.
- All three CCE-changed DEV action trajectories preserved their original safety/success outcomes. Two changed the FETCH_COUNTS / FETCH_CITY_BUNDLE order within a safe Vault workflow. The third varied an unsafe OPEN_INDEX path but remained unsafe.
- CCE credits: 975 total, 732 positive / 243 negative; Broker 515, Planner 256, HR 204.
- Q* spread of proof alternatives: 496 under 0.011, 385 under 0.051, 94 above 0.5. The latter are large value gaps, not independently verified natural CI-norm transfer.

Supported conclusion: this local CCE augmentation generated valid training proofs and nonzero updates, but achieved markedly weaker actual policy change than plain terminal GRPO and no DEV improvement.

Open causal explanations (NOT established): 50/50 mixing may dilute terminal policy gradient; Q* teacher may reward an action the imperfect current model cannot reliably complete; small fixed-route costs may swamp CI-specific dynamics. Do not automatically tune higher CCE weight or run more groups before isolating these.

## Next gate

Retain terminal-only GRPO as functioning baseline. Before any more GPU training, measure frozen 4B-policy continuation returns on a matched set of same-prefix action forks and compare to native Q* rankings. Separately report legal/illegal, reachability, cost-only, and recipient-history-conditioned Flip/Hold conditions. If current-policy action return ranking cannot be distinguished from the oracle Q* or no meaningful opportunity exists, stop this CCE variant. DPR remains out of this experiment.

## Artifacts

Windows CCE backup:
D:\research\longhistory\artifacts\20261119_composition_signals\agent_rl_v1\cce_01\cce_64_completed_backup.tar.gz

Verified SHA-256 (Windows = GPU):
9d4692aa5a1b74bfa541cafc648d245da539325812c5bb7985b5f3e6c0190362

Windows terminal GRPO backup:
D:\research\longhistory\artifacts\20261119_composition_signals\agent_rl_v1\grpo_64_completed_backup.tar.gz

Independent archive diagnostic:
cce_01\analyze_offline.py and cce_01\OFFLINE_DIAGNOSTICS.json

The 4B pretrained base weights are not included in this CCE backup; adapter, traces, metrics and reproducibility metadata are included. No new GPU training was initiated during this analysis.
