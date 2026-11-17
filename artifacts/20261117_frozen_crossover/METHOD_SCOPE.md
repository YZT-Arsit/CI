# What the crossover can and cannot identify

For a fixed checkpoint and observation, the actor is a deterministic free-generation mapping from observation plus textual estimate map to an action. Exchanging only that estimate map tests sensitivity to the explicit text channel. The supplying model is not named in the prompt. All maps come from the same legal observation; oracle maps are kept in separately sealed cells.

The learned off-diagonal combinations are diagnostics with two checkpoints, not the required final shared-weight policy. Only diagonal cells retain one checkpoint for both queries and action. Shared parameters across execution roles remain the research requirement. No rule-based selector,actionmask or hidden native label is used in the learned cells.

[Geiger et al.,NeurIPS2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html) align neural representations with high-level causal variables and test interventions on those representations. This experiment does not implement that internal-representation method. Text-channel sensitivity alone cannot establish an internal causal abstraction, a learned graph algorithm, or novelty. The Exa connector failed during lookup; the original proceedings source was consulted through web search.

Correct oracle inputs are not guaranteed to improve a learned actor; the model may ignore or misuse them. These cells measure behavior under correct inputs, not an unconditional upper bound. Likewise, true-bit TRAIN performance is exact seen-input fit only for teacher_sft; own_sft originally saw some different self-predicted inputs.

All states are matched, but intervention inputs may be off the actor's training distribution. Count changed maps,changed actions,repaired errors and new errors. Identical-prompt comparisons across source columns and repeated preflight inputs check inference stability. Prior-online comparisons also expose any batch-path change. Disagreements must remain visible; they limit attribution rather than justify silently selecting a favorable path.

Native rollouts are reconstructed after output sealing by following the frozen state-policy table in the actual transition model. This is legitimate finite-state replay of those recorded choices, but it is not a new online interaction or another independently sampled episode. Correct answers at individual states cannot substitute for joint safety/utility of the reconstructed trajectories.

The same DEV and one training seed are reused for diagnosis. Strong results would motivate a new preregistered shared-model repair with fresh validation; they would not establish source transfer,natural-norm validity,32-step collaboration or large-GPU readiness.
