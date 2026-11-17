# Loss-only test of value-equivalent action consistency

Previous goal turn: **progress**. Native replay verified a real missing-sample repair, but it did not resolve action binding. Repaired SFT retained correct base Planner choices while downstream roles stopped unnecessarily; full conditional JS retained fixed-letter Planner behavior. No live job from that turn remains.

Question the current constraint before scaling: full action-distribution invariance constrains choices among equally valued actions. Equal native value does not imply identical action identity is required. Full JS imposes stronger invariance than value optimality requires; it is compatible with an optimum, and any training harm remains a hypothesis. Coarsening also reduces regularization magnitude, so a benefit alone would not isolate value semantics from regularization strength.

Hypothesis: retaining only value-class actor consistency may avoid this unnecessary constraint while retaining distinctions between safe completion, safe failure and unsafe actions. Alternative: the query consistency term itself is sufficient to cause the failure. Test both with a matched ablation, not a post-hoc weight search.

Four arms share the previously repaired sampler, initial adapter,80updates,seed17,lr0.0008,AdamW state reset,role schedule,views and supervised terms. Reuse `repaired_sft` and `repaired_js`; train exactly two new arms: query-only JS, and query JS plus actor value-class JS. All JS weights stay1. Actor consistency remains conditional on matching model-generated estimate maps across views. Merge actions only when their **full training value tuple** is identical within the same state; keep OTHER as a separate bin. Values are training supervision, never provided during free-generation evaluation.

Before training: mathematical regression tests preserve total mass, OTHER, safety/completion distinctions, gradients through both views, and equality to the original loss in full-action mode; real-model preflight checks all three roles. Freeze source/data hashes and newDEV generated from seed310000000 offsets. NewDEV has the same finite families and sizes as the prior audit but new worlds. It remains same-source, single-seed DEV, not independent task transfer.

Primary diagnostic is base success across both histories and both presentations. Report all families, query/format failures, safety and strict success jointly. No successful repair claim if complete-task gains depend on extra privacy violations or unexplained per-role failures. Passing this pilot does not establish multi-hop/source generalization, natural-norm validity or32-step collaboration. No result-triggered retries, new seeds, altered weights or length budgets.

Each training update has a9.5s watchdog, and existing8s generation bound is unchanged. No GPU or long RL. Mathematical and real-model tests are correctness checks; the learned outcome remains uncertain. Earlier archives and core code stay unchanged.
