# Fixed-state diagnosis before another repair

Previous turn: progress. Main dd026b3 preserves a successful causal fault localization and an unsuccessful matched consistency repair. Full goal stays active: relation knowledge must reliably affect language-model actions; actual long multi-role collaboration and natural-policy transfer remain unproved.

This run does not tune on the previous DEV results or launch training. It evaluates the two frozen, matched post-training checkpoints on the same exhaustive finite state set from the previous 112 presentation-world versions (56 semantic worlds,36 groups). Each world has5 reachable predecision states, including both provider commitments. This prevents a model that stops early from being evaluated on easier downstream states.

1. Build560 fixed observation versions from native state enumeration; verify labels separately using independent BFS. Model inference receives observations, not target labels. Keep the old DEV/source qualification: this is a reused diagnosis, not a new test.
2. Freely generate each action's completion estimate, then an action using only those self-estimates. Seal all outputs before scoring. Report per-role query correctness, complete estimate-map correctness, optimal-action correctness, and safety. Separate errors despite all-correct estimates from those with incorrect estimates.
3. Only after those records are sealed, run an explicitly labelled oracle-estimate actor diagnostic on the same observations. True bits are supplied solely to localize the actor ceiling; these outputs are never counted as learned policy success.
4. Inspect the four gradient components on fixed representative training batches if timing permits. Gradient agreement is a local diagnostic, not evidence of a causal privacy gradient or a demonstrated repair.

Inference calls retain the existing8second generation bound and9.5second validity threshold. No GPU, no online policy replacement, no model optimization, no fresh-source or three-seed claim. Do not retry failed outputs or choose a new loss before this attribution is available.
