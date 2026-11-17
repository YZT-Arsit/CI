# Fixed-distribution fallback weight experiment

Hypothesis: the 18/320 zero-completion Worker actor presentations are underweighted in joint supervision, leaving an option-position shortcut. Test a fixed multiplier 4 versus 1; this is ordinary loss reweighting, not a new method or group-DRO implementation. One seed, no tuning or checkpoint selection.

All arms start at the sealed `continue_lower_lr` checkpoint. `uniform` and `weighted` each use the identical original 80 batches, two views, learning rate 0.0002, fresh AdamW, clip 1, query loss and format loss. Only the semantic actor loss weights change. Weights use fixed full-schedule normalization `1 + (m-1)*18/320`, not batch normalization. No additional features, runtime oracle, filtering, or rule-based action correction. `initial` is frozen.

Before training verify m=1 is numerically equivalent to the original loss on a real mixed Worker batch. Retain the witnessed unit-test failure and correction. New DEV seed 360000000/360010000 preserves all 24 original condition presentations (112 episodes/model). No training IDs overlap.

Diagnostics: rerun all 216 previous TRAIN oracle-input permutations (regression only). Separately generate 24 fresh Worker states, 12 with and 12 without a safe completion, chosen by SHA256 of canonical state from fresh DEV, and enumerate all six options (144 oracle-input prompts/model). Selection uses labels but never model outputs; these are diagnostics, not autonomous policy evidence. Preserve optimal ties. No training on fresh states.

Primary criterion for a repair in this pilot: weighted improves fresh zero-case all-six optimal rate versus both references, has zero unsafe actions in the fresh zero oracle diagnostic, and does not reduce strict free-policy successes or CI versus either reference on the same DEV. Report deeper-family complete-pair metrics separately; any failure keeps the full objective unproved. TRAIN fit alone cannot pass. No GPU scale-up unless broader relation/agent/norm gates are established.

Every update is limited to 12 seconds, generation retains the existing 9.5-second validity rule. Preserve failures and never restart over existing evidence. No natural-norm or 32-step collaboration claim follows from this experiment.

Primary literature consulted: Sagawa et al., https://arxiv.org/abs/1911.08731 . Its group-generalization warning motivates reporting held-out group results; it does not establish effectiveness of this fixed weighting.
