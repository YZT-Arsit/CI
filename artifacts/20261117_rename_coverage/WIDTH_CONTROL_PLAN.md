# Supplementary numerical-path calibration

Registered as a separate follow-up after finding a batch-width difference during the main run. Main training, outputs, gates and decisions stay unchanged; this is not a retry of selected failed cases. No supplementary output replaces a primary score.

All168 third-palette TRAIN prompts have the same own token length as the original prompts. However, the separate third panel groups candidates differently: only144/168 also match the original batch's padded width. The fixed-versus-renamed comparison on an identical third panel remains matched, but within-model original-versus-third comparison needs this numerical control.

After the main run is sealed, freeze both final checkpoints and run the full336-query layout per model. Keep the original two rows of each four-row group and replace the known-fresh rows with their third-palette counterparts. This preserves batch size4, ordering of the paired worlds, own lengths and padded width relative to the original main query panel. Run all168 third prompts, not only the24 affected by different padding. Exactly672 new responses, zero updates, no intervention in autonomous trajectories.

Report how many original outputs reproduce byte-for-byte (168 per model), how many third outputs match the separate panel, every changed output, parsing validity, all336 batch widths per model and48 changing third-palette pairs. A numerical-control pass requires all original outputs reproduced, all336 widths matched and no invalid responses. Do not assume it passes. Whether any third predictions change is empirical; preserve both results. No new main success threshold or GPU authorization follows.

Decoding is unchanged greedy generation, batch4, max_new_tokens16, max_time8 and validity below9.5seconds. The registration records these settings and exact checkpoint/source/prompt hashes. Report parsed-answer agreement as well as byte agreement and invalid/timing-cutoff counts separately for both primary and calibration panels. This matches the original panel's initial padded width and batch layout; generation lengths and stopping behavior can still differ. It does not isolate padding from every other batch-layout change in the earlier separate panel.

The user's standing authorization permits autonomous diagnostic inference and modestly longer CPU experiments, without approval. That broader authorization covers this separate follow-up. The frozen main experiment's no-retry rule remains intact: no primary call is replaced, no failed subset is selected, and no main gate is changed.
