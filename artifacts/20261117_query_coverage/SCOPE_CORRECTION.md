# Training-lineage correction to the frozen depth wording

During the running coverage experiment, an ancestry check found that the original `20261116_relation_solution/language_models/trace_17` training recipe already sampled from TRAIN depths1/2/3. Its frozen source is `train_language.py::pairs()`, which loads all TRAIN states without a depth filter. Later query/action continuation samplers restrict their data to depth1.

Consequently, the frozen PLAN.md wording that depth3 "becomes seen-depth" must be read as applying only to the recent continuation, not the complete model lineage. **Neither current arm can establish first-ever depth3 exposure or genuine depth3 extrapolation.** The valid comparison is restoring depth2/3 coverage in the current query/action supervision versus continuing with depth1, from identical ancestry. The earlier trace format differs from the current individual query format; this distinction does not justify calling depth3 wholly unseen.

The original plan, data, code, primary thresholds and running models are left unchanged. This correction restricts interpretation; it does not select results or relax the gate. Depth4 and the internal/cycle families are not included in the current continuation. Any broader training-lineage holdout claim must follow the explicit ancestry evidence, not just this round's sampler.

The original base model's pretraining corpus is not audited here. All holdout statements concern the declared experimental supervision, not an assertion that a pretrained model has never encountered an analogous abstract relation.

[LINEAGE_AUDIT.json](LINEAGE_AUDIT.json) verifies seven linked checkpoint stages and the original trace training source hashes. Its deterministic seed17 sampler reconstruction gives212/190/198pairs at depths1/2/3 across600updates. The old logs did not save sampled row IDs, so these counts are identified as reconstructed rather than per-step observations. The eligibility of all three depths already makes a depth3-unseen claim unjustified.
