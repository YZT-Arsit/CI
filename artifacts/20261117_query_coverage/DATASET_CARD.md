# Dataset scope and comparators

Interpret depth claims with [SCOPE_CORRECTION.md](SCOPE_CORRECTION.md): the original trace ancestor sampled depths1/2/3, even though recent query/action continuations used depth1. This round restores multi-hop coverage in the recent continuation; it is not first-ever depth3 exposure.

Existing TRAIN data supplies960factor-aligned history-pair groups at each depth1/2/3. The frozen96jointbatches select192statepresentations before the two renderings per batch. Both arms have64presentations per role and identical binary target counts:215positive/233negative before rendering duplication. Actual unique observation exposure differs: one_hop136from70basegroups; mixed_depth145from89basegroups. Thus the intervention changes depth and the resulting input diversity/token exposure; it does not separately identify graph depth, raw example novelty, entity count or compute effects.

The27correctionbatches reuse the previous108TRAINprompts, corresponding to7distinct semantic observations from5basegroups. These are not new independent supervision. No DEV query or oracle-action prompt is used as a training target.

DEV has128presentations,64semanticworlds,40underlyingbasegroups. A single world ID can be reused by a history intervention, so unique world-ID counts are not semantic-world counts. The fixed branch-state panel has640statepresentations and1536queries:256Planner,512Source,768Worker. Its target distribution is1096zero/440one; a constant-zero predictor scores1096/1536 and must not be mistaken for relational learning. The auditor retains positive counts, false positives/negatives, per-family/role metrics, and statewise all-query correctness alongside aggregate accuracy.

The fresh Worker action panel includes12zero-completion states from11basegroups, and12positive states from9basegroups. All positive states have unique Full optima; all zero states have tied Minimal/Stop optima. Each is shown in allsixoptionorders. It supplies true estimate bits and therefore measures conditional action binding only.

Source task computations remain the existing meeting/query sources with synthetic information-flow and task-obligation rules. No source-task holdout, independent natural business-norm validation or32stepcollaboration result is established by this dataset. Full counts are machine-readable in `DATASET_SCOPE.json`.
