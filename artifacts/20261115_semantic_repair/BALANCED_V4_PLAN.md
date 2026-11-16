# Factorial repair before any held-out policy predictions

Inspection of specification(), not model test results, found that original
training depth and permission/obligation factors were correlated. V2/V3 are
preserved and must be reported as such. This is a pre-evaluation repair.

V4 crosses task kind (2), depth (1/2), irrelevant history count (0/2), mandatory
binding (2), authorization (2), delivered history (2), and matched/unrelated
history (2): 128 base conditions. Each has a paired history-key intervention,
with four independent replicates, giving 1024 worlds. For matched worlds the intervention switches which provider
connects to the same sensitive value. For unrelated worlds it is a null control.
All worlds receive new group IDs. Both counterfactual members stay in training.
This does NOT introduce new independent source instances or CI grammars.

Same frozen model implementations, same final 1200 updates, batch32, seeds
17/29/41. Graph: non-Planner-only, combined, shuffled, no-relations. Token:
non-Planner-only, combined, shuffled. Combined batches balance decision stages;
token pairs use shared symbol bijections and fact permutations. No test-based
checkpoint selection. Training chunks remain 5 steps with a 9.5s watchdog;
checkpoints every 100 steps plus final reduce filesystem overhead.

The historical `ci_only` identifier means non-Planner supervision in this code,
NOT pure immediate-CI labels: Source/Worker pairs also use terminal task utility.
Report this naming limitation and avoid claiming clean CI-only causal isolation.

Add a paired history swap to the held-out test worlds, retaining subject,
provider declarations, task, history length and all presentation factors. Both
members must be safe-successful and choose different semantic providers. Add
same-world optional/authorized/draft controls. Freeze all V4 checkpoints before
examining V2/V3/V4 predictions. Keep all versions, not just the best seed.

Pre-result lineage audit found balanced seed range120000..120511 overlaps the
previous synthetic source-holdout seed range, despite distinct suffixed world
IDs. V4 therefore uses source_holdout_v3, the SAME eight native sources/factors
with fresh synthetic seeds320000..320063. V2/V3 retain v2 source holdouts, which
do not overlap their seed10000 training. No policy predictions were consulted.
