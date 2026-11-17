# Restore and inspect this repair

Restore older dependencies through `../20261117_fallback_order/REPRODUCE.md`, then run from the repository root:

```powershell
python -B tools/restore_contrast_repair_artifacts.py
python -B artifacts/20261117_query_coverage/audit.py
python -B artifacts/20261117_query_coverage/audit_probe.py
python -B artifacts/20261117_query_coverage/audit_pair_queries.py
python -B artifacts/20261117_query_coverage/audit_lineage.py
python -B artifacts/20261117_query_coverage/audit_surface_controls.py
python -B artifacts/20261117_contrast_repair/test_schedule.py
python -B artifacts/20261117_contrast_repair/audit.py
python -B artifacts/20261117_contrast_repair/audit_pair_queries.py
python -B artifacts/20261117_contrast_repair/audit_violations.py
python -B artifacts/20261117_contrast_repair/train_probe.py audit
```

The new restore helper covers both this repair and its query-coverage parent. It verifies compressed and original hashes and refuses to overwrite different evidence. Model source assets remain pinned by the earlier relation-solution manifest; adapters and records are retained here. Auditors run without model inference.

The original CPU workers were:

```powershell
wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_contrast_repair/run.py'
wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_contrast_repair/train_probe.py run'
```

Workers refuse existing outputs. Preserve completed evidence and declare a separate output location for any new replication. The second command requires the first worker's complete seal and performs no parameter updates. Its 1,152 responses are scored separately from the main experiment.

Both trained arms restart from the original corrected checkpoint. There are 123 updates per arm: 96 joint query/action batches and 27 unchanged fallback batches. Of the joint batches, 54 force value-reversing TRAIN pairs and 42 retain the parent anchors. The sampler test preserves the original failure and its repaired passing run. `DATASET_SCOPE.json` discloses changed condition frequencies and label balance; this is a sampling-recipe comparison, not an isolated causal effect of contrast density.

The initial and earlier reference arms are exact copies of parent evidence, with explicit receipts and zero new model calls. Their checkpoints, raw outputs and all source files are compared byte for byte. Counts of scored evidence and newly generated responses are separate. This repair adaptively reuses the parent's DEV set and is not independent confirmation.

The supplementary TRAIN probe tests every changing query in the 54 forced batches using both original training renderings and same-world fresh entity names. It checks original/renamed prompt token counts, retains failures, and cannot establish independent-world generalization. The main gate is unchanged.

`audit_violations.py` is a post-result descriptive analysis, with no additional model calls. It verifies the sealed main outputs, finds the first native violation in each violating episode and records the model's own estimate for its selected action. Positive estimates before violations are observed false positives. No new counterfactual rollout or intervention is performed.

Read the parent's `SCOPE_CORRECTION.md`: the original trace ancestor already sampled depths 1, 2 and 3. Three-hop success would not establish first-ever depth extrapolation. Natural business norms, cross-task transfer and 32-step agent collaboration remain separate unproved requirements.
