# Reproduce and audit this continuation

This run depends on the unchanged `20261116_relation_solution` and `20261117_action_binding` archives and their pinned CPU Python/model environment. No base model weights or dependencies are republished here. Restore archived JSONL first, from the repository root:

```powershell
python -B tools/restore_relation_artifacts.py
python -B tools/restore_action_binding_artifacts.py
python -B tools/restore_binding_factors_artifacts.py
```

Existing evidence is never overwritten by the restore helpers. `compression_manifest.json` validates original and compressed hashes. Training and inference scripts refuse reuse of their original PLAN paths. To rerun an experiment, use an isolated full repository copy with a new output directory; do not delete or overwrite the historical run to make a script start.

Read-only recomputation of results (the audits rewrite only their derived summaries):

```powershell
python -X utf8 -B artifacts/20261117_binding_factors/audit.py
python -X utf8 -B artifacts/20261117_binding_factors/audit_clamp.py
python -X utf8 -B artifacts/20261117_binding_factors/audit_value_channel.py
wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_binding_factors/audit_repair.py'
python -X utf8 -B artifacts/20261117_binding_factors/audit_behavior.py
```

Scripts and records:

- `build.py`, `render.py`, `PLAN.json`: factorized presentation and exact old-prompt identity checks.
- `probe.py`, `queries.jsonl`, `actions.jsonl`: free model predictions, sealed before separate label scoring.
- `clamp.py`, `clamped_actions.jsonl`: frozen model's own baseline estimates, not teacher labels.
- `two_view_loss.py`, `test_consistency_math.py`, `smoke_repair.py`, `smoke_varied.py`: semantic JS, identity and actual model gradient checks. These smoke checks do not update parameters.
- `repair.py`, `repair/PLAN.json`: matched two-view control and conditional consistency treatment, one seed, 80 updates per arm, identical input batches and optimizer schedule. New checkpoints and all failures retained.
- `check_repair_data.py`, `REPAIR_DATA_CHECK.json`: independent finite labels and no old DEV world ID overlap. This does not establish source holdout.
- `audit_repair.py`, `repair/AUDIT.json`, `repair/VIEW_AUDIT.json`: native state replay, free query/action validity and joint success across views and counterfactual histories.
- `VALUE_CHANNEL_AUDIT.json`: existing TRAIN label sufficiency check, not model efficacy.
- `../20261117_native_roles/REPORT.md`: isolated native role session and oracle replay checks, with zero model decisions.

The inference helpers retain the existing bounded generation and update watchdogs. Data creation, model loading and aggregate repeated evaluation are setup or orchestration operations; they are not individual 5–10 second tests.
