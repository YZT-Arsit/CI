# Reproduce matched background repair

Restore all dependencies following ../20261117_witness_binding/REPRODUCE.md, then from the repository root:

```powershell
python -B tools/restore_relation_factors_artifacts.py
python -B artifacts/20261117_relation_factors/audit_factors.py
python -B artifacts/20261117_join_repair/test_join.py
python -B artifacts/20261117_join_repair/data_audit.py
python -B artifacts/20261117_join_repair/test_data_audit_guard.py
python -B artifacts/20261117_join_repair/audit_join.py
python -B artifacts/20261117_join_repair/diagnostics.py
python -B artifacts/20261117_join_repair/training_fit/audit_fit.py
```

These checks do not call a model. Native audit reconstructs data, verifies frozen inputs/checkpoints/output seals, checks runtime tokens, optimizer steps, losses, parameter identity and exact reused bytes, then recomputes scores.

Original worker command in the pinned CPU FP32/four-thread WSL environment:

```powershell
wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_join_repair/run_join.py'
```

The worker refuses an existing study directory. Do not remove evidence to rerun; replications require separate declared output paths. Final checkpoints only, no confirmation-based selection. Each update has a15second hard watchdog. Greedy generation uses16tokens,max_time8and a9.5second scoring cutoff; the cutoff is not a hard process timeout. Initial actor responses are bytewise reused and all actor retention inputs contain oracle judgments. Zero autonomous episodes.

The separately frozen selected-TRAIN diagnostic runs only after the main study is sealed:

```powershell
wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_join_repair/training_fit/run_fit.py'
```

It adds336responses from six final models, without optimization or initial-model comparison.56queries per model include24switch prompts for each background and8shared controls.32are exact selected own-training prompts;24use the other background on the same trained cores. These are never pooled with confirmation scores. Its auditor binds the parent seal, regenerates all prompts and checks pair lengths, checkpoints, identity and counts.
