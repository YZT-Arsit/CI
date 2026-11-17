# Reproduce the frozen factor diagnostic

Restore dependencies through `../20261117_witness_binding/REPRODUCE.md`, then restore and audit this delivery from the repository root:

```powershell
python -B tools/restore_relation_factors_artifacts.py
python -B artifacts/20261117_relation_factors/test_factors.py
python -B artifacts/20261117_relation_factors/audit_factors.py
```

These commands make no model calls. The audit reconstructs every prompt/label, verifies parent and current seals, checks exact runtime token counts and model identity, then recomputes component scores and matched correctness transitions.

Original inference command in the pinned CPU FP32/four-thread WSL environment:

```powershell
wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_relation_factors/run_factors.py'
```

The worker refuses an existing output directory. A replication needs separately declared output paths; do not remove evidence to rerun. Each generation uses the earlier greedy16-token,max_time8configuration. The9.5second scoring cutoff is not a hard process timeout. No training or actor policy execution is performed.

The twelve base groups and all factor variants are correlated diagnostic examples. Names are changed through two developed word pools, not unseen pretrained-model tokens. Do not pool echo into relation accuracy, claim frontier-size effects independently of length/position, or treat perfect elementary probes as completion of natural-norm, long-horizon CI or32step Agent requirements.
