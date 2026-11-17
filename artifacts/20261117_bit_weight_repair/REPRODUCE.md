# Restore and reproduce the bit-weight comparison

Restore earlier dependencies through `../20261117_contrast_repair/REPRODUCE.md`, then:

```powershell
python -B tools/restore_small_fit_artifacts.py
python -B artifacts/20261117_small_fit/audit.py
python -B artifacts/20261117_small_fit/audit_bit_weights.py
python -B artifacts/20261117_bit_weight_repair/test_bit_balance.py artifacts/20261117_bit_weight_repair/fixed_loss.py
python -B artifacts/20261117_bit_weight_repair/audit.py
python -B artifacts/20261117_bit_weight_repair/audit_error_direction.py
```

To reproduce the original valid RED regression, run the same test script with `artifacts/20261117_teacher_binding/binding_loss.py` as its argument; four subtest assertions should fail. This is expected negative evidence. The earlier harness error is preserved separately and is not the valid failing test.

The original worker command was:

```powershell
wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_bit_weight_repair/run.py'
```

The worker requires a completed, audited small_fit control and refuses an existing output folder. A fresh replication must declare a separate output location without overwriting evidence. The fixed loss differs only in the bit term and removal of unused label bookkeeping; the verifier checks the exact specified source transformation.

The native auditor re-scores both the parent's joint control and repaired outputs. It requires exact agreement with the parent's results, source seals, initial weights, schedules, teacher inputs and per-batch token exposure. Control calls are not repeated or counted as new. It separately reports changing/fixed queries, original/renamed names and oracle Worker actions. No independent-world or autonomous-policy claim follows.
