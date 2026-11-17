# Reproduce the fixed-state diagnosis

This continuation uses the two frozen80-update adapters from `20261117_binding_factors/repair`, its112DEV presentation-world versions, and the unchanged pinned SmolLM2 CPU environment from `20261116_relation_solution/environment.json`. It performs no parameter updates.

Restore earlier JSONL dependencies and this run before auditing:

```powershell
python -B tools/restore_relation_artifacts.py
python -B tools/restore_action_binding_artifacts.py
python -B tools/restore_binding_factors_artifacts.py
python -B tools/restore_fixed_state_artifacts.py
python -X utf8 -B artifacts/20261117_fixed_state_diagnosis/audit.py
python -X utf8 -B artifacts/20261117_fixed_state_diagnosis/audit_oracle.py
python -X utf8 -B artifacts/20261117_fixed_state_diagnosis/audit_training.py
python -X utf8 -B artifacts/20261117_fixed_state_diagnosis/value_class_check.py
python -X utf8 -B artifacts/20261117_fixed_state_diagnosis/simulate_coverage.py
```

The audits validate raw hashes, reconstruct prompts from observations, parse free model outputs, independently verify graph labels, and compare the same states. Derived summaries may be recomputed; historical raw files and checkpoints should stay unchanged.

`build.py` enumerates both provider commitments and downstream native states; it writes separate observations and labels. `probe.py` reads no label file. Its `RESPONSES_SEALED.json` must exist before `oracle_probe.py` starts. The latter deliberately reads true bits and labels its outputs as oracle-assisted actor diagnostics. These are not autonomous trajectories or learned-policy successes.

To rerun model work, use a fresh repository copy/new output directory with the same earlier artifacts restored. Do not remove original PLAN files to bypass overwrite guards. Run `probe.py`, then `oracle_probe.py`, then `gradient_probe.py` in the pinned WSL Python environment. These workers should run sequentially so CPU contention does not distort timing. The inherited free-generation helper uses an8second generation bound and9.5second validity threshold; the optional four-component gradient trial has a15second watchdog, with zero optimizer updates.

The first two training batches per role are selected from the previous frozen plan for gradient measurement. Checkpoint and batch reuse is intentional. This is a restricted local diagnostic, not an independent-seed or source-held-out experiment. See `GRADIENT_SCOPE.md` for literature and interpretation limits.
