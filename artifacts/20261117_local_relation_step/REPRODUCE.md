# Restore and rescore the frozen step probe

First restore the dependency chain in `../20261117_rename_coverage/REPRODUCE.md`. From the repository root:

```powershell
python -B tools/restore_local_step_artifacts.py
python -B artifacts/20261117_local_relation_step/test_step.py
python -B artifacts/20261117_local_relation_step/audit_probe.py
```

These commands require no model inference. The auditor rebuilds observations/labels, verifies frozen inputs and checkpoint metadata, and rescores all1,280 responses. The model loader verifies referenced adapter bytes; final independent review separately rehashed both referenced adapters. The frozen probe auditor does not itself directly rehash those two adapter directories. The tests include4,096 graph/frontier states, JSON round trips and a direction-only counterfactual. Test counts are evaluator checks, not model calls.

The original tokenizer-only preflight and model worker ran in the pinned WSL environment, CPUFP32/four threads. The original worker command was:

```powershell
wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_local_relation_step/probe.py'
```

The worker refuses existing outputs. Preserve these outputs; declare a separate location for a replication. `TOKEN_PREFLIGHT.json` checks both own and initial padded token lengths for256 paired groups. Prelaunch source/plan revisions and failed checks are retained under `pre_implementation`, `prelaunch_json_revision`, and `prelaunch_direction_revision`. The placeholder RED test is a new-function test, not evidence of a defect in an earlier production model.

The conditional learning follow-up has separate data, schedules, seals and audit commands in `../20261117_step_learning/REPRODUCE.md`. No graph solver supplies answers to standalone model queries. See `REPORT.md` for scope limitations and negative results.
