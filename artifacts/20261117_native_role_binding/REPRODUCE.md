# Same-task role-conditioning follow-up

Restore with `python -B tools/restore_teacher_binding_artifacts.py`, then follow the [primary dependencies](../20261117_native_agent_probe/REPRODUCE.md). No new packages or source environment are needed.

After restoring raw records:

```powershell
python -X utf8 -B artifacts/20261117_native_role_binding/audit.py
python -X utf8 -B artifacts/20261117_native_role_binding/check_completion.py
```

The first command replays all sealed responses and native commands. The second records initial state assertions and successful reference-action matches; it never provides evaluation data to the model. Both auditors were source-frozen before follow-up inference. No official benchmark score is claimed.

`TREATMENT_CHECK.json` verifies matched task/model/budgets and byte-identical runner,transport,auditor and observations. `check_treatment.py` is deliberately a prelaunch check and refuses an existing RUN_PLAN; use the saved receipt and source hashes after the run. The original worker command is `python -X utf8 -B artifacts/20261117_native_role_binding/run.py`. It refuses in-place reruns. A new replication must be separately declared and must not erase prior outputs.

The final `tokens.py` replaces the generic Qwen system instruction with an explicit role-simulation/history-boundary instruction. Full observation/history remains in the user message. Native policies,schemas,private instructions and role routing are unchanged. The final PLAN pins every worker dependency and the primary plan/output seal. An earlier unrun placement draft is kept in `prelaunch_revision/`; it contributes zero model calls. The same task is reused after the primary failure, so this is not independent validation or a second task.
