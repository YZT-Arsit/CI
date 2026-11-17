# Restore and audit

Run from the repository root. CPU only; historical sources and checkpoints are dependencies, not regenerated data.

1. Restore historical JSONL with the earlier `tools/restore_*_artifacts.py` helpers documented in the linked reports. Restore this continuation with `python -B tools/restore_teacher_binding_artifacts.py`.
2. `python -B artifacts/20261117_teacher_binding/audit.py` replays all five frozen checkpoints' recorded episodes against the native synthetic transition model, reconstructs queries/actions and checks independent labels, raw hashes and checkpoint hashes. It does not call the model.
3. `python -B artifacts/20261117_teacher_binding/summarize.py` checks teacher TRAIN inputs against native labels, unchanged batches and source hashes, then recomputes summary tables.

`PLAN.json` was frozen before training. `PREFLIGHT.json` records six real differentiable checks: three roles crossed with query-consistency weights 0/1. Own-input mode exactly matches the frozen prior loss; teacher mode changes actor inputs only. `repair/*/training.jsonl` records predictions and actual actor inputs separately. Each new checkpoint has 80 updates; three reference checkpoints have zero new updates. The five models each receive 112 fresh DEV presentations. These are repeated presentations of 56 semantic worlds from 36 base groups, not 560 independent tasks.

The original worker command is `wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_teacher_binding/run.py'`. It deliberately refuses to overwrite an existing run. A new replication needs a separate declared directory and must preserve sources, pinned model revision, initial checkpoint, seed, batches and all baseline seals. The two new adapters and three reused copies are archived; source paths and hashes are in the plans and seals.

`build.py` uses the fixed fresh DEV seed range 330000000 and native/BFS checks. `preflight.py` and `run.py` require the same WSL dependencies and pinned SmolLM2-135M-Instruct snapshot as `../20261116_relation_solution/lm_common.py`. No model weight download or dependency installation is performed by the replay auditors. All inference uses the model's own estimates; no oracle actor input is admitted at evaluation.

The separate [native probe](../20261117_native_agent_probe/REPRODUCE.md) uses frozen Qwen and a different environment. Its results must not be pooled with these trained adapters.
