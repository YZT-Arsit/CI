# Reproduce the task-rehearsal audit

Restore historical dependencies following [join repair instructions](../20261117_join_repair/REPRODUCE.md), then run from the repository root:

```powershell
python -B tools/restore_task_rehearsal_artifacts.py
python -B artifacts/20261117_task_rehearsal/test_rehearsal.py
python -B artifacts/20261117_task_rehearsal/audit_rehearsal.py
python -B artifacts/20261117_task_rehearsal/policy_audit.py
python -B artifacts/20261117_task_rehearsal/diagnose_initial.py
```

These commands do not call a model. They reconstruct training records and native worlds, verify the frozen code/data/dependency hashes and parent output seal, compare reused bytes, check final adapter bindings, update counts/loss components/tokens, and independently replay all native actions and task outcomes. `policy_audit.py` also runs `audit_rehearsal.py`; the separate command is useful for inspecting fixed-panel accounting.

The final checkpoint of each training run is used. All three starts reference the preceding study's targeted adapters. Initial fixed-panel outputs are copied byte-for-byte: 2,736 reused responses. Six trained models generate 5,472 fixed-panel responses. Native episode responses are new for all nine models and counted from the traces. Initial state decomposition adds no calls and does not change the predeclared progress gate. It was registered separately while inference was in progress, before prediction scores were inspected.

Original commands in the pinned Python3.12/Torch/Transformers CPU environment were:

```powershell
wsl -d Ubuntu -- python3 -u /mnt/d/research/longhistory/artifacts/20261117_task_rehearsal/token_preflight.py
wsl -d Ubuntu -- python3 -u /mnt/d/research/longhistory/artifacts/20261117_task_rehearsal/run_rehearsal.py
```

The base model is SmolLM2-135M at the historical pinned revision, with CPU FP32, four threads and q/v rank8 LoRA. Dependency and base-model verification are inherited from the preceding sealed study. Token preflight checks the actual tokenizer and answer suffix boundaries. Training has a15second hard watchdog per update. Greedy generation has16new tokens and max_time8, with9.5seconds used as the validity cutoff; generation's max_time is not a hard process watchdog.

Do not remove evidence to rerun. The worker refuses an existing study directory; a replication requires a new declared output location and a new plan. The adapter files, raw responses, labels and failure cases are delivered. JSONL files are gzip-compressed without loss; the restorer verifies hashes and refuses to overwrite different evidence. No external API credentials or GPU are needed for the audit.

Read [PLAN.md](PLAN.md) before interpreting results: this is a developed synthetic comparison with correlated presentations, repeated small TRAIN support, unequal token counts between treatments, and conditional-oracle retention reported separately from autonomous task completion. It is not a natural business benchmark, proof of internal reasoning, or32step effective Agent collaboration.
