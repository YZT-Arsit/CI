# Reproduce the original-policy transfer audit

Restore parent evidence following ../20261117_join_repair/REPRODUCE.md, then from the repository root:

```powershell
python -B tools/restore_relation_factors_artifacts.py
python -B artifacts/20261117_join_policy_transfer/test_policy.py
python -B artifacts/20261117_join_policy_transfer/audit_policy.py
python -B artifacts/20261117_join_policy_transfer/diagnose_initial.py
python -B artifacts/20261117_join_policy_transfer/handoff/audit_handoff.py
```

These commands call no model. Native replay reconstructs every query/actor prompt, observation, selected action, transition, terminal flag and independent candidate label. It checks referenced checkpoints instead of duplicating them. Derived study/AUDIT.json and RESULTS.json are reproducible outputs.

Original inference in the pinned CPU FP32/four-thread WSL environment:

```powershell
wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_join_policy_transfer/run_policy.py'
```

The worker requires the completed parent seal and refuses an existing study directory. Replicate under separately declared output paths; do not remove evidence. Nine models,32new native synthetic episodes each, no optimization/retries/oracle estimates/action filtering. At most three role phases per episode. Generation uses16tokens,max_time8; the9.5second invalidity cutoff is not a hard process timeout. These are simulated environment transitions, not external tool calls or32stepAgent collaboration. The original global task-query interface is used; the learned local-step operator is not invoked as a separate iterative procedure.

The separately frozen judgment-channel diagnostic follows the sealed native study:

```powershell
wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_join_policy_transfer/handoff/run_handoff.py'
```

It requires its sealed tokenizer preflight and produces288fresh responses with no autonomous transitions. Truthful and inverted oracle judgments are explicit injected treatments,not model-generated reasoning. Sixteen state pairs have equal actual prompt/padding lengths and exactly two changed token positions. Do not pool these choices into native success or treat channel influence as proof of relational origin. Outputs/checkpoint identity/source bindings are independently audited.
