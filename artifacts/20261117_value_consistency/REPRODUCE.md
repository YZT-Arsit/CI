# Reproduce the loss ablation

Restore the earlier relation, action-binding, binding-factor and fixed-state archives using their corresponding `tools/restore_*_artifacts.py` scripts, then restore this continuation. Raw JSONL is archived losslessly; hashes cover original and compressed bytes. The pinned WSL CPU dependency/model environment remains `20261116_relation_solution/environment.json`.

`build.py` freezes a loss-only comparison on the previous repaired80batch schedule. It independently verifies560DEVstate versions and checks both history reversal and presentation identity. `preflight.py` checks probability grouping, both gradient paths and exact equality to the old implementation when full-action mode is requested. It performs zero updates; six real-model new-mode forward/backward trials take under5seconds each.

The worker `run.py` trains query_js and value_js for80updates each from the exact same initial adapter and fresh AdamW state. It reuses sampler_sft and full_js checkpoints from the preceding run, with explicit origins. Every evaluation uses freely generated estimates and freely generated actions; no label is provided to the model at evaluation. Native audits reconstruct actual prompts, parses, transitions and values.

Run after restoring this continuation's archive:

```powershell
python -X utf8 -B artifacts/20261117_value_consistency/audit.py
python -X utf8 -B artifacts/20261117_value_consistency/summarize.py
python -X utf8 -B artifacts/20261117_value_consistency/contract_audit.py
python -X utf8 -B artifacts/20261117_value_consistency/audit_reference.py
```

`replay.py` is the earlier native auditor copied with only its worker-source lookup parameterized: explicit `run.py` for the primary run and `reference.py` for the secondary fixed checkpoint. The historical auditor is unchanged. `audit.py` verifies the frozen source/data/plan dependencies before replay. The additional reference is registered separately in REFERENCE_PLAN.md; its112episodes are not pooled into the primary448episode counts.

For new inference/training use a fresh isolated full repository copy/output directory, retaining earlier dependencies. Never delete an existing PLAN to overwrite evidence. Run preflight, then build, then run in the pinned WSL environment. The archived plan's source hashes must match the corresponding sources, and runtime requires the preflight loss hash. New auditors and reporting scripts do not alter the running worker.

This is one seed on the same two source tasks. Failed variants must be retained. Passing a shallow diagnostic cannot establish the full research objective. See INTERPRETATION.md for why weaker regularization is a confound and why native free-generation success takes priority over class loss.
