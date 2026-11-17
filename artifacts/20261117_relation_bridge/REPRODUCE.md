# Reproduce the primitive-fact bridge audit

Restore the parent study and its dependencies using [task rehearsal instructions](../20261117_task_rehearsal/REPRODUCE.md), then run from the repository root:

```powershell
python -B tools/restore_relation_bridge_artifacts.py
python -B artifacts/20261117_relation_bridge/test_bridge.py
python -B artifacts/20261117_relation_bridge/audit_bridge_v1.py
python -B artifacts/20261117_relation_bridge/history_pairs.py
python -B artifacts/20261117_relation_bridge/diagnose_errors.py
```

The audit regenerates all16developed semantic worlds,80fixed states,192questions and labels; verifies native task labels against the independent relation/conditional formula; verifies the frozen parent checkpoints, model identities, code/data hashes, and the archived failed preflight; reconstructs every actor prompt from the corresponding model-generated task estimates; and scores true-world decisions separately from conditional consistency under the inverted note. It makes no model calls.

The parent trajectory localization can separately be reproduced with:

```powershell
python -B artifacts/20261117_relation_bridge/localize_parent.py
```

It recomputes the same observed288episodes and804decisions from the parent raw records. It adds no model calls or autonomous episodes. It is post-declared descriptive localization, not a demonstration that changing one judgment would repair the entire trajectory.

Original inference used the pinned Ubuntu CPU environment:

```powershell
wsl -d Ubuntu -- python3 -u /mnt/d/research/longhistory/artifacts/20261117_relation_bridge/token_preflight.py
wsl -d Ubuntu -- python3 -u /mnt/d/research/longhistory/artifacts/20261117_relation_bridge/run_bridge.py
```

Do not overwrite the delivered study to rerun it. The worker refuses an existing study directory, and token preflight refuses an existing result. Replications require a new declared output path and plan. The three checkpoints are referenced from the parent; no new model is trained here.

The first tokenizer preflight failed before model inference because the inline unknown field was one token shorter. The complete failed freeze is preserved underpreflight_v0 and hash-bound byPREFLIGHT_REVISION.json. These files are historical snapshots, not scripts to execute in that nested location; their original relative paths assume the study root. Revised formatting puts all note values after a newline. Actual-tokenizer preflight verifies all192truth/flip pairs and all192truth/masked pairs differ in one token at equal lengths, hence equal batch padding membership and widths.

Truth and flipped relation facts are externally computed diagnostic inputs. Actors receive model-generated task judgments. The80states are externally enumerated and held fixed; there are zero autonomous episodes in this study. The underlying meeting/query task kernels share the developed synthetic disclosure contract and compact prompt format, so this does not establish cross-business generalization, natural-norm validity or32step effective collaboration. Graph relation derivation by the language model remains a separate required step.

The frozen original auditor is retained. Its structural equality check compared native tuples to JSON lists; audit_bridge_v1 normalizes regenerated records through JSON before that check. AUDIT_REPAIR.json and AUDIT_V0_FAILURE.log preserve the repair evidence. No scores, targets, gates or model outputs changed. The error-cell analysis is explicitly post-score and descriptive.
