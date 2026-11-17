# Restore and verify rehearsal experiments

Restore the dependency chain through `../20261117_step_learning/REPRODUCE.md`. Native restore/audit commands from repository root:

```powershell
python -B tools/restore_step_replay_artifacts.py
python -B artifacts/20261117_step_replay/test_replay.py
python -B artifacts/20261117_step_replay/audit.py
python -B artifacts/20261117_step_replay/presentation_factors/probe.py audit
python -B artifacts/20261117_step_replay/presentation_factors/changes.py
```

They make no model calls. The auditor reconstructs exact local/replay schedules and the new-seed confirmation panel, verifies sources/checkpoints and every training token total, checks432optimizer updates, rescores2,672newresponses and verifies576reused oracle responses byte-for-byte. The latter remain conditional development-set action evidence. It distinguishes unchanged full necessary gates from the narrower matched retention diagnostic.

The separately frozen2×2presentation diagnosis adds576newresponses, no updates. It reconstructs the previous16selected TRAIN prompts per schedule exactly, then changes only names, only order, or both. It verifies all underlying edges/frontier/targets remain unchanged. Tokenizer preflight checks72pair groups and48four-condition equal-length comparisons. Its results concern these selected TRAIN examples, not independent tasks. The original serial launch, after the main worker completed, is `python3 -B artifacts/20261117_step_replay/presentation_factors/probe.py run` in the same pinned WSL environment.

`changes.py` is explicitly post-scoring descriptive analysis of the sealed factor outputs. It counts semantic response changes, correct-to-wrong and wrong-to-correct transitions against the same seen inputs. It makes no new model calls and changes no gate.

The original worker is:

```powershell
wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_step_replay/run.py'
```

It refuses an existing `study` directory. Preserve the evidence; a replication requires separately declared outputs. Three new adapters share the original warm-start,144updates each, fresh AdamW lr0.0002, CPUFP32/four threads. The local-only references are frozen existing adapters, not retrained or selected by new scores. Actor rehearsal adds tokens/compute; no compute-matched method advantage is claimed. The15second update watchdog is hard; greedy generation max_time8 and9.5second invalidation are not a hard process timeout.

The query panel reuses developed families/vocabularies with new480million seeds. Oracle Worker states are reused from development; only their prompts contain correct estimates, explicitly marked oracle. These checks do not evaluate model-generated multiround closure, autonomous actions, natural norms or32step Agent collaboration.

Rehearsal and action-retention measurement cover Worker only. They do not establish that Planner, Source or the old global-query interface is retained. The complete shared-model policy must still be evaluated after integrating model-generated local states.
