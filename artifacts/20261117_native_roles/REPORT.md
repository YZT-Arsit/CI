# Native role boundaries before learned collaboration

Implemented a local role session over the previously pinned [native telecom release](https://github.com/sierra-research/tau2-bench/tree/37199f36924c8896f5e048360691f8476cd89ba1). Upstream source remains unchanged. The policy-facing view exposes only the role's instructions, own tool schemas and own history. The user scenario stays with the user. Task IDs, tickets, evaluator descriptions, answer actions, evaluator criteria and direct database dumps are excluded.

The114published tasks have no prepopulated message history; the adapter rejects that unsupported case rather than silently misrouting historical tool results. Tool outcomes stay local to the requesting role. Explicit messages are the only direct history-sharing operation. Native environment synchronization remains active, so another role can still discover relevant shared-world changes through its own later tools.

Checks completed on all114tasks:

- All114annotated plans still pass native state assertions.
-516tool-result isolation checks pass, and516deliberate wrong-role calls are rejected without database mutation or disclosure into the other role's history.
- Initial assistant views are identical across tasks; changing hidden task/evaluator metadata or the user's private scenario does not change the assistant view.
- Dropped synthetic messages do not appear in the receiver's view; explicitly delivered messages do. Mutating a returned view does not change stored histories.
- Maximum complete check duration0.186seconds.

These are transport and isolation checks, with **zero learned model decisions**. Recorded1,032submitted tool calls include516deliberate rejection tests. The228sent/114delivered messages are synthetic canaries, not task collaboration. There are492net-mutating tool calls across all replayed plans. The tests do not establish task success dependence on messages, complete independence of shared state, full policy compliance, or32meaningful learned steps.

The next collaboration worker must connect actual role-local model decisions to this interface, record raw outputs before evaluation, and test message omissions/reordering without exposing annotated action plans. GitLab's organization clauses are not transplanted into this benchmark's telecom policy.

A read-only reviewer verified the source hashes, raw seal, all114record counts and counter qualifications. See PLAN.json, RESULTS.json and raw.jsonl for evidence. Native Windows replay uses `python -X utf8 -B artifacts/20261117_native_roles/check.py` in a fresh output copy; the existing PLAN protects this run against overwrite. Dependencies are the same isolated native environment as the prior tau2 run.

The original PowerShell wrapper returned exit code 1 after treating the native logger's stderr as an error stream, although all application results were written. This is retained in `check.log`. An isolated rerun with Python subprocess capture completed with native return code 0 and matched the 114 tasks, 114 successful plans, 516 isolation checks, 516 wrong-role rejections, and zero model decisions. `NATIVE_RETURN_CODE.json` records both statuses and source/raw hashes; these are repeated checks, not additional independent tasks.

The isolated rerun's exact files are archived under `receipt_replay/`; its raw hash matches the receipt. The receipt retains its original temporary execution path for provenance. Restore both runs' JSONL with `python -B tools/restore_binding_factors_artifacts.py`.
