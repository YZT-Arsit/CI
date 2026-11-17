# Native telecom execution feasibility

Pinned original release: [sierra-research/tau2-bench, 37199f3](https://github.com/sierra-research/tau2-bench/tree/37199f36924c8896f5e048360691f8476cd89ba1), dated 2025-06-10. All 29 selected source/data files match official Git blob bytes; MIT attribution is retained. This selected subset supports the native telecom environment, not the complete benchmark runner.

| Check | Result |
|---|---:|
| Published task entries replayed | 114 |
| Annotated-order plans passing native environment assertions | 114/114 |
| Full/omission/reversed-order trials | 744 |
| Single-action omissions that break a previously successful plan | 489/516 |
| Reversals that break a previously successful plan | 29/114 |
| Plans containing both assistant and user tools | 71/114 |
| Maximum calls changing either database after execution + synchronization | 12 |
| Maximum trial duration, including initialization | 0.0147 s |

The original tool router enforces the declared user/assistant tool side; synchronization propagates actual backend/device changes. Tools return native results, and native assertions evaluate the final state. An independent read-only review recomputed all record counts and verified source integrity.

These runs replay ground-truth annotated action lists. There are zero learned policy decisions and zero communication messages. They establish an executable external environment and within-plan action dependencies, not a learned collaboration score. Removing an action tests necessity relative to that particular plan and initialization; it does not rule out alternative solutions. Reversal tests order sensitivity, not individual dependency edges or message dependence. Database-change counts do not enumerate every internal transition. Native outcome assertions do not establish complete policy compliance.

The largest annotated repair plan has 12 real mutations, not 32. No waits or repeated actions were added to reach a target length. The next collaboration experiment must connect a policy to role-local observations and actual message exchanges, then separately count decisions, messages, tool calls and state changes.

Native Windows Python 3.13 initially failed before the first trial because upstream text loading inherited GBK. That attempt and its frozen plan remain in `tau2_replay/`. The successful separate `tau2_replay_utf8/` run used `python -X utf8`; upstream source was unchanged. Dependencies are isolated under `.cache/tau2_deps_20261117`, excluded from publication, and their resolved versions are recorded in the successful plan. This is not a claim of reproducing the paper's agent scores or original dependency environment.

The policies here are benchmark-authored telecom rules. They remain distinct from the official GitLab organization clauses in `policy_cases_v2/`.
