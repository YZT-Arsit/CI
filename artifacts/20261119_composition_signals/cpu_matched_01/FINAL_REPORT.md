# KCI-Align — bounded matched CPU LoRA study

**Main research goal unchanged:** dynamic recipient-knowledge-conditioned long-horizon CI action policy learning.

This is a small **batch-organization diagnosis**, not proof of a novel algorithm.
One SmolLM2-135M base+LoRA start (archived truth17), 32 updates for each treated arm;
the two arms see the SAME 128 native query targets and the SAME 32 archived actor replay batches.
Only whether each flip/hold pair is colocated versus shuffled changes.

| Arm | New DEV query /128 | Flip both /32 | Hold both /32 | Original query /192 | Original optimal action /80 | Unsafe /80 |
|---|---:|---:|---:|---:|---:|---:|
| initial | 105 | 9 | 32 | 175 | 74 | 4 |
| paired | 106 | 14 | 31 | 177 | 73 | 1 |
| shuffled | 111 | 15 | 32 | 174 | 69 | 0 |

## Predeclared diagnostic tradeoffs
- paired_flip_strictly_better_than_both: FAIL
- paired_hold_not_worse_than_both: FAIL
- paired_native_actions_not_worse_than_both: FAIL
- paired_native_unsafe_not_worse_than_both: FAIL
- paired_native_queries_not_worse_than_both: PASS

**Joint no-regression gate: FAIL**.
All published panels are **development diagnostics**; no independent held-out task family exists here.
In particular, if flip performance improves while safety, hold, task utility or actions worsen, it is not a successful KCI repair.

## Evidence and limitations
- The independent file/output/model auditor passed; raw outputs and adapter hashes are sealed in cpu_matched_01.
- Paired vs shuffled padded-query token counts: {'paired': 28860, 'shuffled': 29332}. Equal examples and steps; not equal FLOPs.
- All gold labels come from the existing source-adapted deterministic environment, not a GPT self-judge.
- The learned model has not completed the original 32-step multi-agent objective, and no external social norms were verified.
- No GPU or paid API calls. Continue large-GPU NO-GO.

## Reproduction
Run Windows: py -3 artifacts\20261119_composition_signals\audit_cpu.py
and: py -3 artifacts\20261119_composition_signals\analyze_cpu.py
Do not rerun training over the same output folder. Preserve frozen checkpoints and all negative cells.
