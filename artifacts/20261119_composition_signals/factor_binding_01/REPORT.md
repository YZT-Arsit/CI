# KCI-Align Factor-to-Action Matched Study

**Research direction unchanged:** dynamic recipient-knowledge-conditioned long-horizon CI strategy learning.

**Pre-registered joint gate: NO_GO.** Single seed, fixed model start, bounded synthetic source tasks.

## Frozen paired comparison

| Arm | DEV task/128 | DEV relation/128 | Task Flip/32 | Task Hold/32 | Relation Flip/48 | Depth4 task/192 | Depth4 rel/192 | Old native optimal/80 | Unsafe/80 | Self-factor optimal/80 | Self-factor unsafe/80 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| initial | 105 | 59 | 9 | 32 | 0 | 179 | 96 | 74 | 4 | 63 | 2 |
| task_replay_control | 113 | 46 | 18 | 31 | 0 | 179 | 93 | 74 | 0 | 63 | 6 |
| relation_factor | 106 | 61 | 10 | 32 | 2 | 178 | 95 | 72 | 6 | 56 | 1 |

## Frozen gates (not changed after seeing model scores)
- dev_task_flip_margin: FAIL
- dev_task_hold_retention: PASS
- dev_relation_gain: FAIL
- original_native_action_retention: FAIL
- original_native_unsafe_retention: FAIL

## Mechanistic evidence boundary
Insufficient learned relation recognition on paired DEV under this training budget; no evidence for reliable autonomous knowledge update.

All relation/permission/required/utility labels were independently computed from the existing finite native task. Task-source documents and social CI norms are synthetic adaptations, not human validated. Relation evidence at inference comes from the model's generated text, not an injected oracle.

This study does not show new long-horizon optimization or certified DPO performance. Pair supervision is a small component diagnostic, not yet an ACL method claim.

## Independent checks and next decision
Audit checked raw outputs, adapter files and all three starts. Same model start verified.
STOP local loss/format/batching tweaks. Review data task realism and model capacity; do not scale factor-supervision without new independent evidence.

All raw model responses and training traces are in this folder's study/ directory. Prior 20261117 archives are untouched. No paid API or large GPU used.
