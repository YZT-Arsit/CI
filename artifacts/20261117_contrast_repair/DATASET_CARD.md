# Data used by the sampling repair

This experiment reuses existing TRAIN worlds; it does not introduce an external benchmark or real-norm training corpus. The source computations are Meeting/Query tasks with synthetic information-flow and obligation rules. Source-world splits remain those of the existing dataset.

| Item | One-hop arm | Mixed-depth arm |
|---|---:|---:|
| Joint batches | 96 | 96 |
| State presentations before two renderings | 192 | 192 |
| Unique observations | 119 | 157 |
| Underlying base groups | 42 | 59 |
| Binary labels, zero / one before renderings | 285 / 163 | 285 / 163 |
| Unchanged fallback batches | 27 | 27 |

There are 54 forced contrast slots and 42 unchanged parent anchors. Every forced slot uses a history pair with at least one changing completion label, under no permission and full completion required. The mixed arm has at least six forced batches per role/depth cell, with three Meeting and three Query batches in each cell. Counting contrasts in unchanged anchors as well gives 59 of 96 reversing batches: Planner 6/6/6, Source 7/7/6, Worker 6/8/7 across depths 1/2/3. All 17 original condition buckets remain represented, but their frequencies change.

The parent had only ten reversing batches. Its labels were 233 zero / 215 one, and its unique observation counts were 136 / 145 for one-hop / mixed-depth. Thus contrasts, label distribution, condition frequency and observation diversity change together. Both current arms remain aligned with one another on factors, labels, views and updates; they differ in depth and consequent token/entity/instance exposure.

The unchanged fallback supervision contains 108 rendered prompts from seven distinct semantic observations and five base groups. Repeating them in another continuation does not create new independent evidence.

DEV is reused adaptively: 128 episode presentations, 64 semantic worlds and 40 base groups. The fixed query panel has 640 state presentations and 1,536 queries; the oracle Worker panel has 24 semantic states, each in all six option orders. Its labels are never used for parameter updates, but earlier DEV results informed the sampling repair. Reference model outputs are reused byte for byte and not counted as new inference.

The separately registered TRAIN probe has 1,152 generated responses: for each trained arm, 144 changing query pairs in the original vocabulary and 144 in a renamed vocabulary. These are presentations of the actual 54 forced training batches, not independent new worlds. The two renderings and repeated base groups remain correlated.

Read the parent [scope correction](../20261117_query_coverage/SCOPE_CORRECTION.md) before any depth claim: the ancestor trace model already sampled depths 1, 2 and 3. Neither natural-norm validity nor cross-task transfer nor 32-step agent collaboration follows from these data. Machine-readable counts are in `DATASET_SCOPE.json` and the frozen plans.
