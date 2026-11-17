# Teacher-estimate training as an action-binding intervention

Previous goal turn: progress, with two completed loss ablations,560nativeepisodes and an external abstention diagnosis. No earlier model worker remains live. Full goal remains active; this continuation prioritizes relation-to-action learning.

Evidence: in the repaired training schedule, each old arm has12/160Worker presentations whose predicted-positive actions are disjoint from native optimal actions. The model is trained on correct action labels alongside some incorrect self-predicted intermediate estimates. Correcting estimates at inference in earlier oracle diagnostics did not fully restore action selection. These findings motivate testing the training interface, but do not prove it causes failure.

Primary2x2: actor training uses own estimates versus true TRAIN estimates, crossed with query-consistency weight0versus1. Actor consistency is absent in all four. Reuse `sampler_sft` and `query_js` from the prior continuation; train `teacher_sft` and `teacher_query_js` for80updates each. Keep original repaired batches, views, roles, seed17,initialadapter,lr0.0008,freshAdamW,formatweight and supervised labels unchanged. Include the earlier stronger SFT checkpoint as a predeclared fifth reference, not part of the matched2x2.

Only teacher arms replace actor training input bits with existing native TRAIN labels. Query prediction/loss still run, and both predicted and actual actor-input maps are logged. All free-generation evaluations use the model's own estimates and actions. No teacher bit, label, action mask or rule override enters evaluation.

This can introduce train/deployment mismatch: success with correct teacher inputs does not imply success with erroneous own predictions. [Ross et al.,2011](https://proceedings.mlr.press/v15/ross11a.html) analyze policy-induced training distributions; this experiment is not DAgger and inherits no guarantee from it. It is an isolated hypothesis test, not a proposed new method.

Freeze fresh DEV seed330000000 offsets before training:36basegroups,56semanticworlds,112presentations per arm,560episodes for five arms. Native/BFS check all560state versions, maintain history reversal and identity across views. Same two source tasks and one seed remain limitations.

Primary quantities: all-four base success, per-family strict success and CI, and per-role reached-state actions. A repair must preserve safety and task completion together; gains over a degraded control alone are insufficient. Keep all deeper, optional, authorized, undelivered-history and impossible-completion conditions. No outcome-dependent retries, new seeds or loss changes within this experiment.9.5supdate watchdog and inherited8sgeneration bound; CPU only. Historical artifacts and core code remain unchanged.
