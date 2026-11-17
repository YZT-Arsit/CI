# Small-set fitting diagnostic

The selection is the first forced contrast batch in each role x depth1/2/3 x task-kind cell of the existing contrast-repair TRAIN schedule. This yields18pairs,36distinct observations and14underlyingbasegroups. It does not create18independentworlds. No model correctness or generated response is consulted when selecting batches.

All original two-view rendering seeds are retained. Each epoch uses every selected pair once, with a fixed shuffled order; eight epochs produce144updates per trained arm. The two arms see the same query and teacher-actor text in the same order. The only loss difference is whether actor_loss participates in backpropagation. Shared parameters remain shared; removing actor_loss is not freezing actor parameters.

The query panel has168original-vocabulary prompts and168same-world renamed prompts. Per vocabulary,96prompts belong to48label-changing pairs, and72have labels unchanged across the paired history intervention. The 336prompts are all distinct; there are no exact-prompt contradictory labels. Original and renamed queries are in-sample/same-world diagnostics, not independent generalization. Native audit reports changing and fixed-label queries separately, plus all84four-prompt token-count controls per model.

The actor panel is the parent's144oracleWorkerprompts:24states, six option orders each,12zero-completion and12positive. It is not new data and does not cover every Planner/Source condition. Initial oracle responses are byte-reused from the identical source checkpoint; only the two trained arms generate new actor responses. Oracle retention cannot demonstrate full autonomous safety or task utility.

Earlier ancestry already contains depths1/2/3 and whole-path supervision. No first-ever multihop-learning claim follows. This diagnostic uses synthetic information-flow/obligation rules around the existing source task computations. It does not establish natural-norm validity, task transfer or32stepAgentcollaboration.
