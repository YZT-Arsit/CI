# Token-policy repair before test-result inspection

V2's token combined training loss remains near the no-relation floor after 600
updates. No held-out policy results have been inspected. Repair input/learning:

- Use one symbol vocabulary derived from the authorized observation, independent
  of which candidate action is being scored. Preserve the same prefix identity.
- Apply fresh shared per-pair entity bijections and triple-order permutations
  during training. These preserve the actual world relations and labels.
- Balance decision stages in combined batches instead of letting easy Source/
  Worker examples dominate the relatively few Planner future preferences.
- Compare ci_only, combined and shuffled at the SAME 2400 update budget, batch32,
  learning rate .001 and seeds17/29/41; same four-layer width32 model and fixed
  random entity embeddings. No test-driven early stopping or checkpoint choice.
- Keep V2 checkpoints and all failures. Seal all V3 checkpoints before inspecting
  either V2 or V3 held-out predictions. No oracle mask/fallback in evaluation.

These are finite structured-token action models, not pretrained text LLMs.
