# Factor-binding study: scope and failure interpretation

The plan and data were frozen before model calls. This document clarifies a
methodology limitation; it does not change the frozen questions or gates.

- Both training arms consume the same task-query supervision and historical
  Worker actor replay. The task-replay control duplicates task-query targets
  as the additional branch; the factor arm uses new-relation targets.
- The factor arm DOES NOT jointly condition the training task answer on its
  own generated factor value. Self-generated factor notes are introduced at
  evaluation as an intervention on the input interface.
- The starting checkpoint came from the 20261117 composition repair, where
  teacher truth/masked relation notes were part of training. Thus, note use
  is not entirely novel at evaluation, but factor errors and distribution
  shifts can still degrade it.
- If the factor arm improves relation classification but hurts task decisions,
  this single auxiliary-objective design fails; it does NOT prove factorized
  training in general cannot work. Any subsequent joint objective would need
  a separately justified algorithm and preregistered matched controls.
- A clean oracle factor injected at inference is a diagnostic control, never
  evidence of model self-generated relational knowledge.
- Small-source adapted Meeting/Query, synthetic CI norm, same symbol grammar,
  one CPU LoRA seed, and dev/confirm are not external business generalization.
- No new iteration or extra training is authorized by an isolated metric gain;
  the frozen gate governs the stage decision.


Additional interpretation safeguard (documented before factor-model scoring):
A high overall relation-task accuracy could be achieved by always emitting the
majority relation label (TRAIN relation labels are 80 zero / 48 one). Thus the
separately reported relation Flip/Hold **both-side accuracy** must be inspected.
The frozen primary gate is unchanged; a pass in total relation accuracy alone
cannot support a claim that the model learned relational counterfactuals.
