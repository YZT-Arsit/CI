# Source and protocol interpretation

- Primary language configurations were frozen before any primary language
  predictions were inspected. The later balanced recipe was frozen from DEV
  diagnosis before the first primary language result. Other already-frozen arms
  were observed while that fixed worker continued. `test_seen: false` in worker
  seals means that the training worker read no test data or feedback; it does
  not mean nobody ever viewed another locked arm's results during its runtime.
- The test-label generator is known and auditable. These are held-out grouped
  tests with frozen model selection, not independent human-blinded annotation.
- `relation_solution_data.py` subsequently gained the extended source adapter.
  The initial source is restored under `frozen_sources` and matches the exact
  SHA256 recorded by the data jobs. Existing training worlds do not use the new
  source variant. The ragged-cache failure's source is also restored against its
  pre-existing hash. Restoration is not represented as a new experiment.
- The first replay's tuple/list comparison failure was in the audit, not the
  execution output. Original predictions and first audit reports remain intact;
  the corrected comparison uses canonical JSON and reproduces all terminal
  states. This correction never changes a model action or score.
- Inference masks, teacher proofs and exact completion oracles are not model
  inputs. Exact completion is used to create offline labels and to score the
  saved forced-state outputs after they are written.
- Word palettes are disjoint as entity-name pools. Every word is known to the
  pretrained tokenizer; some can also occur in instruction text. The claim is
  new entity assignments/names during adaptation, not completely unseen tokens.
- Direct-learning and representation repairs are explicitly post-hoc diagnostic
  follow-ups with new seeds. They do not replace failed primary results. Both
  direct-learning samplers retain all three seeds, including refusal collapse.
- Additional language diagnostics were initially stopped before inference to
  preserve the three-hour deadline. The user then explicitly extended the time
  allowance; diagnostics resumed in fresh directories. Historical stop records
  remain under `deferred_before_user_extension`, and are not final missing-result
  claims. The separate 12-second timing probe does not overwrite primary scores.
