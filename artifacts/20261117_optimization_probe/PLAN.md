# Shared-model optimization diagnosis

Previous frozen crossover established text-estimate effects but incomplete TRAIN query and Worker fit. This is progress, not a completed solution. Keep Proposal v0.6 shared model, all roles, natural-norm transfer and meaningful multi-step collaboration requirements unchanged.

Before another training intervention, test whether current query/action updates actually harm one another. Use the sealed teacher_sft checkpoint from teacher_binding. Select the first original TRAIN batch for every one of its17semantic sampler buckets, in original order, without consulting error counts. A companion is the last original batch in that bucket; for singleton buckets keep the same pair with each presentation seed increased by1000003. These companions are TRAIN diagnostics, not held-out validation.

At each anchor, restore exactly the same trainable parameters. Compute query and action gradients from the original teacher-input loss with consistency weight0. Compare no update, query-only, action-only and summed-gradient updates. Each update starts fresh AdamW state,lr0.0008,weight_decay0,global norm clipping1,matching the historical optimizer configuration but NOT its unavailable historical moments. Evaluate both losses at the anchor and companion before/after. Restore and hash parameters before every branch; never persist a trained checkpoint. A zero-update branch checks numerical repeatability. All sources, batches, TRAIN files and initial checkpoint hashes are frozen before execution.

Gradient angle is diagnostic, not proof of harmful interference. The important local evidence is actual loss change under a matched optimizer step, including companion changes. These finite one-step TRAIN losses do not establish free-generation improvement, generalization, natural norms, internal neural causal abstraction, or novelty. No new DEV query or model selection. Do not deploy separate query/action models.

CPUFP32,seed17,4threads. Time every gradient computation and each evaluation separately,with a15second hard guard per operation; usually under10seconds. Stop on nonfinite gradients/loss or failed restore/control. Preserve failures. No GPU,dependency install,or unrelated process changes.

The next repair must follow these findings. If no consistent harmful interaction appears, favor a matched additional-training control over adding a gradient algorithm. If harmful interaction appears, first distinguish anchor and companion behavior; a negative gradient cosine alone is insufficient to justify PCGrad.

Source consulted through the requested Exa connector: [Yu et al., Gradient Surgery for Multi-Task Learning, NeurIPS2020](https://arxiv.org/abs/2001.06782). Its analysis distinguishes conflict, gradient magnitude and curvature; this diagnostic does not reproduce its theorem or benchmark, and uses no PCGrad update.
