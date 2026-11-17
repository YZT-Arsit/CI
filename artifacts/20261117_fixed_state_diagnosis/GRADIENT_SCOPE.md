# What the gradient diagnostic can establish

Exa successfully retrieved the primary [Gradient Surgery for Multi-Task Learning paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf) and method sections2.2–2.3 from [the authors' arXiv version](https://arxiv.org/abs/2001.06782). Its discussion distinguishes conflicting directions, unequal magnitudes, and curvature; a negative inner product alone does not establish harmful optimization. PCGrad changes conflicting gradient components, but that does not prove it solves this project's relation or safety task.

Our probe records norms and inner products of query supervision, action supervision, query JS, and action JS on six preselected TRAIN batches at three frozen checkpoints. It does not estimate curvature, replay optimizer states, perform gradient surgery, or optimize any parameter. The result can reject an unsupported story about universal gradient alignment or obvious magnitude dominance. It cannot establish the cause of the observed policy degradation or select PCGrad without an additional matched intervention.

The two batches per role are chosen as the first two in the previous frozen training plan, without using current diagnostic results. Report their restricted coverage rather than treating18batch/checkpoint measurements as independent task evidence.
