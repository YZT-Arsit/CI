# KCI-Align Agent RL v0.1 — executable actions, verified rewards, LoRA GRPO baseline

## Status

Implemented in the original research repository. Windows CPU environment and rollout checks are passing. A local 135M causal LM was used for a genuine tokenizer/action-selection/inference smoke. A real 4B CUDA probe and 4B GRPO training have NOT been performed, and no final research outcome is claimed.

This package is standard constrained-action RL. It does not claim to implement the novel CCE or DPR additions.

## Executable contract

The pipeline reuses code/kci/dynamic_tasks.py and the pinned Meeting/Query source assets. The acting Agent receives its original authorized observation with real delivery receipts. The LM sees a numbered list of valid executable actions and samples ONE single-token option letter. The native executor consumes the corresponding action, delivers actual messages, updates state and computes privacy/task outcomes.

Only the letter is optimized. The policy never receives hidden evaluator flags, task answer, simulated future world, oracle witness, or reward in its prompt.

Terminal reward: -1 for CI violation; 0 for safe incomplete; 1 minus 0.01 times steps (capped at 30) for safe completion. Group size G=4–8, multiple complete trajectories from one identical starting state. Normalized group-relative advantage. Zero-variance groups are recorded and skipped.

The GRPO-style loss is a clipped ratio between current action-letter logprob and the logged old-policy letter logprob, with optional KL on the legal action distribution against a frozen base LM. Fresh LoRA is trainable; the reference is the adapter-disabled base. An existing LoRA adapter can be used for inference-only probes, not as the trainable starting reference in v0.1.

The baseline is not open-text tool call generation, not a full general agent, and not stock TRL GRPO. It is a valid bounded MDP/action policy for our current source-adapted executor.

## Local CPU checks

Run in Windows PowerShell from D:\research\longhistory:

    py -3 tools\run_kci_agent_rl.py preflight
    py -3 -m unittest discover -s code\tests -p test_agent_rl.py -v
    py -3 -m unittest discover -s code\tests -p test_agent_rl_grpo.py -v
    py -3 tools\run_kci_agent_rl.py probe --backend random --world-limit 8 --group-size 4 --output artifacts\20261119_composition_signals\agent_rl_v1\random_new_unique

Always choose a new output name. The tests run without downloading a pretrained model.

The 135M LM inference smoke used the LOCAL existing WSL model and the saved dependency versions from the frozen earlier research project, only to test action-token output. It is not a 4B outcome estimate.

## GPU server setup and commands

Prepare a new CUDA environment, with PyTorch, Transformers, PEFT, Accelerate, Safetensors and optionally bitsandbytes. See requirements_gpu.txt for version bounds. Do not overwrite the Windows CPU environment or the pinned prior experiment.

Copy the repository with the existing source anchors (artifacts/20261113_iteration2/source_anchors), or provide their path via --source-root. Source SHA-256 is checked.

Run the basic source preflight:

    python tools/run_kci_agent_rl.py preflight

On 4B GPU, first run an inference-only signal probe with a local model:

    python tools/run_kci_agent_rl.py probe --backend hf --model /models/LOCAL_4B --bits4 --dtype bf16 --split dev --world-limit 8 --group-size 4 --output artifacts/20261119_composition_signals/agent_rl_v1/4b_probe_01

No weights update in probe. The model is loaded without network access by default; --allow-download is required for Hugging Face download.

If actual 4B policy groups show valid action sampling and useful reward/success/violation contrasts, launch a bounded LoRA+GRPO:

    python tools/run_kci_agent_rl.py train --backend hf --model /models/LOCAL_4B --bits4 --dtype bf16 --gradient-checkpointing --max-updates 8 --group-size 4 --dev-world-limit 8 --lr 1e-5 --kl-coef 0.02 --confirm-gpu-training --output artifacts/20261119_composition_signals/agent_rl_v1/4b_grpo_01

The training command is explicit. Without --confirm-gpu-training, nothing trains. A 4B training attempt on CPU is blocked by default. The optional --allow-cpu-smoke flag is only for very small local models and software validation.

## Metrics and study limitations

Track:
- Safe task success, CI violations, safe failure/over-withholding and trajectory length separately.
- Informative groups with genuine safety/success differences, cost-only reward differences, and zero-variance groups.
- Correct action choices for linked/unlinked history, different permission and horizon.
- Model-generated action entropy, importance ratios, KL and gradients.
- Actual DEV before/after trends without harmful tradeoffs, not just a scalar reward mean.

Both Meeting and Query are SOURCE-ADAPTED original examples; their privacy norms and coordination obligations are synthetic. The TRAIN/DEV distinction currently varies horizons, not independent business task families. This cannot support a claim of unseen business generalization or real-world CI validity.

The future algorithm contribution is state-specific long-horizon credit: CCE compares two presently legal actions under a frozen reference continuation policy, and DPR creates actual failure-state repair preferences. Neither is part of this terminal-only GRPO baseline. Validate this baseline before adding them.

## Reproducibility

All code is kept on an independent Git branch. Complete runs write immutable JSONL episodes, selected action logprobs, group metrics, source/code metadata and MANIFEST.json; partial crashes cannot claim completed status. Saved trained artifact is the small LoRA adapter, never an unrelated original checkpoint. No paid API and no GPU training were initiated while constructing the code.
