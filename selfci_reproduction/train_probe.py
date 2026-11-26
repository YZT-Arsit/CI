"""Run a bounded prefix of the official 30-epoch schedule on one GPU.

The Transformers backend samples directly from the current student. Both
backends use the upstream collator, dual teacher prompts, loss and EMA update.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import TrainerCallback
from trl.trainer.base_trainer import BaseTrainer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "upstream"))
from src import run_selfci as official
from src.utils.selfci_trainer import SelfCITrainer


class DirectSamplingTrainer(SelfCITrainer):
    def _sample_model_outputs(self, prompts, num_generations=1):
        model = self._unwrap_student_model()
        tokenizer = self.processing_class
        training = model.training
        model.eval()
        results = []
        try:
            with torch.inference_mode():
                for prompt in prompts:
                    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
                    width = inputs.input_ids.shape[1]
                    with torch.random.fork_rng(devices=[model.device]):
                        torch.manual_seed(self.args.vllm_seed)
                        output = model.generate(
                            **inputs, do_sample=True, temperature=self.args.temperature,
                            top_p=1.0, top_k=0, repetition_penalty=1.0, use_cache=True,
                            num_return_sequences=num_generations, max_new_tokens=self.args.max_completion_length,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    sequences = []
                    for ids in output[:, width:].tolist():
                        stops = model.generation_config.eos_token_id
                        stops = stops if isinstance(stops, list) else [stops]
                        end = next((i + 1 for i, token in enumerate(ids) if token in stops), len(ids))
                        ids = ids[:end]
                        sequences.append(SimpleNamespace(token_ids=ids, text=tokenizer.decode(ids, skip_special_tokens=True)))
                    results.append(SimpleNamespace(prompt_token_ids=inputs.input_ids[0].tolist(), outputs=sequences))
        finally:
            model.train(training)
        return results


def adapter_hash(model):
    digest = hashlib.sha256()
    for name, param in model.named_parameters():
        if "lora_" in name:
            digest.update(name.encode())
            digest.update(param.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-steps", type=int, required=True)
    parser.add_argument("--engine", choices=["transformers", "vllm"], required=True)
    options, rest = parser.parse_known_args()
    sys.argv[1:] = rest
    if options.probe_steps < 2:
        parser.error("At least two optimizer steps are needed with linear warmup.")

    class MeasuredTrainer(DirectSamplingTrainer if options.engine == "transformers" else SelfCITrainer):
        def __init__(self, *args, **kwargs):
            if options.engine == "transformers":
                # Replace only colocation setup. SelfCITrainer still validates
                # objective parameters and installs the official EMA callback.
                from src.utils.onpolicy_trainer import MultiGPUOnPolicyTrainer
                def init_without_vllm(instance, model, args, train_dataset, processing_class, teacher_model, **extra):
                    BaseTrainer.__init__(instance, model=model, args=args, train_dataset=train_dataset,
                                         processing_class=processing_class, **extra)
                    teacher_model.requires_grad_(False).eval()
                    instance.teacher_model_is_local = False
                    instance.teacher_model = instance.accelerator.prepare_model(teacher_model, evaluation_mode=True)
                from unittest.mock import patch
                with patch.object(MultiGPUOnPolicyTrainer, "__init__", init_without_vllm):
                    super().__init__(*args, **kwargs)
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            else:
                super().__init__(*args, **kwargs)
            self.args.eval_on_start = False
            self.add_callback(ProbeLog(self))

        def compute_loss(self, model, inputs, **kwargs):
            loss = super().compute_loss(model, inputs, **kwargs)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite SelfCI loss")
            if model.training:
                log = dict(step=self.state.global_step, tokens=inputs["completion_mask"].sum(-1).tolist(),
                           loss=loss.detach().item(),
                           prompt_ids_sha256=hashlib.sha256(inputs["prompt_ids"].cpu().numpy().tobytes()).hexdigest())
                with (Path(self.args.output_dir) / "microsteps.jsonl").open("a") as file:
                    file.write(json.dumps(log) + "\n")
            return loss

    class ProbeLog(TrainerCallback):
        def __init__(self, trainer):
            self.trainer = trainer

        def on_train_begin(self, args, state, control, **kwargs):
            self.start = time.monotonic()
            self.initial = adapter_hash(self.trainer.model)
            self.teacher_initial = adapter_hash(self.trainer.teacher_model)
            assert self.initial == self.teacher_initial
            torch.cuda.reset_peak_memory_stats()

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            print(f"verified optimizer step {state.global_step}/{state.max_steps} elapsed={time.monotonic()-self.start:.2f}s", flush=True)
            if state.global_step >= options.probe_steps:
                control.should_training_stop = True
                control.should_save = True
            return control

        def on_train_end(self, args, state, control, **kwargs):
            audit = dict(steps=state.global_step, scheduled_steps=state.max_steps, engine=options.engine,
                         initial=self.initial, student=adapter_hash(self.trainer.model),
                         teacher=adapter_hash(self.trainer.teacher_model), teacher_initial=self.teacher_initial,
                         wall_seconds=time.monotonic()-self.start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved(), history=state.log_history)
            path = Path(args.output_dir) / "probe.json"
            path.write_text(json.dumps(audit, indent=2))
            if audit["student"] == self.initial or audit["teacher"] == self.teacher_initial:
                raise RuntimeError("Missing expected adapter update")

    official.SelfCITrainer = MeasuredTrainer
    official.main()


if __name__ == "__main__":
    main()
