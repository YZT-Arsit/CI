"""Constrained Hugging Face causal-LM action policy for KCI multi-agent RL.

Samples one *model-generated* A/B/C action token from executable options.
For QLoRA, reference scores are computed with the trainable adapter disabled,
so the frozen base model is a valid, unchanged reference (fresh LoRA only).
Requires a target GPU and model weights for 4B; no implicit web downloads.
"""
from __future__ import annotations

import math
import random
import json
from pathlib import Path

from .core import ActionSample, LETTERS


def exploratory_logprobs(action_logits, *, temperature: float = 1.0,
                         epsilon: float = 0.0):
    """Differentiable on-policy mixture: (1-eps)*softmax(logits/T) + eps/|A|.

    The same mixture must be used for action sampling, stored OLD log-probs,
    CURRENT log-probs, and frozen-reference KL. No off-policy uniform actions.
    """
    import torch
    if not 0 < temperature <= 8:
        raise ValueError("Temperature must be in (0,8]")
    if not 0 <= epsilon <= 0.75:
        raise ValueError("Exploration epsilon must be in [0,.75]")
    if action_logits.ndim != 1 or action_logits.numel() == 0:
        raise ValueError("Expected a nonempty action logit vector")
    logp = torch.log_softmax(action_logits.float() / temperature, dim=-1)
    if epsilon:
        uniform = torch.full_like(logp, math.log(epsilon / logp.numel()))
        logp = torch.logaddexp(logp + math.log1p(-epsilon), uniform)
    if not torch.isfinite(logp).all():
        raise RuntimeError("Nonfinite exploratory action probabilities")
    return logp


class HFActionPolicy:
    def __init__(self, *, model_path: str, adapter: str | None = None,
                 trainable: bool = False, quantization: str = "none",
                 dtype: str = "bf16", max_prompt_tokens: int = 4096,
                 allow_download: bool = False, lora_rank: int = 8,
                 lora_alpha: int = 16, gradient_checkpointing: bool = False,
                 temperature: float = 1.0, explore_epsilon: float = 0.0,
                 kbit_inference_precision: str = "training"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not 0 < temperature <= 8 or not 0 <= explore_epsilon <= 0.75:
            raise ValueError("Invalid fixed on-policy sampling parameters")
        if quantization not in ("none", "4bit"):
            raise ValueError("Quantization must be none or 4bit")
        if kbit_inference_precision not in ("training", "legacy"):
            raise ValueError("4-bit inference precision must be training or legacy")
        if dtype not in ("fp32", "fp16", "bf16"):
            raise ValueError("Unsupported compute dtype")
        if max_prompt_tokens < 128:
            raise ValueError("Too-short maximum prompt length")
        if trainable and adapter:
            raise ValueError(
                "Training from an existing adapter needs a matched frozen "
                "adapter reference; v1 supports fresh LoRA only."
            )
        if trainable and quantization == "4bit" and not torch.cuda.is_available():
            raise RuntimeError("4-bit LoRA training requires a CUDA GPU")
        if quantization == "4bit" and not torch.cuda.is_available():
            raise RuntimeError("bitsandbytes 4-bit inference requires CUDA")
        if dtype in ("fp16", "bf16") and not torch.cuda.is_available():
            if dtype != "fp32":
                raise RuntimeError("Use fp32 for portable CPU smoke tests")
        torch_dtype = {"fp32": torch.float32, "fp16": torch.float16,
                       "bf16": torch.bfloat16}[dtype]
        self.torch = torch
        self.policy_id = (
            f"hf-categorical:{model_path}:{adapter or ('fresh-lora' if trainable else 'base')}"
            f":temp{temperature}:eps{explore_epsilon}"
        )
        self.kbit_inference_precision = kbit_inference_precision
        if quantization == "4bit":
            self.policy_id += ":kbit-precision=" + (
                "training" if trainable else kbit_inference_precision)
        self.temperature = temperature
        self.explore_epsilon = explore_epsilon
        self.model_path = model_path
        self.trainable = trainable
        self.max_prompt_tokens = max_prompt_tokens
        self.tok = AutoTokenizer.from_pretrained(
            model_path, local_files_only=not allow_download,
            trust_remote_code=False)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        kwargs = {
            "torch_dtype": torch_dtype,
            "local_files_only": not allow_download,
            "trust_remote_code": False,
        }
        if quantization == "4bit":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True)
            kwargs["device_map"] = "auto"
        elif torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if not trainable and quantization == "4bit" and kbit_inference_precision == "training":
            # QLoRA training upcasts non-quantized embeddings/norms/head to
            # FP32. Reproduce that path BEFORE loading an adapter, including
            # for the frozen base comparator. Casting logits after forward
            # cannot recover precision lost inside a BF16 output head.
            from peft import prepare_model_for_kbit_training
            self.model = prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=False)
        if trainable:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            if quantization == "4bit":
                self.model = prepare_model_for_kbit_training(
                    self.model, use_gradient_checkpointing=gradient_checkpointing)
            elif gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
            config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.0,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type="CAUSAL_LM")
            self.model = get_peft_model(self.model, config)
            self.model.config.use_cache = False
            # Training uses model.train() to activate checkpointing; any
            # stochastic dropout would make logged rollout probabilities
            # inconsistent with the differentiable policy distribution.
            nonzero_dropout = [(name, m.p) for name, m in self.model.named_modules()
                               if isinstance(m, torch.nn.Dropout) and m.p > 0]
            attention_dropout = float(getattr(self.model.config, "attention_dropout", 0.0))
            if nonzero_dropout or attention_dropout:
                raise RuntimeError("Action GRPO v1 requires a zero-dropout model: "
                                   + str((nonzero_dropout[:3], attention_dropout)))
        elif adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model, adapter, is_trainable=False,
                local_files_only=not allow_download)
        if not trainable:
            # Frozen reference/continuation evaluations must not retain
            # accidentally trainable pretrained base model parameters.
            self.model.requires_grad_(False)
        self.model.eval()
        self.label_ids = self._verify_action_tokens()
        self.dtype = dtype
        self.quantization = quantization

    def _verify_action_tokens(self):
        result = []
        for letter in LETTERS:
            ids = self.tok.encode(letter, add_special_tokens=False)
            if len(ids) != 1 or self.tok.decode(ids) != letter:
                raise ValueError(
                    f"Action token {letter!r} is not a single stable token; "
                    "use a model/tokenizer with exact option tokens."
                )
            result.append(ids[0])
        if len(set(result)) != len(result):
            raise ValueError("Action letters collide in tokenizer")
        return result

    def _input_ids(self, prompt):
        if self.tok.chat_template:
            encoded = self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True)
        else:
            encoded = self.tok.encode(prompt, add_special_tokens=True)
        if not encoded or len(encoded) > self.max_prompt_tokens:
            raise ValueError(
                f"Input length {len(encoded)} exceeds explicit limit "
                f"{self.max_prompt_tokens}; no truncation permitted"
            )
        device = self.model.get_input_embeddings().weight.device
        return self.torch.tensor([encoded], dtype=self.torch.long, device=device)

    def log_distribution(self, prompt, actions, *, reference=False):
        """Differentiable logπ over the valid action letters only."""
        n = len(actions)
        if not 1 <= n <= len(self.label_ids):
            raise ValueError("Invalid action count")
        if reference and not self.trainable:
            raise ValueError("A frozen base reference requires a fresh LoRA policy")
        ids = self._input_ids(prompt)
        kwargs = {"input_ids": ids, "attention_mask": self.torch.ones_like(ids),
                  "use_cache": False}
        if reference:
            with self.torch.no_grad(), self.model.disable_adapter():
                logits = self.model(**kwargs).logits[0, -1, :]
        else:
            logits = self.model(**kwargs).logits[0, -1, :]
        selected = logits[self.label_ids[:n]].float()
        result = exploratory_logprobs(
            selected, temperature=self.temperature, epsilon=self.explore_epsilon
        )
        if not self.torch.isfinite(result).all():
            raise RuntimeError("Non-finite legal-action probabilities")
        return result

    def choose(self, prompt, actions, rng: random.Random):
        self.model.eval()
        with self.torch.no_grad():
            logp = self.log_distribution(prompt, actions)
            p = logp.exp().detach().cpu().tolist()
        # Sample on a deterministic Python RNG, not a separate GPU RNG stream.
        x = rng.random()
        cdf = 0.
        chosen = len(p) - 1
        for idx, weight in enumerate(p):
            cdf += weight
            if x <= cdf:
                chosen = idx
                break
        entropy = -sum(q * math.log(max(q, 1e-30)) for q in p)
        return ActionSample(
            index=chosen, logprob=math.log(max(p[chosen], 1e-30)),
            entropy=entropy, choice_token_id=self.label_ids[chosen])

    def reference_log_distribution(self, prompt, actions):
        return self.log_distribution(prompt, actions, reference=True)

    def trainable_parameters(self):
        if not self.trainable:
            raise ValueError("Inference-only model cannot be optimized")
        return [p for p in self.model.parameters() if p.requires_grad]

    def save_adapter(self, directory: str | Path):
        if not self.trainable:
            raise ValueError("No trainable LoRA adapter to save")
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError("Refuse to overwrite an existing adapter")
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(directory), safe_serialization=True)
        (directory / "kci_runtime.json").write_text(json.dumps({
            "quantization": self.quantization,
            "compute_dtype": self.dtype,
            "kbit_inference_precision": "training" if self.quantization == "4bit" else None,
            "policy_id": self.policy_id,
        }, indent=2) + "\n", encoding="utf-8")
