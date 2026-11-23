"""CPU regression for the non-quantized layers of the QLoRA load path.

Real tiny Qwen2/PEFT save and reload; only CUDA/4-bit weight loading is
substituted. This does not establish parity of actual NF4 CUDA kernels.
"""
from pathlib import Path
import sys

import pytest
import torch

pytest.importorskip("peft")
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen2Config,
    Qwen2ForCausalLM,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kci.agent_rl.hf_policy import HFActionPolicy


@pytest.fixture
def tiny_base(tmp_path, monkeypatch):
    torch.manual_seed(73)
    folder = tmp_path / "base"
    tokenizer = Tokenizer(WordLevel(
        {"[UNK]": 0, "[EOS]": 1, **{chr(65+i): 2+i for i in range(26)}},
        unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                 unk_token="[UNK]", eos_token="[EOS]")
    tok.save_pretrained(folder)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=28, hidden_size=32, intermediate_size=48,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=128, attention_dropout=0.0))
    model.to(torch.bfloat16).save_pretrained(folder)
    original_load = AutoModelForCausalLM.from_pretrained

    def cpu_nonquantized_layers(*args, **kwargs):
        kwargs.pop("quantization_config", None)
        kwargs.pop("device_map", None)
        loaded = original_load(*args, **kwargs)
        loaded.is_loaded_in_4bit = True
        return loaded

    # The loader is external and requires CUDA. All layer arithmetic and
    # adapter serialization below use real transformers and PEFT code.
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", cpu_nonquantized_layers)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    import peft.utils.save_and_load as peft_io
    import peft.peft_model as peft_model
    monkeypatch.setattr(peft_io, "infer_device", lambda: "cpu")
    monkeypatch.setattr(peft_model, "infer_device", lambda: "cpu")
    # BitsAndBytesConfig only checks the package version here; the above
    # loader deliberately does not instantiate any quantized linear layer.
    import transformers.utils.quantization_config as qc
    real_version = qc.importlib.metadata.version
    monkeypatch.setattr(qc.importlib.metadata, "version",
                        lambda name: "0.49.0" if name == "bitsandbytes" else real_version(name))
    return str(folder)


def test_saved_qlora_preserves_action_probabilities_on_reload(tiny_base, tmp_path):
    before = HFActionPolicy(model_path=tiny_base, trainable=True,
                            quantization="4bit", dtype="bf16")
    with torch.no_grad():
        for name, parameter in before.model.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.05)
        expected = before.log_distribution("A B C", ("raw", "verify"))
    adapter = tmp_path / "adapter"
    before.save_adapter(adapter)
    after = HFActionPolicy(model_path=tiny_base, adapter=str(adapter),
                           quantization="4bit", dtype="bf16")
    with torch.no_grad():
        actual = after.log_distribution("A B C", ("raw", "verify"))
    assert torch.allclose(actual, expected, atol=1e-6, rtol=0), (expected, actual)
    assert not any(p.requires_grad for p in after.model.parameters())
    legacy = HFActionPolicy(model_path=tiny_base, adapter=str(adapter),
                            quantization="4bit", dtype="bf16",
                            kbit_inference_precision="legacy")
    with torch.no_grad():
        old_path = legacy.log_distribution("A B C", ("raw", "verify"))
    assert not torch.allclose(old_path, expected, atol=1e-6, rtol=0)
    assert legacy.policy_id != after.policy_id


def test_frozen_base_matches_adapter_disabled_reference(tiny_base):
    training = HFActionPolicy(model_path=tiny_base, trainable=True,
                              quantization="4bit", dtype="bf16")
    frozen = HFActionPolicy(model_path=tiny_base, quantization="4bit", dtype="bf16")
    with torch.no_grad():
        expected = training.reference_log_distribution("A B C", ("raw", "verify"))
        actual = frozen.log_distribution("A B C", ("raw", "verify"))
    assert torch.allclose(actual, expected, atol=1e-6, rtol=0), (expected, actual)
