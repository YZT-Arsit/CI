"""Invoke the pinned, unmodified author evaluator, including its adapter merge and seed loop."""
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
ARMS = {
    "4b_initial": ("Qwen3-4B-Instruct-2507", None),
    "4b_selfci": ("Qwen3-4B-Instruct-2507", "selfci"),
    "4b_grpo": ("Qwen3-4B-Instruct-2507", "grpo"),
    "4b_ci_sft": ("Qwen3-4B-Instruct-2507", "ci_sft"),
    "7b_initial": ("Qwen2.5-7B-Instruct", None),
    "7b_selfci": ("Qwen2.5-7B-Instruct", "selfci_7b"),
}


def main():
    os.chdir(ROOT)
    for arm in sys.argv[1:] or ARMS:
        out = ROOT / "results" / "upstream" / arm
        if (out / "metrics.json").exists():
            rows = json.loads((out / "responses.json").read_bytes())
            assert len(rows) == 365
            continue
        backbone, adapter = ARMS[arm]
        ref = ROOT / backbone
        if adapter:
            source = ROOT / "adapters" / adapter
            offline = ROOT / "offline_checkpoints" / (backbone + "_" + arm)
            offline.mkdir(parents=True, exist_ok=True)
            for filename in ("adapter_model.safetensors", "tokenizer_config.json", "tokenizer.json",
                             "special_tokens_map.json", "vocab.json", "merges.txt",
                             "added_tokens.json", "chat_template.jinja"):
                if (source / filename).exists() and not (offline / filename).exists():
                    os.link(source / filename, offline / filename)
            config = json.loads((source / "adapter_config.json").read_bytes())
            config["base_model_name_or_path"] = str(ref)
            (offline / "adapter_config.json").write_text(json.dumps(config))
            ref = offline
        subprocess.run([sys.executable, "upstream/run_test.py", "--model-name-or-path", str(ref),
                        "--dataset-name-or-path", "data/raw", "--num-runs", "5", "--vllm-seed", "42",
                        "--temperature", "0.7", "--top-p", "1.0", "--max-output-length", "2048",
                        "--vllm-max-model-len", "8192", "--output-dir", str(out)], check=True)


if __name__ == "__main__":
    main()
