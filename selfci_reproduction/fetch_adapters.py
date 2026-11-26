"""Fetch only inference weights and provenance for the named author checkpoints."""
import hashlib
import json
from pathlib import Path

import requests


CHECKPOINTS = {
    "selfci_7b": ("feedback_asymmetric_fixed_ema_Qwen2.5-7B-Instruct_bw0p5_fw0p5_ema0p999_ep30", "f2e3a91ef377e4dc1566535eb016cb8c9aa215c0"),
    "selfci": ("feedback_asymmetric_fixed_ema_Qwen3-4B-Instruct-2507_bw0p5_fw0p5_ema0p999_ep30", "1e7f55d3a1b026bf1d606d23f6f5a9d13ebe9e1e"),
    "grpo": ("grpo_Qwen3-4B-Instruct-2507_bs16_g16_mb128_lr1e-6_b1e-3_clip0p2_temp0p7_ep30", "be8c902815b98e3de54207983f0e460baeabb55c"),
    "ci_sft": ("ci-sft_Qwen3-4B-Instruct-2507_lr1e-6_ep30", "4adfb522e9c2b933201ae6b854dd721280201db7"),
}


def fetch(label, name, revision):
    url = f"https://huggingface.co/Sangsang/{name}"
    meta = requests.get(f"https://huggingface.co/api/models/Sangsang/{name}/tree/{revision}", timeout=60)
    meta.raise_for_status()
    files = {x["path"]: x for x in meta.json()}
    folder = Path(__file__).resolve().parent / "adapters" / label
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for filename in ("adapter_config.json", "adapter_model.safetensors", "trainer_state.json", "README.md",
                     "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                     "chat_template.jinja", "merges.txt", "vocab.json", "tokenizer.json"):
        if filename not in files and filename in {"chat_template.jinja", "added_tokens.json"}:
            continue
        details = files[filename]
        path = folder / filename
        digest = details.get("lfs", {}).get("oid")
        if not path.exists() or (digest and hashlib.sha256(path.read_bytes()).hexdigest() != digest):
            response = requests.get(f"{url}/resolve/{revision}/{filename}?download=true", timeout=120)
            response.raise_for_status()
            if len(response.content) != details["size"]:
                raise ValueError(f"Wrong size for {filename}")
            if digest and hashlib.sha256(response.content).hexdigest() != digest:
                raise ValueError(f"Wrong SHA-256 for {filename}")
            path.write_bytes(response.content)
        manifest[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    (folder / "source.json").write_text(json.dumps(dict(url=url, revision=revision, sha256=manifest), indent=2))
    print(label, manifest["adapter_model.safetensors"], flush=True)


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(3) as pool:
        jobs = [pool.submit(fetch, label, *source) for label, source in CHECKPOINTS.items()]
        for job in jobs:
            job.result()
