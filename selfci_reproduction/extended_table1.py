"""Stage pinned public files and run the original SelfCI test entry point, a backbone at a time."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import requests

ROOT = Path(__file__).resolve().parent
MODEL_FILES = json.loads((ROOT / "extended_files.json").read_text(encoding="utf-8"))
ASSETS = ROOT / "extended_assets"


def verify(path, info):
    size = path.stat().st_size
    if size != info["bytes"]:
        raise ValueError(f"Size mismatch at {path}: {size} != {info['bytes']}")
    if "sha256" in info:
        digest, expected = hashlib.sha256(), info["sha256"]
    else:
        digest, expected = hashlib.sha1(f"blob {size}\0".encode()), info["git_blob_sha1"]
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(8 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"Pinned digest mismatch at {path}")


def fetch(url, dest, info):
    if dest.exists():
        verify(dest, info)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fetch-", dir=dest.parent) as temp:
        part = Path(temp) / dest.name
        for attempt in range(3):
            try:
                start = part.stat().st_size if part.exists() else 0
                if start != info["bytes"]:
                    headers = {"Range": f"bytes={start}-"} if start else {}
                    with requests.get(url, headers=headers, stream=True, timeout=(20, 120)) as r:
                        r.raise_for_status()
                        if r.status_code == 206 and not r.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                            raise ValueError(f"Unexpected byte range for {dest.name}")
                        with part.open("ab" if start and r.status_code == 206 else "wb") as f:
                            for data in r.iter_content(4 << 20):
                                f.write(data)
                verify(part, info)
                os.link(part, dest)
                return
            except FileExistsError:
                verify(dest, info)
                return
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)


def fetch_adapter(job):
    key, arm, spec = job
    folder = ASSETS / "adapters" / key / arm
    for name, info in spec["files"].items():
        path = folder / name
        existing = ROOT / "adapters/selfci_7b" / name
        if key == "qwen25" and arm == "selfci" and existing.exists() and not path.exists():
            verify(existing, info)
            path.parent.mkdir(parents=True, exist_ok=True)
            os.link(existing, path)
        fetch(f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{name}", path, info)
    print(f"Verified {key}/{arm} at {folder}", flush=True)


def stage_base(key, cfg, dest, existing):
    base = cfg["base"]
    dest.mkdir()
    missing_bytes = sum(v["bytes"] for n, v in base["files"].items()
                        if not (existing / n).is_file() and not v.get("staged_local"))
    mirror = base.get("public_modelscope_repo", base["repo"] if not base["gated"] else None)
    if missing_bytes and not mirror:
        raise RuntimeError(f"{base['repo']} requires an authorized local directory with all pinned files")
    if shutil.disk_usage(dest).free < missing_bytes:
        raise RuntimeError(f"Insufficient staging space for {key}: {missing_bytes:,} bytes needed")
    for name, info in base["files"].items():
        source = existing / name
        if not source.exists() and info.get("staged_local"):
            source = ASSETS / "base_metadata" / key / name
            if not source.exists():
                raise FileNotFoundError(source)
        if source.exists():
            verify(source, info)
            (dest / name).symlink_to(source.resolve())
        else:
            rev = info.get("modelscope_revision")
            if not rev or not mirror:
                raise ValueError(f"No approved matching public mirror for {base['repo']}/{name}")
            url = f"https://modelscope.cn/models/{mirror}/resolve/{rev}/{name}"
            fetch(url, dest / name, info)


def verify_output(folder, max_tokens):
    metric = json.loads((folder / "metrics.json").read_bytes())
    rows = json.loads((folder / "responses.json").read_bytes())
    expected = {(run, idx) for run in range(1, 6) for idx in range(73)}
    if len(rows) != 365 or {(r["run_index"], r["id"]) for r in rows} != expected:
        raise ValueError(f"Incomplete five-run output at {folder}")
    for name, expected_value in {"num_runs": 5, "vllm_seed": 42, "num_examples": 73, "temperature": 0.7,
                                 "top_p": 1.0, "max_output_length": max_tokens, "prompt_style": "ci-cot"}.items():
        if metric[name] != expected_value:
            raise ValueError(f"Unexpected {name} at {folder}")


def run_arm(key, arm, cfg, base, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    spec = cfg["arms"].get(arm)
    with tempfile.TemporaryDirectory(prefix="selfci-merge-", dir="/dev/shm") as tmp:
        ref = base
        if spec:
            required = sum(v["bytes"] for k, v in cfg["base"]["files"].items() if k.endswith(".safetensors"))
            if shutil.disk_usage(tmp).free < required + (128 << 20):
                raise RuntimeError(f"/dev/shm needs approximately {required:,} bytes for the official merge")
            ref = Path(tmp) / (base.name + "_" + arm)
            ref.mkdir()
            source = ASSETS / "adapters" / key / arm
            for name, details in spec["files"].items():
                item = source / name
                verify(item, details)
                if name == "adapter_config.json":
                    config = json.loads(item.read_bytes())
                    if config["base_model_name_or_path"] != cfg["base"]["repo"]:
                        raise ValueError(f"Adapter/base mismatch: {source}")
                    config["base_model_name_or_path"] = str(base)
                    (ref / name).write_text(json.dumps(config), encoding="utf-8")
                else:
                    (ref / name).symlink_to(item.resolve())
        cmd = [sys.executable, str(ROOT / "upstream/run_test.py"), "--model-name-or-path", str(ref),
               "--dataset-name-or-path", str(ROOT / "data/raw"), "--max-output-length", str(cfg["max_output_length"]),
               "--vllm-max-model-len", "8192", "--num-runs", "5", "--vllm-seed", "42", "--output-dir", str(attempt)]
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
               "VLLM_NO_USAGE_STATS": "1", "TOKENIZERS_PARALLELISM": "false"}
        provenance = dict(base=cfg["base"]["repo"], base_revision=cfg["base"]["revision"], arm=arm,
                          adapter=spec["repo"] if spec else None, adapter_revision=spec["revision"] if spec else None,
                          status=spec["evidence_status"] if spec else "initial", command=cmd)
        (attempt / "source.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        print("Launching", key, arm, "log", attempt / "evaluation.log", flush=True)
        with (attempt / "evaluation.log").open("x", encoding="utf-8") as log:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    verify_output(attempt, cfg["max_output_length"])
    attempt.rename(output)
    print("Verified", output, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", choices=MODEL_FILES, default=list(MODEL_FILES))
    p.add_argument("--arms", nargs="+", choices=["initial", "selfci", "grpo", "sft_candidate"])
    p.add_argument("--fetch-adapters", action="store_true", help="Download and verify author adapter/tokenizer files locally")
    p.add_argument("--staging-dir", type=Path, default=ROOT, help="Parent for newly downloaded one-model working directories")
    p.add_argument("--base-dir", type=Path, help="Already available base model directory; use with exactly one model")
    p.add_argument("--kci", action="store_true", help="Evaluate the same base and public student on KCI after CI-RL")
    args = p.parse_args()
    if args.base_dir and len(args.models) != 1:
        p.error("--base-dir requires a single selected model")
    if args.fetch_adapters:
        from concurrent.futures import ThreadPoolExecutor
        jobs = [(key, arm, spec) for key in args.models for arm, spec in MODEL_FILES[key]["arms"].items()
                if not args.arms or arm in args.arms]
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(fetch_adapter, jobs))
        return
    if not Path("/dev/shm").is_dir():
        p.error("The evaluator requires Linux /dev/shm")
    args.staging_dir.mkdir(parents=True, exist_ok=True)
    for key in args.models:
        cfg = MODEL_FILES[key]
        backbone = cfg["base"]["repo"].split("/")[-1]
        source = (args.base_dir or ROOT / backbone).resolve()
        arms = args.arms or ["initial", *cfg["arms"]]
        pending = []
        for arm in arms:
            if arm != "initial" and arm not in cfg["arms"]:
                continue
            label = f"7b_{arm}" if key == "qwen25" and arm in {"initial", "selfci"} else f"{key}_{arm}"
            output = ROOT / "results/upstream" / label
            if output.exists():
                verify_output(output, cfg["max_output_length"])
                print("Using existing", output, flush=True)
            else:
                pending.append((arm, output))
        do_kci = args.kci and "selfci" in cfg["arms"] and key != "qwen25"
        if not pending and not do_kci:
            continue
        if cfg["base"]["gated"] and not cfg["base"].get("public_modelscope_repo") and not source.is_dir() and not args.base_dir:
            print(f"Skipped {key}: gated base not staged.", flush=True)
            continue
        with tempfile.TemporaryDirectory(prefix="selfci-base-", dir=args.staging_dir) as temp:
            base = Path(temp).resolve() / backbone
            stage_base(key, cfg, base, source)
            for arm, output in pending:
                run_arm(key, arm, cfg, base, output)
            if do_kci:
                prefix = f"kci_{key}"
                for name in ("initial", "selfci"):
                    out = ROOT / "results" / f"{prefix}_{name}"
                    cmd = [sys.executable, str(ROOT / "eval_kci.py"), "generate", "--model", str(base),
                           "--prompts", str(ROOT / "kci_inputs/prompts.jsonl.gz"), "--out", str(out),
                           "--engine", "vllm", "--batch", "128", "--context", "4096"]
                    if name == "selfci":
                        cmd += ["--adapter", str(ASSETS / "adapters" / key / "selfci")]
                    env = {**os.environ, "KCI_MERGE_DIR": "/dev/shm",
                           "KCI_MODEL_ID": f"{cfg['base']['repo']}@{cfg['base']['revision']}"}
                    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
