"""Offline CI-RL evaluation using the pinned SelfCI prompt builder and scoring functions."""
import argparse
import gc
import hashlib
import json
import sys
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def scored_row(official, example, index, model, text, tokens, finish):
    repaired = official._repair_response_text(text)
    post_think = example.get("use_post_think_response") or official.should_use_post_think_response(
        model_name_or_path=model, text=repaired
    )
    answer, parsed = official._extract_answer_text(repaired, use_post_think_response=post_think)
    return dict(
        id=index, prompt=example["prompt"], response_raw=text, response=repaired,
        completion_tokens=tokens, finish_reason=finish, evaluation_text=answer, evaluation_parsed=parsed,
        required_keywords=official._extract_keywords(example, "required_keywords", "allowed_keywords"),
        restricted_keywords=official._extract_keywords(example, "restricted_keywords", "disallowed_keywords"),
    )


def make_generator(args, stack):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_ref = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_ref, padding_side="left", local_files_only=True)
    if args.engine == "transformers" or args.adapter:
        device = "cpu" if args.engine == "vllm" else "cuda:0"
        model = AutoModelForCausalLM.from_pretrained(
            model_ref, dtype=torch.bfloat16, device_map={"": device}, local_files_only=True
        )
        if args.adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()

    if args.engine == "vllm":
        from vllm import LLM, SamplingParams

        if args.adapter:
            merged = stack.enter_context(TemporaryDirectory(dir=args.out, prefix="merged-"))
            model.save_pretrained(merged)
            tokenizer.save_pretrained(merged)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            model_ref = merged
        llm = LLM(model=model_ref, tensor_parallel_size=1, gpu_memory_utilization=0.8,
                  max_model_len=args.context, dtype="bfloat16", seed=42)

        def generate(prompts, seed):
            sampling = SamplingParams(n=1, seed=seed, temperature=0.7, top_p=1.0, max_tokens=args.max_tokens)
            for request in llm.generate(prompts, sampling_params=sampling):
                result = request.outputs[0]
                yield result.text, len(result.token_ids), result.finish_reason

        return generate

    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos = model.generation_config.eos_token_id
    stop_ids = set(eos if isinstance(eos, list) else [eos])

    @torch.inference_mode()
    def generate(prompts, seed):
        torch.manual_seed(seed)
        for start in range(0, len(prompts), args.batch):
            texts = prompts[start:start + args.batch]
            inputs = tokenizer(texts, padding=True, add_special_tokens=False, return_tensors="pt").to("cuda:0")
            width = inputs.input_ids.shape[1]
            if width + args.max_tokens > args.context:
                raise ValueError(f"Prompt width {width} exceeds the configured generation context.")
            output = model.generate(
                **inputs, do_sample=True, temperature=0.7, top_p=1.0, top_k=0, repetition_penalty=1.0,
                max_new_tokens=args.max_tokens, pad_token_id=tokenizer.pad_token_id,
            )
            for ids in output[:, width:].tolist():
                stop = next((i for i, token_id in enumerate(ids) if token_id in stop_ids), None)
                if stop is not None:
                    ids = ids[:stop + 1]
                yield tokenizer.decode(ids, skip_special_tokens=True), len(ids), "stop" if stop is not None else "length"
            print(f"seed {seed} generated {min(start + args.batch, len(prompts))}/{len(prompts)}", flush=True)

    return generate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent / "upstream")
    p.add_argument("--data", required=True, help="Directory containing the original raw train parquet")
    p.add_argument("--model", required=True, help="Local base model directory including 'Instruct' for instruction models")
    p.add_argument("--adapter", help="Optional local LoRA directory to merge into the base before inference")
    p.add_argument("--engine", choices=["vllm", "transformers"], default="vllm")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", choices=["eval", "test"], default="test")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    p.add_argument("--batch", type=int, default=8, help="Transformers batch size")
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--max-tokens", type=int, default=2048)
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.max_tokens, args.batch) < 1:
        p.error("Seeds must be distinct and batch/max-tokens positive.")
    sys.path.insert(0, str(args.repo.resolve()))
    import run_test as official

    args.model = str(Path(args.model).resolve(strict=True))
    rows = list(official.prepare_grpo_dataset_splits_from_hf(
        dataset_path=args.data, model_id=args.model, split_seed=42, shuffle=True, prompt_style="ci-cot"
    )[args.split])
    assert len(rows) == {"eval": 66, "test": 73}[args.split]
    prompts = [r["prompt"] for r in rows]
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = dict(model=args.model, adapter=args.adapter, engine=args.engine, split=args.split,
                    batch=args.batch, context=args.context, max_tokens=args.max_tokens, temperature=0.7, top_p=1.0,
                    versions={k: version(k) for k in {"torch", "transformers", "datasets", args.engine}},
                    rows_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest())
    if args.adapter:
        with (Path(args.adapter) / "adapter_model.safetensors").open("rb") as f:
            manifest["adapter_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Output directory is bound to a different evaluation configuration.")
    save_json(manifest_path, manifest)
    per_run = []
    with ExitStack() as stack:
        generate = None
        for seed in args.seeds:
            path = args.out / f"seed-{seed}.json"
            if path.exists():
                responses = json.loads(path.read_text(encoding="utf-8"))
                if [r["prompt"] for r in responses] != prompts:
                    raise ValueError(f"Saved prompts mismatch in {path}")
            else:
                if generate is None:
                    generate = make_generator(args, stack)
                responses = [
                    scored_row(official, row, i, args.model, *result)
                    for i, (row, result) in enumerate(zip(rows, generate(prompts, seed), strict=True))
                ]
            metrics = official._compute_metrics(responses)
            if not path.exists():
                save_json(path, responses)
            per_run.append(dict(seed=seed, **metrics, parsed=sum(r["evaluation_parsed"] for r in responses),
                                length_capped=sum(r["finish_reason"] == "length" for r in responses)))
            print(per_run[-1], flush=True)
    summary = official._summarize_run_metrics(per_run)
    save_json(args.out / "metrics.json", dict(num_examples=len(rows), num_runs=len(per_run),
                                              per_run_metrics=per_run, **summary))


if __name__ == "__main__":
    main()
