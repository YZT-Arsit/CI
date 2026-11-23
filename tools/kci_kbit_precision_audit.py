"""Frozen TRAIN-only precision/reload probe; no optimization or held-out inputs."""
from pathlib import Path
from hashlib import sha256
import argparse
import gc
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "tools")]


def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "adapter", "train", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    occupancy = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"
    ], text=True).strip()
    if occupancy:
        raise RuntimeError("GPU occupied; audit did not load a model: " + occupancy)
    rows = [json.loads(line) for line in Path(args.train).read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 9 and {r["root"] for r in rows} == {"DG31", "DG32", "DG33"}
    assert all(r["source_split"] == "train" for r in rows)
    import torch
    from peft import prepare_model_for_kbit_training
    from kci.agent_rl.hf_policy import HFActionPolicy
    from kci_v4_semantic_action import scoring
    from kci_v4_pairwise_strict_gate import exact_action

    def evaluate(hf):
        records = []
        hf.model.eval()
        with torch.inference_mode():
            for row in rows:
                actions, scores = scoring(hf, row["prompt"], row["original_action_order"])
                scores = dict(zip(actions, scores.float().cpu().tolist()))
                choice = exact_action(scores)
                records.append(dict(root=row["root"], condition=row["condition"],
                                    gold=row["chosen_action"], prediction=choice,
                                    correct=choice == row["chosen_action"], scores=scores))
        return records

    start = time.monotonic()
    kwargs = dict(model_path=args.model, adapter=args.adapter, quantization="4bit", dtype="bf16")
    hf = HFActionPolicy(**kwargs, kbit_inference_precision="legacy")
    head_before = str(hf.model.get_output_embeddings().weight.dtype)
    legacy = evaluate(hf)
    prepare_model_for_kbit_training(hf.model, use_gradient_checkpointing=False)
    head_after = str(hf.model.get_output_embeddings().weight.dtype)
    matched = evaluate(hf)
    del hf
    gc.collect()
    torch.cuda.empty_cache()
    hf = HFActionPolicy(**kwargs)
    reloaded = evaluate(hf)
    parity = max(abs(x["scores"][a] - y["scores"][a])
                 for x, y in zip(matched, reloaded) for a in x["scores"])
    change = max(abs(x["scores"][a] - y["scores"][a])
                 for x, y in zip(legacy, matched) for a in x["scores"])
    result = dict(scope="frozen_checkpoint_TRAIN_only_not_method_efficacy",
                  training_input_sha256=sha256(Path(args.train).read_bytes()).hexdigest(),
                  adapter_sha256=sha256((Path(args.adapter)/"adapter_model.safetensors").read_bytes()).hexdigest(),
                  head_dtype_before=head_before, head_dtype_after=head_after,
                  max_legacy_to_matched_score_change=change,
                  max_matched_reload_score_error=parity,
                  matched_reload_pass=parity <= 1e-5,
                  strict_correct={k:sum(r["correct"] for r in v) for k,v in
                                  (("legacy",legacy),("matched",matched),("reloaded",reloaded))},
                  changed_decisions=sum(x["prediction"] != y["prediction"] for x,y in zip(legacy,matched)),
                  legacy=legacy, matched=matched, reloaded=reloaded,
                  candidate_forwards=81, elapsed_seconds=time.monotonic()-start)
    output.mkdir(parents=True)
    (output/"PRECISION_AUDIT.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("legacy","matched","reloaded")}))
    if not result["matched_reload_pass"]:
        raise RuntimeError("Matched inference reload parity failed")


if __name__ == "__main__":
    main()
