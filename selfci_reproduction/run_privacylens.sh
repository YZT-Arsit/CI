#!/usr/bin/env bash
set -eu -o pipefail
cd "$(dirname "$0")"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 VLLM_NO_USAGE_STATS=1 OMP_NUM_THREADS=4
python=/root/selfci-vllm/bin/python
judge=/root/selfci-models/Mistral-7B-Instruct-v0.2
(
    cd "$judge"
    sha256sum -c - <<'HASH'
63654d601820b88b1fa8b4a98df5714f700fbc5b3df2cc4ecbabdced35096d31  model-00001-of-00003.safetensors
a42716540ecb2385d371f2109835921ff535406cac8fe8ff28f2f0b5fc7895bd  model-00002-of-00003.safetensors
5f86e15cb3ed9078e30ae6e72445e109d0e337d9cde59b9aeea4ce8e44e54a5d  model-00003-of-00003.safetensors
HASH
) | tee results/mistral_verification.log
for arm in initial selfci; do
    extra=()
    if [[ "$arm" == selfci ]]; then extra=(--adapter adapters/selfci); fi
    out="results/privacylens_$arm"
    seeds=()
    for s in 42 43 44 45 46; do
        if [[ ! -s "$out/actions_$s.jsonl" ]]; then seeds+=("$s"); fi
    done
    if (( ${#seeds[@]} )); then
        "$python" privacylens_lr.py generate --data prepared/privacylens --model Qwen3-4B-Instruct-2507 \
             --out "$out" --seeds "${seeds[@]}" "${extra[@]}"
    fi
    if [[ ! -s "$out/lr.json" ]]; then
        "$python" privacylens_lr.py judge --data prepared/privacylens --model "$judge" --out "$out"
    fi
done
