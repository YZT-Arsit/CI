#!/usr/bin/env bash
set -eu -o pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
engine=${ENGINE:-transformers}
python=${PYTHON_BIN:-env/bin/python}
batch=${BATCH:-24}
model=${MODEL:-Qwen3-4B-Instruct-2507}
adapter=${ADAPTER:-adapters/selfci}
prefix=${KCI_OUT_PREFIX:-kci}
context=${CONTEXT:-4096}
max_tokens=${MAX_TOKENS:-384}
for arm in initial selfci; do
    extra=()
    if [[ "$arm" == selfci ]]; then extra=(--adapter "$adapter"); fi
    echo "$(date -u +%FT%TZ) starting KCI $arm"
    "$python" eval_kci.py generate --model "$model" \
        --prompts kci_inputs/prompts.jsonl.gz --out "results/${prefix}_$arm" \
        --engine "$engine" --batch "$batch" --context "$context" --max-tokens "$max_tokens" "${extra[@]}"
    echo "$(date -u +%FT%TZ) completed KCI $arm"
done
