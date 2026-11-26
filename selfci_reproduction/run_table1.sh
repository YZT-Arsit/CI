#!/usr/bin/env bash
set -eu -o pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
model=Qwen3-4B-Instruct-2507
mkdir -p results
while [[ ! -f "$model/model-00001-of-00003.safetensors" || ! -f "$model/model-00002-of-00003.safetensors" || ! -f "$model/model-00003-of-00003.safetensors" ]]; do
    sleep 15
done
(
    cd "$model"
    sha256sum -c - <<'HASH'
75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6  model-00001-of-00003.safetensors
0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1  model-00002-of-00003.safetensors
7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d  model-00003-of-00003.safetensors
HASH
) | tee results/model_verification.log
engine=${ENGINE:-transformers}
python=${PYTHON_BIN:-env/bin/python}
for arm in initial selfci grpo ci_sft; do
    extra=()
    if [[ "$arm" != initial ]]; then extra=(--adapter "adapters/$arm"); fi
    echo "$(date -u +%FT%TZ) starting $arm"
    "$python" evaluate.py --model "$model" --data data/raw --engine "$engine" --batch 24 \
        --out "results/$arm" "${extra[@]}"
    echo "$(date -u +%FT%TZ) finished $arm"
done
