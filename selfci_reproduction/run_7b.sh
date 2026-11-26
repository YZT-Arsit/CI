#!/usr/bin/env bash
set -eu -o pipefail
cd "$(dirname "$0")"
exec 9>results/.gpu.lock
flock 9
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
model=Qwen2.5-7B-Instruct
(
    cd "$model"
    sha256sum -c - <<'HASH'
a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7  model-00001-of-00004.safetensors
f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185  model-00002-of-00004.safetensors
8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5  model-00003-of-00004.safetensors
1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd  model-00004-of-00004.safetensors
HASH
) | tee results/model_7b_verification.log
for arm in initial selfci; do
    extra=()
    if [[ "$arm" == selfci ]]; then extra=(--adapter adapters/selfci_7b); fi
    "${PYTHON_BIN:-env/bin/python}" evaluate.py --model "$model" --data data/raw \
        --engine "${ENGINE:-transformers}" --batch 16 --out "results/7b_$arm" "${extra[@]}"
done
