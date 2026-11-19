#!/usr/bin/env bash
set -Eeuo pipefail
source /root/autodl-tmp/kci_rl_env/bin/activate
ROOT=/root/kci_policy_qpi_20261119
MODEL=/root/autodl-tmp/kci_models/Qwen3-4B-Instruct-2507-ms
SELECT="$ROOT/artifacts/20261119_composition_signals/agent_rl_v1/policy_qpi_01/selection_v2"
OUT="$ROOT/artifacts/20261119_composition_signals/agent_rl_v1/policy_qpi_01/gpu_qwen4b_qpi_01"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
cd "$ROOT"
echo "QPI_INFERENCE_START_UTC=$(date -u +%FT%TZ)"
if test -e "$OUT"; then echo "REFUSE_EXISTING_STUDY"; exit 2; fi
python -u tools/run_kci_policy_qpi.py --model "$MODEL" \
    --selection "$SELECT" --output "$OUT" \
    --bits4 --dtype bf16 --group-continuations 8 \
    --seed 19921119 --temperature 1.0 --explore-epsilon 0.35 \
    --max-prompt-tokens 4096
python -u tools/audit_kci_policy_qpi.py "$OUT"
echo "QPI_INFERENCE_COMPLETE_UTC=$(date -u +%FT%TZ)"
