#!/usr/bin/env bash
set -eu -o pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
model=Qwen3-4B-Instruct-2507
env/bin/python train_probe.py --engine transformers --probe-steps 8 \
    --student-model-name-or-path "$model" --teacher-model-name-or-path "$model" \
    --dataset-name-or-path verified_remote --output-dir results/training_probe \
    --epochs 30 --batch-size 1 --gradient-accumulation-steps 2 --logging-steps 1 \
    --feedback-weight-type fixed --backward-weight 0.5 --forward-weight 0.5 \
    --teacher-update ema --ema-decay 0.999 --kl-support full --max-completion-length 2048 \
    --disable-wandb --eval-strategy no --save-strategy no
