#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
py=/root/selfci-vllm/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export ENGINE=vllm PYTHON_BIN="$py" BATCH=128 CONTEXT=8192 MAX_TOKENS=2048
KCI_OUT_PREFIX=kci_long bash run_kci.sh > kci_long.log 2>&1
echo $? > kci_long.exit
MODEL=Qwen2.5-7B-Instruct ADAPTER=adapters/selfci_7b KCI_OUT_PREFIX=kci7_long bash run_kci.sh > kci7_long.log 2>&1
echo $? > kci7_long.exit
for key in qwen25 qwen3 deepseek llama olmo_think olmo_instruct; do
    args=(--models "$key")
    case "$key" in
        qwen25) args+=(--arms grpo) ;;
        qwen3) args+=(--base-dir /dev/shm/selfci-bases/Qwen3-4B --kci) ;;
        deepseek|llama) args+=(--kci) ;;
    esac
    "$py" extended_table1.py "${args[@]}" > "extra_$key.log" 2>&1
    echo $? > "extra_$key.exit"
done
