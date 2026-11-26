#!/usr/bin/env bash
# Execute the two requested evaluations in order using the upstream vLLM release.
set -eu -o pipefail
cd "$(dirname "$0")"
export PYTHON_BIN=/root/selfci-vllm/bin/python ENGINE=vllm
export VLLM_NO_USAGE_STATS=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
bash run_table1.sh > table1_vllm.log 2>&1
"$PYTHON_BIN" summarize_ci.py results
"$PYTHON_BIN" run_upstream_test.py 4b_initial 4b_selfci 4b_grpo 4b_ci_sft > upstream_4b.log 2>&1
BATCH=128 bash run_kci.sh > kci_vllm.log 2>&1
