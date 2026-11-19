#!/usr/bin/env bash
set -Eeuo pipefail
source /root/autodl-tmp/kci_rl_env/bin/activate
ROOT=/root/kci_policy_cce_20261119
MODEL=/root/autodl-tmp/kci_models/Qwen3-4B-Instruct-2507-ms
OUT="$ROOT/artifacts/20261119_composition_signals/agent_rl_v1/policy_cce_8_01"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
cd "$ROOT"
echo "PCCE_8_START_UTC=$(date -u +%FT%TZ)"
if test -e "$OUT"; then echo "REFUSE_EXISTING_RUN"; exit 2; fi
python -u tools/run_kci_agent_rl.py train --backend hf --model "$MODEL" \
 --bits4 --dtype bf16 --gradient-checkpointing \
 --credit-mode policy_cce --policy-cce-rollouts 4 \
 --policy-cce-min-gap 0.5 --policy-cce-min-consensus 0.75 \
 --policy-cce-max-prefixes 4 --cce-weight 0.5 \
 --max-updates 8 --group-size 8 --dev-world-limit 12 \
 --seed 9111926 --max-prompt-tokens 4096 \
 --temperature 1.0 --explore-epsilon 0.35 \
 --lora-rank 8 --lora-alpha 16 --lr 0.00001 \
 --clip-eps 0.2 --kl-coef 0.02 \
 --confirm-gpu-training --output "$OUT"
python -u tools/audit_kci_agent_rl.py "$OUT"
python -u tools/audit_kci_policy_cce.py "$OUT"
python - <<'PY'
import json
from pathlib import Path
p=Path("/root/kci_policy_cce_20261119/artifacts/20261119_composition_signals/agent_rl_v1/policy_cce_8_01")
s=json.loads((p/"SUMMARY.json").read_text())
g=[json.loads(x) for x in (p/"groups.jsonl").read_text().splitlines()]
brief={"before":{k:s["before"][k] for k in ("rollouts","safe_success","violations","safe_failure","mean_reward")},
"after":{k:s["after"][k] for k in ("rollouts","safe_success","violations","safe_failure","mean_reward")},
"attempted_groups":len(g),
"optimizer_updates":s["optimizer_steps"],
"supported_policy_CCE_proofs":sum(x.get("policy_cce_proof_count",0) for x in g),
"queried_prefixes":sum(x.get("policy_cce_evidence",{}).get("queried_prefixes",0) for x in g),
"supported_prefixes":sum(x.get("policy_cce_evidence",{}).get("supported_prefixes",0) for x in g),
"frozen_continuation_rollouts":sum(x.get("policy_cce_evidence",{}).get("counterfactual_rollouts",0) for x in g),
"nonzero_gradient_updates":sum(x.get("pre_clip_gradient_norm",0)>0 for x in g)}
print("PCCE_8_GATE",json.dumps(brief,ensure_ascii=False),flush=True)
PY
echo "PCCE_8_COMPLETE_UTC=$(date -u +%FT%TZ)"
