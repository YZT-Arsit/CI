#!/usr/bin/env bash
set -Eeuo pipefail
source /root/autodl-tmp/kci_rl_env/bin/activate
ROOT=/root/kci_state_restart_20261119
MODEL=/root/autodl-tmp/kci_models/Qwen3-4B-Instruct-2507-ms
SEL="$ROOT/artifacts/20261119_composition_signals/agent_rl_v1/policy_qpi_01/selection_v2"
OUT="$ROOT/artifacts/20261119_composition_signals/agent_rl_v1/state_restart_probe_16_01"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
cd "$ROOT"
echo "STATE_RESTART_PROBE_START=$(date -u +%FT%TZ)"
if test -e "$OUT"; then echo "REFUSE_OVERWRITE";exit 2;fi
python -u tools/run_kci_state_directed.py probe \
 --backend hf --model "$MODEL" --bits4 --dtype bf16 \
 --selection "$SEL" --max-groups 16 --group-size 8 \
 --probe-credit --cce-rollouts 4 --cce-min-gap 0.5 \
 --cce-min-consensus 0.75 --seed 9111926 \
 --temperature 1.0 --explore-epsilon 0.35 \
 --max-prompt-tokens 4096 --output "$OUT"
python -u tools/audit_kci_state_restart.py "$OUT"
python - <<'PY'
import json
from pathlib import Path
p=Path("/root/kci_state_restart_20261119/artifacts/20261119_composition_signals/agent_rl_v1/state_restart_probe_16_01")
s=json.loads((p/"SUMMARY.json").read_text())
g=[json.loads(line) for line in (p/"groups.jsonl").read_text().splitlines()]
role={}
for x in g:
    v=role.setdefault(x["restart"]["role"],{"n":0,"supported":0,"outcome_contrast":0})
    v["n"]+=1
    v["supported"]+=x["state_cce_evidence"].get("supported_prefixes",0)
    v["outcome_contrast"]+=x["outcome_signal"]=="outcome_contrast"
print("STATE_PROBE_GATE",json.dumps({
 "groups":len(g),"roles":role,
 "outcome_contrast_groups":s["outcome_contrast_groups"],
 "supported_restart_states":s["supported_counterfactual_initial_states"],
 "accepted_local_credit_decisions":s["accepted_local_credit_decisions"],
 "frozen_policy_branch_rollouts":s["extra_policy_continuation_rollouts"],
 "success":s["safe_success"],"violations":s["violations"],
 "model_training_occurred":False
},ensure_ascii=False),flush=True)
PY
echo "STATE_RESTART_PROBE_COMPLETE=$(date -u +%FT%TZ)"
