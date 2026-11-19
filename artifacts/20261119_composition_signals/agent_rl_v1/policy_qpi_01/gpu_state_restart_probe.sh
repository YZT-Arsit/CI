#!/usr/bin/env bash
set -Eeuo pipefail
source /root/autodl-tmp/kci_rl_env/bin/activate
ROOT=/root/kci_state_directed_20261119
MODEL=/root/autodl-tmp/kci_models/Qwen3-4B-Instruct-2507-ms
SELECT="$ROOT/artifacts/20261119_composition_signals/agent_rl_v1/policy_qpi_01/selection_v2"
OUT="$ROOT/artifacts/20261119_composition_signals/agent_rl_v1/policy_qpi_01/state_directed_4b_probe_01"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
cd "$ROOT"
echo "STATE_RESTART_PROBE_START_UTC=$(date -u +%FT%TZ)"
if [ -e "$OUT" ]; then echo "REFUSE_EXISTING_RUN"; exit 2; fi
python -u tools/run_kci_state_directed.py probe \
 --backend hf --model "$MODEL" --bits4 --dtype bf16 \
 --selection "$SELECT" --output "$OUT" \
 --max-groups 16 --group-size 8 --probe-credit \
 --cce-rollouts 4 --cce-min-gap 0.5 --cce-min-consensus 0.75 \
 --seed 9111926 --max-prompt-tokens 4096 \
 --temperature 1.0 --explore-epsilon 0.35
python -u tools/audit_kci_state_restart.py "$OUT"
python - <<'PY'
import json
from pathlib import Path
p=Path('/root/kci_state_directed_20261119/artifacts/20261119_composition_signals/agent_rl_v1/policy_qpi_01/state_directed_4b_probe_01')
summary=json.loads((p/'SUMMARY.json').read_text())
groups=[json.loads(x) for x in (p/'groups.jsonl').read_text().splitlines()]
roles={}
for g in groups:
    k=g['restart']['role']; roles.setdefault(k,{'groups':0,'supported':0,'outcome_contrast':0})
    roles[k]['groups']+=1
    roles[k]['supported']+=g.get('state_cce_evidence',{}).get('supported_prefixes',0)
    roles[k]['outcome_contrast']+=g['outcome_signal']=='outcome_contrast'
print('STATE_RESTART_POLICY_SIGNAL',json.dumps({'groups':len(groups),'role_coverage':roles,
 'supported_distinct_states':summary['supported_counterfactual_initial_states'],
 'accepted_local_credits':summary['accepted_local_credit_decisions'],
 'frozen_policy_extra_rollouts':summary['extra_policy_continuation_rollouts'],
 'policy_updates':summary['model_trainable_gradient_steps'],
 'safe_success':summary['safe_success'],'violations':summary['violations']},ensure_ascii=False),flush=True)
PY
echo "STATE_RESTART_PROBE_COMPLETE_UTC=$(date -u +%FT%TZ)"
