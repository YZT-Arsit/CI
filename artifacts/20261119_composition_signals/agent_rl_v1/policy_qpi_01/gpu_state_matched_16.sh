#!/usr/bin/env bash
set -Eeuo pipefail
source /root/autodl-tmp/kci_rl_env/bin/activate
ROOT=/root/kci_state_directed_matched_20261119
MODEL=/root/autodl-tmp/kci_models/Qwen3-4B-Instruct-2507-ms
SELECT="$ROOT/artifacts/20261119_composition_signals/agent_rl_v1/policy_qpi_01/selection_v2"
ART="$ROOT/artifacts/20261119_composition_signals/agent_rl_v1/policy_qpi_01"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
cd "$ROOT"
echo "STATE_MATCHED_START_UTC=$(date -u +%FT%TZ)"
for ARM in matched_terminal state_pcce; do
  OUT="$ART/state_directed_$ARM"
  if test -e "$OUT"; then echo "REFUSE_EXISTING_RUN:$OUT"; exit 2; fi
  echo "ARM_START=$ARM $(date -u +%FT%TZ)"
  if test "$ARM" = "matched_terminal"; then
    set -- --credit-mode terminal --matched-budget-teacher
  else
    set -- --credit-mode policy_cce
  fi
  python -u tools/run_kci_state_directed.py train \
    --backend hf --model "$MODEL" --bits4 --dtype bf16 \
    --selection "$SELECT" --output "$OUT" \
    --max-groups 16 --group-size 8 --dev-world-limit 12 \
    --seed 9111926 --max-prompt-tokens 4096 \
    --temperature 1.0 --explore-epsilon 0.35 \
    --cce-rollouts 4 --cce-min-gap 0.5 --cce-min-consensus 0.75 \
    --cce-weight 0.5 --lr 0.00001 --clip-eps 0.2 --kl-coef 0.02 \
    --lora-rank 8 --lora-alpha 16 --gradient-checkpointing \
    --confirm-gpu-training "$@"
  python -u tools/audit_kci_state_restart.py "$OUT"
  echo "ARM_AUDITED=$ARM $(date -u +%FT%TZ)"
done
python -u - <<'PY'
import json
from pathlib import Path
root=Path('/root/kci_state_directed_matched_20261119/artifacts/20261119_composition_signals/agent_rl_v1/policy_qpi_01')
results={}
for arm in ('matched_terminal','state_pcce'):
    p=root/('state_directed_'+arm)
    x=json.loads((p/'SUMMARY.json').read_text())
    groups=[json.loads(s) for s in (p/'groups.jsonl').read_text().splitlines()]
    metrics=('safe_success','violations','safe_failure','mean_reward','mean_decision_entropy')
    results[arm]={
        'groups':len(groups),
        'updates':x['model_trainable_gradient_steps'],
        'dev_before':{k:x['dev_before'][k] for k in metrics},
        'dev_after':{k:x['dev_after'][k] for k in metrics},
        'restart_success':x['safe_success'],
        'restart_violations':x['violations'],
        'supported_state_groups':x['supported_counterfactual_initial_states'],
        'evidence_proofs':x['accepted_local_credit_decisions'],
        'counterfactual_trajectories':x['extra_policy_continuation_rollouts'],
        'model_sampled_trajectories':x['rollouts']
    }
out=root/'MATCHED_RESULTS.json'
out.write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
print('STATE_MATCHED_RESULTS',json.dumps(results,ensure_ascii=False),flush=True)
PY
echo "STATE_MATCHED_COMPLETE_UTC=$(date -u +%FT%TZ)"
