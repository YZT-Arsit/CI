"""Post-launch exact repeated-prefix control; no model calls or plan changes."""
import hashlib
import json
from pathlib import Path

R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
rows=lambda p:list(map(json.loads,p.read_text().splitlines()))
new=R/'repair/replay_then_correct/midpoint';old=R.parent/'20261117_fallback_weight';previous=old/'repair/uniform'
newplan=read(R/'PLAN.json');oldplan=read(old/'PLAN.json');oldseal=read(old/'SEALED.json')
assert oldseal['complete'] and newplan['batches']==oldplan['batches']
assert newplan['schedules']['replay_then_correct'][:80]==[['replay',i] for i in range(80)]
for name in ['adapter/adapter_model.safetensors','CHECKPOINT_SEALED.json','train_orders.jsonl']:
    assert sha(previous/name)==oldseal['files']['repair/uniform/'+name]
a=read(new/'CHECKPOINT_SEALED.json');b=read(previous/'CHECKPOINT_SEALED.json')
ha=sha(new/'adapter/adapter_model.safetensors');hb=sha(previous/'adapter/adapter_model.safetensors')
assert ha==a['adapter_hashes']['adapter_model.safetensors'] and hb==b['adapter_hashes']['adapter_model.safetensors']
left=rows(new/'train_orders.jsonl');right=rows(previous/'train_orders.jsonl')
assert len(left)==len(right)==216 and [r['id'] for r in left]==[r['id'] for r in right]
result={'new_midpoint_weights':ha,'previous_uniform_weights':hb,'exact_weight_file_match':ha==hb,
        'new_steps':a['steps'],'previous_steps':b['steps'],'new_tokens':a['input_tokens'],'previous_tokens':b['input_tokens'],
        'same_train_prompts':216,'raw_output_disagreements':sum(x['raw']!=y['raw'] for x,y in zip(left,right)),
        'scope':'Identical80batch prefix control only, not new independent generalization or a whole-goal pass.'}
(R/'EXACT_REPLAY_CONTROL.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
