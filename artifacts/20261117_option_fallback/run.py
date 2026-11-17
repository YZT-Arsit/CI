"""Run only frozen actor order interventions; no label file opened."""
import hashlib
import json
import sys
from pathlib import Path

R=Path(__file__).resolve().parent;ROOT=R.parents[1]
sys.path.insert(0,str(R.parent/'20261117_frozen_crossover'))
from common import verify, setup, set_peft_model_state_dict, load_file, generate
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    plan=json.loads((R/'PLAN.json').read_text());assert not (R/'RUN_PLAN.json').exists()
    assert all(sha(R/n)==h for n,h in plan['sources'].items())
    assert sha(R/'observations.jsonl')==plan['data_hashes']['observations.jsonl']
    assert all(sha(ROOT/n)==h for n,h in plan['dependencies'].items())
    verify()
    checkpoint=ROOT/plan['checkpoint'];seal=plan['checkpoint_seal']
    assert json.loads((checkpoint/'CHECKPOINT_SEALED.json').read_text())==seal
    assert all(sha(checkpoint/'adapter'/p)==h for p,h in seal['adapter_hashes'].items())
    tokenizer,model=setup(17);set_peft_model_state_dict(model,load_file(str(checkpoint/'adapter/adapter_model.safetensors')));model.eval()
    cases=list(map(json.loads,(R/'observations.jsonl').read_text().splitlines()))
    (R/'RUN_PLAN.json').write_text(json.dumps({'plan_sha256':sha(R/'PLAN.json'),'oracle_estimates':True,'reads_action_labels':False},indent=2))
    for start in range(0,len(cases),4):
        batch=cases[start:start+4];raws,seconds=generate(tokenizer,model,[c['prompt'] for c in batch])
        with (R/'responses.jsonl').open('a') as f:
            for case,raw in zip(batch,raws):f.write(json.dumps({'id':case['id'],'raw':raw,'seconds_batch':seconds})+'\n')
        if (start+4)%48==0:print(json.dumps({'completed':start+4}),flush=True)
    (R/'SEALED.json').write_text(json.dumps({'complete':True,'files':{'responses.jsonl':sha(R/'responses.jsonl')},'labels_scored':False},indent=2))


if __name__=='__main__':main()
