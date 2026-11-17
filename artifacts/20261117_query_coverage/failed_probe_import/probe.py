"""Frozen paired interventions on reached downstream states, without training."""
import gc
from probe_data import R,ROOT,OLD,read,rows,sha,verify_prior
import sys
sys.path.insert(0,str(R.parent/'20261117_frozen_crossover'))
from common import load,verify,generate


def main():
    plan=read(R/'PROBE_PLAN.json');assert not (R/'probe_raw.jsonl').exists();verify_prior();verify()
    assert all(sha(R/n)==h for n,h in plan['sources'].items()) and sha(R/'probe_batches.jsonl')==plan['data_sha256']
    assert sha(OLD/'PLAN.json')==plan['prior_plan_sha256'] and sha(OLD/'SEALED.json')==plan['prior_seal_sha256']
    count=0;batches=rows(R/'probe_batches.jsonl')
    for arm in plan['arms']:
        folder=OLD/'repair'/arm;cp={'path':folder.relative_to(ROOT).as_posix(),'seal':read(folder/'CHECKPOINT_SEALED.json')}
        tokenizer,model=load('fixed',{'checkpoints':{'fixed':cp}})
        for i,batch in enumerate(batches):
            if batch[0]['arm']!=arm:continue
            raw,seconds=generate(tokenizer,model,[r['prompt'] for r in batch])
            with (R/'probe_raw.jsonl').open('a',encoding='utf-8') as f:
                for j,value in enumerate(raw):f.write(json.dumps({'batch':i,'index':j,'raw':value,'seconds_batch':seconds})+'\n');count+=1
        print(json.dumps({'arm':arm,'cumulative_responses':count}),flush=True);del model;gc.collect()
    assert count==plan['responses']
    (R/'PROBE_SEALED.json').write_text(json.dumps({'plan_sha256':sha(R/'PROBE_PLAN.json'),'raw_sha256':sha(R/'probe_raw.jsonl'),'responses':count,'model_updates':0},indent=2))


if __name__=='__main__':main()
