"""Post-launch diagnostic: change only initial Planner estimates to truth."""
import gc
import hashlib
import json
import sys
from pathlib import Path

R=Path(__file__).resolve().parent;ROOT=R.parents[1]
sys.path.insert(0,str(R.parent/'20261117_frozen_crossover'))
from common import verify,load,palette,generate,actor_text,seed_for
from worlds import targets
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
rows=lambda p:list(map(json.loads,p.read_text().splitlines()))


def main():
    plan=read(R/'PLANNER_PROBE_PLAN.json');main_plan=read(R/'PLAN.json');seal=read(R/'SEALED.json')
    assert not (R/'planner_oracle.jsonl').exists()
    assert sha(Path(__file__))==plan['source_sha256'] and sha(R/'PLAN.json')==plan['main_plan_sha256']
    assert all(sha(R/n)==h for n,h in seal['files'].items()) and all(sha(ROOT/n)==h for n,h in main_plan['dependencies'].items());verify()
    worlds=read(R/'dev_worlds.json');count=0
    for arm in main_plan['arms']:
        target=R/'repair'/arm;checkpoint={'path':target.relative_to(ROOT).as_posix(),'seal':read(target/'CHECKPOINT_SEALED.json')}
        tokenizer,model=load('probe',{'checkpoints':{'probe':checkpoint}});pool=palette(tokenizer)['fresh']
        for condition,ws in worlds.items():
            source=rows(target/(condition+'.jsonl'))
            for start in range(0,len(ws),2):
                batch=ws[start:start+2];texts=[];metadata=[]
                for j,w in enumerate(batch):
                    case=targets(w)[0];obs=case['observation'];assert obs['role']=='Planner'
                    record=source[start+j]['trace'][0];predicted={q['action']:q['estimate'] for q in record['queries']}
                    truth={c['action']:str(c['safe_completion']) for c in case['candidates']};nonce=seed_for(w['base_group_id'],0)
                    original,order=actor_text(obs,pool,nonce,predicted);text,new_order=actor_text(obs,pool,nonce,truth)
                    assert original==record['prompt'] and order==new_order==record['actions']
                    texts.append(text);metadata.append({'arm':arm,'condition':condition,'index':start+j,'world_id':w['world_id'],
                        'prompt':text,'actions':order,'truth':truth,'original_estimates':predicted,'identical_prompt':text==original})
                raws,seconds=generate(tokenizer,model,texts)
                with (R/'planner_oracle.jsonl').open('a') as f:
                    for meta,raw in zip(metadata,raws):f.write(json.dumps({**meta,'raw':raw,'seconds_batch':seconds})+'\n');count+=1
        print(json.dumps({'arm':arm,'planner_responses':112}),flush=True);del model;gc.collect()
    (R/'PLANNER_PROBE_SEALED.json').write_text(json.dumps({'responses':count,'raw_sha256':sha(R/'planner_oracle.jsonl'),
        'main_seal_sha256':sha(R/'SEALED.json'),'probe_plan_sha256':sha(R/'PLANNER_PROBE_PLAN.json'),'model_updates':0,
        'oracle_input':True,'learned_policy_evidence':False},indent=2))


if __name__=='__main__':main()
