"""Post-freeze capacity stress, reported separately; no tuning on these outcomes."""
from lm_common import *
import hashlib,time
from experiments.receipt_dataset import make_world
from experiments.relation_solution_data import environment
from evaluate_structured import execute
from kci.relation_solution import LocalRulePolicy,encode as relation_encode,collate
from train_structured import short

def run():
    out=R/'scaling_diagnostic';out.mkdir(exist_ok=True);assert not (out/'plan.json').exists()
    plan={'depths':[2,10],'noise':[0,12,32,64,120,128],'worlds_per_cell':16,'seeds':[17,29,41],'timing_batch_sizes':[1,4,16,64],
        'new_world_seed_start':30000000,'purpose':'post-freeze scaling diagnosis, not a replacement main test or model tuning','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'plan.json').write_text(json.dumps(plan,indent=2));summaries=[];allrows=[]
    for seed in plan['seeds']:
        model=LocalRulePolicy();model.load_state_dict(torch.load(R/'structured_models'/f'local_{seed}'/'model.pt',weights_only=True));model.eval()
        for depth in plan['depths']:
            for noise in plan['noise']:
                rows=[];ledger=[]
                for j in range(16):
                    w=make_world(30000000+depth*10000+noise*100+j,depth=depth,noise=noise);w['base_group_id']=w['world_id']
                    try:result=short(lambda:execute(model,'local',w,811),ledger,'episode');row={'seed':seed,'depth':depth,'noise':noise,'status':'executed',**result}
                    except ValueError as e:row={'seed':seed,'depth':depth,'noise':noise,'world_id':w['world_id'],'status':'capacity_rejected','reason':str(e),'safe_success':False,'ci':None}
                    rows.append(row)
                summary={'seed':seed,'depth':depth,'noise':noise,'episodes':16,'success':sum(r['safe_success'] for r in rows),'capacity_rejected':sum(r['status']!='executed' for r in rows),'violations':sum(r['ci'] is False for r in rows),'max_seconds':max(r['seconds'] for r in ledger)}
                summaries.append(summary);allrows+=rows;print(json.dumps(summary),flush=True)
    (out/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in allrows))
    timing=[];model=LocalRulePolicy();model.load_state_dict(torch.load(R/'structured_models/local_17/model.pt',weights_only=True));model.eval()
    w=make_world(31900000,depth=2,noise=120);envcase=environment(w);obs=envcase.observation(envcase.initial());item=relation_encode(obs,obs['actions'][0])
    for batch in (1,4,16,64):
        data=collate([item]*batch);ledger=[]
        with torch.inference_mode():short(lambda:model(*data),ledger,'inference')
        timing.append({'batch':batch,'facts':int(item[3].sum()),'seconds':ledger[0]['seconds']})
    names=json.loads((R/'entity_palettes.json').read_text())['fresh'];capacity=[]
    for depth in plan['depths']:
        for noise in plan['noise']:
            w=make_world(32000000+depth*10000+noise,depth=depth,noise=noise);e=environment(w)
            try:text,_=prompt(e.observation(e.initial()),names,211);record={'depth':depth,'noise':noise,'renderable':True,'characters':len(text)}
            except ValueError as error:record={'depth':depth,'noise':noise,'renderable':False,'reason':str(error)}
            capacity.append(record)
    (out/'RESULTS.json').write_text(json.dumps({'summaries':summaries,'timing':timing,'language_formatter_capacity':capacity,'main_gate_not_redefined':True},indent=2))

if __name__=='__main__':run()
