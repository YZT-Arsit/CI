"""Frozen one-hop name/order and self-estimate intervention check; no training."""
from learned_evidence import *

def run():
    out=RUN/'alias_confirmation';out.mkdir(exist_ok=True);assert not (out/'PLAN.json').exists();worlds=[]
    for n in range(16):
        w=make_world(250000000+n,('meeting','query')[n%2],depth=1,noise=2,mode='one',allow=False,required=True,delivered=True);w['base_group_id']=w['world_id'];worlds.append([w,intervene(w,'swap')])
    data=out/'worlds.json';data.write_text(json.dumps(worlds,ensure_ascii=False),encoding='utf-8');folder=RUN/'coverage_repair/class_balanced';seal=json.loads((folder/'CHECKPOINT_SEALED.json').read_text());plan={'pairs':16,'renders':2,'modes':['self','identity','swap_estimates'],'seed':17,'checkpoint':'coverage_repair/class_balanced','adapter_hashes':seal['adapter_hashes'],'worlds_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'new-world one-hop DEV confirmation; two renderings reuse worlds and are not independent task samples; Planner only, not full episode or source holdout','teacher_estimates':False,'labels_read_before_sealing':False};(out/'PLAN.json').write_text(json.dumps(plan,indent=2))
    while not (RUN/'risk_repair/RESULTS.json').exists():time.sleep(2)
    assert all(hashlib.sha256((folder/'adapter'/p).read_bytes()).hexdigest()==h for p,h in seal['adapter_hashes'].items());tokenizer,model=setup(17);set_peft_model_state_dict(model,load_file(str(folder/'adapter/adapter_model.safetensors')));model.eval();pool=palette(tokenizer)['fresh']
    for group,pair in enumerate(worlds):
        observations=[environment(w).observation(environment(w).initial()) for w in pair]
        for render in range(2):
            nonce=seed_for(pair[0]['base_group_id'],900+render);qtexts=[];indices=[];estimates=[{},{}]
            for side,obs in enumerate(observations):
                for action in obs['actions']:qtexts.append(query(tokenizer,{'observation':obs},action,pool,nonce)[0]);indices.append((side,action))
            raws,seconds=generate(tokenizer,model,qtexts)
            for (side,action),text,raw in zip(indices,qtexts,raws):
                matches=re.findall(r'ANSWER:\s*([01])\b',raw);valid=len(matches)==1 and seconds<9.5;estimate=matches[0] if valid else 'unknown';estimates[side][action]=estimate
                with (out/'queries.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'group':group,'render':render,'side':side,'action':action,'prompt':text,'raw':raw,'estimate':estimate,'valid':valid,'seconds_batch':seconds})+'\n')
            for mode in plan['modes']:
                texts=[];orders=[];used=[]
                for side,obs in enumerate(observations):
                    a=obs['actions'];own=estimates[side];changed={a[0]:own[a[1]],a[1]:own[a[0]]} if mode=='swap_estimates' else own;text,order=actor_text(obs,pool,nonce,changed);texts.append(text);orders.append(order);used.append(changed)
                raws,seconds=generate(tokenizer,model,texts)
                for side,raw in enumerate(raws):
                    matches=re.findall(r'ANSWER:\s*([ABC])\b',raw);valid=len(matches)==1 and ord(matches[0])-65<len(orders[side]) and seconds<9.5;selected=orders[side][ord(matches[0])-65] if valid else None
                    with (out/'actions.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'group':group,'render':render,'side':side,'mode':mode,'prompt':texts[side],'actions':orders[side],'raw':raw,'selected':selected,'valid':valid,'estimates':used[side],'seconds_batch':seconds})+'\n')
        print(json.dumps({'group_completed':group+1}),flush=True)
    (out/'RESPONSES_SEALED.json').write_text(json.dumps({'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.jsonl')},'complete':True,'labels_scored':False},indent=2));print('RESPONSES SEALED',flush=True)

if __name__=='__main__':run()
