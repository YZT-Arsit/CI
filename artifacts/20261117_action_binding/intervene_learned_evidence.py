"""Post-training self-estimate interventions; text-channel diagnosis, not rollout."""
from learned_evidence import *
from experiments.relation_solution_data import targets
from kci.action_binding_metrics import pair_metrics

def run():
    out=RUN/'evidence_interventions';out.mkdir(exist_ok=True);assert not (out/'PLAN.json').exists();modes=['self','identity','swap_estimates','erase_estimates','irrelevant_tag'];(out/'PLAN.json').write_text(json.dumps({'modes':modes,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model':'learned_evidence evidence final','scope':'Planner-only same-DEV causal text-channel diagnostic, not internal activation interchange or full rollout','teacher_estimates':False,'conditions':['base','depth2','depth3'],'additional_note':'All modes reuse the same model-generated estimates; no resampling to improve them.'},indent=2))
    while not (RUN/'learned_evidence/RESULTS.json').exists():time.sleep(2)
    folder=RUN/'learned_evidence/evidence';seal=json.loads((folder/'CHECKPOINT_SEALED.json').read_text());assert all(hashlib.sha256((folder/'adapter'/p).read_bytes()).hexdigest()==h for p,h in seal['adapter_hashes'].items());worlds=json.loads((RUN/'learned_evidence/dev_worlds.json').read_text());tokenizer,model=setup(17);set_peft_model_state_dict(model,load_file(str(folder/'adapter/adapter_model.safetensors')));model.eval();pool=palette(tokenizer)['fresh'];outputs=[];queries=[]
    for condition in ('base','depth2','depth3'):
        for n,(a,b) in enumerate(zip(worlds[condition],worlds[condition+'_swap'])):
            pair=[targets(w)[0] for w in (a,b)];nonce=seed_for(a['base_group_id'],0);qtexts=[];index=[];estimates=[{},{}]
            for side,row in enumerate(pair):
                for action in row['observation']['actions']:
                    qtexts.append(query(tokenizer,row,action,pool,nonce)[0]);index.append((side,action))
            raws,seconds=generate(tokenizer,model,qtexts)
            for (side,action),text,raw in zip(index,qtexts,raws):
                matches=re.findall(r'ANSWER:\s*([01])\b',raw);valid=len(matches)==1 and seconds<9.5;estimate=matches[0] if valid else 'unknown';estimates[side][action]=estimate;queries.append({'condition':condition,'group':n,'side':side,'action':action,'prompt':text,'raw':raw,'estimate':estimate,'valid':valid,'seconds_batch':seconds})
            for mode in modes:
                texts=[];orders=[];used=[]
                for side,row in enumerate(pair):
                    actions=row['observation']['actions'];own=estimates[side]
                    changed=({actions[0]:own[actions[1]],actions[1]:own[actions[0]]} if mode=='swap_estimates' else {action:'unknown' for action in actions} if mode=='erase_estimates' else own)
                    text,order=actor_text(row['observation'],pool,nonce,changed)
                    if mode=='irrelevant_tag':text+='\nAudit tag: neutral.'
                    texts.append(text);orders.append(order);used.append(changed)
                raws,seconds=generate(tokenizer,model,texts)
                for side,row in enumerate(pair):
                    matches=re.findall(r'ANSWER:\s*([ABC])\b',raws[side]);valid=len(matches)==1 and ord(matches[0])-65<len(orders[side]) and seconds<9.5;selected=orders[side][ord(matches[0])-65] if valid else None
                    assert len(row['optimal_indices'])==1
                    outputs.append({'pair_id':a['base_group_id'],'condition':condition,'group':n,'side':side,'mode':mode,'selected':selected,'optimal_action':row['candidates'][row['optimal_indices'][0]]['action'],'actions':orders[side],'prompt':texts[side],'raw':raws[side],'estimates':used[side],'seconds_batch':seconds})
    for name,rows in [('queries',queries),('actions',outputs)]:
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    summaries=[]
    for condition in ('base','depth2','depth3'):
        for mode in modes:summaries.append({'condition':condition,'mode':mode,**pair_metrics([r for r in outputs if r['condition']==condition and r['mode']==mode])})
    own={(r['pair_id'],r['side']):r for r in outputs if r['mode']=='self'};identity=[r for r in outputs if r['mode']=='identity'];assert all(r['prompt']==own[(r['pair_id'],r['side'])]['prompt'] and r['raw']==own[(r['pair_id'],r['side'])]['raw'] for r in identity)
    result={'summaries':summaries,'identical_inputs_identical_raw_outputs':len(identity),'adapter_hashes':seal['adapter_hashes'],'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.jsonl')},'whole_goal_complete':False};(out/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':run()
