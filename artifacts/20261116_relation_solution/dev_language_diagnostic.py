"""Read-only DEV diagnosis of final seed17 checkpoints; unique choices only."""
from evaluate_language import *
from audit_executions import proof_check

def run():
    out=R/'dev_language_diagnostic';out.mkdir(exist_ok=True);reports=[]
    data=load_states('dev');rows=[]
    for role in ('Planner','Source','Worker'):rows += [r for r in data if r['observation']['role']==role and len(r['optimal_indices'])==1][:8]
    for arm in ('plain','trace'):
        folder=R/'language_models'/f'{arm}_17';report=json.loads((folder/'sealed.json').read_text());tokenizer,model=setup(17);set_peft_model_state_dict(model,load_file(str(folder/'adapter/adapter_model.safetensors')));model.eval();pool=palette(tokenizer)['fresh'];records=[]
        for start in range(0,len(rows),4):
            selected=rows[start:start+4];texts=[];orders=[];nonces=[]
            for row in selected:
                nonce=seed_for(row['base_group_id'],299);nonces.append(nonce);text,actions=prompt(row['observation'],pool,nonce)
                text+=('\nGive each option SAFE (immediate), TASK (safe completion possible), PATH (may include mandatory future links), then ANSWER:\nletter.' if arm=='trace' else '\nReply ANSWER:\nletter.');texts.append(text);orders.append(actions)
            inputs=encode(tokenizer,texts);t=time.perf_counter()
            with torch.inference_mode():output=model.generate(**inputs,do_sample=False,max_new_tokens=256 if arm=='trace' else 16,max_time=8.,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
            seconds=time.perf_counter()-t
            for row,tokens,actions,nonce in zip(selected,output,orders,nonces):
                raw=tokenizer.decode(tokens[inputs['input_ids'].shape[1]:],skip_special_tokens=True);m=re.findall(r'ANSWER:\s*([ABC])\b',raw)
                chosen=actions[ord(m[0])-65] if len(m)==1 and ord(m[0])-65<len(actions) else None
                correct=chosen==row['candidates'][row['optimal_indices'][0]]['action'] and seconds<9.5
                records.append({'role':row['observation']['role'],'world_id':row['world_id'],'raw':raw,'selected':chosen,'correct':correct,'seconds_batch':seconds,
                    'proof':proof_check(raw,row['observation'],nonce,pool) if arm=='trace' else None})
        (out/(arm+'.jsonl')).write_text(''.join(canonical(x)+'\n' for x in records))
        summary={'arm':arm,'states':len(records),'roles':{role:{'n':sum(r['role']==role for r in records),'correct':sum(r['correct'] for r in records if r['role']==role)} for role in ('Planner','Source','Worker')},'proof_faithful':sum(all(r['proof'][k] for k in ('complete','labels_correct','paths_correct')) for r in records if r['proof'])};reports.append(summary);print(json.dumps(summary),flush=True)
        del model;import gc;gc.collect()
    (out/'RESULTS.json').write_text(json.dumps(reports,indent=2))

if __name__=='__main__':run()
