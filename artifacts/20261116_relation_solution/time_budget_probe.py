"""Post-hoc 8s->12s sensitivity on capped invalid batches, unchanged token limit."""
from evaluate_language import *

def batches(folder):
    selected=[]
    conditions=CONDITIONS+(['external_complete','external_complete_swap'] if folder.parent.name=='balanced_language_evaluation' else [])
    for condition in conditions:
        rows=[json.loads(l) for l in (folder/(condition+'.jsonl')).read_text().splitlines()]
        for phase in range(3):
            for start in range(0,len(rows),4):
                entries=[(r,next((t for t in r['trace'] if t['phase']==phase),None)) for r in rows[start:start+4]];entries=[(r,t) for r,t in entries if t is not None]
                if any(not t['format_valid'] and (t['seconds_batch']>=7.8 or t['generated_tokens']>=256) for r,t in entries):
                    selected.append((condition,phase,start,entries))
                    if len(selected)==2:return selected
    return selected

def run():
    out=R/'time_budget_probe';out.mkdir(exist_ok=True);assert not (out/'plan.json').exists()
    plan={'models':'all trace and balanced final checkpoints','selection':'first two capped invalid batches per model in frozen condition/phase/batch order; never selected by task reward','max_time_seconds':12,'max_new_tokens':256,'original_time_seconds':8,'original_token_limit':256,'same_prompts_and_batch_members':True,'purpose':'supplemental timing sensitivity, not replacement primary score','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    while not (R/'language_native_transfer/EVALUATION_SEALED.json').exists():time.sleep(2)
    summaries=[]
    for modeldir,evaluation in [('language_models','language_evaluation'),('balanced_language_models','balanced_language_evaluation')]:
        for report in json.loads((R/modeldir/'CHECKPOINTS_SEALED.json').read_text())['models']:
            if report['arm']!='trace':continue
            chosen=batches(R/evaluation/report['model']);tokenizer,model=setup(report['seed']);adapter=R/modeldir/report['model']/'adapter'
            assert all(hashlib.sha256((adapter/p).read_bytes()).hexdigest()==h for p,h in report['adapter_hashes'].items());set_peft_model_state_dict(model,load_file(str(adapter/'adapter_model.safetensors')));model.eval();records=[]
            for condition,phase,start,entries in chosen:
                prompts=[t['prompt'] for r,t in entries];inputs=encode(tokenizer,prompts);t0=time.perf_counter()
                with torch.inference_mode():output=model.generate(**inputs,do_sample=False,max_new_tokens=256,max_time=12.,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
                elapsed=time.perf_counter()-t0;raws=[tokenizer.decode(x[inputs['input_ids'].shape[1]:],skip_special_tokens=True) for x in output]
                with (out/(report['model']+'_raw.jsonl')).open('a',encoding='utf-8') as f:
                    for (row,trace),raw in zip(entries,raws):f.write(canonical({'condition':condition,'phase':phase,'world_id':row['world_id'],'prompt':trace['prompt'],'old_raw':trace['raw'],'new_raw':raw,'seconds_batch':elapsed})+'\n')
                ws={w['world_id']:w for w in worlds('test' if condition=='test_seen' else condition)}
                for (row,trace),raw in zip(entries,raws):
                    match=re.findall(r'ANSWER:\s*([ABC])\b',raw);valid=len(match)==1 and ord(match[0])-65<len(trace['actions']) and elapsed<13.5;action=trace['actions'][ord(match[0])-65] if valid else None
                    e=environment(ws[row['world_id']]);state=e.initial()
                    for old in row['trace']:
                        if old['phase']==phase:break
                        state,_=e.transition(state,old['selected'])
                    values={}
                    for a in e.actions(state):after,audit=e.transition(state,a);values[a]=tuple(e.exact(after)['value'])
                    correct=valid and values[action]==max(values.values())
                    records.append({'condition':condition,'phase':phase,'world_id':row['world_id'],'old_valid':trace['format_valid'],'new_valid':valid,'recovered_format':not trace['format_valid'] and valid,'new_optimal_action':correct,'selected':action,'old_output_is_prefix':raw.startswith(trace['raw']),'seconds_batch':elapsed})
            (out/(report['model']+'.jsonl')).write_text(''.join(canonical(r)+'\n' for r in records))
            summary={'model':report['model'],'batches':len(chosen),'decisions':len(records),'old_invalid':sum(not r['old_valid'] for r in records),'recovered_format':sum(r['recovered_format'] for r in records),'recovered_and_optimal':sum(r['recovered_format'] and r['new_optimal_action'] for r in records),'prefix_consistent':sum(r['old_output_is_prefix'] for r in records),'max_seconds':max((r['seconds_batch'] for r in records),default=0)}
            summaries.append(summary);print(json.dumps(summary),flush=True);del model;import gc;gc.collect()
    (out/'RESULTS.json').write_text(json.dumps({'summaries':summaries,'primary_results_unchanged':True,'no_training_or_model_selection':True},indent=2))

if __name__=='__main__':run()
