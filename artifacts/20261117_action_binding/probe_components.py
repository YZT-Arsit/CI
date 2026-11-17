"""Post-hoc DEV intervention: separate TASK bits from correlated PATH text."""
import probe_frozen as common
from probe_frozen import *

def run():
    out=RUN/'components';out.mkdir(exist_ok=True);assert not (out/'PLAN.json').exists();common.RUN=out
    models=['trace_17','balanced_17'];modes=['task_flipped','path_swapped','path_erased','task_zero'];plan={'models':models,'modes':modes,'scope':'post-hoc same-DEV teacher-prefix component intervention; not autonomous performance','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'PLAN.json').write_text(json.dumps(plan,indent=2));rows=[r for r in map(json.loads,(RUN/'cases.jsonl').read_text(encoding='utf-8').splitlines()) if r['split']=='new_dev'];groups={}
    for row in rows:groups.setdefault(row['pair_id'],[]).append(row)
    summaries=[]
    for name in models:
        tokenizer,model=setup(17);folder=OLD/('balanced_language_models' if name.startswith('balanced') else 'language_models')/name
        seal=json.loads((folder/'sealed.json').read_text());assert all(hashlib.sha256((folder/'adapter'/p).read_bytes()).hexdigest()==h for p,h in seal['adapter_hashes'].items())
        set_peft_model_state_dict(model,load_file(str(folder/'adapter/adapter_model.safetensors')));model.eval();pool=palette(tokenizer)['fresh'];letters=answer_ids(tokenizer)
        for mode in modes:
            records=[]
            for pair_id,pair in groups.items():
                nonce=seed_for(pair_id,7331);items=[]
                for i,row in enumerate(pair):
                    item=inputs_for(tokenizer,row,pair[1-i],pool,'trace','gold_trace',nonce);other=inputs_for(tokenizer,row,pair[1-i],pool,'trace','swapped_trace',nonce)
                    prefix=item['prefix']
                    if mode=='task_flipped':prefix=re.sub(r'TASK ([01])',lambda m:'TASK '+str(1-int(m[1])),prefix)
                    elif mode=='path_swapped':
                        paths=iter(re.findall(r'PATH ([^;\n]+);',other['prefix']));prefix=re.sub(r'PATH [^;\n]+;',lambda m:'PATH '+next(paths)+';',prefix)
                    elif mode=='path_erased':prefix=re.sub(r'PATH [^;\n]+;','PATH none;',prefix)
                    else:prefix=re.sub(r'TASK [01]','TASK 0',prefix)
                    suffix=tokenizer.encode(prefix,add_special_tokens=False);item.update(ids=item['ids'][:item['base_length']]+suffix,suffix_length=len(suffix),prefix=prefix);items.append(item)
                logits,_,seconds=infer(model,tokenizer,items)
                for i,(row,item) in enumerate(zip(pair,items)):
                    allowed=letters[:len(item['actions'])];top=int(logits[i,-1].argmax());selected=item['actions'][allowed.tolist().index(top)] if top in allowed.tolist() else None
                    records.append({'model':name,'mode':mode,'pair_id':pair_id,'side':row['side'],'selected':selected,'optimal_action':row['optimal_action'],'actions':item['actions'],'prompt':item['prompt'],'assistant_prefix':item['prefix'],'full_vocab_top_id':top,'full_vocab_top_token':tokenizer.decode([top]),'legal_probabilities':logits[i,-1].softmax(-1)[allowed].tolist(),'seconds_batch':seconds})
            path=out/(name+'_'+mode+'.jsonl');path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
            from kci.action_binding_metrics import pair_metrics
            result={'model':name,'mode':mode,**pair_metrics(records),'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest()};summaries.append(result);print(json.dumps(result),flush=True)
        del model;gc.collect()
    (out/'RESULTS.json').write_text(json.dumps({'summaries':summaries,'scope':plan['scope']},indent=2))

if __name__=='__main__':run()
