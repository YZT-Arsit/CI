"""Frozen complete presentation factorial; seal before labels are scored."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'20261117_action_binding'))
from learned_evidence import *
sys.path.insert(0,str(HERE))
from render import query as factor_query,actor

def run():
    plan=json.loads((HERE/'PLAN.json').read_text());assert not (HERE/'RUN_PLAN.json').exists();assert hashlib.sha256((HERE/'render.py').read_bytes()).hexdigest()==plan['renderer_sha256'];assert hashlib.sha256((HERE/'cases.json').read_bytes()).hexdigest()==plan['cases_sha256'];cases=json.loads((HERE/'cases.json').read_text());folder=HERE/plan['initial_checkpoint'];seal=json.loads((folder/'CHECKPOINT_SEALED.json').read_text());assert all(hashlib.sha256((folder/'adapter'/p).read_bytes()).hexdigest()==h for p,h in seal['adapter_hashes'].items());(HERE/'RUN_PLAN.json').write_text(json.dumps({'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'adapter_hashes':seal['adapter_hashes'],'query_batch':4,'action_batch':2,'free_generation':True,'teacher_labels':False,'scope':plan['scope']},indent=2));start=time.perf_counter();tokenizer,model=setup(17);set_peft_model_state_dict(model,load_file(str(folder/'adapter/adapter_model.safetensors')));model.eval();pool=palette(tokenizer)['fresh'];(HERE/'SETUP.json').write_text(json.dumps({'seconds':time.perf_counter()-start,'cpu_only':True}))
    for group in range(16):
        pair=sorted((c for c in cases if c['group']==group),key=lambda c:c['side']);assert len(pair)==2
        for profile in plan['profiles']:
            code=''.join(str(profile[k]) for k in plan['factors']);seeds={k:pair[0]['nonces'][v] for k,v in profile.items()};seeds['links']=pair[0]['nonces'][0];texts=[];indices=[];estimates=[{},{}]
            for side,case in enumerate(pair):
                for action in case['observation']['actions']:texts.append(factor_query(case['observation'],pool,seeds,action));indices.append((side,action))
            lengths=encode(tokenizer,texts)['attention_mask'].sum(1).tolist();raws,seconds=generate(tokenizer,model,texts)
            for (side,action),text,raw,length in zip(indices,texts,raws,lengths):
                matches=re.findall(r'ANSWER:\s*([01])\b',raw);valid=len(matches)==1 and seconds<9.5;estimate=matches[0] if valid else 'unknown';estimates[side][action]=estimate
                with (HERE/'queries.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'group':group,'profile':code,'side':side,'action':action,'prompt':text,'prompt_tokens':length,'raw':raw,'valid':valid,'estimate':estimate,'seconds_batch':seconds})+'\n')
            texts=[];orders=[]
            for side,case in enumerate(pair):
                text,order=actor(case['observation'],pool,seeds,estimates[side]);texts.append(text);orders.append(order)
            lengths=encode(tokenizer,texts)['attention_mask'].sum(1).tolist();raws,seconds=generate(tokenizer,model,texts)
            for side,(text,raw,length) in enumerate(zip(texts,raws,lengths)):
                matches=re.findall(r'ANSWER:\s*([ABC])\b',raw);valid=len(matches)==1 and ord(matches[0])-65<len(orders[side]) and seconds<9.5;selected=orders[side][ord(matches[0])-65] if valid else None
                with (HERE/'actions.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'group':group,'profile':code,'side':side,'prompt':text,'prompt_tokens':length,'actions':orders[side],'raw':raw,'valid':valid,'selected':selected,'estimates':estimates[side],'seconds_batch':seconds})+'\n')
        print(json.dumps({'group_completed':group+1}),flush=True)
    (HERE/'RESPONSES_SEALED.json').write_text(json.dumps({'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.jsonl')},'labels_scored':False,'complete':True},indent=2));print('RESPONSES SEALED',flush=True)

if __name__=='__main__':run()
