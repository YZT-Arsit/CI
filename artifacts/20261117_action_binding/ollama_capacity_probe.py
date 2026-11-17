"""Small frozen CPU-only 7B capacity diagnostic, not a training-method result."""
import json,sys,time,hashlib,re,urllib.request
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];sys.path.insert(0,str(ROOT/'code'))
from experiments.receipt_dataset import make_world
from experiments.relation_solution_data import targets,intervene
from kci.relation_language import prompt,seed_for
from kci.action_binding_metrics import pair_metrics
from build_structural_cases import pair as internal_pair

MODEL='qwen2.5:7b-instruct-q4_K_M'
OPTIONS={'temperature':0,'seed':17,'num_ctx':2048,'num_predict':16,'num_gpu':0,'num_thread':4}

def request(endpoint,payload=None,timeout=25):
    body=None if payload is None else json.dumps(payload).encode();req=urllib.request.Request('http://localhost:11434/api/'+endpoint,data=body,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response:return json.load(response)

def run():
    out=R/'ollama_capacity';out.mkdir(exist_ok=True);assert not (out/'PLAN.json').exists();cases=[]
    for condition,depth in [('history1',1),('history3',3),('internal3',3)]:
        for n in range(2):
            seed=215000000+{'history1':0,'history3':100,'internal3':200}[condition]+n;kind=('meeting','query')[n]
            if condition=='internal3':worlds=internal_pair(seed,depth,kind)
            else:
                w=make_world(seed,kind,depth=depth,noise=2,mode='one',allow=False,required=True,delivered=True);w['base_group_id']=f'ollama_{seed}';worlds=[w,intervene(w,'swap')]
            for side,w in enumerate(worlds):
                row=targets(w)[0];row.update(condition=condition,side=side,pair_id=f'{condition}_{n}');cases.append(row)
    data=out/'cases.jsonl';data.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cases),encoding='utf-8');plan={'model':MODEL,'options':OPTIONS,'cases':len(cases),'pairs':6,'conditions':['history1','history3','internal3'],'scope':'new DEV Planner-only zero-shot capacity diagnostic; different base model/quantization, no causal comparison to the trained135M method','request_timeout_seconds':25,'warmup_timeout_seconds':60,'stop_on_first_request_failure':True,'dataset_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'teacher_evidence':False,'forced_legal_action_mask':False};(out/'PLAN.json').write_text(json.dumps(plan,indent=2))
    while not (R/'evidence_interventions/RESULTS.json').exists():time.sleep(2)
    metadata={'version':request('version'),'available':request('tags'),'model':request('show',{'model':MODEL}),'resident_before':request('ps')};assert any(m['name']==MODEL for m in metadata['available']['models']);(out/'MODEL_METADATA.json').write_text(json.dumps(metadata,indent=2));start=time.perf_counter();warmup=request('generate',{'model':MODEL,'prompt':'','stream':False,'keep_alive':'2m','options':OPTIONS},timeout=60);(out/'WARMUP.json').write_text(json.dumps({'wall_seconds':time.perf_counter()-start,'response':warmup,'resident_after':request('ps')},indent=2));outputs=[];error=None
    for row in cases:
        text,actions=prompt(row['observation'],[f'n{i:03d}' for i in range(200)],seed_for(row['pair_id'],971));text+='\nReply ANSWER:\nletter.';start=time.perf_counter()
        try:response=request('chat',{'model':MODEL,'messages':[{'role':'user','content':text}],'stream':False,'keep_alive':'2m','options':OPTIONS})
        except Exception as exc:
            error={'case':row['pair_id'],'side':row['side'],'type':type(exc).__name__,'detail':str(exc),'seconds':time.perf_counter()-start};(out/'REQUEST_FAILURE.json').write_text(json.dumps(error,indent=2));break
        seconds=time.perf_counter()-start;raw=response.get('message',{}).get('content','');matches=re.findall(r'ANSWER:\s*([ABC])\b',raw);valid=len(matches)==1 and ord(matches[0])-65<len(actions) and seconds<25;selected=actions[ord(matches[0])-65] if valid else None;assert len(row['optimal_indices'])==1
        record={'pair_id':row['pair_id'],'condition':row['condition'],'side':row['side'],'prompt':text,'actions':actions,'selected':selected,'optimal_action':row['candidates'][row['optimal_indices'][0]]['action'],'raw':raw,'seconds':seconds,'response':response};outputs.append(record)
        with (out/'raw.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        print(json.dumps({'condition':row['condition'],'side':row['side'],'seconds':seconds,'valid':valid,'correct':selected==record['optimal_action']}),flush=True)
    result={'complete':len(outputs)==len(cases),'completed_calls':len(outputs),'request_failure':error,'conditions':{c:pair_metrics([r for r in outputs if r['condition']==c]) for c in plan['conditions'] if len([r for r in outputs if r['condition']==c])==4},'whole_goal_complete':False,'model_release':'keep_alive expires two minutes after last request'};(out/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':run()
