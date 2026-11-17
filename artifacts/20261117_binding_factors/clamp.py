"""Fix the model's own base-view judgments; vary presentation without teacher labels."""
from probe import *

def run_clamp():
    assert not (HERE/'CLAMP_PLAN.json').exists();plan=json.loads((HERE/'PLAN.json').read_text());seal=json.loads((HERE/'RESPONSES_SEALED.json').read_text());assert all(hashlib.sha256((HERE/p).read_bytes()).hexdigest()==h for p,h in seal['files'].items());cases=json.loads((HERE/'cases.json').read_text());estimates={}
    for row in map(json.loads,(HERE/'queries.jsonl').read_text().splitlines()):
        if row['profile']=='0000':estimates.setdefault((row['group'],row['side']),{})[row['action']]=row['estimate']
    (HERE/'CLAMP_PLAN.json').write_text(json.dumps({'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'queries_sha256':seal['files']['queries.jsonl'],'estimates_source':'fixed model-generated0000view; no groundtruth bits','outputs':512,'scope':'same-DEV causal mediation diagnostic, not a new learned method or independent test'},indent=2));folder=HERE/plan['initial_checkpoint'];weights=json.loads((folder/'CHECKPOINT_SEALED.json').read_text());assert all(hashlib.sha256((folder/'adapter'/p).read_bytes()).hexdigest()==h for p,h in weights['adapter_hashes'].items());tokenizer,model=setup(17);set_peft_model_state_dict(model,load_file(str(folder/'adapter/adapter_model.safetensors')));model.eval();pool=palette(tokenizer)['fresh']
    for group in range(16):
        pair=sorted((c for c in cases if c['group']==group),key=lambda c:c['side'])
        for profile in plan['profiles']:
            code=''.join(str(profile[k]) for k in plan['factors']);seeds={k:pair[0]['nonces'][v] for k,v in profile.items()};seeds['links']=pair[0]['nonces'][0];texts=[];orders=[]
            for side,case in enumerate(pair):
                text,order=actor(case['observation'],pool,seeds,estimates[group,side]);texts.append(text);orders.append(order)
            raws,seconds=generate(tokenizer,model,texts)
            for side,raw in enumerate(raws):
                matches=re.findall(r'ANSWER:\s*([ABC])\b',raw);valid=len(matches)==1 and ord(matches[0])-65<len(orders[side]) and seconds<9.5;selected=orders[side][ord(matches[0])-65] if valid else None
                with (HERE/'clamped_actions.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'group':group,'profile':code,'side':side,'prompt':texts[side],'actions':orders[side],'raw':raw,'valid':valid,'selected':selected,'estimates':estimates[group,side],'seconds_batch':seconds})+'\n')
        print(json.dumps({'group_completed':group+1}),flush=True)
    (HERE/'CLAMP_SEALED.json').write_text(json.dumps({'complete':True,'raw_sha256':hashlib.sha256((HERE/'clamped_actions.jsonl').read_bytes()).hexdigest()},indent=2))

if __name__=='__main__':run_clamp()
