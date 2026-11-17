"""Audit text-channel intervention scope and the separate Ollama capacity probe."""
import json,sys,re,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];OLD=R.parent/'20261116_relation_solution';sys.path[:0]=[str(OLD),str(ROOT/'code')]
from experiments.relation_solution_data import targets
from kci.relation_language import prompt,seed_for
from kci.action_binding_metrics import pair_metrics
from audit_data import check

def run():
    folder=R/'evidence_interventions';result=json.loads((folder/'RESULTS.json').read_text());worlds=json.loads((R/'learned_evidence/dev_worlds.json').read_text());pool=json.loads((OLD/'entity_palettes.json').read_text())['fresh'];actions=list(map(json.loads,(folder/'actions.jsonl').read_text().splitlines()));queries=list(map(json.loads,(folder/'queries.jsonl').read_text().splitlines()));assert len(actions)==240 and len(queries)==96;own={};estimates={};expected={}
    for name,digest in result['files'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest
    for q in queries:
        key=(q['condition'],q['group'],q['side']);w=worlds[q['condition']+('_swap' if q['side'] else '')][q['group']];row=targets(w)[0];assert not check(row);expected[key]=row;text,_=prompt(row['observation'],pool,seed_for(w['base_group_id'],0),question='task',candidate=q['action']);text=text.replace('A yes; B no.','Reply ANSWER: followed by 1 if possible, otherwise 0.');assert q['prompt']==text;matches=re.findall(r'ANSWER:\s*([01])\b',q['raw']);valid=len(matches)==1 and q['seconds_batch']<9.5;estimate=matches[0] if valid else 'unknown';assert q['valid']==valid and q['estimate']==estimate;estimates.setdefault(key,{})[q['action']]=estimate
    for record in actions:
        key=(record['condition'],record['group'],record['side']);row=expected[key];obs=row['observation'];a=obs['actions'];original=estimates[key];mode=record['mode'];used={a[0]:original[a[1]],a[1]:original[a[0]]} if mode=='swap_estimates' else {x:'unknown' for x in a} if mode=='erase_estimates' else original;text,order=prompt(obs,pool,seed_for(row['base_group_id'],0));text+='Own estimates (1 means compliant completion remains possible): '+ '; '.join(chr(65+i)+'='+used[x] for i,x in enumerate(order))+'.\n';text+='\nReply ANSWER:\nletter.'
        if mode=='irrelevant_tag':text+='\nAudit tag: neutral.'
        assert record['prompt']==text and record['actions']==order and record['estimates']==used;matches=re.findall(r'ANSWER:\s*([ABC])\b',record['raw']);valid=len(matches)==1 and ord(matches[0])-65<len(order) and record['seconds_batch']<9.5;selected=order[ord(matches[0])-65] if valid else None;assert record['selected']==selected and record['optimal_action']==row['candidates'][row['optimal_indices'][0]]['action']
        if mode=='self':own[key]=record
    effective=[]
    for summary in result['summaries']:
        rows=[r for r in actions if r['condition']==summary['condition'] and r['mode']==summary['mode']];assert pair_metrics(rows)=={k:v for k,v in summary.items() if k not in ('condition','mode')};changed=[r for r in rows if r['estimates']!=own[(r['condition'],r['group'],r['side'])]['estimates']]
        effective.append({'condition':summary['condition'],'mode':summary['mode'],'states':len(rows),'changed_estimate_content':len(changed),'changed_action_given_content_change':sum(r['selected']!=own[(r['condition'],r['group'],r['side'])]['selected'] for r in changed),'changed_prompt':sum(r['prompt']!=own[(r['condition'],r['group'],r['side'])]['prompt'] for r in rows)})
    cap=R/'ollama_capacity';cases=list(map(json.loads,(cap/'cases.jsonl').read_text().splitlines()));records=list(map(json.loads,(cap/'raw.jsonl').read_text().splitlines()));cap_result=json.loads((cap/'RESULTS.json').read_text());assert len(cases)==len(records)==12;lengths=[]
    for case,record in zip(cases,records):
        assert not check(case);text,order=prompt(case['observation'],[f'n{i:03d}' for i in range(200)],seed_for(case['pair_id'],971));assert record['prompt']==text+'\nReply ANSWER:\nletter.' and record['actions']==order;matches=re.findall(r'ANSWER:\s*([ABC])\b',record['raw']);valid=len(matches)==1 and ord(matches[0])-65<len(order) and record['seconds']<25;assert record['selected']==(order[ord(matches[0])-65] if valid else None);assert record['optimal_action']==case['candidates'][case['optimal_indices'][0]]['action']
    for condition,summary in cap_result['conditions'].items():assert pair_metrics([r for r in records if r['condition']==condition])==summary
    for a,b in zip(records[::2],records[1::2]):assert a['pair_id']==b['pair_id'];lengths.append({'pair_id':a['pair_id'],'prompt_tokens':[a['response']['prompt_eval_count'],b['response']['prompt_eval_count']]})
    audit={'intervention_queries_checked':len(queries),'intervention_action_outputs_checked':len(actions),'effective_interventions':effective,'capacity_outputs_checked':len(records),'capacity_pair_prompt_lengths':lengths,'errors':0};(R/'CHANNEL_CAPACITY_AUDIT.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))

if __name__=='__main__':run()
