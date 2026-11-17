"""Score only sealed responses; reconstruct prompts and independently check labels."""
import json,hashlib,re,sys
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];OLD=R.parent/'20261116_relation_solution';sys.path[:0]=[str(ROOT/'code'),str(OLD)]
from audit_data import check
from experiments.relation_solution_data import targets
from kci.relation_language import prompt,seed_for
from kci.action_binding_metrics import pair_metrics

def main():
    out=R/'alias_confirmation';plan=json.loads((out/'PLAN.json').read_text());seal=json.loads((out/'RESPONSES_SEALED.json').read_text());assert seal['complete'] and not seal['labels_scored'];assert all(hashlib.sha256((out/p).read_bytes()).hexdigest()==h for p,h in seal['files'].items());assert hashlib.sha256((R/'alias_confirmation.py').read_bytes()).hexdigest()==plan['source_sha256'];assert hashlib.sha256((out/'worlds.json').read_bytes()).hexdigest()==plan['worlds_sha256'];worlds=json.loads((out/'worlds.json').read_text(encoding='utf-8'));pool=json.loads((OLD/'entity_palettes.json').read_text())['fresh'];queries=list(map(json.loads,(out/'queries.jsonl').read_text().splitlines()));actions=list(map(json.loads,(out/'actions.jsonl').read_text().splitlines()));assert len(queries)==128 and len(actions)==192
    cases={};labels=0
    for group,pair in enumerate(worlds):
        for side,w in enumerate(pair):
            rows=targets(w);assert not [error for row in rows for error in check(row)];labels+=len(rows);cases[group,side]=rows[0]
    estimates={};query_correct=0;seen=set()
    for q in queries:
        key=(q['group'],q['render'],q['side']);row=cases[q['group'],q['side']];obs=row['observation'];nonce=seed_for(worlds[q['group']][0]['base_group_id'],900+q['render']);text,_=prompt(obs,pool,nonce,question='task',candidate=q['action']);text=text.replace('A yes; B no.','Reply ANSWER: followed by 1 if possible, otherwise 0.');assert q['prompt']==text;matches=re.findall(r'ANSWER:\s*([01])\b',q['raw']);valid=len(matches)==1 and q['seconds_batch']<9.5;estimate=matches[0] if valid else 'unknown';assert q['estimate']==estimate and q['valid']==valid;unique=(*key,q['action']);assert unique not in seen;seen.add(unique);estimates.setdefault(key,{})[q['action']]=estimate;query_correct+=estimate==str(next(c['safe_completion'] for c in row['candidates'] if c['action']==q['action']))
    records=[];index={}
    for r in actions:
        key=(r['group'],r['render'],r['side']);row=cases[r['group'],r['side']];obs=row['observation'];a=obs['actions'];own=estimates[key];used={a[0]:own[a[1]],a[1]:own[a[0]]} if r['mode']=='swap_estimates' else own;nonce=seed_for(worlds[r['group']][0]['base_group_id'],900+r['render']);text,order=prompt(obs,pool,nonce);text+='Own estimates (1 means compliant completion remains possible): '+ '; '.join(chr(65+i)+'='+used[a] for i,a in enumerate(order))+'.\n';text+='\nReply ANSWER:\nletter.';assert r['prompt']==text and r['actions']==order and r['estimates']==used;matches=re.findall(r'ANSWER:\s*([ABC])\b',r['raw']);valid=len(matches)==1 and ord(matches[0])-65<len(order) and r['seconds_batch']<9.5;selected=order[ord(matches[0])-65] if valid else None;assert r['selected']==selected and r['valid']==valid;assert len(row['optimal_indices'])==1;unique=(*key,r['mode']);assert unique not in index;index[unique]=r;records.append({**r,'pair_id':str(r['group']),'optimal_action':row['candidates'][row['optimal_indices'][0]]['action']})
    identities=0;effective=0;changed=0;effective_invalid=0
    for group in range(16):
        for render in range(2):
            for side in range(2):
                own=index[group,render,side,'self'];identity=index[group,render,side,'identity'];swap=index[group,render,side,'swap_estimates'];assert own['prompt']==identity['prompt'] and own['raw']==identity['raw'];identities+=1;effect=own['estimates']!=swap['estimates'];effective+=effect;changed+=effect and own['valid'] and swap['valid'] and own['selected']!=swap['selected'];effective_invalid+=effect and (not own['valid'] or not swap['valid'])
    same=0;both_correct=0;both_invalid=0
    for group in range(16):
        for side in range(2):
            a=index[group,0,side,'self'];b=index[group,1,side,'self'];assert a['prompt']!=b['prompt'];same+=a['valid'] and b['valid'] and a['selected']==b['selected'];both_invalid+=not a['valid'] and not b['valid'];good=cases[group,side];truth=good['candidates'][good['optimal_indices'][0]]['action'];both_correct+=a['selected']==b['selected']==truth
    result={'queries':128,'query_correct':query_correct,'actions':192,'independent_label_states':labels,'identical_raw_identity_controls':identities,'effective_estimate_swaps':effective,'valid_action_changes_among_effective_swaps':changed,'effective_swaps_with_invalid_output':effective_invalid,'same_valid_semantic_action_across_two_renders':same,'both_renders_invalid':both_invalid,'correct_in_both_renders':both_correct,'render_consistency_denominator':32,'paired_results':[{'render':render,'mode':mode,**pair_metrics([r for r in records if r['render']==render and r['mode']==mode])} for render in range(2) for mode in plan['modes']],'errors':0,'scope':plan['scope'],'whole_goal_complete':False};(out/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
