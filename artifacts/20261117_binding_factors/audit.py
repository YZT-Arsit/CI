"""Reconstruct factorial inputs and attribute changes after response sealing."""
import hashlib,itertools,json,re,sys
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];OLD=R.parent/'20261116_relation_solution';PREV=R.parent/'20261117_action_binding';sys.path[:0]=[str(ROOT/'code'),str(OLD)]
from audit_data import check
from experiments.relation_solution_data import targets
from kci.action_binding_metrics import pair_metrics
from render import query,actor

def main():
    plan=json.loads((R/'PLAN.json').read_text());run=json.loads((R/'RUN_PLAN.json').read_text());seal=json.loads((R/'RESPONSES_SEALED.json').read_text());assert seal['complete'] and not seal['labels_scored'];assert all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in seal['files'].items());assert hashlib.sha256((R/'probe.py').read_bytes()).hexdigest()==run['source_sha256'];assert hashlib.sha256((R/'render.py').read_bytes()).hexdigest()==plan['renderer_sha256'];cases=json.loads((R/'cases.json').read_text());assert hashlib.sha256((R/'cases.json').read_bytes()).hexdigest()==plan['cases_sha256'];pool=json.loads((OLD/'entity_palettes.json').read_text())['fresh'];lookup={(c['group'],c['side']):c for c in cases};labels={};checked=0
    for key,case in lookup.items():
        rows=targets(case['world']);assert not [e for row in rows for e in check(row)];labels[key]=rows[0];checked+=len(rows)
    queries=list(map(json.loads,(R/'queries.jsonl').read_text().splitlines()));actions=list(map(json.loads,(R/'actions.jsonl').read_text().splitlines()));assert len(queries)==1024 and len(actions)==512;estimates={};qindex={};aindex={};scored=[]
    def seeds(case,profile):return {**{k:case['nonces'][int(v)] for k,v in zip(plan['factors'],profile)},'links':case['nonces'][0]}
    for q in queries:
        key=(q['group'],q['side']);case=lookup[key];assert q['prompt']==query(case['observation'],pool,seeds(case,q['profile']),q['action']);matches=re.findall(r'ANSWER:\s*([01])\b',q['raw']);valid=len(matches)==1 and q['seconds_batch']<9.5;estimate=matches[0] if valid else 'unknown';assert q['valid']==valid and q['estimate']==estimate;unique=(*key,q['profile'],q['action']);assert unique not in qindex;qindex[unique]=q;estimates.setdefault((*key,q['profile']),{})[q['action']]=estimate;q['correct']=estimate==str(next(c['safe_completion'] for c in labels[key]['candidates'] if c['action']==q['action']))
    for a in actions:
        key=(a['group'],a['side']);case=lookup[key];text,order=actor(case['observation'],pool,seeds(case,a['profile']),estimates[(*key,a['profile'])]);assert a['prompt']==text and a['actions']==order and a['estimates']==estimates[(*key,a['profile'])];matches=re.findall(r'ANSWER:\s*([ABC])\b',a['raw']);valid=len(matches)==1 and ord(matches[0])-65<len(order) and a['seconds_batch']<9.5;selected=order[ord(matches[0])-65] if valid else None;assert a['valid']==valid and a['selected']==selected;unique=(*key,a['profile']);assert unique not in aindex;aindex[unique]=a;row=labels[key];assert len(row['optimal_indices'])==1;truth=row['candidates'][row['optimal_indices'][0]]['action'];a['correct']=selected==truth;scored.append({**a,'pair_id':str(a['group']),'optimal_action':truth})
    profiles=[''.join(map(str,bits)) for bits in itertools.product((0,1),repeat=4)];summaries=[{'profile':profile,'query_correct':sum(q['correct'] for q in queries if q['profile']==profile),'query_count':64,**pair_metrics([a for a in scored if a['profile']==profile])} for profile in profiles];effects=[];option_query_identities=0
    for factor,name in enumerate(plan['factors']):
        comparisons=0;changed=0;query_comparisons=0;query_changes=0;invalid=0;active_actions=0;active_queries=0;changed_with_same_estimates=0;affected_groups=set()
        for profile in profiles:
            if profile[factor]!='0':continue
            other=profile[:factor]+'1'+profile[factor+1:]
            for key,case in lookup.items():
                a=aindex[(*key,profile)];b=aindex[(*key,other)];comparisons+=1;effect=a['valid'] and b['valid'] and a['selected']!=b['selected'];changed+=effect;invalid+=not(a['valid'] and b['valid']);active_actions+=a['prompt']!=b['prompt'];changed_with_same_estimates+=effect and a['estimates']==b['estimates']
                if a['prompt']!=b['prompt']:affected_groups.add(key[0])
                else:assert a['raw']==b['raw']
                for candidate in case['observation']['actions']:
                    x=qindex[(*key,profile,candidate)];y=qindex[(*key,other,candidate)];query_comparisons+=1;query_changes+=x['estimate']!=y['estimate'];active_queries+=x['prompt']!=y['prompt']
                    if x['prompt']==y['prompt']:assert x['raw']==y['raw']
                    if name=='options':assert x['prompt']==y['prompt'] and x['raw']==y['raw'];option_query_identities+=1
        effects.append({'factor':name,'matched_action_comparisons':comparisons,'effective_actor_prompt_changes':active_actions,'no_op_actor_controls':comparisons-active_actions,'affected_world_pairs':len(affected_groups),'valid_semantic_action_changes':changed,'changed_action_with_same_own_estimates':changed_with_same_estimates,'comparisons_with_invalid_action':invalid,'matched_query_comparisons':query_comparisons,'effective_query_prompt_changes':active_queries,'no_op_query_controls':query_comparisons-active_queries,'changed_query_estimates':query_changes,'interpretation':'actor changes include direct presentation and query-mediated effects; comparisons share16world pairs'})
    old_queries=list(map(json.loads,(PREV/'alias_confirmation/queries.jsonl').read_text().splitlines()));old_actions=[r for r in map(json.loads,(PREV/'alias_confirmation/actions.jsonl').read_text().splitlines()) if r['mode']=='self'];replayed=0
    for old in old_queries:
        now=qindex[(old['group'],old['side'],str(old['render'])*4,old['action'])];assert old['prompt']==now['prompt'] and old['raw']==now['raw'];replayed+=1
    for old in old_actions:
        now=aindex[(old['group'],old['side'],str(old['render'])*4)];assert old['prompt']==now['prompt'] and old['raw']==now['raw'];replayed+=1
    robust_states=sum(all(aindex[(*key,p)]['correct'] for p in profiles) for key in lookup);robust_pairs=sum(all(aindex[(group,side,p)]['correct'] for side in (0,1) for p in profiles) for group in range(16));result={'profiles':summaries,'matched_effects':effects,'robust_states_all16views':robust_states,'state_denominator':32,'robust_world_pairs_all16views':robust_pairs,'world_pair_denominator':16,'previous_raws_exactly_replayed':replayed,'identical_query_raws_for_option_only_changes':option_query_identities,'independent_label_states':checked,'query_token_lengths':sorted({q['prompt_tokens'] for q in queries}),'action_token_lengths':sorted({a['prompt_tokens'] for a in actions}),'max_batch_seconds':max(r['seconds_batch'] for r in queries+actions),'errors':0,'scope':plan['scope'],'whole_goal_complete':False};(R/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
