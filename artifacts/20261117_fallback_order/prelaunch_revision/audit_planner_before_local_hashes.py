"""Reconstruct exactly paired initial-state query and actor interventions."""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

R=Path(__file__).resolve().parent;ROOT=R.parents[1]
sys.path[:0]=[str(R.parent/'20261117_binding_factors'),str(ROOT/'code')]
from render import actor,query
from kci.relation_language import seed_for
from worlds import targets
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
rows=lambda p:list(map(json.loads,p.read_text().splitlines()))


def main():
    plan=read(R/'PLANNER_PROBE_PLAN.json');seal=read(R/'PLANNER_PROBE_SEALED.json');main_plan=read(R/'PLAN.json')
    assert sha(Path(__file__))==plan['auditor_sha256'] and sha(R/'planner_probe.py')==plan['source_sha256']
    assert sha(R/'PLAN.json')==plan['main_plan_sha256'] and sha(R/'PLANNER_PROBE_PLAN.json')==seal['probe_plan_sha256']
    assert sha(R/'planner_oracle.jsonl')==seal['raw_sha256'] and sha(R/'SEALED.json')==seal['main_seal_sha256']
    assert all(sha(R/n)==h for n,h in read(R/'SEALED.json')['files'].items())
    assert all(sha(ROOT/n)==h for n,h in main_plan['dependencies'].items())
    worldsets=read(R/'dev_worlds.json');pool=read(R.parent/'20261116_relation_solution/entity_palettes.json')['fresh']
    records=rows(R/'planner_oracle.jsonl');assert len(records)==seal['responses']==448
    timings=[];summaries=[];offset=0
    for arm in main_plan['arms']:
        stats=Counter();families={}
        for condition,ws in worldsets.items():
            source=rows(R/'repair'/arm/(condition+'.jsonl'));condition_stats=Counter()
            for index,(w,old) in enumerate(zip(ws,source)):
                row=records[offset];offset+=1;assert (row['arm'],row['condition'],row['index'],row['world_id'])==(arm,condition,index,w['world_id'])
                case=targets(w)[0];truth={c['action']:str(c['safe_completion']) for c in case['candidates']};entry=old['trace'][0]
                nonce=seed_for(w['base_group_id'],0);seeds={k:nonce for k in ['names','facts','plans','links','options']};predicted={}
                for q in entry['queries']:
                    assert q['prompt']==query(case['observation'],pool,seeds,q['action'])
                    found=re.findall(r'ANSWER:\s*([01])\b',q['raw']);valid=len(found)==1 and q['seconds_batch']<9.5
                    bit=found[0] if valid else 'unknown';assert bit==q['estimate'];predicted[q['action']]=bit
                    stats.update(queries=1,query_correct=int(bit==truth[q['action']]),query_invalid=int(not valid),
                                 false_negative=int(bit=='0' and truth[q['action']]=='1'),false_positive=int(bit=='1' and truth[q['action']]=='0'))
                text,order=actor(case['observation'],pool,seeds,truth);original,old_order=actor(case['observation'],pool,seeds,predicted)
                assert text==row['prompt'] and original==entry['prompt'] and order==old_order==row['actions']==entry['actions']
                assert row['truth']==truth and row['original_estimates']==predicted and row['identical_prompt']==(text==original)
                best={case['candidates'][j]['action'] for j in case['optimal_indices']}
                found=re.findall(r'ANSWER:\s*([ABC])\b',row['raw']);valid=len(found)==1 and ord(found[0])-65<len(order) and row['seconds_batch']<9.5
                chosen=order[ord(found[0])-65] if valid else None
                found_old=re.findall(r'ANSWER:\s*([ABC])\b',entry['raw']);valid_old=len(found_old)==1 and ord(found_old[0])-65<len(order) and entry['seconds_batch']<9.5
                old_chosen=order[ord(found_old[0])-65] if valid_old else None;assert old_chosen==entry['selected']
                stats.update(states=1,all_queries_correct=int(predicted==truth),self_actor_optimal=int(old_chosen in best),oracle_actor_optimal=int(chosen in best),
                    action_changed=int(chosen!=old_chosen),wrong_to_correct=int(old_chosen not in best and chosen in best),correct_to_wrong=int(old_chosen in best and chosen not in best),
                    identical_prompts=int(text==original),identical_prompt_raw_disagreements=int(text==original and row['raw']!=entry['raw']),oracle_invalid=int(not valid))
                condition_stats.update(states=1,self_optimal=int(old_chosen in best),oracle_optimal=int(chosen in best))
                if index%2==0:
                    batch=records[offset-1:offset+min(2,len(ws)-index)-1];assert len({r['seconds_batch'] for r in batch})==1;timings.append(row['seconds_batch'])
            families[condition]=dict(condition_stats)
        summaries.append({'arm':arm,**dict(stats),'conditions':families})
    result={'scope':'Same112initialPlannerstates per checkpoint; oracle-input diagnostic only, not full oracle rollouts or learned-policy success',
            'summaries':summaries,'responses':448,'generation_seconds':sum(timings),'max_seconds':max(timings),'model_updates':0,'errors':0,'whole_goal_complete':False}
    result['intervention_controls_passed']=all(s['identical_prompts']>0 and s['identical_prompt_raw_disagreements']==0 and s['oracle_invalid']==0 for s in summaries)
    (R/'PLANNER_PROBE_RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps({**result,'summaries':[{k:v for k,v in s.items() if k!='conditions'} for s in summaries]}))


if __name__=='__main__':main()
