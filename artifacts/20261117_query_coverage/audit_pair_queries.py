"""Supplementary all-pair query reversal diagnostic; never changes the primary gate."""
import collections
import json
import re
from probe_data import R,ROOT,read,rows,sha
from eval_worlds import targets


def main():
    registration=read(R/'PAIR_QUERY_PLAN.json');plan=read(R/'PLAN.json');seal=read(R/'SEALED.json')
    assert sha(R/'PLAN.json')==registration['main_plan_sha256'] and sha(R/'audit_pair_queries.py')==registration['source_sha256']
    assert seal['complete'] and all(sha(R/n)==h for n,h in seal['files'].items())
    assert all(sha(R/n)==h for n,h in {**plan['source_hashes'],**plan['data_hashes']}.items())
    assert all(sha(ROOT/n)==h for n,h in plan['dependencies'].items())
    assert sha(R/'query_labels.jsonl')==plan['data_hashes']['query_labels.jsonl']
    labels=rows(R/'query_labels.jsonl');index={(l['condition'],l['world_index'],l['state_index'],l['candidate_index']):l for l in labels}
    worlds=read(R/'dev_worlds.json');pairs=[]
    for condition in plan['paired_conditions']:
        other=condition+'_swap'
        for wi,(a,b) in enumerate(zip(worlds[condition],worlds[other])):
            for si,(ca,cb) in enumerate(zip(targets(a),targets(b))):
                assert ca['observation']['role']==cb['observation']['role'] and ca['observation']['committed_ticket']==cb['observation']['committed_ticket']
                for ci,(va,vb) in enumerate(zip(ca['candidates'],cb['candidates'])):
                    assert va['action']==vb['action'];la=index[condition,wi,si,ci];lb=index[other,wi,si,ci]
                    assert la['target']==va['safe_completion'] and lb['target']==vb['safe_completion']
                    if la['target']!=lb['target']:pairs.append((condition,wi,si,ci,la,lb))
    summaries=[]
    for arm in plan['arms']:
        raw=rows(R/'repair'/arm/'query_raw.jsonl');assert len(raw)==len(labels)
        by_id={r['id']:r for r in raw};assert len(by_id)==len(raw);stats=collections.defaultdict(collections.Counter);four=collections.defaultdict(list)
        def correct(l):
            r=by_id[l['id']];found=re.findall(r'ANSWER:\s*([01])\b',r['raw'])
            return len(found)==1 and r['seconds_batch']<9.5 and int(found[0])==l['target']
        for condition,wi,si,ci,a,b in pairs:
            x,y=correct(a),correct(b);family=condition.split('_v')[0]
            stats[family,a['role']].update(pairs=1,both_correct=int(x and y),individual_correct=int(x)+int(y))
            four[family,wi,si,ci].append(x and y)
        assert all(len(v)==2 for v in four.values())
        summaries.append({'arm':arm,'paired_queries':len(pairs),'both_sides_correct':sum(v['both_correct'] for v in stats.values()),
            'all_four_presentations':sum(all(v) for v in four.values()),'four_presentation_groups':len(four),
            'by_family_role':[{'family':f,'role':r,**dict(v)} for (f,r),v in stats.items()]})
    result={'summaries':summaries,'model_calls':0,'errors':0,'primary_gate_changed':False,
        'scope':registration['scope'],'constant_bit_both_correct':0,'whole_goal_complete':False}
    (R/'PAIR_QUERY_RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps({**result,'summaries':[{k:v for k,v in s.items() if k!='by_family_role'} for s in summaries]}))


if __name__=='__main__':main()
