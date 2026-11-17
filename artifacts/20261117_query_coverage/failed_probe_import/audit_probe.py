"""Native prompt/label reconstruction and paired action-transition scoring."""
import re
from collections import Counter,defaultdict
from probe_data import R,OLD,read,rows,sha,verify_prior,construct_batches
import json


def parse(raw,seconds,order):
    found=re.findall(r'ANSWER:\s*([ABC])\b',raw)
    valid=len(found)==1 and ord(found[0])-65<len(order) and seconds<9.5
    return order[ord(found[0])-65] if valid else None


def main():
    plan=read(R/'PROBE_PLAN.json');seal=read(R/'PROBE_SEALED.json');verify_prior()
    assert all(sha(R/n)==h for n,h in plan['sources'].items())
    assert sha(OLD/'PLAN.json')==plan['prior_plan_sha256'] and sha(OLD/'SEALED.json')==plan['prior_seal_sha256']
    assert sha(OLD/'FAILURE_LOCATIONS.json')==plan['prior_failure_locations_sha256']
    assert sha(R/'PROBE_PLAN.json')==seal['plan_sha256'] and sha(R/'probe_raw.jsonl')==seal['raw_sha256']
    batches=rows(R/'probe_batches.jsonl');assert sha(R/'probe_batches.jsonl')==plan['data_sha256']
    assert batches==construct_batches();raws=rows(R/'probe_raw.jsonl');assert len(raws)==seal['responses']==plan['responses']
    stats=defaultdict(Counter);offset=0;times=[];details=[]
    for i,batch in enumerate(batches):
        seconds=[]
        for j,case in enumerate(batch):
            record=raws[offset];offset+=1;assert (record['batch'],record['index'])==(i,j)
            order=case['actions'];old=parse(case['original_raw'],case['original_seconds'],order)
            chosen=parse(record['raw'],record['seconds_batch'],order);assert old==case['original_selected']
            label=case['case'];best={label['candidates'][k]['action'] for k in label['optimal_indices']}
            metrics=dict(states=1,self_optimal=int(old in best),oracle_optimal=int(chosen in best),
                wrong_to_correct=int(old not in best and chosen in best),correct_to_wrong=int(old in best and chosen not in best),
                invalid=int(chosen is None),identical_controls=int(case['identical_prompt']),
                identical_raw_disagreements=int(case['identical_prompt'] and record['raw']!=case['original_raw']))
            role=label['observation']['role'];stats[case['arm'],role,'all_reached'].update(metrics)
            if case['first_nonoptimal_here']:stats[case['arm'],role,'first_nonoptimal'].update(metrics)
            seconds.append(record['seconds_batch']);details.append({'arm':case['arm'],'condition':case['condition'],'index':case['index'],
                'role':role,'original':old,'oracle':chosen,'best':sorted(best),'first_nonoptimal_here':case['first_nonoptimal_here']})
        assert len(set(seconds))==1;times.append(seconds[0])
    summaries=[{'arm':a,'role':r,'subset':s,**dict(v)} for (a,r,s),v in stats.items()]
    gate=all(v['identical_controls']>0 and v['identical_raw_disagreements']==0 and v['invalid']==0 for (a,r,s),v in stats.items() if s=='all_reached')
    result={'summaries':summaries,'details':details,'responses':len(raws),'generation_seconds':sum(times),'max_seconds':max(times),
        'intervention_controls_passed':gate,'errors':0,'model_updates':0,'whole_goal_complete':False,'scope':plan['scope']}
    (R/'PROBE_RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='details'}))


if __name__=='__main__':main()
