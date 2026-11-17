"""Check which surface cues are held fixed in the new native paired worlds."""
import json
from probe_data import R,read,sha
from eval_worlds import targets
from build_degree_controls_v2 import profile


def main():
    plan=read(R/'PLAN.json');assert sha(R/'dev_worlds.json')==plan['data_hashes']['dev_worlds.json']
    worlds=read(R/'dev_worlds.json');records=[]
    for family in ['base','depth3','depth4','internal3','cycle4']:
        for index,(a,b) in enumerate(zip(worlds[family+'_v0'],worlds[family+'_v0_swap'])):
            ca,cb=targets(a)[0],targets(b)[0]
            best=lambda c:{c['candidates'][i]['action'] for i in c['optimal_indices']}
            assert len(best(ca))==len(best(cb))==1 and best(ca)!=best(cb)
            pa,pb=a['providers'],b['providers']
            same_counts=[len(p['links']) for p in pa]==[len(p['links']) for p in pb]
            same_entities=all(sorted(sum(x['links'],[]))==sorted(sum(y['links'],[])) for x,y in zip(pa,pb))
            assert same_counts and same_entities
            record={'family':family,'base_index':index,'unique_optimum_reverses':True,'provider_edge_counts_equal':same_counts,
                'provider_entity_multisets_equal':same_entities,'history_identical':a['histories']==b['histories']}
            if family=='internal3':
                assert a['histories']==b['histories'] and all(x['links'][0]==y['links'][0] and x['links'][-1]==y['links'][-1] for x,y in zip(pa,pb))
                record['first_and_last_edges_equal']=True
                record['degree_profiles_claimed_equal']=False
            if family=='cycle4':
                assert a['histories']==b['histories']
                for x,y in zip(pa,pb):
                    px,py=profile(x,a['histories']),profile(y,b['histories'])
                    assert px==py and px['cycle_lengths']==[4,4] and not px['history_to_ticket_edge'] and not px['ticket_to_history_edge']
                record.update(degrees_and_cycle_lengths_equal=True,direct_ticket_history_cues_absent=True)
            records.append(record)
    result={'base_pairs_checked':len(records),'records':records,'model_calls':0,'errors':0,
        'scope':'Generator-level controlled contrasts, not evidence any learned model uses relationships. Graph-depth intervention also changes entity/token exposure across training arms.'}
    (R/'SURFACE_CONTROL_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()
