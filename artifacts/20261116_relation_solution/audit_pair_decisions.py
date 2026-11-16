"""Separate first-route counterfactual response from later execution failures."""
import json,sys
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];sys.path.insert(0,str(ROOT/'code'))
from experiments.relation_solution_data import environment

def read(path):return list(map(json.loads,path.read_text(encoding='utf-8').splitlines()))

def run():
    records=[]
    for foldername in ('language_evaluation','balanced_language_evaluation'):
        folder=R/foldername;assert (folder/'EVALUATION_SEALED.json').exists()
        for model in sorted(p for p in folder.iterdir() if p.is_dir()):
            for split in ('test','depth10'):
                left={r['world_id']:r for r in read(model/(split+'.jsonl'))};right={r['world_id']:r for r in read(model/(split+'_swap.jsonl'))}
                worlds=[{w['world_id']:w for p in (R/'dataset'/name).glob('worlds_*.jsonl') for w in read(p)} for name in (split,split+'_swap')]
                valid=unchanged=correct=success=0
                assert left.keys()==right.keys()
                for key,a in left.items():
                    b=right[key];first=[row['trace'][0] for row in (a,b)]
                    assert first[0]['actions']==first[1]['actions'] and first[0]['nonce']==first[1]['nonce']
                    ok=all(t['format_valid'] for t in first);valid+=ok;unchanged+=ok and first[0]['selected']==first[1]['selected']
                    optimal=[]
                    for side in range(2):
                        e=environment(worlds[side][key]);s=e.initial();values={}
                        for action in e.actions(s):
                            after,_=e.transition(s,action);values[action]=tuple(e.exact(after)['value'])
                        optimal.append({action for action,value in values.items() if value==max(values.values())})
                    assert len(optimal[0])==len(optimal[1])==1 and optimal[0]!=optimal[1]
                    correct+=ok and all(first[i]['selected'] in optimal[i] for i in range(2));success+=a['safe_success'] and b['safe_success']
                records.append({'model':model.name,'condition':split,'groups':len(left),'both_planner_valid':valid,'same_route_when_both_valid':unchanged,'both_planner_optimal':correct,'both_episodes_safe_success':success})
    (R/'PAIRED_DECISION_DIAGNOSTIC.json').write_text(json.dumps({'records':records,'scope':'offline first-action scoring only; no tuning or intervention'},indent=2),encoding='utf-8')
    for row in records:print(json.dumps(row))

if __name__=='__main__':run()
